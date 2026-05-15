"""
JAX Profiling: Delaunay Datacube Likelihood (Step-by-Step)
==========================================================

Profiles each step of the JAX likelihood function for an ALMA-style datacube —
a list of N ``Interferometer`` channels sharing a single lens model — where
each channel reconstructs its own source with a Delaunay pixelization +
ConstantSplit regularization.

Mirrors the step-by-step structure of
``jax_profiling/jit/interferometer/delaunay.py`` (Phase 2 of the datacube
roadmap, just merged). The key new ingredient is the **channel-invariant vs
channel-variant** split: most steps are computed once for the whole cube
(shared lens, shared mesh, shared mask), only the NUFFT-based inversion-setup
chain, the data vector, the curvature matrix, the reconstruction, and the
log-evidence depend on per-channel data.

The cube total is::

    cube_cost = sum(channel_invariant_costs) + N_channels * sum(channel_variant_costs)

That number quantifies how much the deferred shared-``Lᵀ W̃ L`` optimisation
will save: moving the curvature matrix from per-channel to shared would
subtract ``(N - 1) * curvature_matrix_cost`` from the cube total.

Channel-invariant vs channel-variant taxonomy
---------------------------------------------

For the canonical datacube case where the lens model is shared across all
channels:

============================================  ================  =========================
Step                                          Channel-invariant Computed
============================================  ================  =========================
1. Ray-trace data grid                        yes               once for the cube
2. Ray-trace mesh grid                        yes               once for the cube
3. Inversion setup (border + mapper + NUFFT)  **NUFFT depends   once per channel
                                              on uv_wavelengths**
4. Data vector D                              per channel       once per channel
5. Curvature matrix F                         per channel       once per channel
6. Regularization matrix H                    yes               once for the cube
7. Reconstruction (NNLS)                      per channel       once per channel
8. Mapped recon + log evidence                per channel       once per channel
============================================  ================  =========================

Dataset
-------

This profiler reuses the SMA interferometer dataset
(``jax_profiling/dataset/interferometer/sma/``) loaded N times as a 4-channel
"cube". Each channel has identical visibilities, noise map and uv_wavelengths
— the point here is timing, not science. The N-channel cube log-evidence is
``N × single-channel log-evidence`` exactly, which makes the regression
assertion trivial.

If you want a realistic per-channel-distinct cube, point the loader at the
workspace simulator output at
``../autolens_workspace/dataset/interferometer/datacube/sim_simple/``; the
JIT-cost taxonomy doesn't change because it's a function of which arrays are
loop-variables in ``FitInterferometer``, not the data values themselves.

Measures
--------

1. Eager baseline: ``FitInterferometer`` per channel with ``xp=np``; cube
   reference log-evidence is the sum.
2. Per-step JIT profiling: each pipeline stage gets its own ``jit_profile()``
   call (lower / compile / first-call / steady-state × 10). Channel-invariant
   stages are timed once; channel-variant stages are timed on channel 0 and
   the cube cost is reported as ``N × per-call``.
3. Full-pipeline cube JIT: ``jax.jit`` over the explicit
   ``sum(analysis.log_likelihood_function(instance) for analysis in
   analysis_list)`` — the same shape as the user-facing
   ``datacube/likelihood_function.py`` and the cube modeling scripts'
   internal ``FactorGraphModel`` sum.
4. Correctness: per-step recomputed cube log-evidence and full-pipeline JIT
   log-evidence both match the summed eager ``FitInterferometer.log_evidence``
   at ``rtol=1e-4``.
5. Results JSON + bar chart written to ``results/jit/datacube/`` using the
   same schema as the interferometer sibling. Bar chart shows the cube-total
   form of every step (channel-variant entries pre-multiplied by N).

vmap is **skipped** for the cube profiler. The natural batching dimension is
"datasets" (one entry per channel) not "parameters" (which the
interferometer-sibling vmap exercises). A vmap-over-channels variant would
require a different graph shape and isn't the bottleneck we care about for
the shared-``Lᵀ W̃ L`` optimisation.
"""

import numpy as np
import jax
import jax.numpy as jnp
import os
import time
import subprocess
import sys
from pathlib import Path
from contextlib import contextmanager

import autofit as af
import autolens as al
import autoarray as aa
from autofit.jax import register_model as _register_model_pytrees

# ---------------------------------------------------------------------------
# Instrument configuration
# ---------------------------------------------------------------------------

INSTRUMENTS = {
    "sma": {"pixel_scale": 0.1, "real_space_shape": (256, 256), "mask_radius": 3.0},
    "alma": {"pixel_scale": 0.05, "real_space_shape": (256, 256), "mask_radius": 3.0},
    "hannah": {"pixel_scale": 0.125, "real_space_shape": (40, 40), "mask_radius": 2.3},
}

instrument = "hannah"  # <-- realistic ALMA settings for Hannah's science case

# n_channels = 34 matches Hannah's real ALMA cube. For quick iteration on the
# smaller SMA dataset, drop this to 4 (also flip ``instrument`` back to "sma").
n_channels = 34
overlay_shape = (26, 26)
edge_n_points = 30
regularization_coefficient = 1.0


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------

class Timer:
    """Accumulates named timing measurements and prints a summary."""

    def __init__(self):
        self.records: list[tuple[str, float]] = []

    @contextmanager
    def section(self, label: str):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.records.append((label, elapsed))
        print(f"  [{label}] {elapsed:.4f} s")


def block(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def jit_profile(func, label, *args, n_repeats=10):
    """JIT-compile *func*, time lower / compile / first call / steady state."""
    jitted = jax.jit(func)

    with timer.section(f"{label}_lower"):
        lowered = jitted.lower(*args)

    with timer.section(f"{label}_compile"):
        compiled = lowered.compile()

    with timer.section(f"{label}_first_call"):
        result = compiled(*args)
        block(result)

    with timer.section(f"{label}_steady_x{n_repeats}"):
        for _ in range(n_repeats):
            result = compiled(*args)
            block(result)

    per_call = timer.records[-1][1] / n_repeats
    print(f"    -> per-call avg: {per_call:.6f} s")
    return compiled, result


timer = Timer()
likelihood_steps = []  # (label, per_call_seconds) — cube-total cost for each step

# ===================================================================
# PART A — Setup (not JIT-compiled)
# ===================================================================

# ---------------------------------------------------------------------------
# 1. Dataset loading: reuse SMA interferometer dataset N times
# ---------------------------------------------------------------------------

print(f"\n--- Dataset loading [{instrument}, {n_channels} channels] ---")

_script_dir = Path(__file__).resolve().parent
_workspace_root = _script_dir.parents[2]
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
real_space_shape = INSTRUMENTS[instrument]["real_space_shape"]
dataset_path = Path("jax_profiling") / "dataset" / "interferometer" / instrument

if al.util.dataset.should_simulate(str(dataset_path)):
    print(f"  Simulating {instrument} dataset...")
    subprocess.run(
        [
            sys.executable,
            str(_workspace_root / "jax_profiling" / "dataset_setup" / "interferometer.py"),
            "--instrument", instrument,
        ],
        cwd=str(_workspace_root),
        check=True,
    )

mask_radius = INSTRUMENTS[instrument]["mask_radius"]

real_space_mask = al.Mask2D.circular(
    shape_native=real_space_shape,
    pixel_scales=pixel_scale,
    radius=mask_radius,
)

with timer.section("dataset_list_load"):
    dataset_list = [
        al.Interferometer.from_fits(
            data_path=dataset_path / "data.fits",
            noise_map_path=dataset_path / "noise_map.fits",
            uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
            real_space_mask=real_space_mask,
            transformer_class=al.TransformerDFT,
            # DFT is intentional even at ALMA-scale visibility counts — profiling
            # the JAX-traceable path is the goal, NUFFT (pynufft) is not yet
            # JIT-friendly.
            raise_error_dft_visibilities_limit=False,
        )
        for _ in range(n_channels)
    ]

n_visibilities = dataset_list[0].uv_wavelengths.shape[0]
print(f"  Channels:           {n_channels}")
print(f"  Visibilities/chan:  {n_visibilities}")

# ---------------------------------------------------------------------------
# 2. Image mesh + edge points (Delaunay-specific, channel-invariant)
# ---------------------------------------------------------------------------

print("\n--- Image mesh construction (Delaunay) ---")

with timer.section("image_mesh_overlay"):
    image_mesh = al.image_mesh.Overlay(shape=overlay_shape)
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset_list[0].real_space_mask
    )

with timer.section("edge_points"):
    pre_edge_pixels = image_plane_mesh_grid.shape[0]
    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=(0.0, 0.0),
        radius=mask_radius,
        n_points=edge_n_points,
    )
    edge_pixels_total = image_plane_mesh_grid.shape[0] - pre_edge_pixels

n_mesh_vertices = image_plane_mesh_grid.shape[0]
print(f"  Overlay shape: {overlay_shape}")
print(f"  Mesh vertices (incl. edge): {n_mesh_vertices}")
print(f"  Edge points added: {edge_pixels_total}")

# ---------------------------------------------------------------------------
# 3. Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

with timer.section("model_build"):
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    _lens_mass_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=_lens_mass_ell[0], sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=_lens_mass_ell[1], sigma=0.01)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
    shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

    mesh = al.mesh.Delaunay(
        pixels=n_mesh_vertices,
        zeroed_pixels=edge_pixels_total,
    )
    regularization = al.reg.ConstantSplit(coefficient=regularization_coefficient)
    pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Delaunay pixels: {n_mesh_vertices}")

# ---------------------------------------------------------------------------
# 4. Instantiate concrete objects from prior medians
# ---------------------------------------------------------------------------

print("\n--- Instantiate concrete model ---")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

with timer.section("register_pytrees"):
    _register_model_pytrees(model)

params_tree = jax.tree_util.tree_map(jnp.asarray, instance)

tracer = al.Tracer(galaxies=list(instance.galaxies))

# The adapt_images object is channel-invariant — the image-plane Delaunay mesh
# vertices are shared across channels (the lens model is shared).
adapt_images = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={
        instance.galaxies.source: image_plane_mesh_grid,
    },
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid,
    },
)

print(f"  Tracer planes: {tracer.total_planes}")

# ---------------------------------------------------------------------------
# 5. Configuration summary
# ---------------------------------------------------------------------------

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {instrument}")
print(f"  Channels:                {n_channels}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Real-space mask radius:  {mask_radius} arcsec")
print(f"  Real-space grid shape:   {real_space_shape[0]} x {real_space_shape[1]}")
print(f"  Visibilities/chan:       {n_visibilities}")
print(f"  Overlay shape:           {overlay_shape[0]} x {overlay_shape[1]}")
print(f"  Delaunay vertices:       {n_mesh_vertices}")
print(f"  Edge zeroed pixels:      {edge_pixels_total}")
print(f"  Reg. coefficient:        {regularization_coefficient}")

# ---------------------------------------------------------------------------
# 6. Per-channel eager FitInterferometer baseline
# ---------------------------------------------------------------------------

print(f"\n--- Per-channel eager FitInterferometer baselines ({n_channels} channels) ---")

fit_list = []
log_evidence_per_channel = []
with timer.section(f"eager_fit_per_channel_x{n_channels}"):
    for c, dataset in enumerate(dataset_list):
        f = al.FitInterferometer(
            dataset=dataset,
            tracer=tracer,
            adapt_images=adapt_images,
            xp=np,
        )
        fit_list.append(f)
        log_evidence_per_channel.append(f.log_evidence)

for c, le in enumerate(log_evidence_per_channel):
    print(f"  channel {c}: log_evidence = {le:.6f}")

cube_log_evidence_ref = float(sum(log_evidence_per_channel))
print(f"  cube reference log_evidence (sum) = {cube_log_evidence_ref:.6f}")


# ===================================================================
# PART B — Per-step JIT profiling (channel-invariant + channel-variant)
# ===================================================================

print("\n" + "=" * 70)
print("PER-STEP JIT PROFILING — CUBE")
print("=" * 70)
print(f"  Channel-invariant steps are timed once.")
print(f"  Channel-variant steps are JIT-compiled on channel 0; the reported")
print(f"  cube cost is N × the per-channel steady-state per-call.")

# Reference single-channel context (channel 0)
fit = fit_list[0]
dataset = dataset_list[0]

# Extract raw arrays from autoarray types via .array so they can cross
# JIT boundaries. See CLAUDE.md for rationale.
grid_pix_raw = jnp.array(dataset.grids.pixelization.array)
mesh_grid_raw = jnp.array(image_plane_mesh_grid.array)

# ---------------------------------------------------------------------------
# Step 1: Ray-trace data grid (channel-invariant)
# ---------------------------------------------------------------------------

print("\n--- Step 1: Ray-trace data grid (shared) ---")

with timer.section("ray_trace_data_eager"):
    traced_grids = tracer.traced_grid_2d_list_from(
        grid=dataset.grids.pixelization, xp=jnp
    )
    for tg in traced_grids:
        block(tg)


def ray_trace_data_raw(grid_raw):
    grid = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    traced = tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.stack([tg.array for tg in traced])


_, _ = jit_profile(ray_trace_data_raw, "ray_trace_data_jit", grid_pix_raw)
ray_trace_data_per_call = timer.records[-1][1] / 10
likelihood_steps.append(
    ("Ray-trace data grid (shared)", ray_trace_data_per_call)
)

# ---------------------------------------------------------------------------
# Step 2: Ray-trace mesh grid (channel-invariant)
# ---------------------------------------------------------------------------

print("\n--- Step 2: Ray-trace mesh grid (shared) ---")


def ray_trace_mesh_raw(mesh_raw):
    grid = aa.Grid2DIrregular(values=mesh_raw, xp=jnp)
    traced = tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.stack([tg.array for tg in traced])


_, _ = jit_profile(ray_trace_mesh_raw, "ray_trace_mesh_jit", mesh_grid_raw)
ray_trace_mesh_per_call = timer.records[-1][1] / 10
likelihood_steps.append(
    ("Ray-trace mesh grid (shared)", ray_trace_mesh_per_call)
)

# ---------------------------------------------------------------------------
# Extract inversion matrices from channel 0
# ---------------------------------------------------------------------------

print("\n--- Extracting inversion matrices from channel 0 ---")

inversion = fit.inversion

with timer.section("extract_inversion_matrices"):
    transformed_mm_ref = jnp.asarray(inversion.operated_mapping_matrix)
    mapping_matrix_ref = jnp.asarray(inversion.mapping_matrix)

    inv_mapper = inversion.cls_list_from(cls=al.Mapper)[0]
    neighbors = inv_mapper.neighbors
    neighbors_array = jnp.array(np.asarray(neighbors))
    neighbors_sizes = jnp.array(neighbors.sizes)

print(f"  transformed_mapping_matrix shape: {transformed_mm_ref.shape}")
print(f"  transformed_mapping_matrix dtype: {transformed_mm_ref.dtype}")
print(f"  mapping_matrix shape: {mapping_matrix_ref.shape}")

# ---------------------------------------------------------------------------
# Step 3: Inversion setup (per channel — NUFFT depends on uv_wavelengths)
# ---------------------------------------------------------------------------
# Steps 5-8 from the interferometer-sibling numbering (border + Delaunay +
# mapper + mapping matrix + NUFFT), combined and JIT-profiled from a pytree
# ModelInstance. Channel-variant because each channel's NUFFT uses its own
# uv_wavelengths. JIT-compile on channel 0; report cube cost as N × per-call.

print("\n--- Step 3: Inversion setup, incl. NUFFT (per channel) ---")


def transformed_mm_from_params(params_tree):
    """Inversion setup from a pytree ModelInstance — full chain through NUFFT.

    This closes over ``dataset`` (channel 0) for the JIT compilation. In real
    cube usage each channel's `AnalysisFactor` closes over its own
    `dataset`, so the steady-state per-call cost is what we want to scale by N.
    """
    t = al.Tracer(galaxies=list(params_tree.galaxies))
    adapt_images_jax = al.AdaptImages(
        galaxy_image_plane_mesh_grid_dict={
            params_tree.galaxies.source: image_plane_mesh_grid,
        },
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid,
        },
    )
    fit_jax = al.FitInterferometer(
        dataset=dataset,
        tracer=t,
        adapt_images=adapt_images_jax,
        xp=jnp,
    )
    return jnp.asarray(fit_jax.inversion.operated_mapping_matrix)


_, transformed_mm_jit = jit_profile(
    transformed_mm_from_params, "inversion_setup_jit", params_tree
)
inversion_setup_per_channel = timer.records[-1][1] / 10
likelihood_steps.append(
    (
        f"Inversion setup, incl. NUFFT (per channel × {n_channels})",
        n_channels * inversion_setup_per_channel,
    )
)

print(f"  per-channel: {inversion_setup_per_channel:.6f} s")
print(f"  cube cost (× {n_channels}): {n_channels * inversion_setup_per_channel:.6f} s")

# Use the reference real / imag arrays for the linear-algebra steps
transformed_mm_real_jnp = jnp.real(transformed_mm_ref)
transformed_mm_imag_jnp = jnp.imag(transformed_mm_ref)
data_real_jnp = jnp.array(dataset.data.real)
data_imag_jnp = jnp.array(dataset.data.imag)
noise_real_jnp = jnp.array(dataset.noise_map.real)
noise_imag_jnp = jnp.array(dataset.noise_map.imag)

# ---------------------------------------------------------------------------
# Step 4: Data vector D (per channel)
# ---------------------------------------------------------------------------

print("\n--- Step 4: Data vector D (per channel) ---")


def compute_data_vector(
    transformed_mm_real, transformed_mm_imag, data_real, data_imag,
    noise_real, noise_imag,
):
    weighted_data_real = data_real / (noise_real ** 2)
    weighted_data_imag = data_imag / (noise_imag ** 2)
    return jnp.matmul(transformed_mm_real.T, weighted_data_real) + jnp.matmul(
        transformed_mm_imag.T, weighted_data_imag
    )


with timer.section("data_vector_eager"):
    data_vector = compute_data_vector(
        transformed_mm_real_jnp, transformed_mm_imag_jnp,
        data_real_jnp, data_imag_jnp, noise_real_jnp, noise_imag_jnp,
    )
    block(data_vector)

_, data_vector = jit_profile(
    compute_data_vector, "data_vector_jit",
    transformed_mm_real_jnp, transformed_mm_imag_jnp,
    data_real_jnp, data_imag_jnp, noise_real_jnp, noise_imag_jnp,
)
data_vector_per_channel = timer.records[-1][1] / 10
likelihood_steps.append(
    (
        f"Data vector D (per channel × {n_channels})",
        n_channels * data_vector_per_channel,
    )
)

# ---------------------------------------------------------------------------
# Step 5: Curvature matrix F (per channel)
# ---------------------------------------------------------------------------

print("\n--- Step 5: Curvature matrix F (per channel) ---")

no_reg_list = list(inversion.no_regularization_index_list)


def compute_curvature_matrix(
    transformed_mm_real, transformed_mm_imag, noise_real, noise_imag,
):
    real_curv = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=transformed_mm_real,
        noise_map=noise_real,
        settings=fit.settings,
        add_to_curvature_diag=True,
        no_regularization_index_list=no_reg_list,
        xp=jnp,
    )
    imag_curv = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=transformed_mm_imag,
        noise_map=noise_imag,
        settings=fit.settings,
        add_to_curvature_diag=False,
        no_regularization_index_list=no_reg_list,
        xp=jnp,
    )
    return real_curv + imag_curv


with timer.section("curvature_matrix_eager"):
    curvature_matrix = compute_curvature_matrix(
        transformed_mm_real_jnp, transformed_mm_imag_jnp, noise_real_jnp, noise_imag_jnp,
    )
    block(curvature_matrix)

_, curvature_matrix = jit_profile(
    compute_curvature_matrix, "curvature_matrix_jit",
    transformed_mm_real_jnp, transformed_mm_imag_jnp, noise_real_jnp, noise_imag_jnp,
)
curvature_matrix_per_channel = timer.records[-1][1] / 10
likelihood_steps.append(
    (
        f"Curvature matrix F (per channel × {n_channels})",
        n_channels * curvature_matrix_per_channel,
    )
)

# ---------------------------------------------------------------------------
# Step 6: Regularization matrix H (channel-invariant)
# ---------------------------------------------------------------------------

print("\n--- Step 6: Regularization matrix H (shared) ---")

with timer.section("regularization_matrix_eager"):
    regularization_matrix = jnp.array(inversion.regularization_matrix)
    block(regularization_matrix)

likelihood_steps.append(
    ("Regularization matrix H (shared)", timer.records[-1][1])
)

# ---------------------------------------------------------------------------
# Step 7: Reconstruction NNLS (per channel)
# ---------------------------------------------------------------------------

print("\n--- Step 7: Regularized reconstruction (per channel) ---")


def compute_reconstruction(data_vector, curvature_matrix, regularization_matrix):
    curvature_reg_matrix = curvature_matrix + regularization_matrix
    return al.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
        xp=jnp,
    )


with timer.section("reconstruction_eager"):
    reconstruction = compute_reconstruction(
        jnp.array(data_vector),
        jnp.array(curvature_matrix),
        jnp.array(regularization_matrix),
    )
    block(reconstruction)

_, reconstruction = jit_profile(
    compute_reconstruction, "reconstruction_jit",
    jnp.array(data_vector), jnp.array(curvature_matrix), jnp.array(regularization_matrix),
)
reconstruction_per_channel = timer.records[-1][1] / 10
likelihood_steps.append(
    (
        f"Reconstruction NNLS (per channel × {n_channels})",
        n_channels * reconstruction_per_channel,
    )
)

# ---------------------------------------------------------------------------
# Step 8: Mapped recon + log evidence (per channel)
# ---------------------------------------------------------------------------

print("\n--- Step 8: Mapped recon + log evidence (per channel) ---")


def compute_log_evidence(
    data_real, data_imag, noise_real, noise_imag,
    transformed_mm_real, transformed_mm_imag,
    reconstruction, curvature_matrix, regularization_matrix, mapper_indices,
):
    mapped_real = jnp.matmul(transformed_mm_real, reconstruction)
    mapped_imag = jnp.matmul(transformed_mm_imag, reconstruction)

    chi_real = jnp.sum(((data_real - mapped_real) / noise_real) ** 2)
    chi_imag = jnp.sum(((data_imag - mapped_imag) / noise_imag) ** 2)
    chi_squared = chi_real + chi_imag

    regularization_term = jnp.dot(
        reconstruction, jnp.dot(regularization_matrix, reconstruction)
    )

    curvature_reg_matrix = curvature_matrix + regularization_matrix
    creg_reduced = curvature_reg_matrix[mapper_indices][:, mapper_indices]
    reg_reduced = regularization_matrix[mapper_indices][:, mapper_indices]
    log_det_curvature_reg = 2.0 * jnp.sum(
        jnp.log(jnp.diag(jnp.linalg.cholesky(creg_reduced)))
    )
    log_det_regularization = 2.0 * jnp.sum(
        jnp.log(jnp.diag(jnp.linalg.cholesky(reg_reduced)))
    )

    noise_normalization = (
        jnp.sum(jnp.log(2 * jnp.pi * noise_real ** 2))
        + jnp.sum(jnp.log(2 * jnp.pi * noise_imag ** 2))
    )

    return -0.5 * (
        chi_squared + regularization_term + log_det_curvature_reg
        - log_det_regularization + noise_normalization
    )


mapper_indices_jnp = jnp.array(np.asarray(inversion.mapper_indices))
inv_recon_jnp = jnp.asarray(inversion.reconstruction)
inv_curv_jnp = jnp.asarray(inversion.curvature_matrix)
reg_jnp = jnp.array(regularization_matrix)

with timer.section("log_evidence_eager"):
    log_evidence_one_channel = compute_log_evidence(
        data_real_jnp, data_imag_jnp, noise_real_jnp, noise_imag_jnp,
        transformed_mm_real_jnp, transformed_mm_imag_jnp,
        reconstruction, curvature_matrix, reg_jnp, mapper_indices_jnp,
    )
    block(log_evidence_one_channel)

_, log_evidence_one_channel = jit_profile(
    compute_log_evidence, "log_evidence_jit",
    data_real_jnp, data_imag_jnp, noise_real_jnp, noise_imag_jnp,
    transformed_mm_real_jnp, transformed_mm_imag_jnp,
    reconstruction, curvature_matrix, reg_jnp, mapper_indices_jnp,
)
log_evidence_per_channel_cost = timer.records[-1][1] / 10
likelihood_steps.append(
    (
        f"Mapped recon + log evidence (per channel × {n_channels})",
        n_channels * log_evidence_per_channel_cost,
    )
)

print(f"  channel 0 log_evidence (step-by-step) = {log_evidence_one_channel}")

# Correctness check: recompute per-channel log_evidence using each channel's
# inversion matrices and sum to get the cube log-evidence. Should match the
# summed eager FitInterferometer.log_evidence at rtol=1e-4.
log_evidence_check_per_channel = []
for c, f in enumerate(fit_list):
    inv_c = f.inversion
    tm_real = jnp.real(jnp.asarray(inv_c.operated_mapping_matrix))
    tm_imag = jnp.imag(jnp.asarray(inv_c.operated_mapping_matrix))
    le_c = compute_log_evidence(
        jnp.array(dataset_list[c].data.real),
        jnp.array(dataset_list[c].data.imag),
        jnp.array(dataset_list[c].noise_map.real),
        jnp.array(dataset_list[c].noise_map.imag),
        tm_real, tm_imag,
        jnp.asarray(inv_c.reconstruction),
        jnp.asarray(inv_c.curvature_matrix),
        jnp.array(inv_c.regularization_matrix),
        jnp.array(np.asarray(inv_c.mapper_indices)),
    )
    log_evidence_check_per_channel.append(float(le_c))

cube_log_evidence_check = float(sum(log_evidence_check_per_channel))
print(
    f"\n  cube log_evidence (per-step recompute) = {cube_log_evidence_check:.6f}"
)
print(f"  cube log_evidence (reference)          = {cube_log_evidence_ref:.6f}")

np.testing.assert_allclose(
    cube_log_evidence_check,
    cube_log_evidence_ref,
    rtol=1e-4,
    err_msg=(
        "Per-step cube log_evidence does not match summed FitInterferometer.log_evidence"
    ),
)
print(
    "  Assertion PASSED: per-step cube log_evidence matches summed "
    "FitInterferometer.log_evidence at rtol=1e-4"
)


# ===================================================================
# PART C — Full-pipeline cube JIT (sum of per-channel log_likelihoods)
# ===================================================================

print("\n" + "=" * 70)
print("FULL-PIPELINE CUBE JIT (for comparison)")
print("=" * 70)

# Part C is expensive at large n_channels: lower + compile build a graph
# proportional to n_channels (e.g. ~70s for n_channels=34 on a laptop CPU),
# and the steady-state first-call follows. Default to skipping; opt in with
# CUBE_FULL_JIT=1 when the full-pipeline timing matters (e.g. comparing
# step-by-step total against single-JIT).
_run_full_cube_jit = os.environ.get("CUBE_FULL_JIT") == "1"

if _run_full_cube_jit:
    analysis_list = [
        al.AnalysisInterferometer(dataset=d, adapt_images=adapt_images, use_jax=True)
        for d in dataset_list
    ]

    def full_cube_pipeline_from_params(params_tree):
        """Cube log-evidence via the explicit per-channel sum.

        Same shape as the user-facing ``datacube/likelihood_function.py``:
        feeds the shared instance to every per-channel
        ``AnalysisInterferometer.log_likelihood_function`` and sums.
        """
        total = jnp.zeros(())
        for analysis in analysis_list:
            total = total + analysis.log_likelihood_function(instance=params_tree)
        return total

    _full_cube_n_repeats = 3
    _, full_cube_result = jit_profile(
        full_cube_pipeline_from_params,
        "full_cube_pipeline",
        params_tree,
        n_repeats=_full_cube_n_repeats,
    )
    full_pipeline_per_call = timer.records[-1][1] / _full_cube_n_repeats

    print(f"  full cube log_evidence (JIT) = {full_cube_result}")

    np.testing.assert_allclose(
        float(full_cube_result),
        cube_log_evidence_ref,
        rtol=1e-4,
        err_msg="Full-pipeline cube JIT log_evidence does not match summed eager FitInterferometer.log_evidence",
    )
    print("  Eager-vs-JIT cube correctness PASSED")
else:
    full_cube_result = None
    full_pipeline_per_call = float("nan")
    print(
        "  Full-pipeline cube JIT SKIPPED — opt-in via CUBE_FULL_JIT=1. "
        f"At n_channels={n_channels} the lower + compile alone is on the order of "
        f"{n_channels * 2}-{n_channels * 3}s, so it's gated to keep the default "
        "runtime usable; the per-step Part B JIT data above is what feeds the "
        "shared-Lᵀ W̃ L analysis."
    )

# ===================================================================
# PART D — vmap (skipped for cube)
# ===================================================================
#
# The natural batching axis for a cube fit is "datasets" (one entry per
# channel), not "parameters" (which the interferometer-sibling vmap exercises).
# vmap-over-channels would require a different graph shape and isn't where the
# shared-Lᵀ W̃ L optimisation lives. Skipped.

print("\n--- vmap (skipped) ---")
print(
    "  Cube batching dimension is 'datasets', not 'parameters'. The "
    "interferometer-sibling vmap pattern doesn't map cleanly here. Skipped."
)

# ===================================================================
# Summary + JSON + bar chart
# ===================================================================

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

print("\n" + "=" * 70)
print(f"JAX LIKELIHOOD FUNCTION SUMMARY — CUBE {instrument.upper()} × {n_channels} — v{al_version}")
print("=" * 70)
print(f"  Instrument:              {instrument}")
print(f"  Channels:                {n_channels}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Real-space mask radius:  {mask_radius} arcsec")
print(f"  Real-space grid shape:   {real_space_shape[0]} x {real_space_shape[1]}")
print(f"  Visibilities/chan:       {n_visibilities}")
print(f"  Delaunay vertices:       {n_mesh_vertices}")
print(f"  Edge zeroed pixels:      {edge_pixels_total}")
print("-" * 70)
print(f"  Cube reference log_evidence:  {cube_log_evidence_ref}")
if full_cube_result is not None:
    print(f"  Cube JIT log_evidence:        {float(full_cube_result)}")
else:
    print(f"  Cube JIT log_evidence:        SKIPPED (CUBE_FULL_JIT=1 to enable)")
print("-" * 70)

max_label = max(len(label) for label, _ in likelihood_steps)
step_total = 0.0
for i, (label, per_call) in enumerate(likelihood_steps, 1):
    print(f"  {i:>2}. {label:<{max_label}}  {per_call:>12.6f} s")
    step_total += per_call

# Shared-Lᵀ W̃ L optimisation savings estimate:
# Moving the curvature matrix from per-channel to shared would save
# (n_channels - 1) × per-channel curvature matrix cost.
shared_lwl_savings = (n_channels - 1) * curvature_matrix_per_channel

print("-" * 70)
print(f"      {'TOTAL (step-by-step cube cost)':<{max_label}}  {step_total:>12.6f} s")
if np.isfinite(full_pipeline_per_call):
    print(f"      {'Full pipeline cube (single JIT)':<{max_label}}  {full_pipeline_per_call:>12.6f} s")
else:
    print(f"      {'Full pipeline cube (single JIT)':<{max_label}}  SKIPPED")
print(f"      {f'Shared-Lᵀ W̃ L savings (curvature only, est.)':<{max_label}}  {shared_lwl_savings:>12.6f} s")
print("=" * 70)

# --- Save results dictionary ---

likelihood_summary = {
    "autolens_version": al_version,
    "instrument": instrument,
    "model": "delaunay",
    "n_channels": n_channels,
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "real_space_shape": list(real_space_shape),
        "visibilities_per_channel": int(n_visibilities),
        "overlay_shape": list(overlay_shape),
        "edge_n_points": edge_n_points,
        "delaunay_vertices": int(n_mesh_vertices),
        "edge_zeroed_pixels": int(edge_pixels_total),
        "regularization_coefficient": regularization_coefficient,
    },
    "cube_log_evidence_eager": cube_log_evidence_ref,
    "cube_log_evidence_jit": (
        float(full_cube_result) if full_cube_result is not None else None
    ),
    "log_evidence_per_channel_eager": [float(le) for le in log_evidence_per_channel],
    "steps_cube_cost": {label: per_call for label, per_call in likelihood_steps},
    "per_channel_costs": {
        "inversion_setup": inversion_setup_per_channel,
        "data_vector": data_vector_per_channel,
        "curvature_matrix": curvature_matrix_per_channel,
        "reconstruction": reconstruction_per_channel,
        "log_evidence": log_evidence_per_channel_cost,
    },
    "total_step_by_step_cube": step_total,
    "full_pipeline_cube_single_jit": full_pipeline_per_call,
    "shared_lwl_savings_estimate": shared_lwl_savings,
    "vmap": "SKIPPED — cube batching axis is 'datasets', not 'parameters'",
}

results_dir = _workspace_root / "jax_profiling" / "results" / "jit" / "datacube"
results_dir.mkdir(parents=True, exist_ok=True)

dict_path = results_dir / f"delaunay_likelihood_summary_{instrument}_v{al_version}.json"
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart ---

labels = [label for label, _ in likelihood_steps]
times = [per_call for _, per_call in likelihood_steps]

fig, ax = plt.subplots(figsize=(11, 6))
y_pos = range(len(labels))
# Different colours for shared vs per-channel
colors = ["#55A868" if "(shared)" in label else "#4C72B0" for label in labels]
bars = ax.barh(y_pos, times, color=colors, edgecolor="white", height=0.6)

for bar, t in zip(bars, times):
    ax.text(
        bar.get_width() + max(times) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.6f} s",
        va="center",
        fontsize=9,
    )

if np.isfinite(full_pipeline_per_call):
    ax.axvline(
        full_pipeline_per_call,
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=f"Full pipeline cube (single JIT): {full_pipeline_per_call:.6f} s",
    )
ax.axvline(
    shared_lwl_savings,
    color="#8172B2",
    linestyle=":",
    linewidth=1.5,
    label=f"Shared-Lᵀ W̃ L savings est.: {shared_lwl_savings:.6f} s",
)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Cube cost per call (s)", fontsize=11)
fig.suptitle(
    f"Delaunay Datacube Likelihood — {instrument.upper()} × {n_channels} channels",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f"AutoLens v{al_version}  |  {pixel_scale}\"/px  |  "
    f"{real_space_shape[0]}x{real_space_shape[1]} real-space  |  "
    f"{n_visibilities} visibilities/chan  |  {n_mesh_vertices} Delaunay verts  |  "
    f"step-by-step total: {step_total:.6f} s",
    fontsize=9,
)
ax.legend(loc="lower right", fontsize=9)
ax.margins(x=0.18)
fig.tight_layout()

chart_path = results_dir / f"delaunay_likelihood_summary_{instrument}_v{al_version}.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")


# ===================================================================
# Regression assertion — deterministic cube log-evidence
# ===================================================================
#
# Identical channels = exact N × single-channel log-evidence (for "sma").
# For "hannah" the per-channel literal isn't pinned yet, so the assertion is
# skipped until the value below is filled in from a clean run.
EXPECTED_LOG_EVIDENCE_PER_CHANNEL = {
    "sma": -3167.5258928840763,
    "alma": None,
    "hannah": -204838.07924622478,
}

_per_channel = EXPECTED_LOG_EVIDENCE_PER_CHANNEL.get(instrument)
expected_cube_log_evidence = (
    n_channels * _per_channel if _per_channel is not None else None
)

if expected_cube_log_evidence is None:
    print(
        f"\n  Cube regression assertion SKIPPED for [{instrument}] — "
        f"capture this run's eager cube log_evidence ({cube_log_evidence_ref}), "
        f"divide by n_channels ({n_channels}) to get the per-channel value "
        f"({cube_log_evidence_ref / n_channels}), and paste that into "
        f"EXPECTED_LOG_EVIDENCE_PER_CHANNEL[{instrument!r}]."
    )
else:
    np.testing.assert_allclose(
        cube_log_evidence_ref,
        expected_cube_log_evidence,
        rtol=1e-4,
        err_msg=(
            f"datacube/delaunay[{instrument}]: regression — eager cube log_evidence "
            f"drifted (got {cube_log_evidence_ref}, expected {expected_cube_log_evidence})"
        ),
    )
    print(
        f"\n  Eager cube regression assertion PASSED: log_evidence matches "
        f"{expected_cube_log_evidence:.6f}"
    )
    if full_cube_result is not None:
        np.testing.assert_allclose(
            float(full_cube_result),
            expected_cube_log_evidence,
            rtol=1e-4,
            err_msg=f"datacube/delaunay[{instrument}]: regression — full cube log_evidence drifted",
        )
        print(f"  Full-pipeline cube regression assertion PASSED")
