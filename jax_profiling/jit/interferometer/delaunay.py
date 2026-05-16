"""
JAX Profiling: Delaunay Interferometer Likelihood
=================================================

Profiles the JAX likelihood function for an interferometer dataset where the
source galaxy is reconstructed using a Delaunay pixelization with cross-
derivative (``ConstantSplit``) regularization, and the lens galaxy is an
Isothermal + ExternalShear.

Mirrors ``jax_profiling/interferometer/pixelization.py`` (Phase 2) with the
``RectangularUniform`` source replaced by a ``Delaunay`` mesh — matching
``jax_profiling/imaging/delaunay.py`` so imaging vs interferometer Delaunay
results can be compared side-by-side.

Matches the step-by-step pedagogy of ``jax_profiling/jit/imaging/delaunay.py``
applied to the visibility-space pipeline. The 11 per-step JIT-profiled stages
map 1:1 onto sections in
``autolens_workspace/scripts/interferometer/features/datacube/likelihood_function.py``
and its single-channel parent
``interferometer/features/pixelization/likelihood_function.py``.

Pipeline steps (matching the imaging-delaunay numbering for cross-reference;
the two lens-light steps from the imaging sibling are dropped since the
interferometer pixelization model has no parametric lens light):

 1. Ray-trace data grid to source plane.
 2. Ray-trace mesh grid (image-plane Overlay vertices) to source plane.
 5. Border relocation (data grid + mesh grid).
 6. Delaunay triangulation + interpolation + mapper.
 7. Mapping matrix.
 8. Transformed mapping matrix (NUFFT) — interferometer-specific. Replaces
    imaging's PSF-convolved blurred mapping matrix; the difference is the
    Fourier transform to visibility space rather than image-space convolution.
 9. Data vector D — visibility-space (real and imaginary components).
 10. Curvature matrix F — real and imaginary curvatures summed.
 11. Regularization matrix H — ConstantSplit (same as imaging).
 12. Reconstruction s = NNLS(F + H, D) (same NNLS path as imaging).
 13. Mapped reconstructed visibilities + log evidence (visibility-space χ²).

Measures:

1. Eager baseline: ``FitInterferometer`` with ``xp=np``, print
   ``figure_of_merit`` / ``log_likelihood``.
2. Per-step JIT profiling: each pipeline stage above gets its own
   ``jit_profile()`` call (lower / compile / first-call / steady-state ×10).
3. Full-pipeline JIT: ``jax.jit(analysis.log_likelihood_function)`` on a
   pytree-registered ``ModelInstance``. Measure lower / compile / first-call /
   steady-state per-call.
4. Batched evaluation (opt-in via ``DELAUNAY_VMAP=1``): ``jax.jit(jax.vmap(...))``.
   Skipped by default because Delaunay vmap compilation can take 20+ minutes
   on CPU due to triangulation + interpolation graph size.
5. Correctness: eager vs JIT log-evidence agreement at ``rtol=1e-4`` for both
   the per-step recomputation and the full pipeline.
6. Static memory analysis of the batched program (only when vmap runs).
7. Results JSON + PNG written to ``results/`` with per-step entries that
   slot into the same bar-chart shape as ``jax_profiling/jit/imaging/delaunay.py``.

JIT-blocker notes
-----------------

Per-step decomposition risks missing cross-step XLA fusion and hitting
library-level JAX blockers. Caveats from the previous opt-out version that
still apply:

- ``dataset.transformer.transform_mapping_matrix`` is JIT-friendly for
  ``TransformerDFT`` (a single matrix multiply) and the default SMA preset
  uses it. ``TransformerNUFFT`` (pynufft-based) is not JIT-friendly; if you
  swap the transformer the step-8 timing will fall back to eager-only.
- The visibility-space χ² in step 13 separates the complex visibilities and
  noise into real/imag components inside the JIT body (matching the
  ``pixelization/likelihood_function.py`` reference). Complex-valued JIT
  with autoarray ``Visibilities`` wrappers is avoided.

Pytree-native parameter inputs
------------------------------

Uses ``af.ModelInstance`` as the JIT input via PyAutoFit's opt-in pytree
registration (``autofit.jax.register_model``). Exercises the ``TuplePrior``
pytree support landed in PyAutoFit#1222.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import time
import subprocess
import sys
from pathlib import Path
from contextlib import contextmanager

import autofit as af
import autolens as al
from autofit.jax import register_model as _register_model_pytrees

# ---------------------------------------------------------------------------
# Instrument configuration
# ---------------------------------------------------------------------------

INSTRUMENTS = {
    "sma": {"pixel_scale": 0.1, "real_space_shape": (256, 256), "mask_radius": 3.0},
    "alma": {"pixel_scale": 0.05, "real_space_shape": (256, 256), "mask_radius": 3.0},
    "hannah": {"pixel_scale": 0.125, "real_space_shape": (40, 40), "mask_radius": 2.3},
}

instrument = "sma"  # <-- change this to profile a different instrument

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
    """Call block_until_ready if available (JAX arrays)."""
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
likelihood_steps = []  # (label, per_call_seconds) for the final summary

# ===================================================================
# PART A — Setup (not JIT-compiled)
# ===================================================================

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

print(f"\n--- Dataset loading [{instrument}] ---")

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

with timer.section("dataset_load"):
    dataset = al.Interferometer.from_fits(
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

with timer.section("apply_sparse_operator"):
    # Precompute the NUFFT precision-matrix preload so per-fit curvature
    # assembly uses the FFT-based sparse path instead of dense DFT for every
    # source pixel. Unblocked by PyAutoArray#316 (the Pmax > 1 extent-indexing
    # fix); on Delaunay this was previously guarded with NotImplementedError.
    dataset = dataset.apply_sparse_operator(use_jax=True, show_progress=True)

n_visibilities = dataset.uv_wavelengths.shape[0]
print(f"  Total visibilities: {n_visibilities}")

# ---------------------------------------------------------------------------
# 2. Image mesh + edge points (Delaunay-specific)
# ---------------------------------------------------------------------------

print("\n--- Image mesh construction (Delaunay) ---")

with timer.section("image_mesh_overlay"):
    image_mesh = al.image_mesh.Overlay(shape=overlay_shape)
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=dataset.real_space_mask)

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
    # GaussianPrior(mean=truth, sigma=small) centres prior-median at the
    # simulator truth while keeping params free so gradient diagnostics
    # have dimensionality.
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
print(f"  Zeroed edge pixels: {edge_pixels_total}")

# ---------------------------------------------------------------------------
# 4. Instantiate concrete objects from prior medians
# ---------------------------------------------------------------------------

print("\n--- Instantiate concrete model ---")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

with timer.section("register_pytrees"):
    _register_model_pytrees(model)

# JIT input: the instance itself, with all parameter leaves promoted to JAX
# arrays. The eager NumPy instance is retained for the eager FitInterferometer
# baseline below.
params_tree = jax.tree_util.tree_map(jnp.asarray, instance)

tracer = al.Tracer(galaxies=list(instance.galaxies))

# AdaptImages tells FitInterferometer / AnalysisInterferometer where the
# Delaunay mesh vertices live in the image-plane (separate from the source-
# plane vertices that get computed by ray-tracing).
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
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Real-space mask radius:  {mask_radius} arcsec")
print(f"  Real-space grid shape:   {real_space_shape[0]} x {real_space_shape[1]}")
print(f"  Visibilities:            {n_visibilities}")
print(f"  Overlay shape:           {overlay_shape[0]} x {overlay_shape[1]}")
print(f"  Delaunay vertices:       {n_mesh_vertices}")
print(f"  Edge zeroed pixels:      {edge_pixels_total}")
print(f"  Reg. coefficient:        {regularization_coefficient}")

# ---------------------------------------------------------------------------
# 6. Full-pipeline reference (FitInterferometer) — eager baseline
# ---------------------------------------------------------------------------

print("\n--- Full FitInterferometer (eager baseline) ---")

with timer.section("fit_interferometer_eager"):
    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        xp=np,
    )
    figure_of_merit_ref = fit.figure_of_merit
    log_likelihood_ref = fit.log_likelihood

print(f"  figure_of_merit = {figure_of_merit_ref}")
print(f"  log_likelihood  = {log_likelihood_ref}")


# ===================================================================
# PART B — Per-step JIT profiling
# ===================================================================

print("\n" + "=" * 70)
print("PER-STEP JIT PROFILING")
print("=" * 70)

import autoarray as aa

# Extract raw arrays from autoarray types via .array so they can cross
# JIT boundaries.  See CLAUDE.md for rationale.
grid_pix_raw = jnp.array(dataset.grids.pixelization.array)
mesh_grid_raw = jnp.array(image_plane_mesh_grid.array)
data_real_jnp = jnp.array(dataset.data.real)
data_imag_jnp = jnp.array(dataset.data.imag)
noise_real_jnp = jnp.array(dataset.noise_map.real)
noise_imag_jnp = jnp.array(dataset.noise_map.imag)

# ---------------------------------------------------------------------------
# Step 1: Ray-trace data grid to source plane
# ---------------------------------------------------------------------------
# Same operation as ``pixelization/likelihood_function.py:__Ray Tracing__``
# applied to the data grid (one of the two grids the inversion uses).

print("\n--- Step 1: Ray-trace data grid ---")

with timer.section("ray_trace_data_eager"):
    traced_grids = tracer.traced_grid_2d_list_from(
        grid=dataset.grids.pixelization, xp=jnp
    )
    for tg in traced_grids:
        block(tg)

print(f"  Number of planes traced: {len(traced_grids)}")


def ray_trace_data_raw(grid_raw):
    grid = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    traced = tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.stack([tg.array for tg in traced])


_, traced_data_grids_raw = jit_profile(
    ray_trace_data_raw, "ray_trace_data_jit", grid_pix_raw
)
likelihood_steps.append(("Ray-trace data grid", timer.records[-1][1] / 10))

print(f"  traced_data_grids shape: {traced_data_grids_raw.shape}")

# ---------------------------------------------------------------------------
# Step 2: Ray-trace mesh grid (image-plane vertices) to source plane
# ---------------------------------------------------------------------------
# Delaunay-specific: the source-plane mesh vertices are computed in the
# image-plane via the ``Overlay`` mesh and ray-traced to source-plane. This is
# the same per-step shape as imaging-delaunay; the underlying tracer call is
# identical.

print("\n--- Step 2: Ray-trace mesh grid ---")

with timer.section("ray_trace_mesh_eager"):
    traced_mesh = tracer.traced_grid_2d_list_from(
        grid=al.Grid2DIrregular(image_plane_mesh_grid), xp=jnp
    )
    for tg in traced_mesh:
        block(tg)


def ray_trace_mesh_raw(mesh_raw):
    grid = aa.Grid2DIrregular(values=mesh_raw, xp=jnp)
    traced = tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.stack([tg.array for tg in traced])


_, traced_mesh_grids_raw = jit_profile(
    ray_trace_mesh_raw, "ray_trace_mesh_jit", mesh_grid_raw
)
likelihood_steps.append(("Ray-trace mesh grid", timer.records[-1][1] / 10))

print(f"  traced_mesh_grids shape: {traced_mesh_grids_raw.shape}")

# ---------------------------------------------------------------------------
# Step 5: Border relocation (data grid + mesh grid)
# ---------------------------------------------------------------------------
# Steps 3-4 from the imaging sibling (lens-light pre-PSF image and PSF-blurred
# image) don't exist for the interferometer pixelization model — there's no
# parametric lens light. We jump straight to step 5.
#
# Same as ``pixelization/likelihood_function.py:__Border Relocation__``;
# scipy-based so eager-only.

print("\n--- Step 5: Border relocation ---")

from autoarray.inversion.mesh.border_relocator import BorderRelocator

with timer.section("border_relocator_setup"):
    border_relocator = BorderRelocator(mask=dataset.mask, sub_size=1)

traced_source_grid = tracer.traced_grid_2d_list_from(
    grid=dataset.grids.pixelization, xp=jnp
)[-1]
traced_mesh_source = tracer.traced_grid_2d_list_from(
    grid=al.Grid2DIrregular(image_plane_mesh_grid), xp=jnp
)[-1]

with timer.section("border_relocation_eager"):
    relocated_grid = border_relocator.relocated_grid_from(grid=traced_source_grid)
    relocated_mesh_grid = border_relocator.relocated_mesh_grid_from(
        grid=traced_source_grid,
        mesh_grid=traced_mesh_source,
    )
    block(relocated_grid)
    block(relocated_mesh_grid)

print(f"  relocated_data_grid shape: {relocated_grid.array.shape}")
print(f"  relocated_mesh_grid shape: {relocated_mesh_grid.array.shape}")

# ---------------------------------------------------------------------------
# Step 6: Delaunay triangulation + interpolation + mapper
# ---------------------------------------------------------------------------
# scipy-based; same as imaging-delaunay step 6.

print("\n--- Step 6: Delaunay triangulation + Interpolation + Mapper ---")

pixelization_obj = instance.galaxies.source.pixelization

with timer.section("delaunay_interpolation_and_mapper"):
    interpolator = al.InterpolatorDelaunay(
        mesh=pixelization_obj.mesh,
        mesh_grid=relocated_mesh_grid,
        data_grid=relocated_grid,
    )
    mapper = al.Mapper(
        interpolator=interpolator,
        image_plane_mesh_grid=image_plane_mesh_grid,
        xp=jnp,
    )

print(f"  mapper.pixels (source): {mapper.pixels}")
print(f"  pix_indexes shape: {mapper.pix_indexes_for_sub_slim_index.shape}")

# ---------------------------------------------------------------------------
# Steps 7-13: Extract matrices from FitInterferometer.inversion for consistency
# ---------------------------------------------------------------------------
# The FitInterferometer pipeline handles edge-pixel zeroing, curvature-diagonal
# adjustments, and settings that are difficult to replicate manually. We extract
# the production matrices upfront and then JIT-profile the linear-algebra ops
# on the reference inputs, matching the imaging-sibling pattern.

print("\n--- Extracting inversion matrices from FitInterferometer ---")

inversion = fit.inversion

with timer.section("extract_inversion_matrices"):
    # ``operated_mapping_matrix`` is the NUFFT-transformed mapping matrix
    # (complex-valued, shape (n_vis, source_pixels)). Imaging's equivalent is
    # the PSF-convolved blurred mapping matrix.
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
# Step 7: Mapping matrix
# ---------------------------------------------------------------------------
# Same as ``pixelization/likelihood_function.py:__Mapping Matrix__``.

print("\n--- Step 7: Mapping matrix ---")

with timer.section("mapping_matrix"):
    mapping_matrix = inv_mapper.mapping_matrix

print(f"  mapping_matrix shape: {mapping_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 8: Transformed mapping matrix (NUFFT) — interferometer-specific
# ---------------------------------------------------------------------------
# Replaces the imaging sibling's "Blurred mapping matrix (PSF convolution)".
#
# For ``TransformerDFT`` this is a single complex matrix multiply
# ``D @ M`` where D is the discrete Fourier matrix (n_vis × n_image) and
# M is the real mapping matrix (n_image × source_pixels). JIT-friendly.
# For ``TransformerNUFFT`` (pynufft-based, ALMA-scale) this step would fall
# back to eager — flag if you swap the transformer.

print("\n--- Step 8: Transformed mapping matrix (NUFFT) ---")

with timer.section("transformed_mapping_matrix_eager"):
    transformed_mapping_matrix = dataset.transformer.transform_mapping_matrix(
        mapping_matrix=mapping_matrix_ref
    )
    block(transformed_mapping_matrix)


def compute_transformed_mapping_matrix(mapping_matrix):
    return dataset.transformer.transform_mapping_matrix(mapping_matrix=mapping_matrix)


# JIT-profile the full inversion setup pipeline (steps 5-8 combined) from a
# pytree ModelInstance. This is the cube-relevant per-channel cost: per
# AnalysisFactor the NUFFT mapping-matrix construction has to rerun because
# uv_wavelengths is channel-specific.
def transformed_mm_from_params(params_tree):
    """Inversion setup from a pytree ModelInstance — full chain through NUFFT."""
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
likelihood_steps.append(
    ("Inversion setup (steps 5-8 combined, incl. NUFFT)", timer.records[-1][1] / 10)
)

print(f"  transformed_mapping_matrix (JIT) shape: {transformed_mm_jit.shape}")
print(f"  transformed_mapping_matrix (JIT) dtype: {transformed_mm_jit.dtype}")

# Use the reference matrices for the linear-algebra steps below.
transformed_mm_real_jnp = jnp.real(transformed_mm_ref)
transformed_mm_imag_jnp = jnp.imag(transformed_mm_ref)

# ---------------------------------------------------------------------------
# Step 9: Data vector (D) — visibility-space
# ---------------------------------------------------------------------------
# Same as ``pixelization/likelihood_function.py:__Data Vector (D)__``.

print("\n--- Step 9: Data vector (D) ---")


def compute_data_vector(
    transformed_mm_real, transformed_mm_imag, data_real, data_imag,
    noise_real, noise_imag,
):
    # Visibility-space data vector: D_i = sum_j f_ij d_j / sigma_j^2 (real + imag).
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
likelihood_steps.append(("Data vector (D)", timer.records[-1][1] / 10))

print(f"  data_vector shape: {data_vector.shape}")

# ---------------------------------------------------------------------------
# Step 10: Curvature matrix (F)
# ---------------------------------------------------------------------------
# Same as ``pixelization/likelihood_function.py:__Curvature Matrix (F)__``:
# F = sum_j f_ij f_kj / sigma_j^2, computed separately for real / imag and summed.

print("\n--- Step 10: Curvature matrix (F) ---")

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
likelihood_steps.append(("Curvature matrix (F)", timer.records[-1][1] / 10))

print(f"  curvature_matrix shape: {curvature_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 11: Regularization matrix (H) — ConstantSplit scheme
# ---------------------------------------------------------------------------
# Same as ``pixelization/likelihood_function.py:__Regularization Matrix (H)__``;
# extracted from the inversion for consistency with the production setup.

print("\n--- Step 11: Regularization matrix (ConstantSplit) ---")

with timer.section("regularization_matrix_eager"):
    regularization_matrix = jnp.array(inversion.regularization_matrix)
    block(regularization_matrix)

likelihood_steps.append(("Regularization matrix (H)", timer.records[-1][1]))

print(f"  regularization_matrix shape: {regularization_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 12: Regularized reconstruction: s = NNLS(F + H, D)
# ---------------------------------------------------------------------------
# Same NNLS path as the imaging sibling. For well-conditioned ConstantSplit at
# the prior median this reduces to a linear solve (no negative source pixels),
# but we use the NNLS solver to match the production AnalysisInterferometer
# behaviour exactly.

print("\n--- Step 12: Regularized reconstruction ---")


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
likelihood_steps.append(("Regularized reconstruction", timer.records[-1][1] / 10))

print(f"  reconstruction shape: {reconstruction.shape}")

# ---------------------------------------------------------------------------
# Step 13: Mapped reconstructed visibilities + log evidence
# ---------------------------------------------------------------------------
# Same five-term log-evidence as the imaging sibling but with visibility-space
# χ² (real + imag). Matches the formula in
# ``pixelization/likelihood_function.py:__Likelihood Function — Five Terms__``.

print("\n--- Step 13: Mapped reconstructed visibilities + log evidence ---")


def compute_log_evidence(
    data_real, data_imag, noise_real, noise_imag,
    transformed_mm_real, transformed_mm_imag,
    reconstruction, curvature_matrix, regularization_matrix, mapper_indices,
):
    """Visibility-space log-evidence — five-term formula matching the production
    ``FitInterferometer.log_evidence``.

    -2 ln e = chi^2 + s^T H s + ln[det(F+H)] - ln[det(H)] + noise_norm
    """
    # Mapped reconstructed visibilities (real / imag separately)
    mapped_real = jnp.matmul(transformed_mm_real, reconstruction)
    mapped_imag = jnp.matmul(transformed_mm_imag, reconstruction)

    # χ² in visibility space (real + imag)
    chi_real = jnp.sum(((data_real - mapped_real) / noise_real) ** 2)
    chi_imag = jnp.sum(((data_imag - mapped_imag) / noise_imag) ** 2)
    chi_squared = chi_real + chi_imag

    # s^T H s
    regularization_term = jnp.dot(
        reconstruction, jnp.dot(regularization_matrix, reconstruction)
    )

    # Complexity terms (Cholesky log-det matching production)
    curvature_reg_matrix = curvature_matrix + regularization_matrix
    creg_reduced = curvature_reg_matrix[mapper_indices][:, mapper_indices]
    reg_reduced = regularization_matrix[mapper_indices][:, mapper_indices]
    log_det_curvature_reg = 2.0 * jnp.sum(
        jnp.log(jnp.diag(jnp.linalg.cholesky(creg_reduced)))
    )
    log_det_regularization = 2.0 * jnp.sum(
        jnp.log(jnp.diag(jnp.linalg.cholesky(reg_reduced)))
    )

    # Noise normalisation (real + imag)
    noise_normalization = (
        jnp.sum(jnp.log(2 * jnp.pi * noise_real ** 2))
        + jnp.sum(jnp.log(2 * jnp.pi * noise_imag ** 2))
    )

    return -0.5 * (
        chi_squared + regularization_term + log_det_curvature_reg
        - log_det_regularization + noise_normalization
    )


# For the JIT correctness check we recompute log_evidence using the inversion's
# own reconstruction and curvature matrix to avoid accumulated FP drift
# (matching the imaging sibling's pattern).
mapper_indices_jnp = jnp.array(np.asarray(inversion.mapper_indices))
inv_recon_jnp = jnp.asarray(inversion.reconstruction)
inv_curv_jnp = jnp.asarray(inversion.curvature_matrix)
reg_jnp = jnp.array(regularization_matrix)

with timer.section("log_evidence_eager"):
    log_evidence = compute_log_evidence(
        data_real_jnp, data_imag_jnp, noise_real_jnp, noise_imag_jnp,
        transformed_mm_real_jnp, transformed_mm_imag_jnp,
        reconstruction, curvature_matrix, reg_jnp, mapper_indices_jnp,
    )
    block(log_evidence)

_, log_evidence = jit_profile(
    compute_log_evidence, "log_evidence_jit",
    data_real_jnp, data_imag_jnp, noise_real_jnp, noise_imag_jnp,
    transformed_mm_real_jnp, transformed_mm_imag_jnp,
    reconstruction, curvature_matrix, reg_jnp, mapper_indices_jnp,
)
likelihood_steps.append(("Mapped recon + log evidence", timer.records[-1][1] / 10))

print(f"  log_evidence (step-by-step) = {log_evidence}")

# Correctness check: use the inversion's own reconstruction and curvature matrix
log_evidence_check = compute_log_evidence(
    data_real_jnp, data_imag_jnp, noise_real_jnp, noise_imag_jnp,
    transformed_mm_real_jnp, transformed_mm_imag_jnp,
    inv_recon_jnp, inv_curv_jnp, reg_jnp, mapper_indices_jnp,
)
print(f"  log_evidence (inv matrices) = {log_evidence_check}")
print(f"  log_evidence (reference)    = {figure_of_merit_ref}")

np.testing.assert_allclose(
    float(log_evidence_check),
    float(figure_of_merit_ref),
    rtol=1e-4,
    err_msg=(
        "Per-step log_evidence from inversion matrices does not match "
        "FitInterferometer.log_evidence"
    ),
)
print(
    "  Assertion PASSED: inversion-matrix log_evidence matches "
    "FitInterferometer.log_evidence"
)


# ===================================================================
# PART C — Full-pipeline JIT (for comparison)
# ===================================================================

print("\n" + "=" * 70)
print("FULL-PIPELINE JIT")
print("=" * 70)

analysis = al.AnalysisInterferometer(
    dataset=dataset, adapt_images=adapt_images, use_jax=True
)

def full_pipeline_from_params(params_tree):
    """Full interferometer likelihood from a pytree-shaped ``ModelInstance``.

    No flat-vector unpacking inside the trace — the instance crosses the JIT
    boundary directly, with constants (redshifts, etc.) kept static via the
    ``aux_data`` partition set up by ``autofit.jax.register_model``.
    """
    return analysis.log_likelihood_function(instance=params_tree)

_, full_result = jit_profile(full_pipeline_from_params, "full_pipeline", params_tree)
full_pipeline_per_call = timer.records[-1][1] / 10

print(f"  full log_evidence = {full_result}")

# Correctness: for inversion models (pixelization + regularization), the
# analysis "log_likelihood_function" actually returns the log-evidence
# (= figure_of_merit), which includes the regularization/determinant terms.
# Match against figure_of_merit_ref, not log_likelihood_ref.
np.testing.assert_allclose(
    float(full_result),
    float(figure_of_merit_ref),
    rtol=1e-4,
    err_msg="interferometer/delaunay: JIT log-evidence does not match eager figure_of_merit",
)
print("  Eager-vs-JIT correctness PASSED")

# ===================================================================
# PART D — vmap (opt-in) + correctness
# ===================================================================
#
# Delaunay vmap compilation can take 20+ minutes on CPU due to the size of
# the triangulation + interpolation XLA graph. Skipped by default — set
# DELAUNAY_VMAP=1 to opt in.

print("\n--- vmap batched evaluation ---")

run_vmap = os.environ.get("DELAUNAY_VMAP", "0") == "1"

batch_size = 3
vmap_batch_time = None
vmap_per_call = None
vmap_speedup = None
result_vmap = None
vmapped_full = None
parameters = None

_n_leaves = len(jax.tree_util.tree_leaves(params_tree))
if not run_vmap:
    print("  SKIPPED: opt-in via DELAUNAY_VMAP=1 (compilation can take 20+ minutes).")
elif _n_leaves == 0:
    print(f"  SKIPPED: model has 0 free parameters (all fixed to truth); "
          f"vmap requires at least one array leaf.")
else:
    parameters = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (batch_size, *leaf.shape)),
        params_tree,
    )

    vmapped_full = jax.jit(jax.vmap(full_pipeline_from_params))

    with timer.section("vmap_first_call"):
        result_vmap = vmapped_full(parameters)
        block(result_vmap)

    n_vmap_repeats = 10
    with timer.section(f"vmap_steady_x{n_vmap_repeats}"):
        for _ in range(n_vmap_repeats):
            result_vmap = vmapped_full(parameters)
            block(result_vmap)

    vmap_batch_time = timer.records[-1][1] / n_vmap_repeats
    vmap_per_call = vmap_batch_time / batch_size
    vmap_speedup = full_pipeline_per_call / vmap_per_call

    print(f"  batch results = {result_vmap}")
    print(f"  vmap batch of {batch_size}:   {vmap_batch_time:.6f} s")
    print(f"  vmap per call:         {vmap_per_call:.6f} s")
    print(f"  single JIT per call:   {full_pipeline_per_call:.6f} s")
    print(f"  vmap speedup:          {vmap_speedup:.1f}x faster per likelihood")

    np.testing.assert_allclose(
        np.array(result_vmap),
        float(full_result),
        rtol=1e-4,
        err_msg="interferometer/delaunay: JAX vmap likelihood mismatch",
    )
    print("  vmap-vs-single-JIT correctness PASSED")

# ===================================================================
# PART E — Static memory analysis (only if vmap ran)
# ===================================================================

print("\n--- Static memory analysis ---")

if vmapped_full is None:
    print("  SKIPPED: vmap path was not exercised this run.")
    memory_analysis = None
else:
    lowered_batched = vmapped_full.lower(parameters)
    compiled_batched = lowered_batched.compile()

    memory_analysis = compiled_batched.memory_analysis()
    print(f"  Output size:  {memory_analysis.output_size_in_bytes / 1024**2:.3f} MB")
    print(f"  Temp size:    {memory_analysis.temp_size_in_bytes / 1024**2:.3f} MB")
    print(
        f"  Total:        "
        f"{(memory_analysis.output_size_in_bytes + memory_analysis.temp_size_in_bytes) / 1024**2:.3f} MB"
    )

# ===================================================================
# JAX Likelihood Function Summary + artefacts
# ===================================================================

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

print("\n" + "=" * 70)
print(f"JAX LIKELIHOOD FUNCTION SUMMARY — {instrument.upper()} — v{al_version}")
print("=" * 70)
print(f"  Instrument:              {instrument}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Real-space mask radius:  {mask_radius} arcsec")
print(f"  Real-space grid shape:   {real_space_shape[0]} x {real_space_shape[1]}")
print(f"  Visibilities:            {n_visibilities}")
print(f"  Delaunay vertices:       {n_mesh_vertices}")
print(f"  Edge zeroed pixels:      {edge_pixels_total}")
print("-" * 70)
print(f"  Eager log_likelihood:    {log_likelihood_ref}")
print(f"  Eager figure_of_merit:   {figure_of_merit_ref}  (log-evidence)")
print(f"  JIT  log-evidence:       {float(full_result)}")
print("-" * 70)

max_label = max(len(label) for label, _ in likelihood_steps)
step_total = 0.0
for i, (label, per_call) in enumerate(likelihood_steps, 1):
    print(f"  {i:>2}. {label:<{max_label}}  {per_call:>12.6f} s")
    step_total += per_call

print("-" * 70)
print(f"      {'TOTAL (step-by-step)':<{max_label}}  {step_total:>12.6f} s")
print(f"      {'Full pipeline (single JIT)':<{max_label}}  {full_pipeline_per_call:>12.6f} s")
if vmap_per_call is not None:
    print(f"      {'vmap batch (per call)':<{max_label}}  {vmap_per_call:>12.6f} s")
    print(f"      {'vmap speedup vs single JIT':<{max_label}}  {vmap_speedup:>11.1f}x")
else:
    print(f"      {'vmap':<{max_label}}  {'SKIPPED':>12}")
print("=" * 70)

# --- Save results dictionary ---

if vmap_per_call is None:
    if not run_vmap:
        vmap_payload = "SKIPPED — opt-in via DELAUNAY_VMAP=1"
    else:
        vmap_payload = "SKIPPED — model has 0 free parameters (all fixed to truth)"
else:
    vmap_payload = {
        "batch_size": batch_size,
        "batch_time": vmap_batch_time,
        "per_call": vmap_per_call,
        "speedup_vs_single_jit": round(vmap_speedup, 1),
    }

likelihood_summary = {
    "autolens_version": al_version,
    "instrument": instrument,
    "model": "delaunay",
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "real_space_shape": list(real_space_shape),
        "visibilities": int(n_visibilities),
        "overlay_shape": list(overlay_shape),
        "edge_n_points": edge_n_points,
        "delaunay_vertices": int(n_mesh_vertices),
        "edge_zeroed_pixels": int(edge_pixels_total),
        "regularization_coefficient": regularization_coefficient,
    },
    "log_likelihood_eager": float(log_likelihood_ref),
    "figure_of_merit_eager": float(figure_of_merit_ref),
    "log_evidence_jit": float(full_result),
    "steps": {label: per_call for label, per_call in likelihood_steps},
    "total_step_by_step": step_total,
    "full_pipeline_single_jit": full_pipeline_per_call,
    "vmap": vmap_payload,
    "memory_mb": None if memory_analysis is None else {
        "output": memory_analysis.output_size_in_bytes / 1024**2,
        "temp": memory_analysis.temp_size_in_bytes / 1024**2,
    },
}

results_dir = _workspace_root / "jax_profiling" / "results" / "jit" / "interferometer"
results_dir.mkdir(parents=True, exist_ok=True)

dict_path = results_dir / f"delaunay_likelihood_summary_{instrument}_v{al_version}.json"
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart ---

labels = [label for label, _ in likelihood_steps]
times = [per_call for _, per_call in likelihood_steps]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = range(len(labels))
bars = ax.barh(y_pos, times, color="#4C72B0", edgecolor="white", height=0.6)

for bar, t in zip(bars, times):
    ax.text(
        bar.get_width() + max(times) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.6f} s",
        va="center",
        fontsize=9,
    )

ax.axvline(
    full_pipeline_per_call,
    color="#C44E52",
    linestyle="--",
    linewidth=1.5,
    label=f"Full pipeline (single JIT): {full_pipeline_per_call:.6f} s",
)
if vmap_per_call is not None:
    ax.axvline(
        vmap_per_call,
        color="#55A868",
        linestyle="--",
        linewidth=1.5,
        label=f"vmap batch per call: {vmap_per_call:.6f} s ({vmap_speedup:.1f}x faster)",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Time per call (s)", fontsize=11)
fig.suptitle(
    f"Delaunay Interferometer Likelihood — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f"AutoLens v{al_version}  |  {pixel_scale}\"/px  |  "
    f"{real_space_shape[0]}x{real_space_shape[1]} real-space  |  "
    f"{n_visibilities} visibilities  |  {n_mesh_vertices} Delaunay verts  |  "
    f"total: {step_total:.6f} s",
    fontsize=9,
)
ax.legend(loc="lower right", fontsize=9)
ax.margins(x=0.15)
fig.tight_layout()

chart_path = results_dir / f"delaunay_likelihood_summary_{instrument}_v{al_version}.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")


# ===================================================================
# Regression assertion — realistic-scale deterministic log-evidence
# ===================================================================
#
# Simulator truth parameters via GaussianPrior(mean=truth, sigma=small)
# make the full-pipeline log-evidence deterministic at the prior median.
# Pinned empirically per instrument; ``None`` means "skip the assertion and
# print the value so it can be pasted in here on a clean run".
EXPECTED_LOG_EVIDENCE = {
    "sma": -3167.5258928840763,
    "alma": None,
    "hannah": -204838.07924622478,
}

expected_log_evidence = EXPECTED_LOG_EVIDENCE.get(instrument)

if expected_log_evidence is None:
    print(
        f"\n  Regression assertion SKIPPED for [{instrument}] — "
        f"capture this run's eager log_evidence ({figure_of_merit_ref}) "
        f"and paste it into EXPECTED_LOG_EVIDENCE[{instrument!r}]."
    )
else:
    np.testing.assert_allclose(
        figure_of_merit_ref,
        expected_log_evidence,
        rtol=1e-4,
        err_msg=(
            f"interferometer/delaunay[{instrument}]: regression — eager log_evidence "
            f"drifted (got {figure_of_merit_ref}, expected {expected_log_evidence})"
        ),
    )
    print(
        f"  Eager regression assertion PASSED: log_evidence matches "
        f"{expected_log_evidence:.6f}"
    )
    np.testing.assert_allclose(
        float(full_result),
        expected_log_evidence,
        rtol=1e-4,
        err_msg=f"interferometer/delaunay[{instrument}]: regression — full log_evidence drifted",
    )
    print(f"  Full-pipeline regression assertion PASSED")
    if result_vmap is not None:
        np.testing.assert_allclose(
            np.array(result_vmap),
            expected_log_evidence,
            rtol=1e-4,
            err_msg=f"interferometer/delaunay[{instrument}]: regression — vmap log_evidence drifted",
        )
        print(f"  vmap regression assertion PASSED")
