"""
JAX Profiling: MGE Imaging Likelihood (Step-by-Step)
=====================================================

Profiles each step of the JAX likelihood function for an imaging dataset where
the lens galaxy's light is modelled with a multi-Gaussian expansion (MGE).

Rather than timing the whole likelihood as a single JIT-compiled block (which
hides internal bottlenecks), this script JIT-compiles and times each step of
the pipeline individually:

1. Instance from parameter vector
2. Build Tracer
3. Ray-trace grids through the lens
4. Compute mapping matrix (per-profile images before PSF)
5. Compute blurred mapping matrix (PSF convolution)
6. Compute data vector  (D)
7. Compute curvature matrix  (F)
8. Reconstruction via positive-only NNLS
9. Map reconstruction back to image plane
10. Chi-squared and log likelihood

Note: because the MGE model uses only linear light profiles (lp_linear),
there is no non-linear blurred image or profile-subtracted image step.

Caveat: XLA may fuse operations differently when compiled as one program vs
separate pieces, so per-step timings are approximate. They are still useful
for identifying which step dominates.

All JAX timings use `block_until_ready()` to force synchronous measurement.

Pytree-native parameter inputs (recommended pattern)
----------------------------------------------------

This script uses ``af.ModelInstance`` as the JIT input via PyAutoFit's
opt-in pytree registration (``autofit.jax.register_model(model)``). The
JIT'd closures consume the instance directly, so:

* ``model.instance_from_vector`` is no longer called inside the JIT trace —
  parameter unpacking happens once at registration time and JAX walks the
  pytree on every call.
* Parameter identity is preserved through ``jax.jit`` and ``jax.vmap``;
  XLA cache keys reflect the structured pytree, not a flat vector shape.
* ``vmap`` batching is ``jax.tree_util.tree_map`` over the instance leaves
  — callers no longer have to stack a ``(batch, N)`` array.

New profiling scripts should follow this pattern. The flat-vector path in
``Fitness.call`` / ``model.instance_from_vector(..., xp=jnp)`` remains the
production likelihood entry point and is intentionally untouched here.
"""

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
import autoarray as aa
from autofit.jax import register_model as _register_model_pytrees

# ---------------------------------------------------------------------------
# Instrument configuration
# ---------------------------------------------------------------------------

INSTRUMENTS = {
    "euclid": {"pixel_scale": 0.1},
    "hst": {"pixel_scale": 0.05},
    "jwst": {"pixel_scale": 0.03},
    "ao": {"pixel_scale": 0.01},
}

instrument = "hst"  # <-- change this to profile a different instrument


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------

class Timer:
    """Accumulates named timing measurements and prints a summary."""

    def __init__(self):
        self.records: list[tuple[str, float]] = []

    @contextmanager
    def section(self, label: str):
        """Context manager that records wall-clock time for *label*."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.records.append((label, elapsed))
        print(f"  [{label}] {elapsed:.4f} s")

    def summary(self):
        print("\n" + "=" * 70)
        print("PROFILING SUMMARY")
        print("=" * 70)
        max_label = max(len(r[0]) for r in self.records)
        total = 0.0
        for label, elapsed in self.records:
            print(f"  {label:<{max_label}}  {elapsed:>10.4f} s")
            total += elapsed
        print("-" * 70)
        print(f"  {'TOTAL':<{max_label}}  {total:>10.4f} s")
        print("=" * 70)


def block(x):
    """Call block_until_ready if available (JAX arrays)."""
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def jit_profile(func, label, *args, n_repeats=10):
    """JIT-compile *func*, time first call and steady-state average.

    Returns the compiled function and its result.
    """
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

print(f"\n--- Dataset loading & masking [{instrument}] ---")

_script_dir = Path(__file__).resolve().parent
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
dataset_path = Path("jax_profiling") / "imaging" / "dataset" / "imaging" / instrument

if al.util.dataset.should_simulate(str(dataset_path)):
    print(f"  Simulating {instrument} dataset...")
    subprocess.run(
        [
            sys.executable,
            str(_script_dir / "simulators" / "imaging.py"),
            "--instrument", instrument,
        ],
        cwd=str(_script_dir),
        check=True,
    )

with timer.section("dataset_load"):
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=pixel_scale,
    )

with timer.section("mask_and_oversample"):
    mask_radius = 3.5

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)
    dataset = dataset.apply_over_sampling(over_sample_size_lp=4)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

with timer.section("model_build"):
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )

    mass = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)

    lens = af.Model(
        al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear
    )

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")

# ---------------------------------------------------------------------------
# 3. Instantiate concrete objects from prior medians
# ---------------------------------------------------------------------------

print("\n--- Instantiate concrete model ---")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

# Register every concrete `model.cls` (Galaxy, profile classes, ModelInstance,
# Collection, …) with `jax.tree_util` so the instance can cross JIT/vmap
# boundaries directly. This must happen AFTER the model is built, because
# registration walks the model's class graph.
with timer.section("register_pytrees"):
    _register_model_pytrees(model)

# JIT input: the instance itself, with all parameter leaves promoted to JAX
# arrays. We keep `instance` (the eager NumPy version) around for any
# non-JIT setup that needs to read parameter values directly.
params_tree = jax.tree_util.tree_map(jnp.asarray, instance)

tracer = al.Tracer(galaxies=list(instance.galaxies))

print(f"  Tracer planes: {tracer.total_planes}")

# ---------------------------------------------------------------------------
# Key configuration that dictates run time
# ---------------------------------------------------------------------------

n_image_pixels = dataset.data.shape[0]
n_over_sampled_pixels = dataset.grids.lp.over_sampled.shape[0]
n_linear_gaussians = len(tracer.cls_list_from(cls=al.lp_linear.LightProfileLinear))

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {instrument}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Mask radius:             {mask_radius} arcsec")
print(f"  Image pixels (masked):   {n_image_pixels}")
print(f"  Over-sampled pixels:     {n_over_sampled_pixels}")
print(f"  Linear Gaussians:        {n_linear_gaussians}")

# ---------------------------------------------------------------------------
# 4. Full-pipeline reference (FitImaging) — eager baseline
# ---------------------------------------------------------------------------

print("\n--- Full FitImaging (eager baseline) ---")

with timer.section("fit_imaging_eager"):
    fit = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        settings=al.Settings(use_border_relocator=True),
        xp=np,
    )
    log_evidence_ref = fit.figure_of_merit
    log_likelihood_ref = fit.log_likelihood

print(f"  figure_of_merit (log_evidence) = {log_evidence_ref}")
print(f"  log_likelihood                 = {log_likelihood_ref}")


# ===================================================================
# PART B — Per-step JIT profiling
# ===================================================================

print("\n" + "=" * 70)
print("PER-STEP JIT PROFILING")
print("=" * 70)

# Extract raw arrays from autoarray types via .array so they can cross
# JIT boundaries.  See CLAUDE.md for rationale.

grid_lp_raw = jnp.array(dataset.grids.lp.array)
data_array = jnp.array(dataset.data.array)
noise_map_array = jnp.array(dataset.noise_map.array)

# Keep autoarray objects for eager calls that need them.
grid_lp = dataset.grids.lp

# ---------------------------------------------------------------------------
# Step 1: Ray-trace grids
# ---------------------------------------------------------------------------

print("\n--- Step 1: Ray-trace grids ---")

with timer.section("ray_trace_eager"):
    traced_grids = tracer.traced_grid_2d_list_from(grid=grid_lp, xp=jnp)
    for tg in traced_grids:
        block(tg)

print(f"  Number of planes traced: {len(traced_grids)}")

def ray_trace_raw(grid_raw):
    """Wraps ray-tracing so inputs/outputs are raw arrays."""
    grid = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    traced = tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.stack([tg.array for tg in traced])

_, traced_grids_raw = jit_profile(ray_trace_raw, "ray_trace_jit", grid_lp_raw)
likelihood_steps.append(("Ray-trace grids", timer.records[-1][1] / 10))

print(f"  traced_grids shape: {traced_grids_raw.shape}")

# ---------------------------------------------------------------------------
# Step 2: Build linear objects and mapping matrix
# ---------------------------------------------------------------------------

print("\n--- Step 2: Mapping matrix (linear profile images) ---")

with timer.section("linear_obj_setup"):
    tracer_to_inv = al.TracerToInversion(
        dataset=aa.DatasetInterface(
            data=fit.profile_subtracted_image,
            noise_map=dataset.noise_map,
            grids=dataset.grids,
            psf=dataset.psf,
            sparse_operator=dataset.sparse_operator,
        ),
        tracer=tracer,
        settings=al.Settings(use_border_relocator=True),
    )

    lp_linear_func_galaxy_dict = tracer_to_inv.lp_linear_func_list_galaxy_dict

    lp_linear_funcs = list(lp_linear_func_galaxy_dict.keys())

# mapping_matrix and operated_mapping_matrix_override already return raw arrays.
with timer.section("mapping_matrix"):
    mapping_matrices = [func.mapping_matrix for func in lp_linear_funcs]
    mapping_matrix = np.hstack(mapping_matrices) if len(mapping_matrices) > 1 else mapping_matrices[0]

print(f"  mapping_matrix shape: {mapping_matrix.shape}")

def mapping_matrix_from_params(params_tree):
    """Compute mapping matrix from a pytree-shaped ``ModelInstance``.

    No flat-vector unpacking inside the trace: ``params_tree`` is the
    structured instance directly (registered as a JAX pytree above).
    """
    t = al.Tracer(galaxies=list(params_tree.galaxies))
    tti = al.TracerToInversion(
        dataset=aa.DatasetInterface(
            data=fit.profile_subtracted_image,
            noise_map=dataset.noise_map,
            grids=dataset.grids,
            psf=dataset.psf,
            sparse_operator=dataset.sparse_operator,
        ),
        tracer=t,
        settings=al.Settings(use_border_relocator=True),
        xp=jnp,
    )
    funcs = list(tti.lp_linear_func_list_galaxy_dict.keys())
    matrices = [f.mapping_matrix for f in funcs]
    return jnp.hstack(matrices) if len(matrices) > 1 else matrices[0]

_, mm_jit = jit_profile(mapping_matrix_from_params, "mapping_matrix_jit", params_tree)
likelihood_steps.append(("Mapping matrix", timer.records[-1][1] / 10))

print(f"  mapping_matrix (JIT) shape: {mm_jit.shape}")

# ---------------------------------------------------------------------------
# Step 3: Blurred mapping matrix (PSF convolution of each profile)
# ---------------------------------------------------------------------------

print("\n--- Step 3: Blurred mapping matrix ---")

with timer.section("blurred_mapping_matrix"):
    blurred_matrices = [func.operated_mapping_matrix_override for func in lp_linear_funcs]
    blurred_mapping_matrix = np.hstack(blurred_matrices) if len(blurred_matrices) > 1 else blurred_matrices[0]

print(f"  blurred_mapping_matrix shape: {blurred_mapping_matrix.shape}")

def blurred_mm_from_params(params_tree):
    """Compute blurred mapping matrix from a pytree-shaped ``ModelInstance``."""
    t = al.Tracer(galaxies=list(params_tree.galaxies))
    tti = al.TracerToInversion(
        dataset=aa.DatasetInterface(
            data=fit.profile_subtracted_image,
            noise_map=dataset.noise_map,
            grids=dataset.grids,
            psf=dataset.psf,
            sparse_operator=dataset.sparse_operator,
        ),
        tracer=t,
        settings=al.Settings(use_border_relocator=True),
        xp=jnp,
    )
    funcs = list(tti.lp_linear_func_list_galaxy_dict.keys())
    matrices = [f.operated_mapping_matrix_override for f in funcs]
    return jnp.hstack(matrices) if len(matrices) > 1 else matrices[0]

_, bmm_jit = jit_profile(blurred_mm_from_params, "blurred_mm_jit", params_tree)
likelihood_steps.append(("Blurred mapping matrix", timer.records[-1][1] / 10))

print(f"  blurred_mapping_matrix (JIT) shape: {bmm_jit.shape}")

# ---------------------------------------------------------------------------
# Step 4: Data vector (D)
# ---------------------------------------------------------------------------

print("\n--- Step 4: Data vector ---")

def compute_data_vector(blurred_mapping_matrix, image, noise_map):
    return al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=blurred_mapping_matrix,
        image=image,
        noise_map=noise_map,
    )

bmm_jnp = jnp.array(blurred_mapping_matrix)
noise_jnp = jnp.array(dataset.noise_map.array)

with timer.section("data_vector_eager"):
    data_vector = compute_data_vector(bmm_jnp, data_array, noise_jnp)
    block(data_vector)

_, data_vector = jit_profile(
    compute_data_vector, "data_vector_jit", bmm_jnp, data_array, noise_jnp
)
likelihood_steps.append(("Data vector (D)", timer.records[-1][1] / 10))

print(f"  data_vector shape: {data_vector.shape}")

# ---------------------------------------------------------------------------
# Step 5: Curvature matrix (F)
# ---------------------------------------------------------------------------

print("\n--- Step 5: Curvature matrix ---")

n_linear = bmm_jnp.shape[1]

def compute_curvature_matrix(blurred_mapping_matrix, noise_map):
    return al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix,
        noise_map=noise_map,
        add_to_curvature_diag=True,
        no_regularization_index_list=list(range(n_linear)),
        xp=jnp,
    )

with timer.section("curvature_matrix_eager"):
    curvature_matrix = compute_curvature_matrix(bmm_jnp, noise_jnp)
    block(curvature_matrix)

_, curvature_matrix = jit_profile(
    compute_curvature_matrix, "curvature_matrix_jit", bmm_jnp, noise_jnp
)
likelihood_steps.append(("Curvature matrix (F)", timer.records[-1][1] / 10))

print(f"  curvature_matrix shape: {curvature_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 6: Reconstruction (positive-only NNLS)
# ---------------------------------------------------------------------------

print("\n--- Step 6: Reconstruction (NNLS) ---")

def compute_reconstruction(data_vector, curvature_matrix):
    return al.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_matrix,
        xp=jnp,
    )

with timer.section("reconstruction_eager"):
    reconstruction = compute_reconstruction(
        jnp.array(data_vector), jnp.array(curvature_matrix)
    )
    block(reconstruction)

_, reconstruction = jit_profile(
    compute_reconstruction, "reconstruction_jit",
    jnp.array(data_vector), jnp.array(curvature_matrix)
)
likelihood_steps.append(("Reconstruction (NNLS)", timer.records[-1][1] / 10))

print(f"  reconstruction shape: {reconstruction.shape}")

# ---------------------------------------------------------------------------
# Step 7: Map reconstruction back to image plane
# ---------------------------------------------------------------------------

print("\n--- Step 7: Mapped reconstructed image ---")

def compute_mapped_recon(blurred_mapping_matrix, reconstruction):
    return al.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix,
        reconstruction=reconstruction,
        xp=jnp,
    )

with timer.section("mapped_recon_eager"):
    mapped_recon = compute_mapped_recon(bmm_jnp, jnp.array(reconstruction))
    block(mapped_recon)

_, mapped_recon = jit_profile(
    compute_mapped_recon, "mapped_recon_jit", bmm_jnp, jnp.array(reconstruction)
)
likelihood_steps.append(("Mapped reconstructed image", timer.records[-1][1] / 10))

print(f"  mapped_reconstructed_image shape: {mapped_recon.shape}")

# ---------------------------------------------------------------------------
# Step 8: Chi-squared and log likelihood
# ---------------------------------------------------------------------------

print("\n--- Step 8: Chi-squared & log likelihood ---")

def compute_log_likelihood(data, noise_map, mapped_recon):
    residual = data - mapped_recon
    chi_squared = jnp.sum((residual / noise_map) ** 2)
    noise_norm = jnp.sum(jnp.log(2 * jnp.pi * noise_map ** 2))
    return -0.5 * (chi_squared + noise_norm)

mapped_recon_jnp = jnp.array(mapped_recon)

with timer.section("log_likelihood_eager"):
    log_like = compute_log_likelihood(
        data_array, noise_jnp, mapped_recon_jnp
    )
    block(log_like)

_, log_like = jit_profile(
    compute_log_likelihood, "log_likelihood_jit",
    data_array, noise_jnp, mapped_recon_jnp
)
likelihood_steps.append(("Chi-squared & log likelihood", timer.records[-1][1] / 10))

print(f"  log_likelihood = {log_like}")

# Assert step-by-step result matches FitImaging.log_likelihood
# (log_likelihood = -0.5 * (chi_squared + noise_norm), same formula as compute_log_likelihood)
np.testing.assert_allclose(
    float(log_like),
    float(log_likelihood_ref),
    rtol=1e-4,
    err_msg="Step-by-step log_likelihood does not match FitImaging.log_likelihood",
)
print("  Assertion PASSED: step-by-step matches FitImaging.log_likelihood")

# ===================================================================
# PART C — Full-pipeline JIT for comparison
# ===================================================================

print("\n" + "=" * 70)
print("FULL-PIPELINE JIT (for comparison)")
print("=" * 70)

# Build the analysis with ``use_jax=True`` so its ``log_likelihood_function``
# threads ``xp=jnp`` through every internal call (border relocation, profile
# evaluation, inversion, etc.). This is the same wiring that ``Fitness.call``
# uses in production — we just feed it our pytree-native instance directly
# instead of going through ``model.instance_from_vector(parameters, xp=jnp)``.
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

def full_pipeline_from_params(params_tree):
    """Full likelihood from a pytree-shaped ``ModelInstance``.

    No flat-vector unpacking inside the trace — the instance crosses the JIT
    boundary directly, with constants (redshifts, etc.) kept static via the
    ``aux_data`` partition set up by ``autofit.jax.register_model``.
    """
    return analysis.log_likelihood_function(instance=params_tree)

_, full_result = jit_profile(full_pipeline_from_params, "full_pipeline", params_tree)
full_pipeline_per_call = timer.records[-1][1] / 10

print(f"  full log_likelihood = {full_result}")

# ===================================================================
# PART D — vmap + correctness
# ===================================================================

print("\n--- vmap batched evaluation ---")

batch_size = 3

# Build the batched pytree: every leaf gets a fresh leading batch axis. No
# flat-vector reshaping required — JAX walks the pytree via the registration
# we added in PART A.
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
    err_msg="mge: JAX vmap likelihood mismatch",
)
print("  Correctness check PASSED")

# ===================================================================
# PART E — Static memory analysis
# ===================================================================

print("\n--- Static memory analysis ---")

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
# JAX Likelihood Function Summary
# ===================================================================

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

print("\n" + "=" * 70)
print(f"JAX LIKELIHOOD FUNCTION SUMMARY — {instrument.upper()} — v{al_version}")
print("=" * 70)
print(f"  Instrument:            {instrument}")
print(f"  Pixel scale:           {pixel_scale} arcsec/pixel")
print(f"  Mask radius:           {mask_radius} arcsec")
print(f"  Image pixels (masked): {n_image_pixels}")
print(f"  Over-sampled pixels:   {n_over_sampled_pixels}")
print(f"  Linear Gaussians:      {n_linear_gaussians}")
print("-" * 70)

max_label = max(len(label) for label, _ in likelihood_steps)
step_total = 0.0
for i, (label, per_call) in enumerate(likelihood_steps, 1):
    print(f"  {i:>2}. {label:<{max_label}}  {per_call:>12.6f} s")
    step_total += per_call

print("-" * 70)
print(f"      {'TOTAL (step-by-step)':<{max_label}}  {step_total:>12.6f} s")
print(f"      {'Full pipeline (single JIT)':<{max_label}}  {full_pipeline_per_call:>12.6f} s")
print(f"      {f'vmap batch={batch_size} (per call)':<{max_label}}  {vmap_per_call:>12.6f} s")
print(f"      {f'vmap speedup vs single JIT':<{max_label}}  {vmap_speedup:>11.1f}x")
print("=" * 70)

# --- Save results dictionary ---

likelihood_summary = {
    "autolens_version": al_version,
    "instrument": instrument,
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "image_pixels_masked": int(n_image_pixels),
        "over_sampled_pixels": int(n_over_sampled_pixels),
        "linear_gaussians": int(n_linear_gaussians),
    },
    "steps": {label: per_call for label, per_call in likelihood_steps},
    "total_step_by_step": step_total,
    "full_pipeline_single_jit": full_pipeline_per_call,
    "vmap": {
        "batch_size": batch_size,
        "batch_time": vmap_batch_time,
        "per_call": vmap_per_call,
        "speedup_vs_single_jit": round(vmap_speedup, 1),
    },
}

results_dir = _script_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

dict_path = results_dir / f"mge_likelihood_summary_{instrument}_v{al_version}.json"
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart ---

labels = [label for label, _ in likelihood_steps]
times = [per_call for _, per_call in likelihood_steps]

fig, ax = plt.subplots(figsize=(10, 5))
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
ax.axvline(
    vmap_per_call,
    color="#55A868",
    linestyle="--",
    linewidth=1.5,
    label=f"vmap batch={batch_size} per call: {vmap_per_call:.6f} s ({vmap_speedup:.1f}x faster)",
)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Time per call (s)", fontsize=11)
fig.suptitle(
    f"MGE Imaging Likelihood — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f"AutoLens v{al_version}  |  {pixel_scale}\"/px  |  {n_image_pixels} pixels  |  "
    f"{n_over_sampled_pixels} over-sampled  |  {n_linear_gaussians} Gaussians  |  "
    f"total: {step_total:.6f} s",
    fontsize=9,
)
ax.legend(loc="lower right", fontsize=9)
ax.margins(x=0.15)
fig.tight_layout()

chart_path = results_dir / f"mge_likelihood_summary_{instrument}_v{al_version}.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")


# ===================================================================
# Regression assertion — realistic-scale deterministic likelihood
# ===================================================================
#
# Seeded simulator (noise_seed=1 in simulators/imaging.py) + fixed model
# parameters make the full-pipeline log-likelihood deterministic at this
# HST-scale dataset. Hardcoded value guards against silent regressions in
# the light-profile / blurring / chi-squared stack.
EXPECTED_LOG_LIKELIHOOD_HST = -159736.35504220804

np.testing.assert_allclose(
    log_likelihood_ref,
    EXPECTED_LOG_LIKELIHOOD_HST,
    rtol=1e-4,
    err_msg=(
        f"imaging/mge[{instrument}]: regression — eager log_likelihood drifted "
        f"(got {log_likelihood_ref}, expected {EXPECTED_LOG_LIKELIHOOD_HST})"
    ),
)
print(
    f"  Eager regression assertion PASSED: log_likelihood matches "
    f"{EXPECTED_LOG_LIKELIHOOD_HST:.6f}"
)
np.testing.assert_allclose(
    float(full_result),
    EXPECTED_LOG_LIKELIHOOD_HST,
    rtol=1e-4,
    err_msg=f"imaging/mge[{instrument}]: regression — full log_likelihood drifted",
)
np.testing.assert_allclose(
    np.array(result_vmap),
    EXPECTED_LOG_LIKELIHOOD_HST,
    rtol=1e-4,
    err_msg=f"imaging/mge[{instrument}]: regression — vmap log_likelihood drifted",
)
print(f"  Regression assertion PASSED: log_likelihood matches {EXPECTED_LOG_LIKELIHOOD_HST:.6f}")
