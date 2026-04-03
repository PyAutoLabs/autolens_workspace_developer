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

Caveat: XLA may fuse operations differently when compiled as one program vs
separate pieces, so per-step timings are approximate. They are still useful
for identifying which step dominates.

All JAX timings use `block_until_ready()` to force synchronous measurement.
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
from pathlib import Path
from contextlib import contextmanager

import autofit as af
import autolens as al
import autoarray as aa


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

# ===================================================================
# PART A — Setup (not JIT-compiled)
# ===================================================================

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

print("\n--- Dataset loading & masking ---")

with timer.section("dataset_load"):
    dataset_name = "source_complex"
    dataset_path = Path("dataset") / "imaging" / dataset_name

    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=0.05,
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
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )

    mass = af.Model(al.mp.NFWSph)

    total_gaussians = 3
    mask_radius = 3.0
    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    gaussian_list = af.Collection(
        af.Model(al.lmp_linear.GaussianGradient) for _ in range(total_gaussians)
    )

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]
        gaussian.mass_to_light_ratio = 10.0
        gaussian.mass_to_light_gradient = 1.0

    bulge_gaussian_list = list(gaussian_list)

    bulge = af.Model(
        al.lp_basis.Basis,
        profile_list=bulge_gaussian_list,
    )

    shear = af.Model(al.mp.ExternalShear)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

    mask_radius = 3.0
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")

# ---------------------------------------------------------------------------
# 3. Instantiate concrete objects from prior medians
# ---------------------------------------------------------------------------

print("\n--- Instantiate concrete model ---")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

tracer = al.Tracer(galaxies=list(instance.galaxies))

print(f"  Tracer planes: {tracer.total_planes}")

# ---------------------------------------------------------------------------
# 4. Full-pipeline reference (FitImaging) — eager baseline
# ---------------------------------------------------------------------------

print("\n--- Full FitImaging (eager baseline) ---")

with timer.section("fit_imaging_eager"):
    fit = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        settings=al.Settings(use_border_relocator=True),
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
grid_blurring_raw = jnp.array(dataset.grids.blurring.array)
data_array = jnp.array(dataset.data.array)
noise_map_array = jnp.array(dataset.noise_map.array)

# Keep autoarray objects for eager calls that need them.
grid_lp = dataset.grids.lp
grid_blurring = dataset.grids.blurring

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

print(f"  traced_grids shape: {traced_grids_raw.shape}")

# ---------------------------------------------------------------------------
# Step 2: Blurred image of non-linear light profiles
# ---------------------------------------------------------------------------

print("\n--- Step 2: Blurred image (non-linear profiles) ---")

with timer.section("blurred_image_eager"):
    blurred_image = tracer.blurred_image_2d_from(
        grid=grid_lp,
        psf=dataset.psf,
        blurring_grid=grid_blurring,
        xp=jnp,
    )
    block(blurred_image)

print(f"  blurred_image shape: {blurred_image.array.shape}")

# ---------------------------------------------------------------------------
# Step 3: Profile-subtracted image
# ---------------------------------------------------------------------------

print("\n--- Step 3: Profile-subtracted image ---")

def profile_subtract(data, blurred_image):
    return data - blurred_image

with timer.section("profile_subtract_eager"):
    blurred_img_jnp = jnp.array(blurred_image.array)
    profile_subtracted = profile_subtract(data_array, blurred_img_jnp)
    block(profile_subtracted)

_, profile_subtracted = jit_profile(
    profile_subtract, "profile_subtract_jit", data_array, blurred_img_jnp
)

print(f"  profile_subtracted shape: {profile_subtracted.shape}")

# ---------------------------------------------------------------------------
# Step 4: Build linear objects and mapping matrix
# ---------------------------------------------------------------------------

print("\n--- Step 4: Mapping matrix (linear profile images) ---")

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

# ---------------------------------------------------------------------------
# Step 5: Blurred mapping matrix (PSF convolution of each profile)
# ---------------------------------------------------------------------------

print("\n--- Step 5: Blurred mapping matrix ---")

with timer.section("blurred_mapping_matrix"):
    blurred_matrices = [func.operated_mapping_matrix_override for func in lp_linear_funcs]
    blurred_mapping_matrix = np.hstack(blurred_matrices) if len(blurred_matrices) > 1 else blurred_matrices[0]

print(f"  blurred_mapping_matrix shape: {blurred_mapping_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 6: Data vector (D)
# ---------------------------------------------------------------------------

print("\n--- Step 6: Data vector ---")

def compute_data_vector(blurred_mapping_matrix, image, noise_map):
    return al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=blurred_mapping_matrix,
        image=image,
        noise_map=noise_map,
    )

bmm_jnp = jnp.array(blurred_mapping_matrix)
profile_sub_jnp = jnp.array(fit.profile_subtracted_image.array)
noise_jnp = jnp.array(dataset.noise_map.array)

with timer.section("data_vector_eager"):
    data_vector = compute_data_vector(bmm_jnp, profile_sub_jnp, noise_jnp)
    block(data_vector)

_, data_vector = jit_profile(
    compute_data_vector, "data_vector_jit", bmm_jnp, profile_sub_jnp, noise_jnp
)

print(f"  data_vector shape: {data_vector.shape}")

# ---------------------------------------------------------------------------
# Step 7: Curvature matrix (F)
# ---------------------------------------------------------------------------

print("\n--- Step 7: Curvature matrix ---")

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

print(f"  curvature_matrix shape: {curvature_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 8: Reconstruction (positive-only NNLS)
# ---------------------------------------------------------------------------

print("\n--- Step 8: Reconstruction (NNLS) ---")

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

print(f"  reconstruction shape: {reconstruction.shape}")

# ---------------------------------------------------------------------------
# Step 9: Map reconstruction back to image plane
# ---------------------------------------------------------------------------

print("\n--- Step 9: Mapped reconstructed image ---")

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

print(f"  mapped_reconstructed_image shape: {mapped_recon.shape}")

# ---------------------------------------------------------------------------
# Step 10: Chi-squared and log likelihood
# ---------------------------------------------------------------------------

print("\n--- Step 10: Chi-squared & log likelihood ---")

def compute_log_likelihood(data, noise_map, blurred_image, mapped_recon):
    model_data = blurred_image + mapped_recon
    residual = data - model_data
    chi_squared = jnp.sum((residual / noise_map) ** 2)
    noise_norm = jnp.sum(jnp.log(2 * jnp.pi * noise_map ** 2))
    return -0.5 * (chi_squared + noise_norm)

blurred_img_jnp = jnp.array(blurred_image.array)
mapped_recon_jnp = jnp.array(mapped_recon)

with timer.section("log_likelihood_eager"):
    log_like = compute_log_likelihood(
        data_array, noise_jnp, blurred_img_jnp, mapped_recon_jnp
    )
    block(log_like)

_, log_like = jit_profile(
    compute_log_likelihood, "log_likelihood_jit",
    data_array, noise_jnp, blurred_img_jnp, mapped_recon_jnp
)

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

from autofit.non_linear.fitness import Fitness

analysis = al.AnalysisImaging(dataset=dataset)

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

jnp_params = jnp.array(model.physical_values_from_prior_medians)

_, full_result = jit_profile(fitness.call, "full_pipeline", jnp_params)

print(f"  full log_likelihood = {full_result}")

# ===================================================================
# PART D — vmap + correctness
# ===================================================================

print("\n--- vmap batched evaluation ---")

batch_size = 3
parameters = jnp.tile(jnp_params, (batch_size, 1))

with timer.section("vmap_first_call"):
    result_vmap = fitness._vmap(parameters)
    block(result_vmap)

with timer.section("vmap_cached_call"):
    result_vmap = fitness._vmap(parameters)
    block(result_vmap)

print(f"  batch results = {result_vmap}")

np.testing.assert_allclose(
    np.array(result_vmap),
    -93643.31852545,
    rtol=1e-4,
    err_msg="mge: JAX vmap likelihood mismatch",
)
print("  Correctness check PASSED")

# ===================================================================
# PART E — Static memory analysis
# ===================================================================

print("\n--- Static memory analysis ---")

batched_call = jax.jit(jax.vmap(fitness.call))
lowered_batched = batched_call.lower(parameters)
compiled_batched = lowered_batched.compile()

memory_analysis = compiled_batched.memory_analysis()
print(f"  Output size:  {memory_analysis.output_size_in_bytes / 1024**2:.3f} MB")
print(f"  Temp size:    {memory_analysis.temp_size_in_bytes / 1024**2:.3f} MB")
print(
    f"  Total:        "
    f"{(memory_analysis.output_size_in_bytes + memory_analysis.temp_size_in_bytes) / 1024**2:.3f} MB"
)

try:
    cost = compiled_batched.cost_analysis()
    if cost is not None:
        for i, device_cost in enumerate(cost):
            print(f"\n  Device {i} cost analysis:")
            for key, value in sorted(device_cost.items()):
                print(f"    {key}: {value}")
except Exception as e:
    print(f"  cost_analysis not available: {e}")

# ===================================================================
# Summary
# ===================================================================

timer.summary()
