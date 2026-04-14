"""
JAX Gradient Testing: MGE Imaging Likelihood (Step-by-Step)
============================================================

Tests whether ``jax.value_and_grad`` can differentiate each step of the MGE
imaging likelihood pipeline.  The existing ``mge.py`` profiles each step
under ``jax.jit``; this companion script replaces JIT profiling with gradient
testing so you can isolate exactly which step breaks ``jax.grad``.

Because the MGE model uses only linear light profiles (``lp_linear``), there
is no non-linear blurred image step — all light is reconstructed via the
linear inversion.

Each step defines a function ``params -> scalar`` and calls
``jax.value_and_grad``.  The output is a summary table showing which steps
produce valid (finite, non-zero) gradients and which do not.

Run from this directory:

    cd jax_profiling/imaging
    python mge_gradients.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import traceback
import subprocess
import sys
from pathlib import Path

import autofit as af
import autolens as al
import autoarray as aa

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

instrument = "hst"

INSTRUMENTS = {
    "euclid": {"pixel_scale": 0.1},
    "hst": {"pixel_scale": 0.05},
    "jwst": {"pixel_scale": 0.03},
    "ao": {"pixel_scale": 0.01},
}

# ---------------------------------------------------------------------------
# Gradient test helper
# ---------------------------------------------------------------------------

results = []  # (label, status, detail)


def test_grad(label, func, params):
    """Test ``jax.value_and_grad(func)(params)`` and record the result.

    Parameters
    ----------
    label
        Human-readable name for the pipeline step.
    func
        A function ``params -> scalar`` to differentiate.
    params
        JAX array of model parameters.

    Returns
    -------
    value, grad or None, None on failure.
    """
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    try:
        value, grad = jax.value_and_grad(func)(params)

        # Force evaluation.
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()
        if hasattr(grad, "block_until_ready"):
            grad.block_until_ready()

        val_f = float(value)
        grad_np = np.array(grad)

        print(f"  value       = {val_f:.8g}")
        print(f"  grad shape  = {grad_np.shape}")
        print(f"  grad norm   = {np.linalg.norm(grad_np):.8g}")
        print(f"  grad min    = {grad_np.min():.8g}")
        print(f"  grad max    = {grad_np.max():.8g}")
        print(f"  # non-zero  = {np.count_nonzero(grad_np)} / {grad_np.size}")

        if not np.isfinite(val_f):
            status = "FAIL"
            detail = "value is not finite"
        elif not np.all(np.isfinite(grad_np)):
            n_bad = np.count_nonzero(~np.isfinite(grad_np))
            status = "FAIL"
            detail = f"{n_bad}/{grad_np.size} gradient entries are non-finite"
        elif np.all(grad_np == 0.0):
            status = "FAIL"
            detail = "gradient is all zeros"
        else:
            status = "PASS"
            detail = f"norm={np.linalg.norm(grad_np):.6g}"

        print(f"\n  --> {status}: {detail}")
        results.append((label, status, detail))
        return value, grad

    except Exception:
        tb = traceback.format_exc()
        # Show last 15 lines to keep output manageable.
        tb_short = "\n".join(tb.strip().splitlines()[-15:])
        print(f"\n  --> ERROR:\n{tb_short}")
        results.append((label, "ERROR", tb.strip().splitlines()[-1]))
        return None, None


# ===================================================================
# PART A -- Setup (identical to mge.py)
# ===================================================================

print("\n" + "=" * 70)
print("PART A -- SETUP")
print("=" * 70)

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

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
)

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

print(f"  Image pixels (masked): {dataset.data.shape[0]}")

# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

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
# 3. Parameter vector
# ---------------------------------------------------------------------------

print("\n--- Parameter vector ---")

jnp_params = jnp.array(model.physical_values_from_prior_medians)

# Perturb ell_comps away from (0,0) to avoid degenerate arctan2 gradients.
key = jax.random.PRNGKey(42)
perturbation = jax.random.uniform(
    key, shape=jnp_params.shape, minval=0.01, maxval=0.05
)
jnp_params = jnp_params + perturbation

print(f"  param_vector shape: {jnp_params.shape}")

# ---------------------------------------------------------------------------
# 4. Eager baseline (to get reference objects for later steps)
# ---------------------------------------------------------------------------

print("\n--- Eager baseline ---")

instance = model.instance_from_vector(vector=jnp_params)
tracer = al.Tracer(galaxies=list(instance.galaxies))

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    settings=al.Settings(use_border_relocator=True),
)

print(f"  log_likelihood = {fit.log_likelihood}")

# Extract raw arrays for intermediate-step tests.
grid_lp = dataset.grids.lp
data_array = jnp.array(dataset.data.array)
noise_map_array = jnp.array(dataset.noise_map.array)


# ===================================================================
# PART B -- Per-step gradient testing
# ===================================================================

print("\n" + "=" * 70)
print("PART B -- PER-STEP GRADIENT TESTING")
print("=" * 70)

# ---------------------------------------------------------------------------
# Step 1: Ray-trace grids
# ---------------------------------------------------------------------------

def step_ray_trace(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    grid_raw = jnp.array(grid_lp.array)
    grid = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    traced = t.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.sum(jnp.stack([tg.array for tg in traced]))

test_grad("Step 1: Ray-trace grids", step_ray_trace, jnp_params)

# ---------------------------------------------------------------------------
# Step 2: Mapping matrix (linear profile images)
# ---------------------------------------------------------------------------

def step_mapping_matrix(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
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
    mm = jnp.hstack(matrices) if len(matrices) > 1 else matrices[0]
    return jnp.sum(mm)

test_grad("Step 2: Mapping matrix", step_mapping_matrix, jnp_params)

# ---------------------------------------------------------------------------
# Step 3: Blurred mapping matrix (PSF convolution of each profile)
# ---------------------------------------------------------------------------

def step_blurred_mapping_matrix(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
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
    bmm = jnp.hstack(matrices) if len(matrices) > 1 else matrices[0]
    return jnp.sum(bmm)

test_grad("Step 3: Blurred mapping matrix", step_blurred_mapping_matrix, jnp_params)

# ---------------------------------------------------------------------------
# Step 4: Data vector (D)
# ---------------------------------------------------------------------------

def step_data_vector(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))

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
    bmm = jnp.hstack(matrices) if len(matrices) > 1 else matrices[0]

    data_vector = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=bmm,
        image=data_array,
        noise_map=noise_map_array,
    )
    return jnp.sum(data_vector)

test_grad("Step 4: Data vector (D)", step_data_vector, jnp_params)

# ---------------------------------------------------------------------------
# Step 5: Curvature matrix (F)
# ---------------------------------------------------------------------------

def step_curvature_matrix(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
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
    bmm = jnp.hstack(matrices) if len(matrices) > 1 else matrices[0]

    n_linear = bmm.shape[1]
    curvature = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=bmm,
        noise_map=noise_map_array,
        add_to_curvature_diag=True,
        no_regularization_index_list=list(range(n_linear)),
        xp=jnp,
    )
    return jnp.sum(curvature)

test_grad("Step 5: Curvature matrix (F)", step_curvature_matrix, jnp_params)

# ---------------------------------------------------------------------------
# Step 6: Reconstruction (NNLS)
# ---------------------------------------------------------------------------

def step_reconstruction(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))

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
    bmm = jnp.hstack(matrices) if len(matrices) > 1 else matrices[0]

    data_vector = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=bmm,
        image=data_array,
        noise_map=noise_map_array,
    )

    n_linear = bmm.shape[1]
    curvature = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=bmm,
        noise_map=noise_map_array,
        add_to_curvature_diag=True,
        no_regularization_index_list=list(range(n_linear)),
        xp=jnp,
    )

    reconstruction = al.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature,
        xp=jnp,
    )
    return jnp.sum(reconstruction)

test_grad("Step 6: Reconstruction (NNLS)", step_reconstruction, jnp_params)

# ---------------------------------------------------------------------------
# Step 7: Mapped reconstructed image
# ---------------------------------------------------------------------------

def step_mapped_recon(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))

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
    bmm = jnp.hstack(matrices) if len(matrices) > 1 else matrices[0]

    data_vector = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=bmm,
        image=data_array,
        noise_map=noise_map_array,
    )

    n_linear = bmm.shape[1]
    curvature = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=bmm,
        noise_map=noise_map_array,
        add_to_curvature_diag=True,
        no_regularization_index_list=list(range(n_linear)),
        xp=jnp,
    )

    reconstruction = al.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature,
        xp=jnp,
    )

    mapped_recon = al.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=bmm,
        reconstruction=reconstruction,
        xp=jnp,
    )
    return jnp.sum(mapped_recon)

test_grad("Step 7: Mapped reconstructed image", step_mapped_recon, jnp_params)

# ---------------------------------------------------------------------------
# Step 8: Log likelihood (chi-squared)
# ---------------------------------------------------------------------------

def step_log_likelihood(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))

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
    bmm = jnp.hstack(matrices) if len(matrices) > 1 else matrices[0]

    data_vector = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=bmm,
        image=data_array,
        noise_map=noise_map_array,
    )

    n_linear = bmm.shape[1]
    curvature = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=bmm,
        noise_map=noise_map_array,
        add_to_curvature_diag=True,
        no_regularization_index_list=list(range(n_linear)),
        xp=jnp,
    )

    reconstruction = al.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature,
        xp=jnp,
    )

    mapped_recon = al.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=bmm,
        reconstruction=reconstruction,
        xp=jnp,
    )

    residual = data_array - mapped_recon
    chi_squared = jnp.sum((residual / noise_map_array) ** 2)
    noise_norm = jnp.sum(jnp.log(2 * jnp.pi * noise_map_array ** 2))
    return -0.5 * (chi_squared + noise_norm)

test_grad("Step 8: Log likelihood", step_log_likelihood, jnp_params)


# ===================================================================
# PART C -- Full pipeline gradient (via Fitness)
# ===================================================================

print("\n" + "=" * 70)
print("PART C -- FULL PIPELINE GRADIENT (via Fitness)")
print("=" * 70)

from autofit.non_linear.fitness import Fitness

analysis = al.AnalysisImaging(dataset=dataset)

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

test_grad("Full pipeline (Fitness.call)", fitness.call, jnp_params)


# ===================================================================
# PART D -- Summary
# ===================================================================

print("\n" + "=" * 70)
print("GRADIENT TEST SUMMARY")
print("=" * 70)

max_label = max(len(r[0]) for r in results)
for label, status, detail in results:
    marker = {"PASS": "+", "FAIL": "-", "ERROR": "!"}[status]
    print(f"  [{marker}] {label:<{max_label}}  {status:<5}  {detail}")

n_pass = sum(1 for _, s, _ in results if s == "PASS")
n_fail = sum(1 for _, s, _ in results if s == "FAIL")
n_error = sum(1 for _, s, _ in results if s == "ERROR")

print("-" * 70)
print(f"  {n_pass} passed, {n_fail} failed, {n_error} errors out of {len(results)} tests")
print("=" * 70)
