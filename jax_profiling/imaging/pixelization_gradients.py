"""
JAX Gradient Testing: Pixelization Imaging Likelihood (Step-by-Step)
=====================================================================

Companion to ``pixelization.py``. Replaces JIT profiling with
``jax.value_and_grad`` at each stage of the pipeline so we can isolate
which step (if any) breaks gradients for a Sersic lens + rectangular
pixelization source model.

Run from the workspace root:

    python jax_profiling/imaging/pixelization_gradients.py

Historical bug: rectangular interpolator gradient explosion
-----------------------------------------------------------
Steps 4-6 (mapping matrix, data vector D, curvature matrix F) used to
emit gradients of order ~1e-24 for ``RectangularAdaptDensity`` -- i.e.
effectively zero, and NaN-poisoning the downstream NNLS. Root cause
lived inside
``autoarray.inversion.mesh.interpolator.rectangular.create_transforms``:

* It builds a rank-space transform from the ray-traced source grid via
  ``sort_points = jnp.sort(traced_points); transform = jnp.interp(q,
  sort_points, t)``.
* Ray-traced source grids contain large numbers of exact-duplicate
  coordinates (Isothermal + circular mask: ~50% of sorted gaps are
  exactly 0). See the ``mapper_grad_isolate.py`` probe output.
* ``jnp.interp``'s vjp (a) divides by ``sort_points[i+1] -
  sort_points[i]`` in the knot-gradient term, producing O(1e24)
  cotangents, and (b) returns a slope ``(yp[i+1]-yp[i]) /
  (xp[i+1]-xp[i])`` in the query-gradient term, which blows up for the
  same reason.

Fix (see ``rectangular.py::create_transforms`` for the full comment):

1. ``jax.lax.stop_gradient`` on ``sort_points`` -- drops the knot-
   gradient term. Semantically correct because its downstream consumer
   (``floor``/``ceil`` bin assignment in
   ``adaptive_rectangular_mappings_weights_...``) already discards
   bin-boundary derivatives.
2. Add a strictly-monotonic jitter ``arange(N) * 1e-7`` to
   ``sort_points`` before freezing -- guarantees non-zero knot gaps so
   the query-gradient slope is bounded (worst case ~650 for N~1.5e4).
   Forward interpolation output shifts by at most ~1.5e-3 in scaled
   source-plane units, well below the ``(source_grid_size - 3)``
   multiplier's sub-pixel sensitivity.

After this fix, steps 4-6 return finite gradients that agree with
finite differences up to O(1) ratio noise from bin-boundary
discontinuities. Steps 8-10 (NNLS / mapped reconstruction) may still
NaN for other reasons -- that is a separate NNLS-conditioning issue.
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

mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)


# ---------------------------------------------------------------------------
# Gradient test helper (identical to mge_gradients.py)
# ---------------------------------------------------------------------------

results = []


def test_grad(label, func, params):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    try:
        value, grad = jax.value_and_grad(func)(params)
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
            status, detail = "FAIL", "value is not finite"
        elif not np.all(np.isfinite(grad_np)):
            n_bad = np.count_nonzero(~np.isfinite(grad_np))
            status, detail = "FAIL", f"{n_bad}/{grad_np.size} gradient entries are non-finite"
        elif np.all(grad_np == 0.0):
            status, detail = "FAIL", "gradient is all zeros"
        else:
            status, detail = "PASS", f"norm={np.linalg.norm(grad_np):.6g}"

        print(f"\n  --> {status}: {detail}")
        results.append((label, status, detail))
        return value, grad
    except Exception:
        tb = traceback.format_exc()
        tb_short = "\n".join(tb.strip().splitlines()[-15:])
        print(f"\n  --> ERROR:\n{tb_short}")
        results.append((label, "ERROR", tb.strip().splitlines()[-1]))
        return None, None


# ===================================================================
# PART A -- Setup (mirrors pixelization.py)
# ===================================================================

print("\n" + "=" * 70)
print("PART A -- SETUP")
print("=" * 70)

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
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=4, over_sample_size_pixelization=1,
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size, over_sample_size_pixelization=1,
)

print(f"  Image pixels (masked): {dataset.data.shape[0]}")

# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

# GaussianPrior(mean=truth, sigma=small) centres prior-median at the
# simulator truth while keeping params free so gradient diagnostics
# have dimensionality.
lens_bulge = af.Model(al.lp.Sersic)
lens_bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
lens_bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
_lens_bulge_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
lens_bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=_lens_bulge_ell[0], sigma=0.01)
lens_bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=_lens_bulge_ell[1], sigma=0.01)
lens_bulge.intensity = af.GaussianPrior(mean=2.0, sigma=0.1)
lens_bulge.effective_radius = af.GaussianPrior(mean=0.6, sigma=0.05)
lens_bulge.sersic_index = af.GaussianPrior(mean=3.0, sigma=0.2)

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

lens = af.Model(
    al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear,
)

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)
source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Mesh shape: {mesh_shape}, source pixels: {mesh_pixels_yx ** 2}")

# ---------------------------------------------------------------------------
# Parameter vector (perturbed from prior medians)
# ---------------------------------------------------------------------------

jnp_params = jnp.array(model.physical_values_from_prior_medians)
key = jax.random.PRNGKey(42)
perturbation = jax.random.uniform(
    key, shape=jnp_params.shape, minval=0.01, maxval=0.05
)
jnp_params = jnp_params + perturbation
print(f"  param_vector shape: {jnp_params.shape}")

# ---------------------------------------------------------------------------
# Eager baseline
# ---------------------------------------------------------------------------

print("\n--- Eager baseline ---")

instance = model.instance_from_vector(vector=jnp_params)
tracer = al.Tracer(galaxies=list(instance.galaxies))

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    settings=al.Settings(use_border_relocator=True),
    xp=np,
)
log_evidence_ref = fit.figure_of_merit
log_likelihood_ref = fit.log_likelihood

print(f"  log_evidence   = {log_evidence_ref}")
print(f"  log_likelihood = {log_likelihood_ref}")

EXPECTED_LOG_EVIDENCE_HST = -66270.78281169113

np.testing.assert_allclose(
    log_evidence_ref,
    EXPECTED_LOG_EVIDENCE_HST,
    rtol=1e-4,
    err_msg=(
        f"imaging/pixelization_gradients[{instrument}]: regression — eager "
        f"log_evidence drifted (got {log_evidence_ref}, expected "
        f"{EXPECTED_LOG_EVIDENCE_HST})"
    ),
)
print(
    f"  Eager regression assertion PASSED: log_evidence matches "
    f"{EXPECTED_LOG_EVIDENCE_HST:.6f}"
)

grid_lp = dataset.grids.lp
grid_blurring = dataset.grids.blurring
data_array = jnp.array(dataset.data.array)
noise_map_array = jnp.array(dataset.noise_map.array)


# ===================================================================
# PART B -- Per-step gradient testing
# ===================================================================

print("\n" + "=" * 70)
print("PART B -- PER-STEP GRADIENT TESTING")
print("=" * 70)


# ---------------------------------------------------------------------------
# Step 1: Ray-trace pixelization grid
# ---------------------------------------------------------------------------

def step_ray_trace(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    grid_raw = jnp.array(dataset.grids.pixelization.array)
    grid = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    traced = t.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.sum(jnp.stack([tg.array for tg in traced]))

test_grad("Step 1: Ray-trace grids", step_ray_trace, jnp_params)


# ---------------------------------------------------------------------------
# Step 2: Blurred lens light image (Sersic through PSF)
# ---------------------------------------------------------------------------

def step_blurred_image(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    blurred = t.blurred_image_2d_from(
        grid=grid_lp, psf=dataset.psf, blurring_grid=grid_blurring, xp=jnp,
    )
    return jnp.sum(blurred.array)

test_grad("Step 2: Blurred lens light image", step_blurred_image, jnp_params)


# ---------------------------------------------------------------------------
# Step 3: Profile-subtracted image
# ---------------------------------------------------------------------------

def step_profile_subtracted(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    blurred = t.blurred_image_2d_from(
        grid=grid_lp, psf=dataset.psf, blurring_grid=grid_blurring, xp=jnp,
    )
    return jnp.sum(data_array - blurred.array)

test_grad("Step 3: Profile-subtracted image", step_profile_subtracted, jnp_params)


# ---------------------------------------------------------------------------
# Inversion matrix extraction helper
#
# Rebuilds FitImaging inside the trace so we get gradients through
# border relocation + mapper construction + PSF convolution.
# ---------------------------------------------------------------------------

def _fit_jax(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    return al.FitImaging(
        dataset=dataset, tracer=t,
        settings=al.Settings(use_border_relocator=True), xp=jnp,
    )


# ---------------------------------------------------------------------------
# Step 4: Blurred mapping matrix (inversion setup: border relocate + mapper
# + mapping matrix + PSF convolution)
# ---------------------------------------------------------------------------

def step_blurred_mapping_matrix(params):
    fj = _fit_jax(params)
    bmm = jnp.array(fj.inversion.operated_mapping_matrix)
    return jnp.sum(bmm)

test_grad("Step 4: Blurred mapping matrix", step_blurred_mapping_matrix, jnp_params)


# ---------------------------------------------------------------------------
# Step 5: Data vector (D)
# ---------------------------------------------------------------------------

def step_data_vector(params):
    fj = _fit_jax(params)
    bmm = jnp.array(fj.inversion.operated_mapping_matrix)
    profile_subtracted = jnp.array(fj.profile_subtracted_image.array)
    D = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=bmm,
        image=profile_subtracted,
        noise_map=noise_map_array,
    )
    return jnp.sum(D)

test_grad("Step 5: Data vector (D)", step_data_vector, jnp_params)


# ---------------------------------------------------------------------------
# Step 6: Curvature matrix (F)
# ---------------------------------------------------------------------------

def step_curvature_matrix(params):
    fj = _fit_jax(params)
    bmm = jnp.array(fj.inversion.operated_mapping_matrix)
    F = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=bmm,
        noise_map=noise_map_array,
        settings=fj.settings,
        add_to_curvature_diag=False,
        xp=jnp,
    )
    return jnp.sum(F)

test_grad("Step 6: Curvature matrix (F)", step_curvature_matrix, jnp_params)


# ---------------------------------------------------------------------------
# Step 7: Regularization matrix (H)
# ---------------------------------------------------------------------------

def step_regularization_matrix(params):
    fj = _fit_jax(params)
    H = fj.inversion.regularization_matrix
    return jnp.sum(jnp.array(H))

test_grad("Step 7: Regularization matrix (H)", step_regularization_matrix, jnp_params)


# ---------------------------------------------------------------------------
# Step 8: Reconstruction (NNLS on F + H)
# ---------------------------------------------------------------------------

def step_reconstruction(params):
    fj = _fit_jax(params)
    bmm = jnp.array(fj.inversion.operated_mapping_matrix)
    profile_subtracted = jnp.array(fj.profile_subtracted_image.array)
    D = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=bmm,
        image=profile_subtracted,
        noise_map=noise_map_array,
    )
    F = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=bmm,
        noise_map=noise_map_array,
        settings=fj.settings,
        add_to_curvature_diag=False,
        xp=jnp,
    )
    H = jnp.array(fj.inversion.regularization_matrix)
    s = al.util.inversion.reconstruction_positive_only_from(
        data_vector=D,
        curvature_reg_matrix=F + H,
        xp=jnp,
    )
    return jnp.sum(s)

test_grad("Step 8: Reconstruction (NNLS)", step_reconstruction, jnp_params)


# ---------------------------------------------------------------------------
# Step 9: Mapped reconstructed image
# ---------------------------------------------------------------------------

def step_mapped_recon(params):
    fj = _fit_jax(params)
    bmm = jnp.array(fj.inversion.operated_mapping_matrix)
    profile_subtracted = jnp.array(fj.profile_subtracted_image.array)
    D = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=bmm,
        image=profile_subtracted,
        noise_map=noise_map_array,
    )
    F = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=bmm,
        noise_map=noise_map_array,
        settings=fj.settings,
        add_to_curvature_diag=False,
        xp=jnp,
    )
    H = jnp.array(fj.inversion.regularization_matrix)
    s = al.util.inversion.reconstruction_positive_only_from(
        data_vector=D,
        curvature_reg_matrix=F + H,
        xp=jnp,
    )
    mapped = al.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=bmm,
        reconstruction=s,
        xp=jnp,
    )
    return jnp.sum(mapped)

test_grad("Step 9: Mapped reconstructed image", step_mapped_recon, jnp_params)


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
