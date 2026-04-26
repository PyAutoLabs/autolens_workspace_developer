"""
JAX Gradient Testing: Delaunay Imaging Likelihood (Step-by-Step)
=================================================================

Companion to ``delaunay.py``. Replaces JIT profiling with
``jax.value_and_grad`` at each stage of the pipeline so we can isolate
which step (if any) breaks gradients for a Sersic lens + Delaunay
pixelization source model.

Modelled on ``pixelization_gradients.py`` (rectangular) and
``mge_gradients.py`` (linear-light source). The Delaunay mapper has two
specific gradient-killers that MGE / rectangular do not share:

- **Frozen triangulation**: ``scipy.spatial.Delaunay`` is built outside
  the JIT boundary, so the triangulation is a constant under tracing.
  Gradients that need to flow through it are silently zeroed.
- **Cell-boundary discontinuities**: barycentric weights are
  piecewise-linear in source-plane position with undefined gradients at
  cell boundaries — typically spiky / zero / NaN on a handful of leaves
  rather than catastrophic failure.

The script runs all stages, classifies each PASS / FAIL / ERROR, and
prints a summary table. It does not raise on FAIL — this is a
diagnostic, not a regression gate.

Run from the workspace root:

    python jax_profiling/imaging/delaunay_gradients.py

Current result on main (PyAutoArray @ 4ea58e1a)
-----------------------------------------------
**3 PASS / 0 FAIL / 8 ERROR.**

- Steps 1-3 (ray-trace, blurred lens light, profile-subtracted) PASS:
  the pre-inversion path is fully differentiable.
- Steps 4-11 ERROR with the same root cause:
  ``ValueError: Pure callbacks do not support JVP. Please use
  jax.custom_jvp to use callbacks while taking gradients.``
  The error is raised from
  ``PyAutoArray/autoarray/inversion/mesh/interpolator/delaunay.py:80``
  where ``jax_delaunay`` host-calls ``scipy.spatial.Delaunay`` via
  ``jax.pure_callback``. ``pure_callback`` has no registered JVP rule,
  so any ``jax.value_and_grad`` through the Delaunay inversion path
  hard-errors rather than silently zeroing.
- This means the prompt's "frozen triangulation" concern is not just
  hypothetical: the Delaunay path is currently *un-differentiable*
  end-to-end, not merely emitting zero gradients. A library fix would
  need to wrap the host call in ``jax.custom_jvp`` (most likely with a
  zero-JVP rule, since the triangulation is itself a discrete
  combinatorial structure) so the rest of the gradient pipeline can
  flow around it.
- Because Steps 4-11 raise before producing a curvature matrix, the
  PART B.5 NNLS kappa diagnostic also reports nothing useful — its
  ``_build_Q_q`` helper hits the same ``pure_callback`` error. Once
  the Delaunay JVP is added it should run normally.
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

# Delaunay-specific: image-plane mesh + edge points, copied from delaunay.py
overlay_shape = (26, 26)
edge_n_points = 30
mask_radius = 3.5


# ---------------------------------------------------------------------------
# Gradient test helper (identical to pixelization_gradients.py)
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
    except Exception as e:
        tb = traceback.format_exc()
        tb_short = "\n".join(tb.strip().splitlines()[-15:])
        print(f"\n  --> ERROR:\n{tb_short}")
        # Use exception type+message directly: traceback.format_exc()'s last
        # line can be a JAX traceback-filtering footer rather than the
        # exception itself, which makes the summary table unreadable.
        results.append((label, "ERROR", f"{type(e).__name__}: {e}"))
        return None, None


# ===================================================================
# PART A -- Setup (mirrors delaunay.py)
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
# Image-plane mesh + edge points (Delaunay-specific)
# ---------------------------------------------------------------------------

print("\n--- Image mesh construction (Delaunay) ---")

image_mesh = al.image_mesh.Overlay(shape=overlay_shape)
image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=dataset.mask)

edge_pixels_total = image_plane_mesh_grid.shape[0]
image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=(0.0, 0.0),
    radius=mask_radius,
    n_points=edge_n_points,
)
edge_pixels_total = image_plane_mesh_grid.shape[0] - edge_pixels_total

n_mesh_vertices = image_plane_mesh_grid.shape[0]
print(f"  Overlay shape: {overlay_shape}")
print(f"  Mesh vertices (incl. edge): {n_mesh_vertices}")
print(f"  Edge points added: {edge_pixels_total}")

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

mesh = al.mesh.Delaunay(
    pixels=n_mesh_vertices,
    zeroed_pixels=edge_pixels_total,
)
pixelization = al.Pixelization(
    mesh=mesh,
    regularization=al.reg.ConstantSplit(coefficient=1.0),
)
source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Delaunay pixels: {n_mesh_vertices}, edge zeroed: {edge_pixels_total}")

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

adapt_images = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={
        instance.galaxies.source: image_plane_mesh_grid,
    },
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid,
    },
)

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    adapt_images=adapt_images,
    settings=al.Settings(use_border_relocator=True),
    xp=np,
)
log_evidence_ref = fit.figure_of_merit
log_likelihood_ref = fit.log_likelihood

print(f"  log_evidence   = {log_evidence_ref}")
print(f"  log_likelihood = {log_likelihood_ref}")

# Reference value for the perturbed-params setup (PRNGKey(42), uniform 0.01-0.05).
# Differs from delaunay.py's +29179.95 (un-perturbed prior medians) — same
# pattern as pixelization_gradients.py vs pixelization.py.
EXPECTED_LOG_EVIDENCE_HST = -62305.31055677842

np.testing.assert_allclose(
    log_evidence_ref,
    EXPECTED_LOG_EVIDENCE_HST,
    rtol=1e-4,
    err_msg=(
        f"imaging/delaunay_gradients[{instrument}]: setup regression — eager "
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
# border relocation + Delaunay triangulation + barycentric interpolation
# + mapper construction + PSF convolution. AdaptImages is reconstructed
# against the new instance.galaxies.source so the dict-by-identity lookup
# resolves inside the trace (Delaunay-specific vs. pixelization_gradients.py).
# ---------------------------------------------------------------------------

def _fit_jax(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    adapt_images_jax = al.AdaptImages(
        galaxy_image_plane_mesh_grid_dict={
            inst.galaxies.source: image_plane_mesh_grid,
        },
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid,
        },
    )
    return al.FitImaging(
        dataset=dataset, tracer=t, adapt_images=adapt_images_jax,
        settings=al.Settings(use_border_relocator=True), xp=jnp,
    )


# ---------------------------------------------------------------------------
# Step 4: Delaunay mapping matrix (pre-PSF)
#
# Exercises border-relocate -> scipy.Delaunay -> barycentric weights ->
# mapper -> mapping matrix construction. This is the stage most likely to
# surface frozen-triangulation / cell-boundary discontinuity issues.
# ---------------------------------------------------------------------------

def step_mapping_matrix(params):
    fj = _fit_jax(params)
    mm = jnp.array(fj.inversion.mapping_matrix)
    return jnp.sum(mm)

test_grad("Step 4: Delaunay mapping matrix", step_mapping_matrix, jnp_params)


# ---------------------------------------------------------------------------
# Step 5: Blurred mapping matrix (PSF-convolved)
# ---------------------------------------------------------------------------

def step_blurred_mapping_matrix(params):
    fj = _fit_jax(params)
    bmm = jnp.array(fj.inversion.operated_mapping_matrix)
    return jnp.sum(bmm)

test_grad("Step 5: Blurred mapping matrix", step_blurred_mapping_matrix, jnp_params)


# ---------------------------------------------------------------------------
# Step 6: Data vector (D)
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

test_grad("Step 6: Data vector (D)", step_data_vector, jnp_params)


# ---------------------------------------------------------------------------
# Step 7: Curvature matrix (F)
#
# add_to_curvature_diag=False so the gradient flows through F itself
# rather than the constant diagonal floor (matches pixelization_gradients.py).
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

test_grad("Step 7: Curvature matrix (F)", step_curvature_matrix, jnp_params)


# ---------------------------------------------------------------------------
# Step 8: Regularization matrix (H) — ConstantSplit cross-derivative
# ---------------------------------------------------------------------------

def step_regularization_matrix(params):
    fj = _fit_jax(params)
    H = fj.inversion.regularization_matrix
    return jnp.sum(jnp.array(H))

test_grad("Step 8: Regularization matrix (H)", step_regularization_matrix, jnp_params)


# ---------------------------------------------------------------------------
# Step 9: Reconstruction (NNLS on F + H)
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

test_grad("Step 9: Reconstruction (NNLS)", step_reconstruction, jnp_params)


# ---------------------------------------------------------------------------
# Step 10: Mapped reconstructed image
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

test_grad("Step 10: Mapped reconstructed image", step_mapped_recon, jnp_params)


# ===================================================================
# PART B.5 -- NNLS kappa diagnostic loop
# ===================================================================
#
# If the NNLS step poisons gradients, increasing target_kappa from the
# library's 1e-2 default may repair them. Drives jaxnnls primitives
# directly (not jax.grad internals) on the real (Q, q) produced by the
# Delaunay pipeline.

print("\n" + "=" * 70)
print("PART B.5 -- NNLS KAPPA DIAGNOSTIC")
print("=" * 70)


def _build_Q_q(params):
    """Rebuild (curvature_reg_matrix, data_vector) for the given params."""
    fj = _fit_jax(params)
    bmm = jnp.array(fj.inversion.operated_mapping_matrix)
    profile_subtracted = jnp.array(fj.profile_subtracted_image.array)

    q = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
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
    return F + H, q


try:
    Q_eval, q_eval = _build_Q_q(jnp_params)
    Q_np = np.array(Q_eval)
    q_np = np.array(q_eval)

    print(f"\n--- Inputs to NNLS ---")
    print(f"  Q shape        : {Q_np.shape}")
    print(f"  q shape        : {q_np.shape}")
    print(f"  Q symmetry err : {np.max(np.abs(Q_np - Q_np.T)):.6g}")
    print(f"  Q cond (2-norm): {np.linalg.cond(Q_np):.6g}")
    eigs = np.linalg.eigvalsh(0.5 * (Q_np + Q_np.T))
    print(f"  Q eig min/max  : {eigs.min():.6g} / {eigs.max():.6g}")
    print(f"  Q is pos-def   : {eigs.min() > 0}")
    print(f"  q finite       : {np.all(np.isfinite(q_np))}")

    from jaxnnls.pdip import solve_nnls, factorize_kkt
    from jaxnnls.pdip_relaxed import solve_relaxed_nnls
    import jaxnnls.diff_qp as _dq

    def _diagnose_kappa(Q, q, target_kappa):
        print(f"\n--- target_kappa = {target_kappa:g} ---")
        x, s, z, conv_fw, iter_fw = solve_nnls(Q, q)
        x_np = np.array(x)
        print(f"  forward converged  : {int(conv_fw)}  iters: {int(iter_fw)} (cap 50)")
        print(f"  x (primal) min/max : {x_np.min():.6g} / {x_np.max():.6g}")
        print(f"  x finite           : {np.all(np.isfinite(x_np))}")
        print(f"  # active (x<=eps)  : {int(np.sum(x_np <= 1e-12))} / {x_np.size}")

        xr, sr, zr, conv_rx, iter_rx = solve_relaxed_nnls(
            Q, q, x, s, z, target_kappa=target_kappa
        )
        sr_np = np.array(sr)
        zr_np = np.array(zr)
        print(f"  relaxed converged  : {int(conv_rx)}  iters: {int(iter_rx)} (cap 50)")
        print(f"  sr finite          : {int(np.isfinite(sr_np).sum())}/{sr_np.size}")
        print(f"  zr finite          : {int(np.isfinite(zr_np).sum())}/{zr_np.size}")
        prod = sr_np * zr_np
        if np.any(np.isfinite(prod)):
            fprod = prod[np.isfinite(prod)]
            print(f"  sr*zr min/max (fin): {fprod.min():.6g} / {fprod.max():.6g} "
                  f"(target: {target_kappa:g})")

        try:
            P_inv_vec_j, L_H_pack = factorize_kkt(Q, sr, zr)
            L_H_mat = L_H_pack[0] if isinstance(L_H_pack, tuple) else L_H_pack
            L_H_np = np.array(L_H_mat)
            print(f"  L_H finite         : {np.all(np.isfinite(L_H_np))}")
            diag_abs = np.abs(np.diag(L_H_np))
            print(f"  L_H diag min/max   : {diag_abs.min():.6g} / {diag_abs.max():.6g}")
        except Exception as e:
            print(f"  factorize_kkt raised: {type(e).__name__}: {e}")

        def _loss(p):
            Q_p, q_p = _build_Q_q(p)
            x_p = _dq.solve_nnls_primal(Q_p, q_p, target_kappa=target_kappa)
            return jnp.sum(x_p)

        try:
            val, grad = jax.value_and_grad(_loss)(jnp_params)
            grad_np = np.asarray(grad).ravel()
            n_nan = int(np.sum(~np.isfinite(grad_np)))
            print(f"  grad finite entries: {grad_np.size - n_nan}/{grad_np.size}")
            if n_nan < grad_np.size:
                finite_g = grad_np[np.isfinite(grad_np)]
                print(f"  grad norm (finite) : {np.linalg.norm(finite_g):.6g}")
            if n_nan == 0:
                print(f"  *** kappa={target_kappa:g} PRODUCES FULLY FINITE GRADIENTS ***")
        except Exception as e:
            print(f"  grad raised        : {type(e).__name__}: {e}")

    print(f"\n  JAX x64 enabled    : {jax.config.read('jax_enable_x64')}")
    print(f"  Q dtype            : {Q_eval.dtype}")

    for kappa in (1e-3, 1e-2, 1e-1, 1.0):
        _diagnose_kappa(Q_eval, q_eval, kappa)
except Exception:
    tb = traceback.format_exc()
    tb_short = "\n".join(tb.strip().splitlines()[-15:])
    print(f"\n  Kappa diagnostic skipped (build_Q_q failed):\n{tb_short}")


# ===================================================================
# PART C -- Full pipeline gradient (via Fitness)
# ===================================================================

print("\n" + "=" * 70)
print("PART C -- FULL PIPELINE GRADIENT (via Fitness)")
print("=" * 70)

from autofit.non_linear.fitness import Fitness

analysis = al.AnalysisImaging(dataset=dataset, adapt_images=adapt_images)
fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

test_grad("Step 11: Full pipeline (Fitness.call)", fitness.call, jnp_params)


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
