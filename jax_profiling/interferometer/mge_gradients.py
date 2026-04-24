"""
JAX Gradient Testing: MGE Interferometer Likelihood (Step-by-Step)
==================================================================

Companion to ``jax_profiling/interferometer/mge.py``. Where that script JIT-
profiles the forward pass, this one walks the interferometer MGE likelihood
pipeline stage-by-stage and reports whether ``jax.value_and_grad`` returns
finite, non-zero gradients at each stage (PASS / FAIL / ERROR).

Mirrors ``jax_profiling/imaging/mge_gradients.py`` exactly in structure; the
per-step bodies swap PSF-blurring for NUFFT/DFT visibility transforms and
work in complex visibility space for D, F, residuals, and the log likelihood.
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
from autofit.jax import register_model as _register_model_pytrees

# JAX x64 is auto-enabled by autoconf at import time (see
# autoconf/jax_wrapper.py: sets JAX_ENABLE_X64="True" before jax imports),
# so we do not need to call jax.config.update here.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

instrument = "sma"

INSTRUMENTS = {
    "sma": {"pixel_scale": 0.1, "real_space_shape": (256, 256)},
    "alma": {"pixel_scale": 0.05, "real_space_shape": (256, 256)},
}

# ---------------------------------------------------------------------------
# Gradient test helper
# ---------------------------------------------------------------------------

results = []  # (label, status, detail)


def test_grad(label, func, params):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    try:
        value, grad = jax.value_and_grad(func)(params)

        if hasattr(value, "block_until_ready"):
            value.block_until_ready()

        val_f = float(value)
        grad_leaves = jax.tree_util.tree_leaves(grad)
        for leaf in grad_leaves:
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        grad_np = (
            np.concatenate([np.asarray(l).ravel() for l in grad_leaves])
            if grad_leaves
            else np.array([])
        )

        print(f"  value       = {val_f:.8g}")
        print(f"  grad leaves = {len(grad_leaves)}")
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
        tb_short = "\n".join(tb.strip().splitlines()[-15:])
        print(f"\n  --> ERROR:\n{tb_short}")
        results.append((label, "ERROR", tb.strip().splitlines()[-1]))
        return None, None


# ===================================================================
# PART A -- Setup (matches interferometer/mge.py)
# ===================================================================

print("\n" + "=" * 70)
print("PART A -- SETUP")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

print(f"\n--- Dataset loading [{instrument}] ---")

_script_dir = Path(__file__).resolve().parent
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
real_space_shape = INSTRUMENTS[instrument]["real_space_shape"]
dataset_path = Path("jax_profiling") / "interferometer" / "dataset" / "interferometer" / instrument

if al.util.dataset.should_simulate(str(dataset_path)):
    print(f"  Simulating {instrument} dataset...")
    subprocess.run(
        [
            sys.executable,
            str(_script_dir / "simulators" / "interferometer.py"),
            "--instrument", instrument,
        ],
        cwd=str(_script_dir),
        check=True,
    )

mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=real_space_shape,
    pixel_scales=pixel_scale,
    radius=mask_radius,
)

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

n_visibilities = dataset.uv_wavelengths.shape[0]
print(f"  Total visibilities: {n_visibilities}")

# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

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

# Simulator truth source centre is (0.1, 0.1); set via mge_model_from's
# centre kwarg so the shared centre prior's median lands there.
source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    centre_prior_is_uniform=False,
    centre=(0.1, 0.1),
    centre_sigma=0.005,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")

# ---------------------------------------------------------------------------
# 3. Parameter vector / pytree instance
# ---------------------------------------------------------------------------

print("\n--- Parameter vector ---")

param_vector = model.physical_values_from_prior_medians
instance = model.instance_from_vector(vector=param_vector)

# Perturb ell_comps away from (0,0) to avoid degenerate arctan2 gradients.
# Done at the flat-vector layer so downstream tree structure is preserved.
jnp_params = jnp.array(param_vector)
key = jax.random.PRNGKey(42)
perturbation = jax.random.uniform(
    key, shape=jnp_params.shape, minval=0.01, maxval=0.05
)
jnp_params = jnp_params + perturbation
instance = model.instance_from_vector(vector=np.array(jnp_params))

_register_model_pytrees(model)
params_tree = jax.tree_util.tree_map(jnp.asarray, instance)

tracer = al.Tracer(galaxies=list(instance.galaxies))

# ---------------------------------------------------------------------------
# 4. Eager baseline
# ---------------------------------------------------------------------------

print("\n--- Eager baseline ---")

fit = al.FitInterferometer(
    dataset=dataset,
    tracer=tracer,
    xp=np,
)
figure_of_merit_ref = fit.figure_of_merit
log_likelihood_ref = fit.log_likelihood

print(f"  figure_of_merit = {figure_of_merit_ref}")
print(f"  log_likelihood  = {log_likelihood_ref}")

EXPECTED_LOG_LIKELIHOOD_SMA = -3153.4301153111696

np.testing.assert_allclose(
    log_likelihood_ref,
    EXPECTED_LOG_LIKELIHOOD_SMA,
    rtol=1e-4,
    err_msg=(
        f"interferometer/mge_gradients[{instrument}]: regression — eager "
        f"log_likelihood drifted (got {log_likelihood_ref}, expected "
        f"{EXPECTED_LOG_LIKELIHOOD_SMA})"
    ),
)
print(
    f"  Eager regression assertion PASSED: log_likelihood matches "
    f"{EXPECTED_LOG_LIKELIHOOD_SMA:.6f}"
)

# Raw arrays for intermediate-step tests.
grid_lp = dataset.grids.lp
data_array = jnp.array(dataset.data.array)           # complex128
noise_map_array = jnp.array(dataset.noise_map.array)  # complex128


# ===================================================================
# PART B -- Per-step gradient testing
# ===================================================================

print("\n" + "=" * 70)
print("PART B -- PER-STEP GRADIENT TESTING")
print("=" * 70)


def _build_tti(params):
    t = al.Tracer(galaxies=list(params.galaxies))
    return al.TracerToInversion(
        dataset=aa.DatasetInterface(
            data=fit.profile_subtracted_visibilities,
            noise_map=dataset.noise_map,
            grids=dataset.grids,
            transformer=dataset.transformer,
            sparse_operator=dataset.sparse_operator,
        ),
        tracer=t,
        settings=al.Settings(use_border_relocator=True),
        xp=jnp,
    )


def _funcs_and_mm(params):
    tti = _build_tti(params)
    funcs = list(tti.lp_linear_func_list_galaxy_dict.keys())
    matrices = [f.mapping_matrix for f in funcs]
    mm = jnp.hstack(matrices) if len(matrices) > 1 else matrices[0]
    return funcs, mm


def _transformed_mm(params):
    _, mm = _funcs_and_mm(params)
    return dataset.transformer.transform_mapping_matrix(
        mapping_matrix=mm, xp=jnp
    )


DIAG_VALUE_OVERRIDE = None  # set to a float to override settings default


def _curvature_and_data_vector(params):
    tm = _transformed_mm(params)
    data_vector = al.util.inversion_interferometer.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=tm,
        visibilities=data_array,
        noise_map=noise_map_array,
    )

    settings = al.Settings()
    F_real = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=tm.real,
        noise_map=noise_map_array.real,
        settings=settings,
        xp=jnp,
    )
    F_imag = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=tm.imag,
        noise_map=noise_map_array.imag,
        settings=settings,
        xp=jnp,
    )
    F = F_real + F_imag
    n_linear = F.shape[0]
    diag_value = (
        DIAG_VALUE_OVERRIDE
        if DIAG_VALUE_OVERRIDE is not None
        else settings.no_regularization_add_to_curvature_diag_value
    )
    F = al.util.inversion.curvature_matrix_with_added_to_diag_from(
        curvature_matrix=F,
        value=diag_value,
        no_regularization_index_list=list(range(n_linear)),
        xp=jnp,
    )
    return data_vector, F, tm


# ---------------------------------------------------------------------------
# Step 1: Ray-trace grids
# ---------------------------------------------------------------------------

def step_ray_trace(params):
    t = al.Tracer(galaxies=list(params.galaxies))
    grid_raw = jnp.array(grid_lp.array)
    grid = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    traced = t.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.sum(jnp.stack([tg.array for tg in traced]))

test_grad("Step 1: Ray-trace grids", step_ray_trace, params_tree)

# ---------------------------------------------------------------------------
# Step 2: Mapping matrix (real space, linear profile images)
# ---------------------------------------------------------------------------

def step_mapping_matrix(params):
    _, mm = _funcs_and_mm(params)
    return jnp.sum(mm)

test_grad("Step 2: Mapping matrix (real space)", step_mapping_matrix, params_tree)

# ---------------------------------------------------------------------------
# Step 3: Transformed mapping matrix (DFT / NUFFT -> visibilities)
# ---------------------------------------------------------------------------

def step_transformed_mapping_matrix(params):
    tm = _transformed_mm(params)
    return jnp.sum(tm.real) + jnp.sum(tm.imag)

test_grad("Step 3: Transformed mapping matrix", step_transformed_mapping_matrix, params_tree)

# ---------------------------------------------------------------------------
# Step 4: Data vector D (visibilities space)
# ---------------------------------------------------------------------------

def step_data_vector(params):
    tm = _transformed_mm(params)
    data_vector = al.util.inversion_interferometer.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=tm,
        visibilities=data_array,
        noise_map=noise_map_array,
    )
    return jnp.sum(data_vector)

test_grad("Step 4: Data vector (D, visibilities)", step_data_vector, params_tree)

# ---------------------------------------------------------------------------
# Step 5: Curvature matrix F (real + imag summed)
# ---------------------------------------------------------------------------

def step_curvature_matrix(params):
    _, F, _ = _curvature_and_data_vector(params)
    return jnp.sum(F)

test_grad("Step 5: Curvature matrix (F)", step_curvature_matrix, params_tree)

# ---------------------------------------------------------------------------
# Stock library defaults from here on. The κ=1e-11 nnls_target_kappa in
# PyAutoArray/autoarray/config/general.yaml yields finite gradients even at
# the default diag=1e-3 (cond(F) ~ 5.7e4 on the interferometer+MGE stack).
# PART B.6 below documents the old failure mode by sweeping diag_value.
# ---------------------------------------------------------------------------
DIAG_VALUE_OVERRIDE = None

# ---------------------------------------------------------------------------
# Step 6: Reconstruction (NNLS)
# ---------------------------------------------------------------------------

def step_reconstruction(params):
    data_vector, F, _ = _curvature_and_data_vector(params)
    reconstruction = al.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=F,
        xp=jnp,
    )
    return jnp.sum(reconstruction)

test_grad("Step 6: Reconstruction (NNLS)", step_reconstruction, params_tree)

# ---------------------------------------------------------------------------
# Step 7: Mapped reconstructed visibilities
# ---------------------------------------------------------------------------

def step_mapped_recon(params):
    data_vector, F, tm = _curvature_and_data_vector(params)
    reconstruction = al.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=F,
        xp=jnp,
    )
    model_vis = al.util.inversion_interferometer.mapped_reconstructed_visibilities_from(
        transformed_mapping_matrix=tm,
        reconstruction=reconstruction,
    )
    return jnp.sum(model_vis.real) + jnp.sum(model_vis.imag)

test_grad("Step 7: Mapped reconstructed visibilities", step_mapped_recon, params_tree)

# ---------------------------------------------------------------------------
# Step 8: Log likelihood (visibilities chi-squared)
# ---------------------------------------------------------------------------

def step_log_likelihood(params):
    data_vector, F, tm = _curvature_and_data_vector(params)
    reconstruction = al.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=F,
        xp=jnp,
    )
    model_vis = al.util.inversion_interferometer.mapped_reconstructed_visibilities_from(
        transformed_mapping_matrix=tm,
        reconstruction=reconstruction,
    )

    residual = data_array - model_vis
    chi_real = jnp.sum((residual.real / noise_map_array.real) ** 2)
    chi_imag = jnp.sum((residual.imag / noise_map_array.imag) ** 2)
    chi_squared = chi_real + chi_imag
    noise_norm = jnp.sum(jnp.log(2 * jnp.pi * noise_map_array.real ** 2)) + \
                 jnp.sum(jnp.log(2 * jnp.pi * noise_map_array.imag ** 2))
    return -0.5 * (chi_squared + noise_norm)

test_grad("Step 8: Log likelihood", step_log_likelihood, params_tree)


# ===================================================================
# PART B.5 -- NNLS backward-pass diagnostics
# ===================================================================
#
# Below we rebuild Q at the *default* diag value to document the NaN that
# motivated this workaround. PART B.6 sweeps diag_value to find stable ones.

DIAG_VALUE_OVERRIDE = None

print("\n" + "=" * 70)
print("PART B.5 -- NNLS BACKWARD-PASS DIAGNOSTICS (default diag=1e-3)")
print("=" * 70)


def _build_Q_q(params):
    data_vector, F, _ = _curvature_and_data_vector(params)
    return F, data_vector


Q_eval, q_eval = _build_Q_q(params_tree)
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


def _diagnose_kappa(Q, q, target_kappa, precondition=False):
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
    print(f"  sr min/max         : {np.nanmin(sr_np):.6g} / {np.nanmax(sr_np):.6g}")
    print(f"  zr min/max         : {np.nanmin(zr_np):.6g} / {np.nanmax(zr_np):.6g}")
    print(f"  sr finite          : {int(np.isfinite(sr_np).sum())}/{sr_np.size}")
    print(f"  zr finite          : {int(np.isfinite(zr_np).sum())}/{zr_np.size}")
    with np.errstate(divide="ignore", invalid="ignore"):
        P_inv = zr_np / sr_np
    print(f"  P_inv_vec finite   : {int(np.isfinite(P_inv).sum())}/{P_inv.size}")
    if np.any(np.isfinite(P_inv)):
        finite_P = P_inv[np.isfinite(P_inv)]
        print(f"  P_inv min/max (fin): {finite_P.min():.6g} / {finite_P.max():.6g}")
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

    import jaxnnls.diff_qp as _dq

    def _loss(params):
        Q_p, q_p = _build_Q_q(params)
        if precondition:
            d = jnp.sqrt(jnp.diag(Q_p))
            D = 1.0 / d
            Q_p = (Q_p * D[:, None]) * D[None, :]
            q_p = q_p * D
            y = _dq.solve_nnls_primal(Q_p, q_p, target_kappa=target_kappa)
            x_p = y * D
        else:
            x_p = _dq.solve_nnls_primal(Q_p, q_p, target_kappa=target_kappa)
        return jnp.sum(x_p)

    try:
        val, grad = jax.value_and_grad(_loss)(params_tree)
        grad_leaves = jax.tree_util.tree_leaves(grad)
        grad_np = (
            np.concatenate([np.asarray(l).ravel() for l in grad_leaves])
            if grad_leaves
            else np.array([])
        )
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


# -----------------------------------------------------------------------------
# Jacobi (diagonal) preconditioning trial
# -----------------------------------------------------------------------------

print("\n--- Jacobi preconditioning trial ---")
d = np.sqrt(np.diag(Q_np))
if np.any(d == 0):
    print("  diag(Q) has zeros, Jacobi preconditioning skipped")
else:
    D = 1.0 / d
    Q_pc_np = (Q_np * D[:, None]) * D[None, :]
    q_pc_np = q_np * D
    print(f"  original cond(Q)   : {np.linalg.cond(Q_np):.6g}")
    print(f"  precond cond(Q)    : {np.linalg.cond(Q_pc_np):.6g}")
    eigs_pc = np.linalg.eigvalsh(0.5 * (Q_pc_np + Q_pc_np.T))
    print(f"  precond eig min/max: {eigs_pc.min():.6g} / {eigs_pc.max():.6g}")

    Q_pc = jnp.array(Q_pc_np)
    q_pc = jnp.array(q_pc_np)
    for kappa in (1e-3, 1e-2, 1e-1):
        _diagnose_kappa(Q_pc, q_pc, kappa, precondition=True)


# -----------------------------------------------------------------------------
# add_to_curvature_diag_value sweep — config default is 1.0e-3, which may be
# too small for interferometer Q (cond ~5.7e4). Rebuild Q with a larger diag
# boost and re-run NNLS forward + backward (target_kappa=1e-2).
# -----------------------------------------------------------------------------

print("\n" + "=" * 70)
print("PART B.6 -- add_to_curvature_diag_value SWEEP")
print("=" * 70)

for diag_value in (1e-3, 1e-1, 1.0, 10.0, 100.0):
    print(f"\n--- diag_value = {diag_value:g} (no Jacobi) ---")
    DIAG_VALUE_OVERRIDE = diag_value
    Q_sweep, q_sweep = _build_Q_q(params_tree)
    Q_sweep_np = np.array(Q_sweep)
    cond = np.linalg.cond(Q_sweep_np)
    eigs_sweep = np.linalg.eigvalsh(0.5 * (Q_sweep_np + Q_sweep_np.T))
    print(f"  cond(Q)        : {cond:.6g}")
    print(f"  eig min/max    : {eigs_sweep.min():.6g} / {eigs_sweep.max():.6g}")
    _diagnose_kappa(Q_sweep, q_sweep, target_kappa=1e-2)

    print(f"\n--- diag_value = {diag_value:g} (WITH Jacobi — mimics library path) ---")
    _diagnose_kappa(Q_sweep, q_sweep, target_kappa=1e-2, precondition=True)

DIAG_VALUE_OVERRIDE = None  # restore default for PART C


# ===================================================================
# PART C -- Full pipeline gradient (via AnalysisInterferometer)
# ===================================================================

print("\n" + "=" * 70)
print("PART C -- FULL PIPELINE GRADIENT (via AnalysisInterferometer)")
print("=" * 70)

analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=True)


def full_pipeline_from_params(params_tree):
    return analysis.log_likelihood_function(instance=params_tree)


test_grad(
    "Full pipeline (AnalysisInterferometer.log_likelihood)",
    full_pipeline_from_params,
    params_tree,
)


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


# ===================================================================
# Regression assertion — all gradient steps must produce finite, non-zero
# gradients under stock library defaults (κ=1e-11, diag=1e-3).
# ===================================================================
assert n_error == 0, (
    f"interferometer/mge_gradients: {n_error} steps raised exceptions under stock defaults"
)
assert n_fail == 0, (
    f"interferometer/mge_gradients: {n_fail} steps produced NaN/zero gradients under stock defaults"
)
assert n_pass == len(results), (
    f"interferometer/mge_gradients: only {n_pass}/{len(results)} steps passed"
)
print(f"  Regression assertion PASSED: all {n_pass}/{len(results)} gradient steps finite")
