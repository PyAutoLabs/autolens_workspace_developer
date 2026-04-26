"""
JAX Gradient Testing: Point-Source Image-Plane Likelihood (Step-by-Step)
=========================================================================

Companion to ``image_plane.py``.  Replaces JIT profiling with
``jax.value_and_grad`` at each stage of the image-plane positions
likelihood so we can isolate which step (if any) breaks gradients for
an ``Isothermal`` lens + ``PointFlux`` source model fitted via
``FitPositionsImagePairAll``.

Run from the workspace root:

    python jax_profiling/point_source/image_plane_gradients.py

Pipeline summary
----------------
``FitPositionsImagePairAll.log_likelihood`` walks the following chain:

1. Tracer construction from the model instance.
2. ``solver.solve(tracer, source_plane_coordinate, xp=jnp)`` -- forward-
   solve the lens equation for the modelled source-plane centre by
   recursively refining a triangle tiling of the image plane.  This is
   the load-bearing stage for differentiability: ``PointSolver`` uses
   ``jnp.where`` masking and integer-indexed neighbour lookups, none of
   which is reverse-mode differentiable in the usual sense.
3. ``square_distance(observed, model)`` over every (observed x model)
   pair.
4. ``log_p`` per pair and ``all_permutations_log_likelihoods`` -- the
   Bayesian all-pairs likelihood from
   https://arxiv.org/abs/2406.15280.
5. ``chi_squared = -2 * (-log(n_perms) + sum(log_p_per_data))`` and
   ``log_likelihood = -0.5 * chi_squared``.

Finding: the image-plane likelihood is not differentiable
---------------------------------------------------------
On ``main`` (and at the time this probe was written) every stage that
chains through ``PointSolver.solve`` returns an **identically zero
gradient** under ``jax.value_and_grad`` while still producing a sensible
forward value:

* Step 1 (solver arrivals)               -> grad norm = 0
* Step 2 (pairwise sq. distances)        -> grad norm = 0
* Step 3 (FitPositionsImagePairAll chi^2)-> grad norm = 0
* Full pipeline (Fitness.call)           -> grad norm = 0

The forward value of the full pipeline matches the eager NumPy
reference to float64, so this is not a NaN-poisoning issue: the
solver's triangle-subdivision path is reverse-mode opaque (integer
indexing on neighbour lookups, ``jnp.where`` masking on retained
triangles, fixed-iteration recursion) and gradients zero out at the
boundary.

A user running NUTS / HMC against ``AnalysisPoint(FitPositionsImagePairAll)``
would see a flat likelihood landscape and an immediate sampler failure
-- this probe surfaces the cause cleanly. Fixing it (re-formulating
the solver as differentiable, switching to a continuous relaxation,
or pre-solving outside the trace and stop-gradienting through it) is
follow-up work and out of scope here.

A future status-flip on this probe would indicate the solver path
became differentiable; that is the regression guard this script
exists to provide.
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


# ---------------------------------------------------------------------------
# Gradient test helper (identical to imaging/pixelization_gradients.py)
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
            status, detail = (
                "FAIL",
                f"{n_bad}/{grad_np.size} gradient entries are non-finite",
            )
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
# PART A -- Setup (mirrors image_plane.py)
# ===================================================================

print("\n" + "=" * 70)
print("PART A -- SETUP")
print("=" * 70)

dataset_name = "simple"

print(f"\n--- Dataset loading [{dataset_name}] ---")

_script_dir = Path(__file__).resolve().parent
dataset_path = (
    Path("jax_profiling") / "point_source" / "dataset" / "point_source" / dataset_name
)

if al.util.dataset.should_simulate(str(dataset_path)):
    print(f"  Simulating {dataset_name} dataset...")
    subprocess.run(
        [
            sys.executable,
            str(_script_dir / "simulators" / "point_source.py"),
            "--name",
            dataset_name,
        ],
        cwd=str(_script_dir),
        check=True,
    )

dataset = al.from_json(
    file_path=dataset_path / "point_dataset_positions_only.json",
)

n_observed_positions = dataset.positions.shape[0]
positions_noise_sigma = float(dataset.positions_noise_map[0])

print(f"  Observed positions:   {n_observed_positions}")
print(f"  Position noise sigma: {positions_noise_sigma}")


print("\n--- Point solver ---")

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.2)
solver = al.PointSolver.for_grid(
    grid=grid,
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
)


# ---------------------------------------------------------------------------
# Model construction (identical to image_plane.py)
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.GaussianPrior(mean=0.01, sigma=0.005)
mass.centre.centre_1 = af.GaussianPrior(mean=0.01, sigma=0.005)
mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.01, sigma=0.01)
mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.01, sigma=0.01)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

point_0 = af.Model(al.ps.PointFlux)
point_0.centre.centre_0 = af.GaussianPrior(mean=0.07, sigma=0.005)
point_0.centre.centre_1 = af.GaussianPrior(mean=0.07, sigma=0.005)
source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")


# ---------------------------------------------------------------------------
# Parameter vector (perturbed from prior medians)
# ---------------------------------------------------------------------------

jnp_params = jnp.array(model.physical_values_from_prior_medians)
key = jax.random.PRNGKey(42)
perturbation = jax.random.uniform(
    key, shape=jnp_params.shape, minval=0.001, maxval=0.005
)
jnp_params = jnp_params + perturbation
print(f"  param_vector shape: {jnp_params.shape}")


# ---------------------------------------------------------------------------
# Eager baseline
# ---------------------------------------------------------------------------

print("\n--- Eager baseline (FitPositionsImagePairAll) ---")

instance_eager = model.instance_from_vector(vector=np.array(jnp_params))
analysis_eager = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAll,
    use_jax=False,
)
fit_eager = analysis_eager.fit_from(instance=instance_eager)
log_likelihood_ref = float(fit_eager.log_likelihood)

print(f"  log_likelihood = {log_likelihood_ref}")

observed_positions_raw = jnp.array(dataset.positions.array)
positions_noise_map_raw = jnp.array(dataset.positions_noise_map.array)


# ===================================================================
# PART B -- Per-step gradient testing
# ===================================================================

print("\n" + "=" * 70)
print("PART B -- PER-STEP GRADIENT TESTING")
print("=" * 70)


# ---------------------------------------------------------------------------
# Step 1: Source-plane guess -> solver-produced image-plane arrivals
#
# Calls PointSolver.solve directly with the JAX-traced tracer and the
# source-plane centre extracted from the params. If the solver does
# not trace under value_and_grad (the expected outcome -- triangle
# subdivision uses jnp.where masking + integer indexing), the harness
# records the traceback's last line.
#
# remove_infinities=False keeps the output static-shaped, which is
# required for any chance at JIT/grad tracing.
# ---------------------------------------------------------------------------

def step_solver(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    centre = inst.galaxies.source.point_0.centre
    source_plane_coordinate = (centre[0], centre[1])
    arrivals = solver.solve(
        tracer=t,
        source_plane_coordinate=source_plane_coordinate,
        xp=jnp,
        remove_infinities=False,
    )
    # Mask the inf-padded sentinel rows out of the reduction so the
    # finite/non-finite check measures whether the solver's *real*
    # arrivals carry gradients, rather than tripping on the static-
    # shape padding that is inherent to PointSolver's JAX path.
    arrivals_arr = arrivals.array
    finite_mask = jnp.isfinite(arrivals_arr).all(axis=1)
    masked = jnp.where(finite_mask[:, None], arrivals_arr, 0.0)
    return jnp.sum(masked)


test_grad(
    "Step 1: Solver image-plane arrivals",
    step_solver,
    jnp_params,
)


# ---------------------------------------------------------------------------
# Step 2: Image-plane residual (square distance to observed positions)
#
# For every (observed x model) pair, compute the squared image-plane
# separation and sum. Chains through the solver, so this stage will
# fail with the solver if Step 1 fails.
# ---------------------------------------------------------------------------

def step_image_plane_residual(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    centre = inst.galaxies.source.point_0.centre
    source_plane_coordinate = (centre[0], centre[1])
    arrivals = solver.solve(
        tracer=t,
        source_plane_coordinate=source_plane_coordinate,
        xp=jnp,
        remove_infinities=False,
    )
    model_data = arrivals.array
    # Pairwise squared separations between every observed and every
    # model image position. inf-padded model rows contribute inf to
    # the sum, so we mask them out before reducing.
    diffs = observed_positions_raw[:, None, :] - model_data[None, :, :]
    sq_distances = jnp.sum(diffs ** 2, axis=-1)
    finite_mask = jnp.isfinite(sq_distances)
    return jnp.sum(jnp.where(finite_mask, sq_distances, 0.0))


test_grad(
    "Step 2: Image-plane pairwise squared distances",
    step_image_plane_residual,
    jnp_params,
)


# ---------------------------------------------------------------------------
# Step 3: Positions chi-squared (FitPositionsImagePairAll formula)
#
# Reproduces the all-permutations Bayesian chi-squared:
#
#   log_p(d, m, sigma) = -log(sqrt(2*pi*sigma^2)) - 0.5 * |d-m|^2 / sigma^2
#   per_data_logL_i    = log( sum_m exp(log_p(d_i, m, sigma_i)) )
#   chi_squared        = -2 * (-log(n_perm) + sum_i per_data_logL_i)
#   log_likelihood     = -0.5 * chi_squared
#
# n_perm = n_finite_model ^ n_observed counts non-NaN solver outputs.
# Chains through the solver -- expected to fail with Step 1.
# ---------------------------------------------------------------------------

def step_positions_chi_squared(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    centre = inst.galaxies.source.point_0.centre
    source_plane_coordinate = (centre[0], centre[1])
    arrivals = solver.solve(
        tracer=t,
        source_plane_coordinate=source_plane_coordinate,
        xp=jnp,
        remove_infinities=False,
    )
    model_data = arrivals.array

    diffs = observed_positions_raw[:, None, :] - model_data[None, :, :]
    sq_distances = jnp.sum(diffs ** 2, axis=-1)

    sigma = positions_noise_map_raw[:, None]
    log_p = -jnp.log(jnp.sqrt(2 * jnp.pi * sigma ** 2)) - 0.5 * sq_distances / sigma ** 2

    # Mask inf-padded model rows out of the log-sum-exp.
    finite_mask = jnp.isfinite(model_data).all(axis=1)
    log_p = jnp.where(finite_mask[None, :], log_p, -jnp.inf)
    per_data_logL = jax.scipy.special.logsumexp(log_p, axis=1)

    n_finite_model = jnp.sum(finite_mask)
    n_observed = observed_positions_raw.shape[0]
    n_perm = n_finite_model ** n_observed

    chi_squared = -2.0 * (-jnp.log(n_perm) + jnp.sum(per_data_logL))
    return -0.5 * chi_squared


test_grad(
    "Step 3: Positions chi-squared (manual FitPositionsImagePairAll)",
    step_positions_chi_squared,
    jnp_params,
)


# ===================================================================
# PART C -- Full pipeline gradient (via Fitness)
# ===================================================================

print("\n" + "=" * 70)
print("PART C -- FULL PIPELINE GRADIENT (via Fitness)")
print("=" * 70)

from autofit.non_linear.fitness import Fitness

analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAll,
    use_jax=True,
)
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
print(
    f"  {n_pass} passed, {n_fail} failed, {n_error} errors out of "
    f"{len(results)} tests"
)
print("=" * 70)
