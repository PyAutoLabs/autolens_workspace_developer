"""
JAX Gradient Testing: Point-Source Source-Plane Likelihood (Step-by-Step)
=========================================================================

Companion to ``source_plane.py``.  Replaces JIT profiling with
``jax.value_and_grad`` at each stage of the source-plane positions
likelihood so we can isolate which step (if any) breaks gradients for an
``Isothermal`` lens + ``PointFlux`` source model fitted via
``FitPositionsSource``.

Run from the workspace root:

    python jax_profiling/point_source/source_plane_gradients.py

Pipeline summary
----------------
``FitPositionsSource.log_likelihood`` walks the following chain:

1. Tracer construction from the model instance.
2. ``tracer.deflections_yx_2d_from(grid=positions, xp=jnp)`` -- ray-trace
   the observed image-plane positions to the source plane.
3. ``model_data = positions - deflections`` -- source-plane positions of
   the observed images.  This is the JIT-able prefix exercised in
   ``source_plane.py``.
4. ``residual_map = ||model_data - source_centre||`` -- distances to the
   modelled source-plane centroid.
5. Magnifications at each observed position via the Hessian of the
   deflection field.
6. ``chi_squared = residual^2 * magnifications^2 / sigma^2`` summed over
   positions, plus ``noise_normalization``.
7. ``log_likelihood = -0.5 * (chi_squared + noise_normalization)``.

Surprising finding: full pipeline IS differentiable
---------------------------------------------------
``source_plane.py`` reports that the *forward* JIT of
``AnalysisPoint(FitPositionsSource).log_likelihood_function`` is blocked
by ``Grid2DIrregular.grid_2d_via_deflection_grid_from`` not propagating
``xp``.  That blocker still applies under ``jax.jit``, but
``jax.value_and_grad`` does not require lowering at the same boundary
and the full-pipeline stage below succeeds with a finite, non-zero
gradient.  Concretely: NUTS / HMC samplers driven through ``Fitness.call``
should work for the source-plane likelihood today, even though the
forward JIT path remains broken.

That makes this probe load-bearing in a different way than the imaging
probes -- it surfaces a usable gradient path that the forward JIT
profiler conceals.
"""

import numpy as np
import jax
import jax.numpy as jnp
import traceback
import subprocess
import sys
from pathlib import Path

import autofit as af
import autoarray as aa
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
# PART A -- Setup (mirrors source_plane.py)
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
# Model construction (identical to source_plane.py)
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

# GaussianPrior(mean=truth, sigma=small) centres prior-median at the
# simulator truth while keeping params free so gradient diagnostics
# have dimensionality.
mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.GaussianPrior(mean=0.01, sigma=0.005)
mass.centre.centre_1 = af.GaussianPrior(mean=0.01, sigma=0.005)
mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.01, sigma=0.005)
mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.01, sigma=0.005)
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

print("\n--- Eager baseline (FitPositionsSource) ---")

instance_eager = model.instance_from_vector(vector=np.array(jnp_params))
analysis_eager = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSource,
    use_jax=False,
)
fit_eager = analysis_eager.fit_from(instance=instance_eager)
log_likelihood_ref = float(fit_eager.log_likelihood)

print(f"  log_likelihood = {log_likelihood_ref}")

# Pre-compute observed positions / noise as raw JAX arrays so the
# per-stage closures stay inside the trace boundary.
observed_positions_raw = jnp.array(dataset.positions.array)
positions_noise_map_raw = jnp.array(dataset.positions_noise_map.array)


# ===================================================================
# PART B -- Per-step gradient testing
# ===================================================================

print("\n" + "=" * 70)
print("PART B -- PER-STEP GRADIENT TESTING")
print("=" * 70)


# ---------------------------------------------------------------------------
# Step 1: Ray-trace observed positions to source plane
#
# This is the JIT-able prefix from source_plane.py: build a tracer,
# compute deflections at each observed image-plane position, and
# subtract to get the source-plane positions. Stays inside raw arrays
# the whole way -- no Grid2DIrregular result crosses the trace
# boundary, so the xp-propagation blocker in
# Grid2DIrregular.grid_2d_via_deflection_grid_from is avoided.
# ---------------------------------------------------------------------------

def step_ray_trace_to_source(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    grid_in = aa.Grid2DIrregular(values=observed_positions_raw, xp=jnp)
    deflections = t.deflections_yx_2d_from(grid=grid_in, xp=jnp)
    source_positions = observed_positions_raw - deflections.array
    return jnp.sum(source_positions)


test_grad(
    "Step 1: Ray-trace positions to source plane",
    step_ray_trace_to_source,
    jnp_params,
)


# ---------------------------------------------------------------------------
# Step 2: Source-plane residual
#
# Source-plane positions minus the modelled source centroid -- the input
# to FitPositionsSource.residual_map. Tests gradient flow through the
# source-position prior leaf as well as the lens mass parameters.
# ---------------------------------------------------------------------------

def step_source_plane_residual(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    grid_in = aa.Grid2DIrregular(values=observed_positions_raw, xp=jnp)
    deflections = t.deflections_yx_2d_from(grid=grid_in, xp=jnp)
    source_positions = observed_positions_raw - deflections.array
    centre = inst.galaxies.source.point_0.centre
    source_centre = jnp.array([centre[0], centre[1]])
    residuals = source_positions - source_centre
    return jnp.sum(residuals ** 2)


test_grad(
    "Step 2: Source-plane residual",
    step_source_plane_residual,
    jnp_params,
)


# ---------------------------------------------------------------------------
# Step 3: Positions chi-squared (residual + noise, no magnification)
#
# A simplified chi-squared that uses only the source-plane residuals
# and the position noise map:
#
#   chi^2          = sum_i ||residual_i||^2 / sigma_i^2
#   noise_norm     = sum_i log(2*pi * sigma_i^2)
#   log_likelihood = -0.5 * (chi^2 + noise_norm)
#
# FitPositionsSource also multiplies by magnification^2 (Hessian-
# derived) to account for image-plane noise being magnified to the
# source plane. We deliberately omit that term here because the
# Hessian path through LensCalc currently has a latent bug
# (``lens_calc._hessian_via_jax`` calls ``jnp.array(grid[:, 0])``
# instead of ``jnp.array(grid.array[:, 0])`` -- see PyAutoLens
# CLAUDE.md, "Use grid.array[:, 0] (not grid[:, 0])"), which raises
# a ``ValueError: object __array__ method not producing an array``
# for ``Grid2DIrregular`` inputs under a JAX trace. The full-pipeline
# stage below exercises the chi-squared *with* magnifications and
# gives the load-bearing answer; this stage isolates whether the
# residual + noise reduction is itself differentiable independent
# of that LensCalc bug.
# ---------------------------------------------------------------------------

def step_positions_chi_squared(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    grid_in = aa.Grid2DIrregular(values=observed_positions_raw, xp=jnp)

    deflections = t.deflections_yx_2d_from(grid=grid_in, xp=jnp)
    source_positions = observed_positions_raw - deflections.array

    centre = inst.galaxies.source.point_0.centre
    source_centre = jnp.array([centre[0], centre[1]])
    residual_sq = jnp.sum((source_positions - source_centre) ** 2, axis=1)

    sigma_squared = positions_noise_map_raw ** 2
    chi_squared = jnp.sum(residual_sq / sigma_squared)
    noise_normalization = jnp.sum(jnp.log(2 * jnp.pi * sigma_squared))
    return -0.5 * (chi_squared + noise_normalization)


test_grad(
    "Step 3: Positions chi-squared (residual + noise)",
    step_positions_chi_squared,
    jnp_params,
)


# ===================================================================
# PART C -- Full pipeline gradient (via Fitness)
# ===================================================================

print("\n" + "=" * 70)
print("PART C -- FULL PIPELINE GRADIENT (via Fitness)")
print("=" * 70)

# source_plane.py reports a forward-JIT blocker for this same
# pipeline (Grid2DIrregular.grid_2d_via_deflection_grid_from does not
# propagate xp). value_and_grad does not require lowering at the same
# boundary, so this stage is expected to PASS with a finite gradient
# even while the forward JIT path is broken. A future status-flip
# would indicate either the forward JIT was fixed (and we now hit a
# different gradient path) or the gradient regressed.

from autofit.non_linear.fitness import Fitness

analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSource,
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
