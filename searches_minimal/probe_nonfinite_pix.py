"""
Localise the pixelized likelihood's non-finite walls (autolens_workspace_developer#104)
---------------------------------------------------------------------------------------

#101 proved *by elimination* that the pixelized objective (kernel-CDF mesh, NNLS
positive-only solve, free Isothermal+shear mass) has hard non-finite walls: every
gradient trajectory dies within ~25-50 steps regardless of learning rate, update
rule, start band or fixed regularization. This script answers the question that
elimination could not: **WHICH intermediate goes non-finite, and where?**

It walks the same objective ``pix_lr_free.py`` optimises (``pix_multi_start``'s
model/analysis — do not rebuild it here, the whole point is that it is the same
likelihood) stage by stage, in dependency order, and reports the FIRST stage that
goes non-finite.

Two walks, because there are two distinct death signatures
=========================================================

``forward`` — the *value* goes non-finite. This is the trajectory death: the loss
is finite for ~25-50 steps and then NaNs. Reading the stages in order names the
first intermediate that broke.

``backward`` — the *value is finite but the gradient is NaN*. #101 recorded 2-3 of
16 broad draws in exactly this state at **step 0**, before any optimizer step. A
forward-only probe finds NOTHING for these points: every stage is finite. The NaN
is created in the backward pass, and the classic generators are

  * ``sqrt(x)`` at x=0        -> value 0.0, derivative 1/(2*sqrt(0)) = inf -> NaN
  * ``where(bad, safe, x)``   -> value safe, derivative 0 * NaN = NaN (JAX's
                                 most-cited autodiff trap: reverse-mode evaluates
                                 BOTH branches)
  * ``cholesky(A)`` near-singular -> value finite, derivative NaN

so the backward walk grads each stage separately and reports the first stage whose
gradient is non-finite while its value is finite. That pair (finite value, NaN
grad) is the localisation.

Why the recorded death points are NOT the reproduction
======================================================

``pix_lr_free.py:206-208`` records ``last_finite_params`` — the params at the last
step that was still finite. Those evaluate FINITE by construction; replaying them
reproduces nothing. The NaN-producing params (one optimizer step later) are never
persisted. So this script reproduces the two real levers instead:

``--mode draws`` (default, cheapest): re-draw the seed-0 broad starts exactly as
``pix_lr_free.py:120-130`` does and KEEP the draws it silently rejects — the ones
with finite loss and NaN gradient. Each is a single-point reproduction: one
forward + one backward pass, no optimizer, no trajectory, ~1GB.

``--mode replay``: the seed-0 trajectory is deterministic, so re-running a rule to
``death_step+1`` (from ``lr_free_results/pix_death_report_330592.txt``) reproduces
a trajectory death exactly. Costlier; use once ``draws`` is exhausted.

Triage the two death classes separately
=======================================

From the death report: 13/16 deaths sit at r_E 2.6-6.4 against truth 1.6, at
log_post ~ -1.6e5 — the regime ``pix_gradient_findings.md`` calls "garbage
evaluation points ... source arcs miss the mesh". Only start 2 (r_E 1.3587) and
start 12 (r_E 1.4268) die near the Nautilus mode (1.31). A NaN wall INSIDE the
physical basin is a genuine bug; a NaN at r_E=6.4 is plausibly invalid model space
that deserves a finite penalty, not a fix. The report tags each point accordingly.

Everything here is READ-ONLY on the libraries — phase 1 localises, phase 2 fixes.

Usage (from the workspace root)::

    python -m searches_minimal.probe_nonfinite_pix                  # seed-0 draws
    python -m searches_minimal.probe_nonfinite_pix --mode replay --rule adam
    python -m searches_minimal.probe_nonfinite_pix --max-points 2   # quicker

Env: ``PIX_START_LOW`` / ``PIX_START_HIGH`` override the start band (defaults
mirror ``lr_free_multistart``'s broad band).
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

# x64 before any autolens import, exactly as the harness under test does.
jax.config.update("jax_enable_x64", True)

from autofit.non_linear.fitness import Fitness  # noqa: E402

from searches_minimal.pix_multi_start import (  # noqa: E402
    build_model,
    build_analysis,
    TRUTH_EINSTEIN_RADIUS,
    BASIN_TOL,
    MESH_SHAPE,
    OS_PIX,
)
from searches_minimal._setup import build_dataset  # noqa: E402
from searches_minimal.lr_free_multistart import (  # noqa: E402
    START_LOW,
    START_HIGH,
)

START_LOW = float(os.environ.get("PIX_START_LOW", START_LOW))
START_HIGH = float(os.environ.get("PIX_START_HIGH", START_HIGH))

NAUTILUS_MODE_R_E = 1.31  # converged Nautilus mode (job 330513)


# ---------------------------------------------------------------------------
# The stage ladder
# ---------------------------------------------------------------------------
# Dependency order through the pixelized likelihood. Each entry pulls one
# intermediate off the fit/inversion. Ordering matters: the FIRST non-finite
# entry is the localisation, so a stage must never appear before something it
# depends on.
#
# The `log_evidence` decomposition (fit_dataset.py:324) is the natural top of the
# ladder — it is exactly five terms, two of which are the prime suspects:
#
#   log_evidence = -0.5 * [ chi_squared + regularization_term
#                           + log_det_curvature_reg_matrix_term
#                           - log_det_regularization_matrix_term
#                           + noise_normalization ]

Stage = Tuple[str, Callable]

INVERSION_STAGES: List[Stage] = [
    # --- mesh / mapper: does the source-plane mesh survive the trace? --------
    ("mapping_matrix", lambda f, i: i.mapping_matrix),
    ("operated_mapping_matrix", lambda f, i: i.operated_mapping_matrix),
    # --- linear algebra setup ------------------------------------------------
    ("data_vector", lambda f, i: i.data_vector),
    ("curvature_matrix", lambda f, i: i.curvature_matrix),
    ("regularization_matrix", lambda f, i: i.regularization_matrix),
    ("curvature_reg_matrix", lambda f, i: i.curvature_reg_matrix),
    # The Jacobi preconditioner inside reconstruction_positive_only_from
    # (inversion_util.py:333-335) is `d = sqrt(diag(curvature_reg_matrix))`,
    # `D = 1/d`. A structurally-unmapped mesh pixel gives an exactly-zero
    # diagonal -> D = inf (forward), and sqrt(0) has an infinite derivative
    # (backward). Both are reproduced explicitly here rather than inferred.
    ("_nnls_precond_d", lambda f, i: jnp.sqrt(jnp.diag(i.curvature_reg_matrix))),
    ("_nnls_precond_D", lambda f, i: 1.0 / jnp.sqrt(jnp.diag(i.curvature_reg_matrix))),
    # --- the solve -----------------------------------------------------------
    ("reconstruction", lambda f, i: i.reconstruction),
    ("mapped_reconstructed_data", lambda f, i: i.mapped_reconstructed_data),
    # --- the five log_evidence terms ----------------------------------------
    ("regularization_term", lambda f, i: i.regularization_term),
    (
        "log_det_curvature_reg_matrix_term",
        lambda f, i: i.log_det_curvature_reg_matrix_term,
    ),
    (
        "log_det_regularization_matrix_term",
        lambda f, i: i.log_det_regularization_matrix_term,
    ),
]

FIT_STAGES: List[Stage] = [
    ("model_data", lambda f, i: f.model_data),
    ("residual_map", lambda f, i: f.residual_map),
    ("chi_squared_map", lambda f, i: f.chi_squared_map),
    ("chi_squared", lambda f, i: f.chi_squared),
    ("noise_normalization", lambda f, i: f.noise_normalization),
    ("log_evidence", lambda f, i: f.log_evidence),
    ("figure_of_merit", lambda f, i: f.figure_of_merit),
]


def _stage_ladder() -> List[Stage]:
    """Inversion internals first, then the fit terms that consume them."""
    return INVERSION_STAGES + FIT_STAGES


def _as_array(value) -> np.ndarray:
    """Concrete numpy view of a stage output (scalars included)."""
    if hasattr(value, "array"):  # autoarray types (Array2D, Grid2D, ...)
        value = value.array
    return np.asarray(value, dtype=float)


def _finite_report(value) -> Tuple[bool, str]:
    """(is_finite, human summary) for a stage output."""
    arr = _as_array(value)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    n = arr.size
    if n_nan == 0 and n_inf == 0:
        lo, hi = (float(arr.min()), float(arr.max())) if n else (0.0, 0.0)
        return True, f"finite  n={n:<8d} range=[{lo:.4e}, {hi:.4e}]"
    return False, f"NON-FINITE  n={n:<8d} nan={n_nan} inf={n_inf}"


# ---------------------------------------------------------------------------
# Building the objective (identical to the harness under test)
# ---------------------------------------------------------------------------


def build_objective():
    dataset = build_dataset()
    analysis = build_analysis(dataset)
    model = build_model()
    fitness = Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=False,
        resample_figure_of_merit=-np.inf,
        convert_to_chi_squared=True,
    )
    return model, analysis, fitness


def _fit_at(analysis, model, params) -> Tuple[object, object]:
    """The fit + inversion at `params`, traced through JAX (eager, un-jitted so
    every intermediate is a concrete array we can read)."""
    instance = model.instance_from_vector(vector=params, xp=jnp)
    fit = analysis.fit_from(instance=instance)
    return fit, fit.inversion


# ---------------------------------------------------------------------------
# The two walks
# ---------------------------------------------------------------------------


def walk_forward(analysis, model, params) -> Optional[str]:
    """Read every stage in order; report the first whose VALUE is non-finite.

    Returns the name of the first non-finite stage, or None if all are finite
    (which is the expected outcome for a step-0 finite-loss/NaN-grad point — see
    walk_backward, that is where its NaN lives).
    """
    print("  forward walk (value finiteness):")
    fit, inversion = _fit_at(analysis, model, params)

    first_bad = None
    for name, getter in _stage_ladder():
        try:
            value = getter(fit, inversion)
        except Exception as exc:  # a raise localises just as well as a NaN
            print(f"    {name:36s} RAISED  {type(exc).__name__}: {exc}")
            if first_bad is None:
                first_bad = f"{name} (raised {type(exc).__name__})"
            continue
        ok, summary = _finite_report(value)
        marker = "   " if ok else ">> "
        print(f"  {marker}{name:36s} {summary}")
        if not ok and first_bad is None:
            first_bad = name
    return first_bad


def walk_backward(analysis, model, params, stages: Optional[List[str]] = None):
    """Grad each stage separately; report the first whose GRADIENT is non-finite.

    This is the walk that matters for the step-0 deaths, where every forward value
    is finite and the NaN is manufactured in the backward pass. Each stage is
    reduced to a scalar by summation before differentiating — summation preserves
    non-finiteness (any NaN element makes the sum's grad NaN), so no NaN can hide.
    """
    print("  backward walk (gradient finiteness):")
    ladder = _stage_ladder()
    if stages:
        ladder = [(n, g) for n, g in ladder if n in stages]

    first_bad = None
    for name, getter in ladder:

        def scalar_of(p, _getter=getter):
            fit, inversion = _fit_at(analysis, model, p)
            return jnp.sum(jnp.asarray(_getter(fit, inversion)))

        try:
            grad = jax.grad(scalar_of)(params)
        except Exception as exc:
            print(f"    {name:36s} GRAD RAISED  {type(exc).__name__}: {exc}")
            if first_bad is None:
                first_bad = f"{name} (grad raised {type(exc).__name__})"
            continue

        grad_np = np.asarray(grad, dtype=float)
        n_nan = int(np.isnan(grad_np).sum())
        n_inf = int(np.isinf(grad_np).sum())
        if n_nan == 0 and n_inf == 0:
            print(f"     {name:36s} grad finite   |g|max={np.abs(grad_np).max():.4e}")
            continue

        bad_idx = np.flatnonzero(~np.isfinite(grad_np))
        names = _param_names(model)
        culprits = ", ".join(f"{names[k]}" for k in bad_idx[:4])
        print(
            f"  >> {name:36s} grad NON-FINITE  nan={n_nan} inf={n_inf}"
            f"  params: {culprits}"
        )
        if first_bad is None:
            first_bad = name
    return first_bad


def _param_names(model) -> List[str]:
    return [str(path) for path, _ in model.path_priors_tuples]


# ---------------------------------------------------------------------------
# Reproductions
# ---------------------------------------------------------------------------


def collect_draws(model, fitness, max_points: int, seed: int = 0):
    """Re-draw the seed-0 broad starts and KEEP the ones pix_lr_free discards.

    pix_lr_free.py:124-130 loops until it has N_STARTS draws with finite loss AND
    finite gradient, throwing the rest away. Those rejects are the cheapest
    reproduction available: a finite-loss/NaN-grad point needs no optimizer and no
    trajectory. Same seed, same band, same objective -> the same draw sequence.
    """
    rng = np.random.default_rng(seed)
    value_and_grad = jax.jit(jax.value_and_grad(fitness.call))

    rejects, tries = [], 0
    print(
        f"drawing seed-{seed} starts in U({START_LOW}, {START_HIGH}) "
        f"(the pix_lr_free sequence); keeping the rejects"
    )
    while len(rejects) < max_points and tries < 30 * max_points:
        tries += 1
        u = rng.uniform(START_LOW, START_HIGH, size=model.prior_count)
        x = jnp.asarray(model.vector_from_unit_vector(unit_vector=list(u), xp=jnp))
        loss, grad = value_and_grad(x)

        loss_finite = bool(np.isfinite(float(loss)))
        grad_finite = bool(np.all(np.isfinite(np.asarray(grad))))
        if loss_finite and grad_finite:
            continue  # this is a draw the harness would have kept

        kind = (
            "finite loss, NaN grad"
            if loss_finite
            else ("non-finite loss" if not loss_finite else "?")
        )
        print(f"  draw {tries:3d}: REJECT ({kind})  loss={float(loss):.6e}")
        rejects.append((x, kind, float(loss)))

    print(f"collected {len(rejects)} reject(s) from {tries} draws\n")
    return rejects


def _tag_point(model, params) -> str:
    """In-basin deaths are bugs; out-of-basin deaths may be invalid model space."""
    r_e = float(
        model.instance_from_vector(
            vector=list(np.asarray(params))
        ).galaxies.lens.mass.einstein_radius
    )
    near_truth = abs(r_e - TRUTH_EINSTEIN_RADIUS) < BASIN_TOL
    near_naut = abs(r_e - NAUTILUS_MODE_R_E) < BASIN_TOL
    if near_truth or near_naut:
        return f"r_E={r_e:.4f}  IN-BASIN (bug candidate)"
    return f"r_E={r_e:.4f}  out-of-basin (may be genuinely-invalid space)"


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("draws", "replay"), default="draws")
    parser.add_argument("--max-points", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stages", nargs="*", default=None, help="restrict the backward walk"
    )
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}  x64={jax.config.jax_enable_x64}")
    print(f"mesh=KernelAdaptDensity{MESH_SHAPE} os_pix={OS_PIX}  (no sparse operator)")

    t0 = time.time()
    model, analysis, fitness = build_objective()
    print(f"Free (non-linear) parameters: {model.prior_count}")

    if args.mode == "replay":
        raise SystemExit(
            "replay mode is not implemented yet — exhaust --mode draws first "
            "(see the module docstring: the recorded death points are LAST-FINITE "
            "params and reproduce nothing; replay must re-run the deterministic "
            "seed-0 trajectory to death_step+1)."
        )

    points = collect_draws(model, fitness, args.max_points, seed=args.seed)
    if not points:
        print(
            "No rejected draws found. Either the band no longer produces them, or "
            "the objective changed since #101 — do NOT proceed as if localised."
        )
        return

    verdicts: Dict[str, List[str]] = {}
    for k, (params, kind, loss) in enumerate(points):
        print(f"\n{'=' * 78}\npoint {k}: {kind}   loss={loss:.6e}")
        print(f"  {_tag_point(model, params)}\n{'=' * 78}")

        fwd = walk_forward(analysis, model, params)
        print()
        bwd = walk_backward(analysis, model, params, stages=args.stages)

        print(f"\n  --> first non-finite VALUE:    {fwd or 'none (all finite)'}")
        print(f"  --> first non-finite GRADIENT: {bwd or 'none (all finite)'}")
        verdicts.setdefault(f"{fwd} | {bwd}", []).append(f"point {k} ({kind})")

    print(f"\n{'=' * 78}\nSUMMARY (first non-finite value | first non-finite gradient)")
    print(f"{'=' * 78}")
    for site, pts in verdicts.items():
        print(f"  {site:60s} <- {', '.join(pts)}")
    print(f"\nwall: {time.time() - t0:.1f} s")
    print(
        "\nNext: map each site above to a verdict in pix_nonfinite_findings.md — "
        "fix (finite-safe formulation) / guard (finite penalty preserving a useful "
        "gradient) / document (genuinely-invalid model space)."
    )


if __name__ == "__main__":
    main()
