"""
Shared setup for the JAX gradient-based optimizer scripts in this folder
(``optax_adam.py``, ``optax_adabelief.py``, ``jaxopt_lbfgs.py``, ...).

Where the nested/ensemble samplers here explore the full posterior, these
scripts run **gradient-based point optimizers** against a *maximum a
posteriori* (MAP) objective on the HST MGE lens likelihood:

    log_posterior(theta) = log_likelihood(theta) + sum(log_prior(theta))

Both terms are pure-JAX and differentiable end-to-end:

  - ``analysis.log_likelihood_function`` on ``AnalysisImaging(use_jax=True)``
    (proven differentiable in ``jax_profiling/gradient/imaging/mge.py``), and
  - ``model.log_prior_list_from_vector(vector, xp=jnp)`` (autofit's own
    JAX-traceable prior — the same term ``af.Fitness`` adds when
    ``fom_is_log_likelihood=False``).

Optimizing in the **physical** parameter vector (not the unit cube) keeps the
prior explicit, so this is a true MAP estimate rather than an MLE. The MGE
light amplitudes are linear and solved by the inversion, so the free
(nonlinear) parameter space is small (~15-D), which suits gradient methods.

Usage
-----

    from searches_minimal._grad_setup import build_map_objective, write_grad_summary

    obj = build_map_objective()
    # obj.value_and_grad(params) -> (neg_log_posterior, grad)   [jitted]
    # obj.x0                     -> prior-median start (physical)
    # obj.log_likelihood(params) -> log L only                  [jitted]
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from searches_minimal._setup import (
    build_analysis,
    build_dataset,
    build_model,
    format_best_fit,
)


@dataclass
class MapObjective:
    model: object
    analysis: object
    ndim: int
    x0: jnp.ndarray
    sigmas: np.ndarray  # per-parameter natural prior scale
    log_likelihood: Callable  # jitted, params -> log L
    log_prior: Callable  # jitted, params -> sum log prior
    neg_log_posterior: Callable  # jitted, params -> -(log L + log prior)
    neg_log_posterior_raw: Callable  # UN-jitted (for vmap / composing, e.g. multi-start)
    value_and_grad: Callable  # jitted value_and_grad of neg_log_posterior


def _natural_sigmas(model) -> np.ndarray:
    """Per-parameter natural scale: Gaussian -> sigma, Uniform -> (hi - lo)."""
    out = []
    for prior in model.priors_ordered_by_id:
        sigma_attr = getattr(prior, "sigma", None)
        if sigma_attr is not None and np.isfinite(sigma_attr):
            out.append(float(sigma_attr))
            continue
        lo = getattr(prior, "lower_limit", None)
        hi = getattr(prior, "upper_limit", None)
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            out.append(float(hi - lo))
            continue
        out.append(1.0)
    return np.asarray(out, dtype=np.float64)


def build_map_objective() -> MapObjective:
    """Build the shared HST MGE MAP objective and its jitted derivatives."""
    dataset = build_dataset()
    model = build_model()
    analysis = build_analysis(dataset, use_jax=True)
    ndim = model.prior_count

    def log_likelihood(params):
        instance = model.instance_from_vector(vector=params, xp=jnp)
        return analysis.log_likelihood_function(instance=instance)

    def log_prior(params):
        return jnp.sum(
            jnp.asarray(model.log_prior_list_from_vector(vector=params, xp=jnp))
        )

    def neg_log_posterior(params):
        return -(log_likelihood(params) + log_prior(params))

    return MapObjective(
        model=model,
        analysis=analysis,
        ndim=ndim,
        x0=jnp.asarray(model.vector_from_unit_vector([0.5] * ndim)),
        sigmas=_natural_sigmas(model),
        log_likelihood=jax.jit(log_likelihood),
        log_prior=jax.jit(log_prior),
        neg_log_posterior=jax.jit(neg_log_posterior),
        neg_log_posterior_raw=neg_log_posterior,
        value_and_grad=jax.jit(jax.value_and_grad(neg_log_posterior)),
    )


def time_compile(obj: MapObjective) -> float:
    """One-shot compile + first eval of value_and_grad at x0. Returns seconds.

    (The gradient at the raw prior median is NaN — the median is a degenerate
    point for this lens model — but that is irrelevant for *timing* the
    compile; use ``robust_cold_start`` for the actual optimisation start.)"""
    t0 = time.time()
    loss, grad = obj.value_and_grad(obj.x0)
    jax.block_until_ready(loss)
    jax.block_until_ready(grad)
    return time.time() - t0


def robust_cold_start(
    obj: MapObjective, seed: int = 0, n_tries: int = 16, spread: float = 0.02
) -> tuple[jnp.ndarray, int]:
    """A cold start with a **finite** value and gradient.

    The exact prior median is a degenerate point for this lens model: the
    elliptical components and external shear sit at (0, 0), where the
    ``arctan2`` / ``sqrt`` that map (ell_comps -> angle, magnitude) have
    singular gradients (NaN). We perturb in **unit-cube** space by a small
    amount — which stays inside every prior by construction and nudges those
    parameters off zero — and return the first draw whose ``value_and_grad``
    is finite. Deterministic given ``seed``, so the start is reproducible.

    Requires ``obj.value_and_grad`` to be compiled already (call
    ``time_compile`` first); the retries are then post-compile and cheap.
    """
    rng = np.random.default_rng(seed)
    base = np.full(obj.ndim, 0.5)
    for t in range(n_tries):
        u = np.clip(base + rng.uniform(-spread, spread, size=obj.ndim), 1e-4, 1.0 - 1e-4)
        x = jnp.asarray(obj.model.vector_from_unit_vector(list(u)))
        loss, grad = obj.value_and_grad(x)
        if np.isfinite(float(loss)) and np.all(np.isfinite(np.asarray(grad))):
            return x, t
    raise RuntimeError(
        f"robust_cold_start: no finite-gradient start found in {n_tries} tries "
        f"(seed={seed}, spread={spread})"
    )


def write_grad_summary(
    *,
    name: str,
    title: str,
    obj: MapObjective,
    best_params: jnp.ndarray,
    log_posterior_history: list[float],
    wall_s: float,
    compile_s: float,
    warm_ms_per_eval: float,
    n_evals: int,
    n_iters: int,
    converged: bool,
    config_line: str,
    tolerance: float = 1.0,
) -> str:
    """Write the standard summary block for a gradient-optimizer run and
    return the one-line comparison row (pipe-delimited) for comparison.txt.

    ``log_posterior_history`` is the per-iteration log-posterior (= -loss); it
    drives evals/time-to-ML. ``Max log L`` is the pure likelihood re-evaluated
    at ``best_params`` so it is directly comparable to the other scripts here.
    """
    best_params_np = np.asarray(best_params)
    best_instance = obj.model.instance_from_vector(vector=list(best_params_np))

    max_log_l = float(obj.log_likelihood(jnp.asarray(best_params)))
    log_prior_best = float(obj.log_prior(jnp.asarray(best_params)))
    max_log_post = max_log_l + log_prior_best

    from searches_minimal._metrics import MLTracker

    evals_to_ml, time_to_ml = MLTracker.from_log_l_history(
        log_posterior_history, total_sampling_time=wall_s, tolerance=tolerance
    )

    summary = f"""\
--- {title} Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_log_l:.4f}     (log posterior = {max_log_post:.4f}, log prior = {log_prior_best:.4f})
Log evidence:    n/a (point optimiser, not nested sampling)

--- Performance ---
Wall time:           {wall_s:.2f} s
Sampling time:       {wall_s - compile_s:.2f} s     (wall - JIT compile)
JIT compile time:    {compile_s:.2f} s     (one-shot value_and_grad warm-up)
Likelihood evals:    {n_evals}     (fwd+grad evals; iterations: {n_iters})
Time per eval:       {warm_ms_per_eval:.1f} ms     (post-compile fwd+grad)
ESS:                 n/a (point optimiser)
Posterior samples:   n/a (point optimiser)
Sampler config:      {config_line}

--- Convergence ---
Converged:           {converged}
Evals to ML:         {evals_to_ml if evals_to_ml is not None else 'n/a'}     (first iter within {tolerance:g} nat of max log posterior)
Time to ML:          {f'{time_to_ml:.2f} s' if time_to_ml is not None else 'n/a'}
"""

    print()
    print(summary)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_summary.txt").write_text(summary)
    print(f"Summary written to: {output_dir / f'{name}_summary.txt'}")

    # Pipe-delimited comparison row (robustness delta is filled in by the
    # comparison writeup once all candidates have a max log L).
    row = (
        f"| {title:<26} | {wall_s:8.1f} | {compile_s:8.1f} | {n_evals:6d} | "
        f"{warm_ms_per_eval:8.1f} | {max_log_l:12.2f} | {max_log_post:12.2f} | "
        f"{'yes' if converged else 'no':>9} |"
    )
    print("\ncomparison row:\n" + row)
    return row
