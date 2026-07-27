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
    JAX-traceable prior — the same prior term an autofit fitness/objective adds
    to the log likelihood when maximising the log posterior rather than the log
    likelihood alone).

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
    neg_log_posterior_raw: (
        Callable  # UN-jitted (for vmap / composing, e.g. multi-start)
    )
    value_and_grad: Callable  # jitted value_and_grad of neg_log_posterior
    # --- Phase-3: shared unconstrained parameterization + residual LS ---------
    # z (unconstrained, in R^ndim) -> physical params, via one sigmoid per param
    # over the model's own inverse-CDF (`vector_from_unit_vector`). Smooth,
    # bijective, no hard clipping; NO Jacobian term is added, so the optimum is
    # the MAP in *physical* coordinates (the transform is a pure preconditioner).
    physical_from_z: Callable = None  # z -> physical params (jax-traceable)
    neg_log_posterior_z_raw: Callable = None  # UN-jitted z -> -(log L + log prior)
    residual_z_raw: Callable = None  # UN-jitted z -> residual vector r(z)
    prior_gauss_idx: np.ndarray = (
        None  # indices of Gaussian priors (for r's prior block)
    )
    prior_gauss_mean: np.ndarray = None
    prior_gauss_sigma: np.ndarray = None


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

    # --- unconstrained parameterization (shared by every gradient optimizer) --
    def physical_from_z(z):
        # sigmoid maps R -> (0, 1) with no clipping; the model's own inverse-CDF
        # (`vector_from_unit_vector`, xp=jnp -> stacked jax array) then maps the
        # unit cube exactly to physical params for every prior type.
        u = jax.nn.sigmoid(z)
        return model.vector_from_unit_vector(u, xp=jnp)

    def neg_log_posterior_z(z):
        return neg_log_posterior(physical_from_z(z))

    # Gaussian-prior residual block: only plain Gaussian priors contribute a
    # (theta - mean) / sigma residual to the least-squares objective. Uniform
    # priors are constant within their bounds (enforced by the sigmoid) and add
    # nothing; other prior normalisation constants are theta-independent. See
    # `build_residual_z` for the imaging block.
    g_idx, g_mean, g_sigma = [], [], []
    for i, prior in enumerate(model.priors_ordered_by_id):
        if type(prior).__name__ == "GaussianPrior":
            g_idx.append(i)
            g_mean.append(float(prior.mean))
            g_sigma.append(float(prior.sigma))
    g_idx = np.asarray(g_idx, dtype=int)
    g_mean = np.asarray(g_mean, dtype=np.float64)
    g_sigma = np.asarray(g_sigma, dtype=np.float64)

    def residual_z(z):
        """Residual vector r(z) with 0.5||r||^2 == neg_log_posterior(theta(z))
        up to theta-independent constants (noise normalisation, Uniform-prior
        and Gaussian-prior normalisation). Imaging block = normalized residual
        map (0.5*sum = 0.5*chi^2); prior block = (theta - mean)/sigma per
        Gaussian prior (0.5*sum = Gaussian -log_prior up to a constant).

        NOTE: this residual is only **reverse-mode** differentiable — the
        positive-only source solve (``jax_nnls.solve_nnls_primal``) is a
        ``custom_vjp`` with no forward rule. Gauss-Newton / Levenberg-Marquardt
        need the forward-mode Jacobian (``jacfwd``/``jvp``) and therefore raise
        ``TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp``.
        Use it for the residual-identity check and gradient-based methods, not
        LM/GN (see ``jaxopt_lm_multistart.py``)."""
        theta = physical_from_z(z)
        instance = model.instance_from_vector(vector=theta, xp=jnp)
        fit = analysis.fit_from(instance=instance)
        r_img = fit.normalized_residual_map.array
        if g_idx.size == 0:
            return r_img
        r_prior = (theta[g_idx] - jnp.asarray(g_mean)) / jnp.asarray(g_sigma)
        return jnp.concatenate([r_img, r_prior])

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
        physical_from_z=physical_from_z,
        neg_log_posterior_z_raw=neg_log_posterior_z,
        residual_z_raw=residual_z,
        prior_gauss_idx=g_idx,
        prior_gauss_mean=g_mean,
        prior_gauss_sigma=g_sigma,
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
        u = np.clip(
            base + rng.uniform(-spread, spread, size=obj.ndim), 1e-4, 1.0 - 1e-4
        )
        x = jnp.asarray(obj.model.vector_from_unit_vector(list(u)))
        loss, grad = obj.value_and_grad(x)
        if np.isfinite(float(loss)) and np.all(np.isfinite(np.asarray(grad))):
            return x, t
    raise RuntimeError(
        f"robust_cold_start: no finite-gradient start found in {n_tries} tries "
        f"(seed={seed}, spread={spread})"
    )


def _logit(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    return np.log(u) - np.log1p(-u)


def make_z_starts(
    obj: MapObjective,
    n_starts: int,
    low: float = 0.15,
    high: float = 0.85,
    seed: int = 0,
    max_factor: int = 30,
) -> tuple[jnp.ndarray, int]:
    """Broad cold starts in the **unconstrained** z-space, filtered to finite
    value+gradient. Draws the same unit-cube spread U(low, high) used by the
    physical-space multi-start scripts, then maps u -> z = logit(u), so the
    physical start points are identical — only the coordinates differ.

    Returns ``(z_stack (n_kept, ndim), n_kept)``. Requires ``obj`` built (the
    z value_and_grad is compiled here on first call)."""
    vag_z = jax.jit(jax.value_and_grad(obj.neg_log_posterior_z_raw))
    rng = np.random.default_rng(seed)
    starts, tries = [], 0
    while len(starts) < n_starts and tries < n_starts * max_factor:
        tries += 1
        u = rng.uniform(low, high, size=obj.ndim)
        z = jnp.asarray(_logit(u))
        loss, grad = vag_z(z)
        if np.isfinite(float(loss)) and np.all(np.isfinite(np.asarray(grad))):
            starts.append(z)
    return jnp.stack(starts), len(starts)


def fd_gradient_check(
    obj: MapObjective, seed: int = 0, n_dirs: int = 5, eps: float = 1e-4
) -> list[tuple[float, float]]:
    """Validate autodiff gradients of the z-space objective against directional
    finite differences. Returns a list of ``(autodiff_dir_deriv, fd_dir_deriv)``
    for ``n_dirs`` random unit directions at a finite cold start. A central
    difference is used; agreement to a few significant figures confirms the
    autodiff path (through the NNLS inversion) is faithful at production
    over-sampling."""
    z0, _ = make_z_starts(obj, 1, seed=seed)
    z0 = z0[0]
    grad_fn = jax.jit(jax.grad(obj.neg_log_posterior_z_raw))
    f_fn = jax.jit(obj.neg_log_posterior_z_raw)
    g = np.asarray(grad_fn(z0))
    rng = np.random.default_rng(seed + 1)
    out = []
    for _ in range(n_dirs):
        d = rng.normal(size=obj.ndim)
        d /= np.linalg.norm(d)
        ad = float(np.dot(g, d))
        fp = float(f_fn(jnp.asarray(np.asarray(z0) + eps * d)))
        fm = float(f_fn(jnp.asarray(np.asarray(z0) - eps * d)))
        fd = (fp - fm) / (2 * eps)
        out.append((ad, fd))
    return out


def _peak_gpu_mb() -> float | None:
    """Peak GPU bytes-in-use for device 0, in MB, or None on CPU / if the
    backend does not expose memory stats."""
    try:
        dev = jax.devices()[0]
        if dev.platform != "gpu":
            return None
        stats = dev.memory_stats() or {}
        peak = stats.get("peak_bytes_in_use")
        return round(peak / 1e6, 1) if peak else None
    except Exception:
        return None


def run_vmapped_map_solver(
    *,
    obj: MapObjective,
    solver,
    z_starts: jnp.ndarray,
    name: str,
    title: str,
    max_iters: int,
    termination: str,
    einstein_truth: float = 1.6,
    einstein_tol: float = 0.3,
    is_residual: bool = False,
) -> str:
    """Run a jaxopt solver from every start in ``z_starts`` under a single
    ``jax.vmap`` (one batched compile), separating cold-compile from warm-solve
    wall time, and write the standard summary.

    ``solver`` is any jaxopt solver whose ``.run(init_params)`` returns an
    ``OptStep(params, state)`` with a ``state.iter_num``. The objective is the
    shared unconstrained z-space MAP (``is_residual=False`` for first-/quasi-
    Newton on ``neg_log_posterior_z``; ``True`` for LM/Gauss-Newton on the
    residual least squares). Scoring is identical either way: map each solved
    ``z`` to physical params and evaluate the true log posterior."""
    n_starts = int(z_starts.shape[0])
    run_batched = jax.jit(jax.vmap(lambda z: solver.run(z)))

    # Cold call = batched compile + first solve; warm call = solve only.
    t0 = time.time()
    step = run_batched(z_starts)
    jax.block_until_ready(step.params)
    cold_s = time.time() - t0

    t0 = time.time()
    step = run_batched(z_starts)
    jax.block_until_ready(step.params)
    warm_s = time.time() - t0
    compile_s = max(cold_s - warm_s, 0.0)

    z_final = np.asarray(step.params)  # (S, ndim)
    iter_nums = np.asarray(getattr(step.state, "iter_num", np.full(n_starts, -1)))

    # Score every start in physical coordinates.
    phys = np.array([np.asarray(obj.physical_from_z(jnp.asarray(z))) for z in z_final])
    logpost = np.array(
        [
            float(obj.log_likelihood(jnp.asarray(p)) + obj.log_prior(jnp.asarray(p)))
            for p in phys
        ]
    )
    logpost = np.where(np.isfinite(logpost), logpost, -np.inf)
    j = int(np.argmax(logpost))
    best_params = phys[j]
    r_e = np.array(
        [
            obj.model.instance_from_vector(
                vector=list(p)
            ).galaxies.lens.mass.einstein_radius
            for p in phys
        ]
    )
    n_in_basin = int(np.sum(np.abs(r_e - einstein_truth) < einstein_tol))
    actual_iters = int(np.max(iter_nums)) if np.all(iter_nums >= 0) else max_iters
    print(
        f"\n{n_in_basin}/{n_starts} starts reached the basin "
        f"(p_hit={n_in_basin / n_starts:.2f}); iters used {int(np.min(iter_nums))}"
        f"..{int(np.max(iter_nums))} (cap {max_iters})"
    )

    # Accounting. LM/GN cost one residual + one Jacobian + one linear solve per
    # iteration (jaxopt damped-GN inner loop); first-/quasi-Newton cost one
    # value+grad per iteration (plus line-search probes we cannot cheaply count
    # under vmap, noted in the doc).
    scalar_iters = (
        int(np.sum(np.clip(iter_nums, 0, None)))
        if np.all(iter_nums >= 0)
        else n_starts * max_iters
    )
    acc = {
        "starts": n_starts,
        "actual_iters": f"{int(np.min(iter_nums))}..{int(np.max(iter_nums))} per start",
        "max_iters": max_iters,
        "batch_latency_ms": round(warm_s / max(actual_iters, 1) * 1e3, 1),
        "scalar_obj_evals": scalar_iters,
        "peak_gpu_mb": _peak_gpu_mb(),
    }
    if is_residual:
        acc["residual_evals"] = scalar_iters
        acc["jacobian_evals"] = scalar_iters
        acc["linear_solves"] = scalar_iters
    else:
        acc["scalar_grad_evals"] = scalar_iters

    return write_grad_summary(
        name=name,
        title=title,
        obj=obj,
        best_params=best_params,
        log_posterior_history=[float(logpost[j])],
        wall_s=compile_s + warm_s,
        compile_s=compile_s,
        warm_ms_per_eval=warm_s / max(actual_iters, 1) * 1e3,
        n_evals=scalar_iters,
        n_iters=actual_iters,
        converged=(n_in_basin > 0),
        config_line=(
            f"n_starts={n_starts}, max_iters={max_iters}, unconstrained z-space, "
            f"{n_in_basin}/{n_starts} in basin (p_hit={n_in_basin / n_starts:.2f})"
        ),
        termination=termination,
        accounting=acc,
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
    termination: str = "user-budget (fixed step count)",
    accounting: dict | None = None,
) -> str:
    """Write the standard summary block for a gradient-optimizer run and
    return the one-line comparison row (pipe-delimited) for comparison.txt.

    ``log_posterior_history`` is the per-iteration log-posterior (= -loss); it
    drives evals/time-to-ML. ``Max log L`` is the pure likelihood re-evaluated
    at ``best_params`` (physical coordinates) so it is directly comparable
    across scripts.

    ``termination`` records whether the iteration count is a user-set budget
    ("user-budget ...") or discovered at run time ("self-terminating ..."), the
    distinction requested for the results table. ``accounting`` is an optional
    dict of extra Part-5 fields (batched steps, batch latency, scalar-equivalent
    obj/grad evals, LM residual/Jacobian/solve counts, peak GPU MB, warm wall
    time); each present key is rendered in an Evaluation-accounting block.
    """
    best_params_np = np.asarray(best_params)
    best_instance = obj.model.instance_from_vector(vector=list(best_params_np))

    max_log_l = float(obj.log_likelihood(jnp.asarray(best_params)))
    log_prior_best = float(obj.log_prior(jnp.asarray(best_params)))
    max_log_post = max_log_l + log_prior_best
    r_e = float(best_instance.galaxies.lens.mass.einstein_radius)

    from searches_minimal._metrics import MLTracker

    evals_to_ml, time_to_ml = MLTracker.from_log_l_history(
        log_posterior_history, total_sampling_time=wall_s, tolerance=tolerance
    )

    acc = accounting or {}
    _labels = {
        "starts": "Starts/particles",
        "batched_steps": "Batched optimizer steps",
        "batch_latency_ms": "Batch latency per step (ms)",
        "scalar_obj_evals": "Scalar-equiv objective evals",
        "scalar_grad_evals": "Scalar-equiv gradient evals",
        "residual_evals": "Residual evals (LM/GN)",
        "jacobian_evals": "Jacobian evals (LM/GN)",
        "linear_solves": "Linear-system solves (LM/GN)",
        "rejected_steps": "Rejected steps (LM/GN)",
        "actual_iters": "Actual iterations",
        "max_iters": "Max iterations (cap)",
        "peak_gpu_mb": "Peak GPU memory (MB)",
        "warm_wall_s": "Warm (reused-compile) wall (s)",
    }
    acc_lines = "".join(
        f"{_labels.get(k, k) + ':':<32} {v}\n" for k, v in acc.items() if v is not None
    )
    acc_block = f"\n--- Evaluation accounting ---\n{acc_lines}" if acc_lines else ""

    summary = f"""\
--- {title} Results ---
Best fit:        {format_best_fit(best_instance)}
Einstein radius: {r_e:.4f}     (truth ~ 1.6)
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
{acc_block}
--- Convergence ---
Termination:         {termination}
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
    term_short = "self-term" if termination.startswith("self") else "user-budget"
    row = (
        f"| {title:<28} | {wall_s:8.1f} | {compile_s:8.1f} | {n_evals:8d} | "
        f"{max_log_l:12.2f} | {r_e:8.4f} | {term_short:>11} | "
        f"{'yes' if converged else 'no':>7} |"
    )
    print("\ncomparison row:\n" + row)
    return row
