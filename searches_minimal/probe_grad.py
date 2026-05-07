"""
Gradient probe for the HST MGE imaging likelihood — sanity check whether
``jax.grad`` of the MGE+inversion+chi-squared stack is robust enough for
HMC-driven sampling (see ``nss_grad.py``).

Computes ``jax.value_and_grad(log_likelihood)`` at five representative
parameter points (prior median, three samples from the same narrow band
``nss_grad.py`` initialises in, one full-prior random draw) and reports:

  - finite-ness of log L and gradient
  - gradient-magnitude condition (max |g_i| / median |g_i|)
  - per-parameter "natural" gradient (g_i * sigma_prior_i)
  - finite-difference cross-check on six analytic-gradient entries

Final verdict line is one of:
  OK_HMC_VIABLE   WARN_ILL_CONDITIONED   FAIL_NAN_OR_INF   FAIL_FD_MISMATCH
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from searches_minimal._setup import (
    build_analysis,
    build_dataset,
    build_model,
)


def natural_sigmas(model) -> np.ndarray:
    """Per-parameter natural scale, prior-aware.

    Gaussian -> sigma; Uniform -> (upper - lower); Log uniform -> width on
    the unit cube mapped through the prior. Falls back to 1.0 if nothing
    sensible exists. Used for both finite-diff steps and "natural" gradient
    units (g_i * sigma_i).
    """
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


def main() -> int:
    print(f"JAX devices: {jax.devices()}")
    if jax.default_backend() != "gpu":
        print("ERROR: JAX is not on GPU. Aborting.", file=sys.stderr)
        return 2

    dataset = build_dataset()
    model = build_model()
    analysis = build_analysis(dataset, use_jax=True)

    ndim = model.prior_count
    sigmas = natural_sigmas(model)
    print(f"Model free parameters: {ndim}")
    print(f"Natural sigmas (per-parameter step scales):")
    for i, s in enumerate(sigmas):
        print(f"  [{i:2d}] sigma={s:.4g}")

    def log_likelihood(params):
        instance = model.instance_from_vector(vector=params, xp=jnp)
        return analysis.log_likelihood_function(instance=instance)

    log_l_only = jax.jit(log_likelihood)
    vag = jax.jit(jax.value_and_grad(log_likelihood))

    rng = np.random.default_rng(seed=42)

    p_med = jnp.asarray(model.vector_from_unit_vector([0.5] * ndim))
    p_band = [
        jnp.asarray(
            model.vector_from_unit_vector(
                rng.uniform(0.4, 0.6, size=ndim).tolist()
            )
        )
        for _ in range(3)
    ]
    p_full = jnp.asarray(
        model.vector_from_unit_vector(rng.uniform(0.0, 1.0, size=ndim).tolist())
    )

    points = {
        "prior_median": p_med,
        "narrow_band_0": p_band[0],
        "narrow_band_1": p_band[1],
        "narrow_band_2": p_band[2],
        "full_prior_random": p_full,
    }

    print("\nJIT-compiling value_and_grad (this is the expensive step)...", flush=True)
    t0 = time.time()
    log_l0, grad0 = vag(p_med)
    log_l0 = float(jax.block_until_ready(log_l0))
    grad0 = np.asarray(jax.block_until_ready(grad0))
    print(f"  Compiled in {time.time() - t0:.1f} s", flush=True)

    results = {}
    any_nan = False
    worst_cond = 0.0

    for name, params in points.items():
        if name == "prior_median":
            log_l, grad = log_l0, grad0
        else:
            log_l_jax, grad_jax = vag(params)
            log_l = float(jax.block_until_ready(log_l_jax))
            grad = np.asarray(jax.block_until_ready(grad_jax))
        finite_logl = bool(np.isfinite(log_l))
        finite_grad = bool(np.all(np.isfinite(grad)))
        any_nan = any_nan or (not finite_logl) or (not finite_grad)
        abs_grad = np.abs(grad)
        nonzero = abs_grad[abs_grad > 0]
        median = float(np.median(nonzero)) if nonzero.size else 0.0
        max_abs = float(np.max(abs_grad))
        cond = max_abs / median if median > 0 else float("inf")
        worst_cond = max(worst_cond, cond if np.isfinite(cond) else 0.0)
        natural_grad = grad * sigmas
        results[name] = dict(
            params=np.asarray(params),
            log_l=log_l,
            grad=grad,
            natural_grad=natural_grad,
            finite_logl=finite_logl,
            finite_grad=finite_grad,
            cond=cond,
            l_inf=max_abs,
            l_2=float(np.linalg.norm(grad)),
        )

        print(f"\n=== {name} ===")
        print(f"  log L = {log_l:.4f}    finite={finite_logl}")
        print(
            f"  ||grad||_inf = {max_abs:.3e}    ||grad||_2 = {results[name]['l_2']:.3e}"
            f"    finite={finite_grad}"
        )
        print(f"  max/median |grad| = {cond:.3g}     (HMC-condition proxy)")
        print(
            f"  natural_grad (g*sigma): max={np.max(np.abs(natural_grad)):.3e}"
            f"  median={float(np.median(np.abs(natural_grad[natural_grad != 0]))):.3e}"
            f"  min_nonzero={float(np.min(np.abs(natural_grad[natural_grad != 0]))):.3e}"
        )

    # Finite-difference cross-check: 3 random params at p_med and p_band_0.
    fd_checks = []
    for label, p in [("prior_median", p_med), ("narrow_band_0", p_band[0])]:
        idxs = rng.choice(ndim, size=3, replace=False)
        for i in idxs:
            eps = 1e-3 * sigmas[i]
            p_plus = np.asarray(p).copy()
            p_plus[i] += eps
            p_minus = np.asarray(p).copy()
            p_minus[i] -= eps
            l_plus = float(jax.block_until_ready(log_l_only(jnp.asarray(p_plus))))
            l_minus = float(jax.block_until_ready(log_l_only(jnp.asarray(p_minus))))
            grad_fd = (l_plus - l_minus) / (2.0 * eps)
            grad_an = float(results[label]["grad"][i])
            denom = max(abs(grad_an), abs(grad_fd), 1e-10)
            rel_err = abs(grad_an - grad_fd) / denom
            fd_checks.append(
                dict(
                    point=label,
                    idx=int(i),
                    eps=float(eps),
                    grad_fd=grad_fd,
                    grad_an=grad_an,
                    rel_err=rel_err,
                )
            )

    print("\n=== Finite-difference cross-check ===")
    for c in fd_checks:
        flag = "OK" if c["rel_err"] < 1e-2 else ("WARN" if c["rel_err"] < 1e-1 else "FAIL")
        print(
            f"  [{c['point']:13s}] idx={c['idx']:2d}  eps={c['eps']:.3e}"
            f"  grad_an={c['grad_an']:+.4e}  grad_fd={c['grad_fd']:+.4e}"
            f"  rel_err={c['rel_err']:.2e}  [{flag}]"
        )

    fd_pass = all(c["rel_err"] < 1e-2 for c in fd_checks)
    fd_warn = all(c["rel_err"] < 1e-1 for c in fd_checks)

    if any_nan:
        verdict = "FAIL_NAN_OR_INF"
    elif not fd_warn:
        verdict = "FAIL_FD_MISMATCH"
    elif not fd_pass:
        verdict = "WARN_FD_LOOSE"
    elif worst_cond > 1e6:
        verdict = "WARN_ILL_CONDITIONED"
    else:
        verdict = "OK_HMC_VIABLE"

    print(f"\n=== VERDICT: {verdict} ===")
    print(f"  worst max/median |grad| across points: {worst_cond:.3g}")
    print(f"  finite-diff worst rel_err: {max(c['rel_err'] for c in fd_checks):.2e}")

    summary_lines = ["--- Gradient probe ---", f"Verdict: {verdict}", ""]
    summary_lines.append("Per-point gradient diagnostics:")
    for name, r in results.items():
        summary_lines.append(
            f"  {name:18s}  log L = {r['log_l']:14.4f}"
            f"  ||g||_inf={r['l_inf']:.3e}  max/median={r['cond']:.3g}"
            f"  finite={r['finite_logl'] and r['finite_grad']}"
        )
    summary_lines.append("")
    summary_lines.append("Finite-difference cross-check (eps = 1e-3 * sigma_prior):")
    for c in fd_checks:
        flag = "OK" if c["rel_err"] < 1e-2 else ("WARN" if c["rel_err"] < 1e-1 else "FAIL")
        summary_lines.append(
            f"  {c['point']:13s} idx={c['idx']:2d}  grad_an={c['grad_an']:+.4e}"
            f"  grad_fd={c['grad_fd']:+.4e}  rel_err={c['rel_err']:.2e} [{flag}]"
        )
    summary_lines.append("")
    summary_lines.append(
        f"worst max/median |grad|: {worst_cond:.3g}    "
        f"worst FD rel_err: {max(c['rel_err'] for c in fd_checks):.2e}"
    )

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "probe_grad_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\nSummary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
