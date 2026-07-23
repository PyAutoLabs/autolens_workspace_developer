"""
Shared warm-start harness for the JAX-native posterior sampler wave.
---------------------------------------------------------------------

Posterior samplers in this workspace are meant to be **warm-started from a JAX
optimizer's maximum-likelihood solution**, not cold-started from the prior. A
cold prior-to-posterior run is not how they are used in practice and is not a
representative benchmark: on the HST MGE likelihood the posterior is ~1000x
narrower than the prior, so a cold sampler spends its whole budget crossing a
scale gap that a gradient optimizer closes in seconds
(see ``smc_gradient_findings.md``).

This module produces **one cached warm-start artifact** so every sampler in the
wave starts from the *identical* point — making the comparison apples-to-apples:

    mle   : (ndim,) physical parameter vector at the best multi-start optimum
    std   : (ndim,) per-parameter scale of a Gaussian *reference* centred on it
    log_l : log likelihood at the MLE

``std`` comes from a Laplace approximation (inverse-Hessian diagonal at the
optimum) where that is finite and positive-definite, falling back to the spread
of the converged multi-start endpoints, then to a fraction of the prior scales.

Why a *reference* and not just an initial point: SMC / nested sampling obtain
their log-evidence by tempering from a **normalised** starting distribution.
Dropping particles at the MLE destroys that (the tempering increments no longer
integrate a valid path), so log Z becomes meaningless. Tempering instead from an
explicit normalised Gaussian centred on the MLE keeps log Z valid *and* keeps the
path short. Pure MCMC samplers (NUTS/HMC/MCLMC) can ignore ``std`` and just use
``mle`` as their initial point.

Usage
-----

    # one-off, writes output/warm_start.json
    python -m searches_minimal._warm_start --n-starts 12 --n-steps 300

    # from a sampler script (computes+caches on first call)
    from searches_minimal._warm_start import load_warm_start
    ws = load_warm_start()
    ws.mle, ws.std, ws.log_l
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
from optax import contrib

from searches_minimal._grad_setup import build_map_objective, time_compile

CACHE_PATH = Path(__file__).parent / "output" / "warm_start.json"

# Broad unit-cube spread for the multi-start draws (matches the other
# multi-start scripts so the starts are directly comparable).
START_LOW = 0.15
START_HIGH = 0.85
MAX_CONSECUTIVE_NAN = 10


@dataclass
class WarmStart:
    mle: np.ndarray  # (ndim,) physical parameters at the best optimum
    std: np.ndarray  # (ndim,) marginal reference scale (diagonal fallback)
    cov: np.ndarray | None  # (ndim, ndim) FULL reference covariance, or None
    log_l: float  # log likelihood at the MLE
    n_starts: int
    n_converged: int
    optimizer: str
    std_source: str  # "laplace" | "multistart_spread" | "prior_fraction"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["mle"] = [float(v) for v in self.mle]
        d["std"] = [float(v) for v in self.std]
        d["cov"] = None if self.cov is None else [[float(v) for v in r] for r in self.cov]
        return d


# --------------------------------------------------------------------------
# Multi-start Prodigy in physical parameter space.
#
# Prodigy is learning-rate-free (a D-Adaptation successor), so there is no step
# size to hand-tune — the reason it is the natural default here. The optimizer
# state is vmapped **per start**: Prodigy estimates global scalars (its adapted
# distance ``d``), and a shared state would couple the starts and silently break
# multi-start independence (see lr_free_multistart.py).
# --------------------------------------------------------------------------


def _finite_gradient_starts(obj, n_starts: int, seed: int) -> jnp.ndarray:
    """Draw broad starts with finite value+gradient (physical space).

    The exact prior median is degenerate for this lens model — elliptical
    components and external shear sit at (0, 0) where the arctan2/sqrt that map
    to angle/magnitude have singular (NaN) gradients. Filtering on finiteness
    avoids it, exactly as ``_grad_setup.robust_cold_start`` documents.
    """
    rng = np.random.default_rng(seed)
    starts, tries = [], 0
    while len(starts) < n_starts and tries < n_starts * 20:
        tries += 1
        u = rng.uniform(START_LOW, START_HIGH, size=obj.ndim)
        x = jnp.asarray(obj.model.vector_from_unit_vector(list(u)))
        loss, grad = obj.value_and_grad(x)
        if np.isfinite(float(loss)) and np.all(np.isfinite(np.asarray(grad))):
            starts.append(x)
    if not starts:
        raise RuntimeError("no finite-gradient starts found")
    print(f"Collected {len(starts)} finite-gradient starts (from {tries} draws)")
    return jnp.stack(starts)


def _laplace_cov(obj, mle: jnp.ndarray) -> np.ndarray | None:
    """Full posterior covariance from the inverse Hessian at the MLE.

    The FULL matrix matters, not just its diagonal: the lens parameters
    (einstein_radius / ellipticity / shear / centres) are strongly correlated, so
    a diagonal-only whitening leaves the posterior a thin *tilted* ridge and a
    spherical MALA/HMC proposal still walks straight off it (measured: diagonal
    whitening lifted acceptance off zero only at the first temperature, then it
    collapsed again). Whitening by the Cholesky factor of this matrix decorrelates
    properly.

    Returns ``None`` if the Hessian is not finite or the covariance is not
    positive-definite (the caller then falls back).
    """
    try:
        hess = np.asarray(jax.hessian(obj.neg_log_posterior_raw)(mle))
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"  Laplace: Hessian failed ({type(exc).__name__}), falling back")
        return None
    if not np.all(np.isfinite(hess)):
        print("  Laplace: Hessian non-finite, falling back")
        return None
    hess = 0.5 * (hess + hess.T)  # symmetrise away round-off
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        print("  Laplace: Hessian singular, falling back")
        return None
    cov = 0.5 * (cov + cov.T)
    if not np.all(np.isfinite(cov)):
        print("  Laplace: covariance non-finite, falling back")
        return None
    try:
        np.linalg.cholesky(cov)  # positive-definite?
    except np.linalg.LinAlgError:
        print("  Laplace: covariance not positive-definite, falling back")
        return None
    evals = np.linalg.eigvalsh(cov)
    print(
        f"  Laplace covariance OK — condition number {evals.max() / evals.min():.3g} "
        f"(this is the anisotropy a diagonal whitening cannot remove)"
    )
    return cov


def compute_warm_start(
    n_starts: int = 12,
    n_steps: int = 300,
    seed: int = 0,
    optimizer: str = "prodigy",
) -> WarmStart:
    """Run the multi-start optimizer and derive the warm-start artifact."""
    obj = build_map_objective()
    print(f"Model free parameters: {obj.ndim}")

    compile_s = time_compile(obj)
    print(f"JIT compile (value_and_grad): {compile_s:.1f} s")

    params = _finite_gradient_starts(obj, n_starts, seed)

    batched_vag = jax.jit(jax.vmap(jax.value_and_grad(obj.neg_log_posterior_raw)))
    batched_log_l = jax.jit(jax.vmap(obj.log_likelihood))

    build = {"prodigy": lambda: contrib.prodigy(), "adam": lambda: optax.adam(1e-2)}[
        optimizer
    ]
    opt = optax.apply_if_finite(build(), max_consecutive_errors=MAX_CONSECUTIVE_NAN)
    opt_states = jax.vmap(opt.init)(params)

    @jax.jit
    def step_update(grads, states, params):
        return jax.vmap(opt.update)(grads, states, params)

    global_best_loss = np.inf
    global_best_params = params[0]

    print(f"\n[{optimizer}] {n_starts}-start x {n_steps} steps")
    t0 = time.time()
    for i in range(n_steps):
        losses, grads = batched_vag(params)
        losses_np = np.where(np.isfinite(np.asarray(losses)), np.asarray(losses), np.inf)
        j = int(np.argmin(losses_np))
        if losses_np[j] < global_best_loss:
            global_best_loss = float(losses_np[j])
            global_best_params = params[j]
        updates, opt_states = step_update(grads, opt_states, params)
        params = optax.apply_updates(params, updates)
        if i % 50 == 0:
            print(f"  step {i:4d}: best log_posterior = {-global_best_loss:.2f}")
    print(f"  optimise wall: {time.time() - t0:.1f} s")

    mle = global_best_params
    log_l = float(batched_log_l(mle[None, :])[0])

    # Converged set = starts whose final loss is within 1 nat of the best.
    final_losses = np.where(
        np.isfinite(np.asarray(batched_vag(params)[0])),
        np.asarray(batched_vag(params)[0]),
        np.inf,
    )
    converged_mask = final_losses <= (global_best_loss + 1.0)
    n_converged = int(converged_mask.sum())
    print(f"  {n_converged}/{n_starts} starts within 1 nat of the best optimum")

    # --- Gaussian reference scale, best source first ------------------------
    cov = _laplace_cov(obj, mle)
    std = None if cov is None else np.sqrt(np.diag(cov))
    std_source = "laplace"
    if std is None:
        endpoints = np.asarray(params)[converged_mask]
        if endpoints.shape[0] >= 3:
            spread = endpoints.std(axis=0)
            if np.all(np.isfinite(spread)) and np.all(spread > 0):
                std, std_source = spread, "multistart_spread"
    if std is None:
        std, std_source = 0.1 * np.asarray(obj.sigmas), "prior_fraction"
    print(f"  reference scale from: {std_source}")

    return WarmStart(
        mle=np.asarray(mle, dtype=np.float64),
        std=np.asarray(std, dtype=np.float64),
        cov=None if cov is None else np.asarray(cov, dtype=np.float64),
        log_l=log_l,
        n_starts=int(n_starts),
        n_converged=n_converged,
        optimizer=optimizer,
        std_source=std_source,
    )


# --------------------------------------------------------------------------
# Cache.
# --------------------------------------------------------------------------


def save_warm_start(ws: WarmStart, path: Path = CACHE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ws.as_dict(), indent=2))
    return path


def load_warm_start(refresh: bool = False, path: Path = CACHE_PATH, **kwargs) -> WarmStart:
    """Load the cached warm start, computing (and caching) it if absent."""
    if path.exists() and not refresh:
        d = json.loads(path.read_text())
        return WarmStart(
            mle=np.asarray(d["mle"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
            cov=(
                None
                if d.get("cov") is None
                else np.asarray(d["cov"], dtype=np.float64)
            ),
            log_l=float(d["log_l"]),
            n_starts=int(d["n_starts"]),
            n_converged=int(d["n_converged"]),
            optimizer=str(d["optimizer"]),
            std_source=str(d["std_source"]),
        )
    ws = compute_warm_start(**kwargs)
    save_warm_start(ws, path)
    return ws


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-starts", type=int, default=12)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", choices=["prodigy", "adam"], default="prodigy")
    parser.add_argument("--refresh", action="store_true", help="recompute even if cached")
    args = parser.parse_args()

    ws = load_warm_start(
        refresh=args.refresh,
        n_starts=args.n_starts,
        n_steps=args.n_steps,
        seed=args.seed,
        optimizer=args.optimizer,
    )
    path = save_warm_start(ws)
    print("\n--- Warm start ---")
    print(f"optimizer:     {ws.optimizer}")
    print(f"max log L:     {ws.log_l:.4f}")
    print(f"converged:     {ws.n_converged}/{ws.n_starts}")
    print(f"ref scale via: {ws.std_source}  (full covariance: {ws.cov is not None})")
    print(f"mle:           {np.array2string(ws.mle, precision=4, max_line_width=100)}")
    print(f"std:           {np.array2string(ws.std, precision=4, max_line_width=100)}")
    print(f"written to:    {path}")


if __name__ == "__main__":
    main()
