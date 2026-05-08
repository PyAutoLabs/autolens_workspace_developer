"""
Minimal Nautilus Example — pure-JAX HST MGE likelihood
------------------------------------------------------

Drives the Nautilus nested sampler against the HST MGE imaging likelihood
running fully under ``jax.jit``. The analysis is built with ``use_jax=True``
and the closure is passed through ``jax.jit`` once, ahead of sampling, so
the JIT compile cost is reported separately from the sampling wall time.

Nautilus itself is a NumPy sampler, so the wrapper does
``np.asarray(jit_loglike(jnp.asarray(params)))`` per call -- the JAX kernel
runs but every evaluation crosses the Python <-> JAX boundary, the same
way ``nss_simple.py`` crosses the Python boundary in the opposite
direction. Compare versus ``nss_jit.py`` (no Python boundary) and
``nautilus_simple.py`` (NumPy likelihood).

``n_live`` and ``n_like_max`` are kept at the smoke-test values used by
``nautilus_simple.py`` -- this is a wiring test, not a converged posterior.

Requirements:
    pip install nautilus-sampler
"""
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from searches_minimal._metrics import MLTracker
from searches_minimal._setup import (
    build_analysis,
    build_dataset,
    build_model,
    format_best_fit,
)

dataset = build_dataset()
model = build_model()
analysis = build_analysis(dataset, use_jax=True)

print(f"Model free parameters: {model.total_free_parameters}")

from nautilus import Sampler


def log_likelihood_jax(params):
    """Pure-JAX log likelihood: flat parameter vector -> scalar log L."""
    instance = model.instance_from_vector(vector=params, xp=jnp)
    return analysis.log_likelihood_function(instance=instance)


jit_log_likelihood = jax.jit(log_likelihood_jax)

# Warm up the JIT once so the compile cost is measured separately.
warmup_unit = [0.5] * model.prior_count
warmup_physical = jnp.asarray(model.vector_from_unit_vector(warmup_unit))
print("JIT-compiling MGE likelihood (one-shot)...", flush=True)
t_jit_start = time.time()
_ = float(jax.block_until_ready(jit_log_likelihood(warmup_physical)))
t_jit = time.time() - t_jit_start
print(f"  Compiled in {t_jit:.2f} s", flush=True)


def prior_transform(cube):
    """Map a unit cube to physical parameters via the model's priors."""
    return np.array(model.vector_from_unit_vector(cube))


n_likelihood_calls = 0
tracker = MLTracker()


def log_likelihood(params):
    """Adapter: NumPy in, JIT'd JAX likelihood, Python float out."""
    global n_likelihood_calls
    n_likelihood_calls += 1
    log_l = float(jit_log_likelihood(jnp.asarray(params)))
    tracker.record(log_l)
    return log_l


n_live = 200

sampler = Sampler(
    prior=prior_transform,
    likelihood=log_likelihood,
    n_dim=model.prior_count,
    n_live=n_live,
)

t_start = time.time()
# Run to Nautilus's default convergence (n_eff=10000, f_live=0.01) on the
# JAX-jitted MGE likelihood. JIT compile is paid once above; per-call cost
# inside sampling is the JAX kernel + Python<->JAX boundary.
sampler.run(verbose=True)
t_elapsed = time.time() - t_start

points, log_w, log_l = sampler.posterior()
best_idx = np.argmax(log_l)
best_instance = model.instance_from_vector(vector=list(points[best_idx]))
max_logl = float(np.max(log_l))

evals_to_ml, time_to_ml = tracker.finalise(max_log_l=max_logl, tolerance=1.0)

summary = f"""\
--- Nautilus (JAX JIT) Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_logl:.4f}
Log evidence:    {float(sampler.log_z):.4f}

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (excludes JIT compile, run ahead of time)
Sampling time:       n/a (Nautilus does not split warmup)
JIT compile time:    {t_jit:.2f} s     (one-shot warm-up before sampling)
Likelihood evals:    {n_likelihood_calls}
Time per eval:       {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms
ESS:                 {float(sampler.n_eff):.1f}
Posterior samples:   {len(points)}
Sampler config:      n_live={n_live}, default n_eff=10000, f_live=0.01

--- Convergence ---
Converged:           yes (Nautilus default n_eff / f_live)
Evals to ML:         {evals_to_ml if evals_to_ml is not None else 'n/a'}     (first eval within 1 nat of max log L)
Time to ML:          {f'{time_to_ml:.2f} s' if time_to_ml is not None else 'n/a'}
"""

print()
print(summary)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
summary_path = output_dir / f"{Path(__file__).stem}_summary.txt"
summary_path.write_text(summary)
print(f"Summary written to: {summary_path}")
