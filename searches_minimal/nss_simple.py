"""
Minimal NSS Example — NumPy likelihood via ``jax.pure_callback`` (HST MGE)
-------------------------------------------------------------------------

Drives the NSS (Nested Slice Sampling) JAX-based sampler against the HST
MGE likelihood by wrapping PyAutoLens's NumPy ``AnalysisImaging`` with
``jax.pure_callback``. This is the slowest NSS path — each draw enters
Python once per callback — but it works with any existing NumPy likelihood
without requiring JAX-traceable code.

Per-call cost is dominated by the NumPy MGE evaluation (~1-2 s on this
problem), so converged runs at production ``n_live`` may exceed an hour.
See ``nss_jit.py`` for the fast, pure-JAX version that runs roughly an
order of magnitude faster per evaluation.

Requirements:
    pip install git+https://github.com/yallup/nss.git
    (pulls handley-lab/blackjax fork with nested sampling support)
"""
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from nss.ns import run_nested_sampling

from searches_minimal._metrics import MLTracker
from searches_minimal._setup import (
    build_analysis,
    build_dataset,
    build_model,
    format_best_fit,
)

dataset = build_dataset()
model = build_model()
analysis = build_analysis(dataset, use_jax=False)

print(f"Model free parameters: {model.total_free_parameters}")

n_likelihood_calls = 0
tracker = MLTracker()


def numpy_log_likelihood(params_np):
    global n_likelihood_calls
    n_likelihood_calls += 1
    instance = model.instance_from_vector(vector=params_np.tolist())
    log_l = float(analysis.log_likelihood_function(instance=instance))
    tracker.record(log_l)
    return np.float64(log_l)


def numpy_log_prior(params_np):
    log_priors = model.log_prior_list_from_vector(vector=params_np.tolist())
    return np.float64(sum(log_priors))


def log_likelihood(params):
    return jax.pure_callback(
        lambda p: jnp.float64(numpy_log_likelihood(np.asarray(p))),
        jax.ShapeDtypeStruct((), jnp.float64),
        params,
        vmap_method="sequential",
    )


def log_prior(params):
    return jax.pure_callback(
        lambda p: jnp.float64(numpy_log_prior(np.asarray(p))),
        jax.ShapeDtypeStruct((), jnp.float64),
        params,
        vmap_method="sequential",
    )


ndim = model.prior_count
n_live = 200
num_mcmc_steps = 5
num_delete = 1
termination = -3
rng_key = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng_key)

# Draw initial samples by mapping unit-cube draws through the model priors
# so every starting point lies in the prior's support (some priors are
# not simple uniforms).
unit_cube = np.asarray(
    jax.random.uniform(init_key, shape=(n_live, ndim), minval=0.0, maxval=1.0)
)
physical = np.array(
    [model.vector_from_unit_vector(list(u)) for u in unit_cube]
)
initial_samples = jnp.asarray(physical)

print(f"Running NSS (autofit NumPy likelihood via pure_callback)...")
print(f"  n_live={n_live}, n_dim={ndim}")
print(f"  Using jax.pure_callback for NumPy likelihood", flush=True)

t_start = time.time()
final_state, results = run_nested_sampling(
    rng_key,
    loglikelihood_fn=log_likelihood,
    prior_logprob=log_prior,
    num_mcmc_steps=num_mcmc_steps,
    initial_samples=initial_samples,
    num_delete=num_delete,
    termination=termination,
)
t_elapsed = time.time() - t_start

positions = final_state.particles.position
log_likelihoods = final_state.particles.loglikelihood

best_idx = int(jnp.argmax(log_likelihoods))
best_instance = model.instance_from_vector(vector=np.asarray(positions[best_idx]).tolist())
max_logl = float(jnp.max(log_likelihoods))
evals_to_ml, time_to_ml = tracker.finalise(max_log_l=max_logl, tolerance=1.0)

summary = f"""\
--- NSS (pure_callback) Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_logl:.4f}
Log evidence:    {float(results.logZs.mean()):.4f}

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (includes JIT compile)
Sampling time:       {float(results.time):.2f} s
Likelihood evals:    {n_likelihood_calls}
Time per eval:       {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms
ESS:                 {float(results.ess):.1f}
Posterior samples:   {len(positions)}
Sampler config:      n_live={n_live}, num_mcmc_steps={num_mcmc_steps}, num_delete={num_delete}, termination={termination}

--- Convergence ---
Converged:           yes (NSS termination={termination})
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
