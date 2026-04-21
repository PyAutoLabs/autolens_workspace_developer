"""
Minimal NSS Example — pure-JAX HST MGE likelihood
-------------------------------------------------

Drives NSS (Nested Slice Sampling) with the HST MGE likelihood running
fully under ``jax.jit``. The analysis is built with ``use_jax=True`` so
every internal call (border relocation, profile evaluation, inversion)
threads ``xp=jnp`` through the pipeline.

The likelihood closure takes a flat JAX parameter vector and calls
``model.instance_from_vector(vector=params, xp=jnp)`` inside the trace —
this is the same entry point used by ``af.Fitness.call`` in production.

Requirements:
    pip install git+https://github.com/yallup/nss.git
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from nss.ns import run_nested_sampling

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


def log_likelihood(params):
    instance = model.instance_from_vector(vector=params, xp=jnp)
    return analysis.log_likelihood_function(instance=instance)


def log_prior(params):
    # Initial samples are drawn from the prior, so a flat log-prior
    # inside the support is sufficient for the sampler.
    return jnp.float64(0.0)


ndim = model.prior_count
n_live = 30
rng_key = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng_key)

# Map unit-cube draws through the model's priors so every initial point
# lies inside the prior's support.
unit_cube = np.asarray(
    jax.random.uniform(init_key, shape=(n_live, ndim), minval=0.0, maxval=1.0)
)
physical = np.array([model.vector_from_unit_vector(list(u)) for u in unit_cube])
initial_samples = jnp.asarray(physical)

print("Running NSS (pure JAX, flat vector) nested sampling...")
print(f"  n_live={n_live}, n_dim={ndim}")
print("  JIT compilation will happen on first step (can take a while)\n")

t_start = time.time()
# ``termination`` is large so this is a smoke test — drop it for a real run.
final_state, results = run_nested_sampling(
    rng_key,
    loglikelihood_fn=log_likelihood,
    prior_logprob=log_prior,
    num_mcmc_steps=2,
    initial_samples=initial_samples,
    num_delete=5,
    termination=1e5,
)
t_elapsed = time.time() - t_start

positions = final_state.particles.position
log_likelihoods = final_state.particles.loglikelihood

best_idx = int(jnp.argmax(log_likelihoods))
best_params = np.asarray(positions[best_idx]).tolist()
best_instance = model.instance_from_vector(vector=best_params)

print("\n--- NSS (pure JAX) Results ---")
print(format_best_fit(best_instance))
print(f"Log evidence:  {float(results.logZs.mean()):.2f}")
print(f"\n--- Performance ---")
print(f"Wall time:          {t_elapsed:.2f} s (includes JIT compile)")
print(f"Sampling time:      {float(results.time):.2f} s")
print(f"Likelihood evals:   {int(results.evals)}")
print(f"Time per eval:      {float(results.time) / max(int(results.evals), 1) * 1e3:.3f} ms")
