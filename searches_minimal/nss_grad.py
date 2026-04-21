"""
Minimal NSS Example — pure-JAX HST MGE likelihood with gradients
----------------------------------------------------------------

Runs the NSS HMC-based Sequential Monte Carlo variant, which uses
``jax.grad`` to accelerate sampling via Hamiltonian Monte Carlo. The
HST MGE likelihood is fully JAX-traceable because the MGE light profile
supports gradients, so ``jax.grad`` propagates through the inversion
and chi-squared stack.

The likelihood closure takes a flat JAX parameter vector and calls
``model.instance_from_vector(vector=params, xp=jnp)`` inside the trace —
identical to ``nss_jit.py`` with HMC replacing slice sampling.

Requirements:
    pip install git+https://github.com/yallup/nss.git
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from nss.smc import run_hmc_sequential_mc

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
    return jnp.float64(0.0)


ndim = model.prior_count
# Small live count + max_steps=1 keeps this a smoke test; every SMC step
# triggers n_live * hmc_trajectory_length gradient evaluations of the MGE
# likelihood, which is expensive even after JIT.
n_live = 8

rng_key = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng_key)

# Start each walker inside a narrow band around the prior median so HMC
# doesn't have to escape the long tails of a wide prior in its warmup.
unit_cube = np.asarray(
    jax.random.uniform(init_key, shape=(n_live, ndim), minval=0.4, maxval=0.6)
)
physical = np.array([model.vector_from_unit_vector(list(u)) for u in unit_cube])
initial_samples = jnp.asarray(physical)

print("Running NSS (HMC + jax.grad) sequential Monte Carlo...")
print("  Note: jax.jit(jax.value_and_grad(log_likelihood)) compiles for a few")
print("  minutes on the MGE likelihood before the first HMC step.")
print(f"  n_live={n_live}, n_dim={ndim}")
print("  JIT compilation will happen on first step (can take a while)", flush=True)

t_start = time.time()
# ``max_steps=1`` stops after a single SMC annealing step so this is a
# smoke test of the HMC + grad path, not a converged run. Drop ``max_steps``
# and raise ``n_live`` / ``warmup_steps`` for a real posterior.
smc_state, results = run_hmc_sequential_mc(
    rng_key,
    loglikelihood_fn=log_likelihood,
    prior_logprob=log_prior,
    num_mcmc_steps=1,
    initial_samples=initial_samples,
    hmc_trajectory_length=2,
    target_ess=0.9,
    warmup_steps=2,
    max_steps=1,
)
t_elapsed = time.time() - t_start

best_idx = int(jnp.argmax(smc_state.weights))
best_params = np.asarray(smc_state.particles[best_idx]).tolist()
best_instance = model.instance_from_vector(vector=best_params)

print("\n--- NSS (HMC + jax.grad) Results ---")
print(format_best_fit(best_instance))
print(f"Log evidence:  {float(results.logZs.mean()):.2f}")
print(f"\n--- Performance ---")
print(f"Wall time:          {t_elapsed:.2f} s (includes JIT compile + warmup)")
print(f"Sampling time:      {float(results.time):.2f} s")
print(f"Gradient evals:     {int(results.evals)}")
print(f"Time per eval:      {float(results.time) / max(int(results.evals), 1) * 1e3:.3f} ms")
print(f"ESS:                {float(results.ess):.0f}")
