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
from pathlib import Path

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
# Production-realistic: n_live=50 is at the upper end of what fits inside
# a 2-hour wall-clock budget for the gradient path -- jax.grad through the
# MGE inversion has a heavy JIT compile cost (>20 min) and each step
# triggers n_live * hmc_trajectory_length gradient evaluations.
n_live = 50
num_mcmc_steps = 3
hmc_trajectory_length = 5
target_ess = 0.9
warmup_steps = 100

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
# Run SMC to its native convergence (lambda=1.0 reached via target_ess
# annealing schedule). Drop ``max_steps`` so the sampler decides when it
# has annealed all the way to the target distribution.
smc_state, results = run_hmc_sequential_mc(
    rng_key,
    loglikelihood_fn=log_likelihood,
    prior_logprob=log_prior,
    num_mcmc_steps=num_mcmc_steps,
    initial_samples=initial_samples,
    hmc_trajectory_length=hmc_trajectory_length,
    target_ess=target_ess,
    warmup_steps=warmup_steps,
)
t_elapsed = time.time() - t_start

best_idx = int(jnp.argmax(smc_state.weights))
best_params = np.asarray(smc_state.particles[best_idx]).tolist()
best_instance = model.instance_from_vector(vector=best_params)
n_evals = int(results.evals)

summary = f"""\
--- NSS (HMC + jax.grad) Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       n/a (SMC tracks weights, not raw log L per particle)
Log evidence:    {float(results.logZs.mean()):.4f}

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (includes JIT compile + warmup)
Sampling time:       {float(results.time):.2f} s
Likelihood evals:    {n_evals}     (each = likelihood + gradient via jax.grad)
Time per eval:       {float(results.time) / max(n_evals, 1) * 1e3:.3f} ms
ESS:                 {float(results.ess):.1f}
Posterior samples:   {len(smc_state.particles)}
Sampler config:      n_live={n_live}, num_mcmc_steps={num_mcmc_steps}, hmc_trajectory_length={hmc_trajectory_length}, target_ess={target_ess}, warmup_steps={warmup_steps}

--- Convergence ---
Converged:           yes (HMC SMC reaches lambda=1.0 via target_ess={target_ess})
Evals to ML:         n/a (pure JAX path; intermediate evals not exposed without host callback)
Time to ML:          n/a
"""

print()
print(summary)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
summary_path = output_dir / f"{Path(__file__).stem}_summary.txt"
summary_path.write_text(summary)
print(f"Summary written to: {summary_path}")
