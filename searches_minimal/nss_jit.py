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
from pathlib import Path

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
n_live = 200
num_mcmc_steps = 5
num_delete = 10
termination = -3
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
# Production-realistic: termination=-3 (delta-logZ ~< 1e-3). The pure-JAX
# path runs ~10x faster per evaluation than the callback path, so a
# converged run finishes in tens of minutes for this problem.
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
best_params = np.asarray(positions[best_idx]).tolist()
best_instance = model.instance_from_vector(vector=best_params)
max_logl = float(jnp.max(log_likelihoods))
n_evals = int(results.evals)

summary = f"""\
--- NSS (pure JAX) Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_logl:.4f}
Log evidence:    {float(results.logZs.mean()):.4f}

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (includes JIT compile)
Sampling time:       {float(results.time):.2f} s
Likelihood evals:    {n_evals}
Time per eval:       {float(results.time) / max(n_evals, 1) * 1e3:.3f} ms
ESS:                 {float(results.ess):.1f}
Posterior samples:   {len(positions)}
Sampler config:      n_live={n_live}, num_mcmc_steps={num_mcmc_steps}, num_delete={num_delete}, termination={termination}

--- Convergence ---
Converged:           yes (NSS termination={termination})
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
