"""
Minimal Emcee Example (HST MGE lens likelihood)
-----------------------------------------------

Drives the Emcee MCMC sampler directly against the HST MGE imaging
likelihood, bypassing ``af.NonLinearSearch``. Useful as a fast end-to-end
smoke test of the real PyAutoLens likelihood under a production sampler.

Walker counts and chain length are kept small so the search finishes in a
few minutes — this is a wiring test, not a converged posterior. Walkers are
initialised in a tight ball around the prior medians so they do not waste
steps falling off the prior edges.

Requirements:
    pip install emcee
"""
import time
import numpy as np

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

import emcee


n_likelihood_calls = 0


def log_posterior(params):
    global n_likelihood_calls
    log_prior = float(sum(model.log_prior_list_from_vector(vector=list(params))))
    if not np.isfinite(log_prior):
        return -np.inf
    n_likelihood_calls += 1
    instance = model.instance_from_vector(vector=list(params))
    log_like = float(analysis.log_likelihood_function(instance=instance))
    return log_like + log_prior


ndim = model.prior_count
nwalkers = 2 * ndim
nsteps = 2

# Initialise each walker near a random draw from the prior so every walker
# starts inside the prior's support.
initial_positions = np.array(
    [model.random_vector_from_priors_within_limits(0.4, 0.6) for _ in range(nwalkers)]
)

sampler = emcee.EnsembleSampler(
    nwalkers=nwalkers,
    ndim=ndim,
    log_prob_fn=log_posterior,
)

t_start = time.time()
sampler.run_mcmc(initial_positions, nsteps=nsteps, progress=True)
t_elapsed = time.time() - t_start

flat_samples = sampler.get_chain(flat=True)
flat_log_prob = sampler.get_log_prob(flat=True)

best_idx = np.argmax(flat_log_prob)
best_instance = model.instance_from_vector(vector=list(flat_samples[best_idx]))

print("\n--- Emcee Results ---")
print(format_best_fit(best_instance))
print(f"Best log posterior:  {flat_log_prob[best_idx]:.2f}")
print(f"\n--- Performance ---")
print(f"Wall time:          {t_elapsed:.2f} s")
print(f"Likelihood calls:   {n_likelihood_calls}")
print(f"Time per call:      {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms")
print(f"Walkers:            {nwalkers}")
print(f"Steps per walker:   {nsteps}")
