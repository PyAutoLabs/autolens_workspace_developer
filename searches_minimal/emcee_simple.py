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
from pathlib import Path

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
best_log_post = float(flat_log_prob[best_idx])

summary = f"""\
--- Emcee Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       n/a (only log posterior tracked; best log posterior = {best_log_post:.4f})
Log evidence:    n/a (Emcee is MCMC, not nested sampling)

--- Performance ---
Wall time:           {t_elapsed:.2f} s
Sampling time:       n/a
Likelihood evals:    {n_likelihood_calls}
Time per eval:       {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms
ESS:                 n/a (autocorr undefined with nsteps={nsteps})
Posterior samples:   {len(flat_samples)}
Sampler config:      nwalkers={nwalkers}, nsteps={nsteps} (smoke test)
"""

print()
print(summary)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
summary_path = output_dir / f"{Path(__file__).stem}_summary.txt"
summary_path.write_text(summary)
print(f"Summary written to: {summary_path}")
