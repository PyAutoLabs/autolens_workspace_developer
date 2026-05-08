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

import emcee


n_likelihood_calls = 0
tracker = MLTracker()


def log_posterior(params):
    global n_likelihood_calls
    log_prior = float(sum(model.log_prior_list_from_vector(vector=list(params))))
    if not np.isfinite(log_prior):
        return -np.inf
    n_likelihood_calls += 1
    instance = model.instance_from_vector(vector=list(params))
    log_like = float(analysis.log_likelihood_function(instance=instance))
    tracker.record(log_like)
    return log_like + log_prior


ndim = model.prior_count
nwalkers = 2 * ndim
# Emcee has no native convergence criterion -- run a long chain and report
# autocorrelation. This is the bare minimum for a high-dim MGE problem;
# the 2-hour wall-clock guard caps total runtime if MGE is too slow.
nsteps = 2000

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
max_logl = float(max(tracker.history_log_l)) if tracker.history_log_l else float('nan')

# Try autocorrelation-based ESS; emcee raises if chain is too short.
try:
    tau = sampler.get_autocorr_time(quiet=False)
    autocorr_ess = nsteps / float(np.mean(tau)) * nwalkers
    ess_str = f"{autocorr_ess:.1f}     (mean autocorr time = {float(np.mean(tau)):.1f} steps)"
    converged_str = "yes (chain length > 50 * autocorr time)"
except Exception as exc:
    ess_str = f"n/a ({type(exc).__name__}: chain too short for reliable autocorr)"
    converged_str = "no (autocorr time exceeds chain length / 50)"

evals_to_ml, time_to_ml = tracker.finalise(max_log_l=max_logl, tolerance=1.0)

summary = f"""\
--- Emcee Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_logl:.4f}     (best log posterior = {best_log_post:.4f})
Log evidence:    n/a (Emcee is MCMC, not nested sampling)

--- Performance ---
Wall time:           {t_elapsed:.2f} s
Sampling time:       n/a
Likelihood evals:    {n_likelihood_calls}
Time per eval:       {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms
ESS:                 {ess_str}
Posterior samples:   {len(flat_samples)}
Sampler config:      nwalkers={nwalkers}, nsteps={nsteps}

--- Convergence ---
Converged:           {converged_str}
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
