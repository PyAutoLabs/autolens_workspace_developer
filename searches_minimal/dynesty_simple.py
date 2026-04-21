"""
Minimal Dynesty Example (HST MGE lens likelihood)
-------------------------------------------------

Drives the Dynesty nested sampler directly against the HST MGE imaging
likelihood, bypassing ``af.NonLinearSearch``. Useful as a fast end-to-end
smoke test of the real PyAutoLens likelihood under a production sampler.

``nlive`` is kept small so the search finishes in a few minutes — this is
a wiring test, not a converged posterior.

Requirements:
    pip install dynesty
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

from dynesty import NestedSampler


def prior_transform(cube):
    return np.array(model.vector_from_unit_vector(cube))


n_likelihood_calls = 0


def log_likelihood(params):
    global n_likelihood_calls
    n_likelihood_calls += 1
    instance = model.instance_from_vector(vector=list(params))
    return float(analysis.log_likelihood_function(instance=instance))


sampler = NestedSampler(
    loglikelihood=log_likelihood,
    prior_transform=prior_transform,
    ndim=model.prior_count,
    nlive=30,
)

t_start = time.time()
sampler.run_nested(print_progress=True, maxiter=30)
t_elapsed = time.time() - t_start

results = sampler.results
best_idx = np.argmax(results.logl)
best_instance = model.instance_from_vector(vector=list(results.samples[best_idx]))

print("\n--- Dynesty Results ---")
print(format_best_fit(best_instance))
print(f"Log evidence:  {results.logz[-1]:.2f}")
print(f"\n--- Performance ---")
print(f"Wall time:          {t_elapsed:.2f} s")
print(f"Likelihood calls:   {n_likelihood_calls}")
print(f"Time per call:      {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms")
print(f"Iterations:         {results.niter}")
print(f"Samples:            {len(results.samples)}")
