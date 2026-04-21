"""
Minimal Nautilus Example (HST MGE lens likelihood)
--------------------------------------------------

Drives the Nautilus nested sampler directly against the HST MGE imaging
likelihood, bypassing ``af.NonLinearSearch``. Useful as a fast end-to-end
smoke test of the real PyAutoLens likelihood under a production sampler.

``n_live`` is kept small so the search finishes in a few minutes — this is
a wiring test, not a converged posterior.

Requirements:
    pip install nautilus-sampler
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

from nautilus import Sampler


def prior_transform(cube):
    """Map a unit cube to physical parameters via the model's priors."""
    return np.array(model.vector_from_unit_vector(cube))


n_likelihood_calls = 0


def log_likelihood(params):
    global n_likelihood_calls
    n_likelihood_calls += 1
    instance = model.instance_from_vector(vector=list(params))
    return float(analysis.log_likelihood_function(instance=instance))


sampler = Sampler(
    prior=prior_transform,
    likelihood=log_likelihood,
    n_dim=model.prior_count,
    n_live=50,
)

t_start = time.time()
# n_like_max caps total likelihood evaluations — keeps this a smoke test.
# Crank up for a real posterior (e.g. n_eff=200, n_like_max=None).
sampler.run(verbose=True, f_live=1.0, n_eff=30, n_like_max=100)
t_elapsed = time.time() - t_start

points, log_w, log_l = sampler.posterior()
best_idx = np.argmax(log_l)
best_instance = model.instance_from_vector(vector=list(points[best_idx]))

print("\n--- Nautilus Results ---")
print(format_best_fit(best_instance))
print(f"Log evidence:  {sampler.log_z:.2f}")
print(f"\n--- Performance ---")
print(f"Wall time:          {t_elapsed:.2f} s")
print(f"Likelihood calls:   {n_likelihood_calls}")
print(f"Time per call:      {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms")
print(f"Posterior samples:  {len(points)}")
