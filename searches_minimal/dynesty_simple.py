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

from dynesty import NestedSampler


def prior_transform(cube):
    return np.array(model.vector_from_unit_vector(cube))


n_likelihood_calls = 0
tracker = MLTracker()


def log_likelihood(params):
    global n_likelihood_calls
    n_likelihood_calls += 1
    instance = model.instance_from_vector(vector=list(params))
    log_l = float(analysis.log_likelihood_function(instance=instance))
    tracker.record(log_l)
    return log_l


nlive = 200

sampler = NestedSampler(
    loglikelihood=log_likelihood,
    prior_transform=prior_transform,
    ndim=model.prior_count,
    nlive=nlive,
)

t_start = time.time()
sampler.run_nested(print_progress=True)
t_elapsed = time.time() - t_start

results = sampler.results
best_idx = np.argmax(results.logl)
best_instance = model.instance_from_vector(vector=list(results.samples[best_idx]))
max_logl = float(results.logl[best_idx])
log_z = float(results.logz[-1])

# ESS for weighted nested-sampling samples: 1 / sum(w^2).
weights = np.exp(results.logwt - log_z)
weights_sum_sq = float(np.sum(weights**2))
ess = 1.0 / weights_sum_sq if weights_sum_sq > 0 else 0.0

evals_to_ml, time_to_ml = tracker.finalise(max_log_l=max_logl, tolerance=1.0)

summary = f"""\
--- Dynesty Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_logl:.4f}
Log evidence:    {log_z:.4f}

--- Performance ---
Wall time:           {t_elapsed:.2f} s
Sampling time:       n/a
Likelihood evals:    {n_likelihood_calls}
Time per eval:       {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms
ESS:                 {ess:.1f}
Posterior samples:   {len(results.samples)}
Sampler config:      nlive={nlive}, default dlogz termination

--- Convergence ---
Converged:           yes (Dynesty default dlogz)
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
