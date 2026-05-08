"""
Minimal L-BFGS-B Example (HST MGE lens likelihood)
--------------------------------------------------

Drives scipy's L-BFGS-B optimiser directly against the HST MGE imaging
likelihood. The model is parameterised in **unit space** (each dimension in
[0, 1]); the optimiser gets simple box bounds and the likelihood function
internally maps unit vectors to physical values via the model's priors.

This side-steps the need to derive finite bounds from unbounded priors like
``GaussianPrior`` and matches the ``prior_transform`` pattern used by the
nested-sampling scripts in this folder.

Requirements:
    scipy (included with autofit)
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

from scipy import optimize


n_likelihood_calls = 0
tracker = MLTracker()


def chi_squared_from_unit(unit_vector):
    """Return -2 log L for a unit-cube vector (scipy minimises this)."""
    global n_likelihood_calls
    n_likelihood_calls += 1
    physical = model.vector_from_unit_vector(list(unit_vector))
    instance = model.instance_from_vector(vector=physical)
    log_like = float(analysis.log_likelihood_function(instance=instance))
    tracker.record(log_like)
    return -2.0 * log_like


ndim = model.prior_count
bounds = [(0.0, 1.0)] * ndim

# Start at the centre of the unit cube (prior median).
x0 = np.full(ndim, 0.5)

t_start = time.time()
result = optimize.minimize(
    fun=chi_squared_from_unit,
    x0=x0,
    method="L-BFGS-B",
    bounds=bounds,
    options={"disp": True},
)
t_elapsed = time.time() - t_start

best_physical = model.vector_from_unit_vector(list(result.x))
best_instance = model.instance_from_vector(vector=best_physical)
max_logl = -0.5 * float(result.fun)

evals_to_ml, time_to_ml = tracker.finalise(max_log_l=max_logl, tolerance=1.0)

summary = f"""\
--- L-BFGS-B Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_logl:.4f}     (chi-squared = {float(result.fun):.4f}, converged = {bool(result.success)})
Log evidence:    n/a (L-BFGS-B is a point optimiser, not nested sampling)

--- Performance ---
Wall time:           {t_elapsed:.2f} s
Sampling time:       n/a
Likelihood evals:    {int(result.nfev)}     (gradient evals: {int(result.njev)}, iterations: {int(result.nit)})
Time per eval:       {t_elapsed / max(int(result.nfev), 1) * 1e3:.3f} ms
ESS:                 n/a (point optimiser)
Posterior samples:   n/a (point optimiser)
Sampler config:      bounds=[(0, 1)] * ndim, scipy default tolerance

--- Convergence ---
Converged:           {bool(result.success)} ({result.message if hasattr(result, 'message') else 'see scipy output'})
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
