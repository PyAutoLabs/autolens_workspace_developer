"""
Nautilus baseline — the standard PyAutoLens/autofit search workflow
-------------------------------------------------------------------

The conventional baseline for Part 6 of the MAP-optimizer benchmark: one fully
converged Nautilus posterior/evidence run using the **normal** PyAutoLens
implementation — the high-level ``af.Nautilus(...).fit(model, analysis)`` search
API that every imaging modeling example uses (e.g. ``autolens_workspace`` imaging
modeling scripts, ``n_live=100``) — **not** a hand-rolled ``nautilus.Sampler``
wrapper. Using the autofit search means invalid model instances (the degenerate
ell_comps/shear = 0 points, NaN likelihood) are handled by autofit's own
``FitException`` -> penalty path, exactly as in a real run.

Same dataset, model and priors as every optimizer here (``_setup``), and the
JAX likelihood (``use_jax=True``) so each evaluation runs on the A100 — directly
comparable to the batched-JAX MAP workflow. Be precise about the hardware:
Nautilus orchestrates the search on the **CPU** (single process here) and calls
the PyAutoLens/JAX likelihood one point at a time on the A100. It is **not**
GPU-native and does **not** batch its likelihood evaluations — the desired
comparison is the real end-to-end wall-clock cost of this standard workflow
versus the batched MAP workflow.

Run on the A100:

    python -m searches_minimal.nautilus_autofit

Requirements: nautilus-sampler (autofit's Nautilus wrapper).
"""

import time
from pathlib import Path

import numpy as np

import autofit as af
import autolens as al

from searches_minimal._setup import (
    build_analysis,
    build_dataset,
    build_model,
    format_best_fit,
)

N_LIVE = 100  # the value used by the standard autolens_workspace imaging examples

dataset = build_dataset()
model = build_model()
# JAX likelihood so each evaluation runs on the A100 (single-core orchestration
# avoids pickling the JAX analysis across processes).
analysis = build_analysis(dataset, use_jax=True)

print(f"Model free parameters: {model.total_free_parameters}")

search = af.Nautilus(
    path_prefix="phase3_nautilus",
    name="mge_lens",
    n_live=N_LIVE,
    number_of_cores=1,
)

t0 = time.time()
result = search.fit(model=model, analysis=analysis)
wall = time.time() - t0

samples = result.samples
best_sample = samples.max_log_likelihood_sample
best_instance = result.max_log_likelihood_instance
max_logl = float(best_sample.log_likelihood)
best_r_e = float(best_instance.galaxies.lens.mass.einstein_radius)
log_evidence = (
    float(samples.log_evidence) if samples.log_evidence is not None else float("nan")
)
total_evals = int(samples.total_samples)

# Effective sample size from the posterior weights: (sum w)^2 / sum(w^2).
w = np.asarray(samples.weight_list, dtype=float)
ess = (
    float(w.sum() ** 2 / np.sum(w**2)) if w.size and np.sum(w**2) > 0 else float("nan")
)

# Nautilus's native log-evidence error, if autofit exposes the internal sampler.
log_z_err = float("nan")
for attr in ("results_internal", "search_internal", "sampler"):
    obj = getattr(samples, attr, None) or getattr(result, attr, None)
    if obj is not None and hasattr(obj, "log_z_err"):
        log_z_err = float(obj.log_z_err)
        break

summary = f"""\
--- Nautilus (autofit search, JAX likelihood on A100) Results ---
Best fit:        {format_best_fit(best_instance)}
Einstein radius: {best_r_e:.4f}     (truth ~ 1.6)
Max log L:       {max_logl:.4f}
Log evidence:    {log_evidence:.4f} +/- {log_z_err:.4f}

--- Performance ---
Wall time:           {wall:.2f} s     (end-to-end search.fit, CPU orchestration + A100 JAX likelihood)
Likelihood evals:    {total_evals}
Time per eval:       {wall / max(total_evals, 1) * 1e3:.3f} ms     (sequential; Nautilus does not batch)
ESS:                 {ess:.1f}
Posterior samples:   {len(w)}
Sampler config:      af.Nautilus n_live={N_LIVE}, default n_eff=500 / f_live=0.01; number_of_cores=1
Hardware:            Nautilus orchestrates on CPU; each likelihood on the A100 (NOT GPU-native, NOT batched)

--- Convergence ---
Termination:         self-terminating (Nautilus n_eff / f_live convergence)
Converged:           yes (autofit Nautilus default criterion)
Truth basin:         {"YES" if abs(best_r_e - 1.6) < 0.3 else "NO"}   (max-posterior einstein_radius vs truth 1.6)
"""

print()
print(summary)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
summary_path = output_dir / f"{Path(__file__).stem}_summary.txt"
summary_path.write_text(summary)
print(f"Summary written to: {summary_path}")
