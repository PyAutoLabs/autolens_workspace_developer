"""
BlackJAXNUTS Example — autofit-wrapped, HST MGE lens model
----------------------------------------------------------

Companion to ``nautilus_jax.py`` and ``nss_jit.py`` that drives
PyAutoFit's ``af.BlackJAXNUTS`` (NUTS via BlackJAX) on the same HST MGE
imaging likelihood. Unlike the rest of the scripts in this folder,
this one is **not strictly "minimal"** — it goes through the full
``af.NonLinearSearch`` machinery rather than calling BlackJAX directly.

What this exercises end-to-end:

 - The autofit ``BlackJAXNUTS`` class (gradient-based MCMC, requires
   ``Analysis(use_jax=True)``).
 - Autofit's ``Fitness`` wrapper as the JAX-traceable log-density target
   (no separate hand-rolled ``log_likelihood_jax`` closure as in
   ``nautilus_jax.py``).
 - JAX pytree registration of the model so ``model.instance_from_vector``
   flows through ``jax.jit``.

Because window adaptation + sampling happen inside ``BlackJAXNUTS._fit``,
we cannot cleanly split the JIT compile time from sampling time the way
the raw-BlackJAX neighbours do — wall time is reported as the whole
``search.fit(...)`` call. NUTS-specific diagnostics (ESS, mean
acceptance, divergences, total leapfrog evals) come from
``result.samples.samples_info``.

``num_warmup`` and ``num_samples`` are kept small — this is a wiring
test on a high-dim (>40) lens model where NUTS warmup over disparate
parameter scales is genuinely slow. Bump them up if you want a
science-quality posterior.

Requirements:
    pip install blackjax
"""
import time
from pathlib import Path

import autofit as af
from autofit.jax.pytrees import enable_pytrees, register_model

from searches_minimal._metrics import MLTracker
from searches_minimal._setup import (
    build_analysis,
    build_dataset,
    build_model,
    format_best_fit,
)

enable_pytrees()

dataset = build_dataset()
model = build_model()
analysis = build_analysis(dataset, use_jax=True)

# Register every concrete cls in the lens + source model so the joint
# ``instance_from_vector`` walks JAX pytrees.
register_model(model)

print(f"Model free parameters: {model.total_free_parameters}")

# Tiny smoke values — the lens model is high-dim and NUTS warmup over
# disparate scales in JAX-grad'd likelihood compiles for minutes per
# adaptation window. 10/10 is a wiring check, not a science run; bump
# to 200/200+ if you have GPU / patience. See the docstring above.
num_warmup = 10
num_samples = 10

search = af.BlackJAXNUTS(
    num_warmup=num_warmup,
    num_samples=num_samples,
    target_accept=0.8,
)

print(
    f"Running BlackJAXNUTS over {model.prior_count} dims "
    f"(num_warmup={num_warmup}, num_samples={num_samples})..."
)

t_start = time.time()
result = search.fit(model=model, analysis=analysis)
t_elapsed = time.time() - t_start

# --------------------------------------------------------------------------
# Results — pull what we can from the autofit samples wrapper.
# --------------------------------------------------------------------------

samples = result.samples
samples_info = samples.samples_info

best_instance = samples.max_log_likelihood()
max_logl = float(samples.max_log_likelihood_sample.log_likelihood)

n_logl_evals = int(samples_info["n_logl_evals"])
ess_min = float(samples_info["ess_min"])
mean_acceptance = float(samples_info["mean_acceptance"])
n_divergent = int(samples_info["n_divergent"])

# Per-sample log-likelihood history isn't stored by SamplesMCMC at the
# Sample level, but Sample objects expose .log_likelihood — pull them out
# in chain order to get the per-sample trace, then feed MLTracker for
# the standard "evals to ML / time to ML" headline.
log_l_history = [s.log_likelihood for s in samples.sample_list]
evals_to_ml, time_to_ml = MLTracker.from_log_l_history(
    log_l_history,
    total_sampling_time=t_elapsed,
    tolerance=1.0,
)

summary = f"""\
--- BlackJAXNUTS (autofit-wrapped) Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_logl:.4f}
Log evidence:    n/a (NUTS does not estimate Z)

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (whole search.fit; warmup + JIT compile + sampling)
Sampling time:       n/a (warmup + sampling not split by autofit wrapper)
JIT compile time:    n/a (folded into _fit; subtract neighbour-sampler timings to estimate)
Likelihood evals:    {n_logl_evals}     (sum of leapfrog integration steps)
Time per eval:       {t_elapsed / max(n_logl_evals, 1) * 1e3:.3f} ms
ESS (min over dims): {ess_min:.1f}
Posterior samples:   {num_samples}
Mean acceptance:     {mean_acceptance:.3f}
Divergences:         {n_divergent}     (post-warmup; non-zero suggests step size too large)
Sampler config:      num_warmup={num_warmup}, num_samples={num_samples}, target_accept=0.8 (smoke test)

--- Convergence ---
Converged:           {'yes' if n_divergent == 0 and ess_min > 50 else 'no (ESS low or divergences present)'}
Evals to ML:         {evals_to_ml if evals_to_ml is not None else 'n/a'}     (sample index — NUTS leapfrog count not exposed per sample)
Time to ML:          {f'{time_to_ml:.2f} s' if time_to_ml is not None else 'n/a'}
"""

print()
print(summary)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
summary_path = output_dir / f"{Path(__file__).stem}_summary.txt"
summary_path.write_text(summary)
print(f"Summary written to: {summary_path}")
