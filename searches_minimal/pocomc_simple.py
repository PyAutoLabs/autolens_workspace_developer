"""
Minimal PocoMC Example — pure-JAX HST MGE likelihood (numpy boundary)
---------------------------------------------------------------------

Drives PocoMC (Preconditioned Monte Carlo, Karamanis et al. 2022/2024)
against the HST MGE imaging likelihood. PocoMC is **not** JAX-native —
it runs on PyTorch + NumPy under the hood — so the likelihood crosses
a ``np.asarray(jit_log_likelihood(jnp.asarray(...)))`` boundary on every
call. Boundary cost is identical to ``nautilus_jax.py``.

The hypothesis being tested: PocoMC's normalizing-flow-preconditioned
SMC handles multi-modal correlated posteriors fundamentally better than
vanilla nested sampling. The 2024 paper reports 25-50x speedups over NS
on cosmology / GW problems with similar dimensionality.

Sampling is in unit-cube space ``[0, 1]^N`` with a uniform PocoMC prior;
the cube → physical step happens Python-side inside the likelihood
adapter (no ``pure_callback`` plumbing needed because PocoMC is not
JIT'd anyway). Compare versus ``nautilus_jax.py`` (NS + same boundary
cost), ``blackjax_smc.py`` (JAX-native SMC + RWM, gradient-free), and
``numpyro_ess.py`` (JAX-native ensemble slice sampling, gradient-free).

PocoMC returns log evidence directly via ``sampler.evidence()`` — a
bridge-sampling estimator, not a path-temperature integral.

Requirements:
    pip install pocomc
"""
import os
# JAX preallocates 75% of GPU memory by default; PocoMC trains a Zuko
# normalizing flow on PyTorch which competes for the same VRAM. Cap JAX
# at 50% so torch has headroom on small cards. Setting before any JAX
# import is required for this env var to take effect.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import time
from pathlib import Path

import numpy as np
from scipy.stats import uniform as scipy_uniform
import jax
import jax.numpy as jnp
import pocomc

from searches_minimal._metrics import MLTracker
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

ndim = model.prior_count


# --------------------------------------------------------------------------
# JAX-jitted likelihood in physical space, identical to ``nautilus_jax.py``.
# --------------------------------------------------------------------------

def log_likelihood_jax(physical):
    instance = model.instance_from_vector(vector=physical, xp=jnp)
    return analysis.log_likelihood_function(instance=instance)


jit_log_likelihood = jax.jit(log_likelihood_jax)

# Warm up the JIT once so the compile cost is measured separately.
warmup_unit = [0.5] * ndim
warmup_physical = jnp.asarray(model.vector_from_unit_vector(warmup_unit))
print("JIT-compiling MGE likelihood (one-shot)...", flush=True)
t_jit_start = time.time()
_ = float(jax.block_until_ready(jit_log_likelihood(warmup_physical)))
t_jit = time.time() - t_jit_start
print(f"  Compiled in {t_jit:.2f} s", flush=True)


# --------------------------------------------------------------------------
# PocoMC adapter — likelihood callback that PocoMC will invoke per particle.
#
# PocoMC samples in the prior's parameter space, which we set to the unit
# cube via ``scipy.stats.uniform``. The adapter maps cube → physical via
# autofit's prior CDFs (Python-side) and then evaluates the JAX-jitted
# likelihood through the np <-> jax boundary, exactly like the
# ``nautilus_jax.py`` callback.
# --------------------------------------------------------------------------

tracker = MLTracker()


def likelihood_for_pocomc(cube_np):
    physical_np = np.asarray(
        model.vector_from_unit_vector(list(np.asarray(cube_np))),
        dtype=np.float64,
    )
    log_l = float(jit_log_likelihood(jnp.asarray(physical_np)))
    tracker.record(log_l)
    return log_l


prior = pocomc.Prior([scipy_uniform(loc=0.0, scale=1.0)] * ndim)


# --------------------------------------------------------------------------
# Sampler config — PocoMC's defaults are n_effective=512, n_active=256,
# n_total=4096, n_evidence=4096. On a 6 GB card these are tight but feasible
# (the flow training is the dominant VRAM consumer); halve them if the user
# hits OOM and report it back to ``sweep_findings.md``-style notes.
# --------------------------------------------------------------------------

n_effective = 512
n_active = 256
n_total = 4096
n_evidence = 4096

sampler = pocomc.Sampler(
    prior=prior,
    likelihood=likelihood_for_pocomc,
    n_dim=ndim,
    n_effective=n_effective,
    n_active=n_active,
    random_state=42,
)

print(
    f"\nRunning PocoMC over {ndim} dims "
    f"(n_effective={n_effective}, n_active={n_active}, "
    f"n_total={n_total}, n_evidence={n_evidence})..."
)

t_start = time.time()
sampler.run(n_total=n_total, n_evidence=n_evidence, progress=True)
t_elapsed = time.time() - t_start


# --------------------------------------------------------------------------
# Results — posterior samples + bridge-sampled log evidence.
# --------------------------------------------------------------------------

samples, weights, logl, logp = sampler.posterior()
logz, logz_err = sampler.evidence()

best_idx = int(np.argmax(logl))
best_cube = np.asarray(samples[best_idx])
best_physical = np.asarray(
    model.vector_from_unit_vector(list(best_cube)), dtype=np.float64
)
best_instance = model.instance_from_vector(vector=list(best_physical))
max_logl = float(np.max(logl))

n_likelihood_calls = len(tracker.history_log_l)

evals_to_ml, time_to_ml = tracker.finalise(max_log_l=max_logl, tolerance=1.0)

summary = f"""\
--- PocoMC Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_logl:.4f}
Log evidence:    {logz:.4f} +/- {logz_err:.4f}     (PocoMC bridge-sampled)

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (excludes JIT compile, run ahead of time)
Sampling time:       {t_elapsed:.2f} s     (preconditioning + sampling + evidence folded)
JIT compile time:    {t_jit:.2f} s     (one-shot likelihood warm-up before sampling)
Likelihood evals:    {n_likelihood_calls}
Time per eval:       {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms
ESS:                 n/a (PocoMC reports per-iteration ESS internally)
Posterior samples:   {len(samples)}     (importance-weighted, post-trim)
Sampler config:      n_effective={n_effective}, n_active={n_active}, n_total={n_total}, n_evidence={n_evidence}

--- Convergence ---
Converged:           yes (PocoMC ran to n_total)
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
