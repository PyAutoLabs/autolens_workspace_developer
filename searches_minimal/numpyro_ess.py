"""
Minimal NumPyro ESS Example — pure-JAX HST MGE likelihood
---------------------------------------------------------

Drives NumPyro's Ensemble Slice Sampling (ESS) kernel — gradient-free,
JAX-native — against the same HST MGE imaging likelihood used by
``nautilus_jax.py`` / ``nss_jit.py`` / ``blackjax_smc.py``.

ESS (Karamanis & Beutler 2021) is built for "correlated and multi-modal"
posteriors. Multiple walker chains run in parallel and share information
to choose better slice directions; no gradients are required, and on
``num_chains >= 2 * ndim`` it scales gracefully with dimension.

Sampling proceeds in unit-cube space ``[0, 1]^N`` with the cube → physical
mapping wrapped in ``jax.pure_callback`` — same boundary cost as
``nautilus_jax.py`` / ``blackjax_smc.py``. ESS is a straight MCMC sampler,
so it produces posterior samples but **no log evidence**; pair with a
nested or SMC sampler if Z is needed.

This is a wiring smoke test, not a tuned production run. ``num_chains``,
``num_warmup``, and ``num_samples`` are conservative defaults.

Requirements:
    pip install numpyro
"""
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, ESS

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
# Cube → physical bridge — autofit's prior CDFs are Python objects, so the
# mapping is wrapped in ``jax.pure_callback`` (same approach as
# ``blackjax_smc.py``). One host roundtrip per likelihood call, matching
# ``nautilus_jax.py`` boundary cost.
# --------------------------------------------------------------------------

def _cube_to_physical_host(cube_np):
    return np.asarray(
        model.vector_from_unit_vector(list(np.asarray(cube_np))),
        dtype=np.float64,
    )


_phys_shape = jax.ShapeDtypeStruct((ndim,), jnp.float64)


def cube_to_physical(cube):
    return jax.pure_callback(
        _cube_to_physical_host, _phys_shape, cube, vmap_method="sequential"
    )


# --------------------------------------------------------------------------
# Log likelihood and ESS potential function.
#
# NumPyro minimises ``potential_fn`` (= negative log joint density). With a
# uniform prior on ``[0, 1]^N`` the joint density is just the likelihood,
# gated to ``-inf`` outside the cube — equivalently ``+inf`` potential.
# --------------------------------------------------------------------------

def log_likelihood(cube):
    in_cube = jnp.all((cube >= 0.0) & (cube <= 1.0))
    safe_cube = jnp.clip(cube, 0.0, 1.0)
    physical = cube_to_physical(safe_cube)
    instance = model.instance_from_vector(vector=physical, xp=jnp)
    log_l = analysis.log_likelihood_function(instance=instance)
    return jnp.where(in_cube, log_l, -jnp.inf)


def potential_fn(cube):
    return -log_likelihood(cube)


# --------------------------------------------------------------------------
# JIT compile the likelihood once on a centred cube point.
# --------------------------------------------------------------------------

print("JIT-compiling MGE likelihood (one-shot)...", flush=True)
t_jit_start = time.time()
_ = float(jax.block_until_ready(jax.jit(log_likelihood)(jnp.full(ndim, 0.5))))
t_jit = time.time() - t_jit_start
print(f"  Compiled in {t_jit:.2f} s", flush=True)


# --------------------------------------------------------------------------
# ESS sampler. The Karamanis & Beutler paper recommends
# ``num_chains >= 2 * ndim``; NumPyro additionally requires ``num_chains``
# divisible by 2 and ``chain_method='vectorized'`` so all walkers are
# vmapped through a single JAX trace.
# --------------------------------------------------------------------------

n_chains = max(32, 2 * ndim + (2 * ndim) % 2)
num_warmup = 200
num_samples = 200

rng_key = jax.random.PRNGKey(42)
rng_key, init_key, run_key = jax.random.split(rng_key, 3)

initial_walkers = jax.random.uniform(
    init_key, shape=(n_chains, ndim), minval=0.0, maxval=1.0
)

kernel = ESS(potential_fn=potential_fn)
mcmc = MCMC(
    kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=n_chains,
    chain_method="vectorized",
    progress_bar=True,
)

print(
    f"\nRunning NumPyro ESS over {ndim} dims "
    f"(n_chains={n_chains}, num_warmup={num_warmup}, num_samples={num_samples})..."
)
print("  JIT compile of the ensemble kernel happens on the first warmup step.\n")

t_start = time.time()
mcmc.run(run_key, init_params=initial_walkers)
jax.block_until_ready(mcmc.get_samples(group_by_chain=False))
t_elapsed = time.time() - t_start


# --------------------------------------------------------------------------
# Results — pull best fit from the post-warmup samples.
#
# ``get_samples(group_by_chain=False)`` returns shape (n_chains*num_samples, ndim).
# Recompute log L per sample via vmap to find the global argmax and to feed
# MLTracker for the headline ``evals_to_ml`` / ``time_to_ml`` numbers.
# --------------------------------------------------------------------------

samples = mcmc.get_samples(group_by_chain=False)  # (n_total, ndim)
samples_by_chain = mcmc.get_samples(group_by_chain=True)  # (n_chains, n_samples, ndim)
n_total = samples.shape[0]

log_l_per_sample = jax.jit(jax.vmap(log_likelihood))(samples)
best_idx = int(jnp.argmax(log_l_per_sample))
best_cube = np.asarray(samples[best_idx])
best_physical = _cube_to_physical_host(best_cube)
best_instance = model.instance_from_vector(vector=list(best_physical))
max_logl = float(jnp.max(log_l_per_sample))

# n_eff and r_hat via NumPyro's diagnostic on the full sample tensor.
import numpyro.diagnostics as diagnostics
n_eff_per_param = diagnostics.effective_sample_size(np.asarray(samples_by_chain))
r_hat_per_param = diagnostics.gelman_rubin(np.asarray(samples_by_chain))
ess_min = float(np.min(n_eff_per_param))
r_hat_max = float(np.max(r_hat_per_param))

# Likelihood eval count: ESS evaluates the potential along each slice. The
# default ``max_steps`` is 10000, so the true count is unknown without
# instrumentation — report (warmup + samples) * n_chains as the *number of
# accepted moves*, with a note. For HPC profiling the per-slice eval count
# matters; ``max_iter`` in the kernel can be tuned to bound it.
n_accepted_moves = (num_warmup + num_samples) * n_chains

evals_to_ml, time_to_ml = MLTracker.from_log_l_history(
    [float(x) for x in log_l_per_sample],
    total_sampling_time=t_elapsed,
    tolerance=1.0,
)

summary = f"""\
--- NumPyro ESS Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_logl:.4f}
Log evidence:    n/a (ESS does not estimate Z)

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (excludes JIT compile, run ahead of time)
Sampling time:       {t_elapsed:.2f} s     (warmup + samples folded; see num_warmup/num_samples)
JIT compile time:    {t_jit:.2f} s     (one-shot likelihood warm-up before sampling)
Accepted moves:      {n_accepted_moves}     ((num_warmup + num_samples) * n_chains; per-slice evals are higher)
Time per move:       {t_elapsed / max(n_accepted_moves, 1) * 1e3:.3f} ms
ESS (n_eff min):     {ess_min:.1f}     (worst across {ndim} parameters)
R_hat (max):         {r_hat_max:.3f}     ({'OK' if r_hat_max < 1.1 else 'NOT CONVERGED'} — < 1.1 indicates well-mixed chains)
Posterior samples:   {n_total}     (n_chains * num_samples, post-warmup)
Sampler config:      n_chains={n_chains}, num_warmup={num_warmup}, num_samples={num_samples}, chain_method=vectorized

--- Convergence ---
Converged:           {'yes' if r_hat_max < 1.1 and ess_min > 50 else 'no (R_hat or ESS poor)'}
Evals to ML:         {evals_to_ml if evals_to_ml is not None else 'n/a'}     (sample index — within 1 nat of max log L)
Time to ML:          {f'{time_to_ml:.2f} s' if time_to_ml is not None else 'n/a'}
"""

print()
print(summary)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
summary_path = output_dir / f"{Path(__file__).stem}_summary.txt"
summary_path.write_text(summary)
print(f"Summary written to: {summary_path}")
