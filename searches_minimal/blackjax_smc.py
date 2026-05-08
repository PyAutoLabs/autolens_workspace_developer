"""
Minimal BlackJAX SMC Example — pure-JAX HST MGE likelihood
----------------------------------------------------------

Drives BlackJAX's adaptive-tempered SMC sampler with a random-walk
Metropolis (RWM) inner kernel — gradient-free — against the same HST
MGE imaging likelihood used by ``nautilus_jax.py`` and ``nss_jit.py``.

Sampling proceeds in unit-cube space ``[0, 1]^N``: the prior is uniform
on the cube (autofit's prior CDFs handle the cube → physical mapping),
RWM step sizes are dimensionless, and bounds enforcement is a single
``jnp.all((cube >= 0) & (cube <= 1))`` check. The cube → physical step
is wrapped in ``jax.pure_callback`` (the autofit ``vector_from_unit_vector``
inverse-CDF is Python-side), so each likelihood evaluation crosses one
host roundtrip — the same boundary cost as ``nautilus_jax.py``.

Adaptive tempering anneals ``lambda`` from 0 (prior) to 1 (posterior),
choosing each next ``lambda`` to maintain ``target_ess``. The log
evidence is recovered as the sum of ``info.log_likelihood_increment``
across temperature steps, at no extra likelihood cost.

This is a wiring smoke test, not a tuned production run — particle count,
RWM sigma, and ``num_mcmc_steps`` are conservative defaults. Compare
versus ``nautilus_jax.py`` (NumPy-sampler boundary, JIT'd likelihood)
and ``nss_jit.py`` (pure-JAX nested sampler).

Requirements:
    pip install blackjax
"""
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import blackjax
import blackjax.smc.resampling as resampling
import blackjax.mcmc.random_walk as rw

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
# Cube → physical bridge.
#
# autofit's ``vector_from_unit_vector`` inverts each prior's CDF using
# Python-side prior objects, so it cannot be traced by JAX directly. Wrap
# it as a host callback. ``vmap_method='sequential'`` is the safe default
# for unbatched Python code; SMC vmaps over particles inside its inner
# kernel and this is the regime ``pure_callback`` runs reliably under.
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
# Log prior + log likelihood, both in unit-cube space.
# --------------------------------------------------------------------------

def log_prior(cube):
    in_cube = jnp.all((cube >= 0.0) & (cube <= 1.0))
    return jnp.where(in_cube, 0.0, -jnp.inf)


def log_likelihood(cube):
    in_cube = jnp.all((cube >= 0.0) & (cube <= 1.0))
    # Clip before the host callback so out-of-cube proposals don't drive
    # the autofit prior CDFs into ranges that may produce NaN. ``log_prior``
    # already returns -inf out-of-cube, so the MH acceptance rejects them
    # regardless of what value we return for log_likelihood here.
    safe_cube = jnp.clip(cube, 0.0, 1.0)
    physical = cube_to_physical(safe_cube)
    instance = model.instance_from_vector(vector=physical, xp=jnp)
    log_l = analysis.log_likelihood_function(instance=instance)
    return jnp.where(in_cube, log_l, 0.0)


# --------------------------------------------------------------------------
# JIT compile the likelihood once on a centred cube point so the compile
# time can be reported separately from the sampling time.
# --------------------------------------------------------------------------

print("JIT-compiling MGE likelihood (one-shot)...", flush=True)
t_jit_start = time.time()
_ = float(jax.block_until_ready(jax.jit(log_likelihood)(jnp.full(ndim, 0.5))))
t_jit = time.time() - t_jit_start
print(f"  Compiled in {t_jit:.2f} s", flush=True)


# --------------------------------------------------------------------------
# Adaptive-tempered SMC with RWM as the gradient-free inner kernel.
#
# BlackJAX's SMC machinery vmaps the inner kernel across particles and
# unpacks ``mcmc_parameters`` whose leading dim is 1 as shared parameters.
# The RWM kernel itself takes ``random_step`` (a callable) which we can't
# pass through ``mcmc_parameters`` — so the wrapper closes over ``sigma``
# (an array) and constructs the proposal each call.
# --------------------------------------------------------------------------

n_particles = 256
num_mcmc_steps = 5
target_ess = 0.5
rmh_sigma_per_dim = 0.05  # 5% of cube width


def smc_step_fn(rng_key, state, logdensity_fn, sigma):
    proposal = rw.normal(sigma)
    kernel = rw.build_additive_step()
    return kernel(rng_key, state, logdensity_fn, proposal)


sigma_init = jnp.eye(ndim)[None, :, :] * rmh_sigma_per_dim  # (1, N, N) → shared

smc = blackjax.adaptive_tempered_smc(
    logprior_fn=log_prior,
    loglikelihood_fn=log_likelihood,
    mcmc_step_fn=smc_step_fn,
    mcmc_init_fn=rw.init,
    mcmc_parameters={"sigma": sigma_init},
    resampling_fn=resampling.systematic,
    target_ess=target_ess,
    num_mcmc_steps=num_mcmc_steps,
)

rng_key = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng_key)
initial_particles = jax.random.uniform(
    init_key, shape=(n_particles, ndim), minval=0.0, maxval=1.0
)
state = smc.init(initial_particles)


print(
    f"\nRunning BlackJAX SMC (adaptive tempered + RWM) over {ndim} dims "
    f"(n_particles={n_particles}, num_mcmc_steps={num_mcmc_steps}, "
    f"target_ess={target_ess}, rmh_sigma={rmh_sigma_per_dim})..."
)
print("  JIT compile of the SMC step kernel happens on the first iteration.\n")

# Outer loop in Python — each ``smc.step`` jit-compiles the inner kernel
# over the particles, so per-iteration overhead is the kernel scan, not
# Python overhead. ``pure_callback`` inside the trace prevents using
# ``lax.while_loop`` here without extra plumbing, and a Python loop is
# clear enough for a smoke test.
log_l_history: list[float] = []
log_z = 0.0
n_smc_steps = 0
vmapped_log_l = jax.jit(jax.vmap(log_likelihood))

t_start = time.time()
while float(state.tempering_param) < 1.0:
    rng_key, sub_key = jax.random.split(rng_key)
    state, info = jax.block_until_ready(smc.step(sub_key, state))
    log_z += float(info.log_likelihood_increment)
    n_smc_steps += 1

    log_l_step = vmapped_log_l(state.particles)
    cur_max_log_l = float(jnp.max(log_l_step))
    log_l_history.append(cur_max_log_l)

    print(
        f"  step {n_smc_steps:3d}: lambda={float(state.tempering_param):.4f}  "
        f"max log L={cur_max_log_l:.2f}  log Z (running)={log_z:.4f}  "
        f"acc rate={float(jnp.mean(info.update_info.acceptance_rate)):.3f}"
    )

t_elapsed = time.time() - t_start


# --------------------------------------------------------------------------
# Results — pull best fit from the final particle batch.
# --------------------------------------------------------------------------

final_log_l = vmapped_log_l(state.particles)
best_idx = int(jnp.argmax(final_log_l))
best_cube = np.asarray(state.particles[best_idx])
best_physical = _cube_to_physical_host(best_cube)
best_instance = model.instance_from_vector(vector=list(best_physical))
max_logl = float(jnp.max(final_log_l))

# Likelihood eval count: each SMC step runs ``num_mcmc_steps`` RWM updates
# per particle (one loglik eval per proposal), plus the n_particles initial
# tempered-density evaluations and the per-step ``vmapped_log_l`` we run for
# tracking. This is an upper bound — RWM always evaluates the proposal even
# when log_prior=-inf short-circuits acceptance, because the JIT trace can't
# branch on the prior value.
n_likelihood_calls = (
    n_particles  # initial weights
    + n_smc_steps * n_particles * (num_mcmc_steps + 1)  # MCMC evals + max-tracking
)

evals_to_ml, time_to_ml = MLTracker.from_log_l_history(
    log_l_history,
    total_sampling_time=t_elapsed,
    tolerance=1.0,
)

summary = f"""\
--- BlackJAX SMC (adaptive tempered + RWM) Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_logl:.4f}
Log evidence:    {log_z:.4f}     (sum of SMC log_likelihood_increment over temperatures)

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (excludes JIT compile, run ahead of time)
Sampling time:       {t_elapsed:.2f} s     (no separate warmup phase)
JIT compile time:    {t_jit:.2f} s     (one-shot warm-up before sampling)
Likelihood evals:    {n_likelihood_calls}     (upper bound; SMC traces all proposals)
Time per eval:       {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms
ESS:                 n/a (SMC adaptive tempered targets ESS at each step)
Posterior samples:   {n_particles}     (final tempered particles)
Sampler config:      n_particles={n_particles}, num_mcmc_steps={num_mcmc_steps}, target_ess={target_ess}, rmh_sigma={rmh_sigma_per_dim}, n_smc_steps={n_smc_steps}

--- Convergence ---
Converged:           yes (tempering reached lambda=1.0)
Evals to ML:         {evals_to_ml if evals_to_ml is not None else 'n/a'}     (SMC step index — not raw eval index — within 1 nat of max)
Time to ML:          {f'{time_to_ml:.2f} s' if time_to_ml is not None else 'n/a'}
"""

print()
print(summary)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
summary_path = output_dir / f"{Path(__file__).stem}_summary.txt"
summary_path.write_text(summary)
print(f"Summary written to: {summary_path}")
