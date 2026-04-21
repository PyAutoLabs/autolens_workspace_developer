"""
Minimal NSS Example — NumPy likelihood via ``jax.pure_callback`` (HST MGE)
-------------------------------------------------------------------------

Drives the NSS (Nested Slice Sampling) JAX-based sampler against the HST
MGE likelihood by wrapping PyAutoLens's NumPy ``AnalysisImaging`` with
``jax.pure_callback``. This is the slowest NSS path — each draw enters
Python once per callback — but it works with any existing NumPy likelihood
without requiring JAX-traceable code.

``n_live`` is kept small because every likelihood call round-trips through
Python. See ``nss_jit.py`` for the fast, pure-JAX version.

Requirements:
    pip install git+https://github.com/yallup/nss.git
    (pulls handley-lab/blackjax fork with nested sampling support)
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import blackjax
from nss.ns import Results, finalise, log_weights, safe_ess

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

n_likelihood_calls = 0


def numpy_log_likelihood(params_np):
    global n_likelihood_calls
    n_likelihood_calls += 1
    instance = model.instance_from_vector(vector=params_np.tolist())
    return np.float64(analysis.log_likelihood_function(instance=instance))


def numpy_log_prior(params_np):
    log_priors = model.log_prior_list_from_vector(vector=params_np.tolist())
    return np.float64(sum(log_priors))


def log_likelihood(params):
    return jax.pure_callback(
        lambda p: jnp.float64(numpy_log_likelihood(np.asarray(p))),
        jax.ShapeDtypeStruct((), jnp.float64),
        params,
        vmap_method="sequential",
    )


def log_prior(params):
    return jax.pure_callback(
        lambda p: jnp.float64(numpy_log_prior(np.asarray(p))),
        jax.ShapeDtypeStruct((), jnp.float64),
        params,
        vmap_method="sequential",
    )


ndim = model.prior_count
n_live = 8
rng_key = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng_key)

# Draw initial samples by mapping unit-cube draws through the model priors
# so every starting point lies in the prior's support (some priors are
# not simple uniforms).
unit_cube = np.asarray(
    jax.random.uniform(init_key, shape=(n_live, ndim), minval=0.0, maxval=1.0)
)
physical = np.array(
    [model.vector_from_unit_vector(list(u)) for u in unit_cube]
)
initial_samples = jnp.asarray(physical)

print(f"Running NSS (autofit NumPy likelihood via pure_callback)...")
print(f"  n_live={n_live}, n_dim={ndim}")
print(f"  Using jax.pure_callback for NumPy likelihood", flush=True)

# Mirror ``nss.ns.run_nested_sampling`` but with a ``max_steps`` guard so
# this terminates in bounded time. Every NSS step has to round-trip through
# Python once per slice-sample shrink, so even a single iteration takes a
# while; a real run would use the upstream function without ``max_steps``.
num_mcmc_steps = 1
num_delete = 4
max_steps = 1

t_start = time.time()
algo = blackjax.nss(
    logprior_fn=log_prior,
    loglikelihood_fn=log_likelihood,
    num_delete=num_delete,
    num_inner_steps=num_mcmc_steps,
)
state = algo.init(initial_samples)


@jax.jit
def one_step(carry, _):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), dead_point


rng_key, sample_key = jax.random.split(rng_key)
# Warmup JIT and block until compiled.
(_, rng_key), _ = jax.block_until_ready(one_step((state, sample_key), None))

dead = []
for _ in range(max_steps):
    (state, rng_key), dead_info = one_step((state, rng_key), None)
    dead.append(dead_info)

final_state = finalise(state, dead)
logw = log_weights(rng_key, final_state)
minimum = jnp.nan_to_num(logw).min()
logzs = jax.scipy.special.logsumexp(jnp.nan_to_num(logw, nan=minimum), axis=0)
results = Results(
    name="NSS",
    time=time.time() - t_start,
    evals=int(final_state.update_info.num_steps.sum()
              + final_state.update_info.num_shrink.sum()),
    ess=int(safe_ess(logw.mean(axis=-1))),
    logZs=logzs,
)
t_elapsed = time.time() - t_start

positions = final_state.particles.position
log_likelihoods = final_state.particles.loglikelihood

best_idx = int(jnp.argmax(log_likelihoods))
best_instance = model.instance_from_vector(vector=np.asarray(positions[best_idx]).tolist())

print("\n--- NSS (pure_callback) Results ---")
print(format_best_fit(best_instance))
print(f"Log evidence:  {float(results.logZs.mean()):.2f}")
print(f"\n--- Performance ---")
print(f"Wall time:          {t_elapsed:.2f} s (includes JIT compile)")
print(f"Sampling time:      {float(results.time):.2f} s")
print(f"Likelihood calls:   {n_likelihood_calls}")
print(f"Time per call:      {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms")
print(f"Total samples:      {len(positions)}")
