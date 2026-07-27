"""
Multi-start Adam (GIGA-Lens recipe) — HST MGE lens likelihood
-------------------------------------------------------------

Phase-1 showed that a *single* cold-start gradient optimizer (Adam, ADABelief,
L-BFGS) all converge to the same wrong basin (einstein_radius pinned at the
prior wall). GIGA-Lens (Gu, Huang et al. 2022; 2.0 in arXiv:2606.30633)
addresses exactly this: run **many** gradient-descent starts in parallel and
keep the best — the first stage of its multi-start -> variational-inference ->
HMC pipeline.

This script implements that first stage: ``N_STARTS`` broad random cold starts
(drawn across the prior, not tiny median perturbations, so they sample
*different* basins), all optimised in parallel with Adam via ``jax.vmap``. It
reports the best fit found and, crucially, **how many starts reach the correct
basin** (einstein_radius near the truth ~1.6) — the robustness signal that a
single start cannot provide.

On CPU each vmapped step costs ~N_STARTS x the single-eval time (no GPU
parallelism), so this is where GIGA-Lens's GPU/multi-GPU design pays off; here
we keep N_STARTS modest.

Run from the workspace root:

    python -m searches_minimal.multi_start_adam

Requirements: optax (JAX).
"""

import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

from searches_minimal._grad_setup import (
    build_map_objective,
    time_compile,
    write_grad_summary,
)

# --- config -----------------------------------------------------------------
N_STARTS = 12
LEARNING_RATE = 1e-2
N_STEPS = 300
START_LOW, START_HIGH = 0.15, 0.85  # unit-cube spread of the random starts
EINSTEIN_TRUTH = 1.6
EINSTEIN_TOL = 0.3  # a start "found the basin" if |r_E - truth| < this

obj = build_map_objective()
print(f"Model free parameters: {obj.ndim}")

compile_s = time_compile(obj)
print(f"JIT compile (value_and_grad): {compile_s:.1f} s")

# Batched value_and_grad over a stack of starts, and a batched log-L for scoring.
batched_vag = jax.jit(jax.vmap(jax.value_and_grad(obj.neg_log_posterior_raw)))
batched_log_l = jax.jit(jax.vmap(obj.log_likelihood))

# --- build N_STARTS broad, finite-gradient starts ---------------------------
rng = np.random.default_rng(0)
starts = []
tries = 0
while len(starts) < N_STARTS and tries < N_STARTS * 20:
    tries += 1
    u = rng.uniform(START_LOW, START_HIGH, size=obj.ndim)
    x = jnp.asarray(obj.model.vector_from_unit_vector(list(u)))
    loss, grad = obj.value_and_grad(x)
    if np.isfinite(float(loss)) and np.all(np.isfinite(np.asarray(grad))):
        starts.append(x)
params = jnp.stack(starts)  # (N_STARTS, ndim)
print(f"Collected {len(starts)} finite-gradient starts (from {tries} draws)")

# The batched graph is a DISTINCT jit from the single value_and_grad used above
# for start-filtering, so it compiles on its first call. Time that compile
# explicitly (otherwise it silently pollutes the warm-eval measurement), then
# time a genuinely warm batched eval.
t0 = time.time()
l, g = batched_vag(params)
jax.block_until_ready(l)
jax.block_until_ready(g)
batched_compile_s = time.time() - t0
print(f"Batched (vmap x{N_STARTS}) compile: {batched_compile_s:.1f} s")

t0 = time.time()
for _ in range(2):
    l, g = batched_vag(params)
    jax.block_until_ready(l)
    jax.block_until_ready(g)
warm_ms = (time.time() - t0) / 2 / N_STARTS * 1e3  # per single start-eval
print(f"Warm batched eval: {warm_ms:.1f} ms/single-start-eval")

opt = optax.adam(learning_rate=LEARNING_RATE)  # Adam self-normalises the ~1e5 grad
opt_state = opt.init(params)

best_history: list[float] = []  # best log-posterior across all starts, per step
global_best_loss = np.inf
global_best_params = params[0]

print(f"\nRunning {N_STARTS}-start Adam for {N_STEPS} steps (lr={LEARNING_RATE})...")
t_start = time.time()
for i in range(N_STEPS):
    losses, grads = batched_vag(params)  # (S,), (S, ndim)
    losses_np = np.asarray(losses)
    losses_np = np.where(np.isfinite(losses_np), losses_np, np.inf)
    j = int(np.argmin(losses_np))
    if losses_np[j] < global_best_loss:
        global_best_loss = float(losses_np[j])
        global_best_params = params[j]
    best_history.append(-global_best_loss)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    if i % 50 == 0:
        print(f"  step {i:4d}: best log_posterior = {-global_best_loss:.2f}")
loop_s = time.time() - t_start

# --- per-start outcome: how many reached the correct basin ------------------
final_log_l = np.asarray(batched_log_l(params))
final_r_e = np.array(
    [
        obj.model.instance_from_vector(
            vector=list(np.asarray(p))
        ).galaxies.lens.mass.einstein_radius
        for p in params
    ]
)
in_basin = np.abs(final_r_e - EINSTEIN_TRUTH) < EINSTEIN_TOL
n_in_basin = int(np.sum(in_basin))

print("\nPer-start outcome:")
for k in range(len(params)):
    tag = "  <-- correct basin" if in_basin[k] else ""
    print(
        f"  start {k:2d}: log L = {final_log_l[k]:12.2f}   r_E = {final_r_e[k]:.3f}{tag}"
    )
print(
    f"\n{n_in_basin}/{N_STARTS} starts reached the correct basin "
    f"(|r_E - {EINSTEIN_TRUTH}| < {EINSTEIN_TOL})"
)

write_grad_summary(
    name="multi_start_adam",
    title="multi-start Adam (MAP)",
    obj=obj,
    best_params=global_best_params,
    log_posterior_history=best_history,
    wall_s=compile_s + batched_compile_s + loop_s,
    compile_s=compile_s + batched_compile_s,  # single (start-filter) + batched (loop)
    warm_ms_per_eval=warm_ms,
    n_evals=N_STARTS * N_STEPS,
    n_iters=N_STEPS,
    converged=(n_in_basin > 0),
    config_line=(
        f"n_starts={N_STARTS}, lr={LEARNING_RATE}, steps={N_STEPS}, "
        f"start_spread=U({START_LOW},{START_HIGH}), {n_in_basin}/{N_STARTS} reached correct basin"
    ),
)
