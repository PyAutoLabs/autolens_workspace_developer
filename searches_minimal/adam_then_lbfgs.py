"""
Adam -> L-BFGS polishing — HST MGE lens MAP
-------------------------------------------

The production-candidate hybrid. Stage 1 runs wide multi-start **Adam** (the
proven basin finder: broad z-starts, 300 steps) to discover the truth basin.
Stage 2 initializes **L-BFGS from every Adam endpoint** (one vmapped batch) and
polishes each to its local optimum using second-order curvature — cheap once
Adam has placed each start inside a basin, and it squeezes out the last fraction
of a nat that first-order Adam leaves on the table.

Both stages share the unconstrained z-parameterization (``neg_log_posterior_z``)
so L-BFGS is properly scaled. Costs of the two stages are reported separately.

Termination: Stage 1 is a user-set budget (``N_STEPS``); Stage 2 is
self-terminating (L-BFGS tolerance, capped at ``LBFGS_MAXITER``).

Run on the A100:

    MULTISTART_N_STARTS=128 python -m searches_minimal.adam_then_lbfgs

Requirements: optax + jaxopt (JAX).
"""

import os
import time

import numpy as np
import jax
import optax
from jaxopt import LBFGS

from searches_minimal._grad_setup import (
    build_map_objective,
    make_z_starts,
    time_compile,
    write_grad_summary,
    _peak_gpu_mb,
)

N_STARTS = int(os.environ.get("MULTISTART_N_STARTS", "128"))
N_STEPS = int(os.environ.get("ADAM_STEPS", "300"))
LEARNING_RATE = float(os.environ.get("ADAM_LR", "1e-2"))
LBFGS_MAXITER = int(os.environ.get("LBFGS_MAXITER", "200"))
LBFGS_TOL = float(os.environ.get("LBFGS_TOL", "1e-3"))
EINSTEIN_TRUTH, EINSTEIN_TOL = 1.6, 0.3

print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
obj = build_map_objective()
print(f"Model free parameters: {obj.ndim}  |  N_STARTS = {N_STARTS}")

compile_s = time_compile(obj)
z_starts, n_kept = make_z_starts(obj, N_STARTS)
print(f"Collected {n_kept} finite-gradient z-starts")

batched_vag = jax.jit(jax.vmap(jax.value_and_grad(obj.neg_log_posterior_z_raw)))

# ---- Stage 1: multi-start Adam (basin discovery) ---------------------------
params = z_starts
t0 = time.time()
l, g = batched_vag(params)
jax.block_until_ready(l)
jax.block_until_ready(g)
adam_compile_s = time.time() - t0
print(f"Adam batched compile: {adam_compile_s:.1f} s")

opt = optax.adam(LEARNING_RATE)
opt_state = opt.init(params)
best_history: list[float] = []
global_best_loss = np.inf
global_best_z = params[0]
print(f"\nStage 1: {n_kept}-start Adam for {N_STEPS} steps (lr={LEARNING_RATE})...")
t_start = time.time()
for i in range(N_STEPS):
    losses, grads = batched_vag(params)
    losses_np = np.where(np.isfinite(np.asarray(losses)), np.asarray(losses), np.inf)
    j = int(np.argmin(losses_np))
    if losses_np[j] < global_best_loss:
        global_best_loss = float(losses_np[j])
        global_best_z = params[j]
    best_history.append(-global_best_loss)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    if i % 50 == 0:
        print(f"  step {i:4d}: best log_posterior = {-global_best_loss:.2f}")
adam_loop_s = time.time() - t_start
z_after_adam = params
adam_best_logpost = -global_best_loss
print(
    f"Stage 1 done: Adam best log_posterior = {adam_best_logpost:.2f}  ({adam_loop_s:.1f} s)"
)

# ---- Stage 2: L-BFGS polish from every Adam endpoint -----------------------
# implicit_diff=False + backtracking line search — see jaxopt_lbfgs_multistart.py
# (default zoom + implicit diff make the vmapped compile hang for >15 min).
solver = LBFGS(
    fun=obj.neg_log_posterior_z_raw,
    maxiter=LBFGS_MAXITER,
    tol=LBFGS_TOL,
    implicit_diff=False,
    linesearch="backtracking",
    maxls=15,
)
run_batched = jax.jit(jax.vmap(lambda z: solver.run(z)))
t0 = time.time()
step = run_batched(z_after_adam)
jax.block_until_ready(step.params)
lbfgs_cold_s = time.time() - t0
t0 = time.time()
step2 = run_batched(z_after_adam)
jax.block_until_ready(step2.params)
lbfgs_warm_s = time.time() - t0
lbfgs_compile_s = max(lbfgs_cold_s - lbfgs_warm_s, 0.0)
z_polished = np.asarray(step.params)
iter_nums = np.asarray(step.state.iter_num)
print(
    f"Stage 2 done: L-BFGS polish, iters {int(np.min(iter_nums))}..{int(np.max(iter_nums))} "
    f"(cap {LBFGS_MAXITER})  compile {lbfgs_compile_s:.1f}s solve {lbfgs_warm_s:.1f}s"
)

# ---- Score polished endpoints ----------------------------------------------
phys = np.array(
    [np.asarray(obj.physical_from_z(jax.numpy.asarray(z))) for z in z_polished]
)
logpost = np.array(
    [
        float(
            obj.log_likelihood(jax.numpy.asarray(p))
            + obj.log_prior(jax.numpy.asarray(p))
        )
        for p in phys
    ]
)
logpost = np.where(np.isfinite(logpost), logpost, -np.inf)
jb = int(np.argmax(logpost))
best_params = phys[jb]
r_e = np.array(
    [
        obj.model.instance_from_vector(
            vector=list(p)
        ).galaxies.lens.mass.einstein_radius
        for p in phys
    ]
)
n_in_basin = int(np.sum(np.abs(r_e - EINSTEIN_TRUTH) < EINSTEIN_TOL))
polish_gain = float(logpost[jb]) - adam_best_logpost
print(
    f"\n{n_in_basin}/{n_kept} polished starts in basin; polish gain over Adam = {polish_gain:+.4f} nat"
)

scalar_adam = n_kept * N_STEPS
scalar_lbfgs = int(np.sum(np.clip(iter_nums, 0, None)))
write_grad_summary(
    name=f"adam_then_lbfgs_n{n_kept}",
    title=f"Adam->L-BFGS ({n_kept}x)",
    obj=obj,
    best_params=best_params,
    log_posterior_history=best_history + [float(logpost[jb])],
    wall_s=compile_s + adam_compile_s + adam_loop_s + lbfgs_cold_s,
    compile_s=compile_s + adam_compile_s + lbfgs_compile_s,
    warm_ms_per_eval=adam_loop_s / N_STEPS / n_kept * 1e3,
    n_evals=scalar_adam + scalar_lbfgs,
    n_iters=N_STEPS + int(np.max(iter_nums)),
    converged=(n_in_basin > 0),
    config_line=(
        f"Adam {n_kept}x{N_STEPS} (lr={LEARNING_RATE}) -> L-BFGS polish "
        f"(maxiter={LBFGS_MAXITER}); {n_in_basin}/{n_kept} in basin; polish +{polish_gain:.4f} nat"
    ),
    termination="hybrid (Stage 1 user-budget; Stage 2 self-terminating)",
    accounting={
        "starts": n_kept,
        "stage1_adam_compile_s": round(compile_s + adam_compile_s, 1),
        "stage1_adam_loop_s": round(adam_loop_s, 1),
        "stage1_scalar_grad_evals": scalar_adam,
        "stage1_adam_best_logpost": round(adam_best_logpost, 4),
        "stage2_lbfgs_compile_s": round(lbfgs_compile_s, 1),
        "stage2_lbfgs_solve_s": round(lbfgs_warm_s, 1),
        "stage2_scalar_grad_evals": scalar_lbfgs,
        "stage2_polish_gain_nat": round(polish_gain, 4),
        "peak_gpu_mb": _peak_gpu_mb(),
    },
)
