"""
Adam -> Levenberg-Marquardt polishing — HST MGE lens MAP
--------------------------------------------------------

Same two-stage hybrid as ``adam_then_lbfgs.py`` but the polish stage is
**Levenberg-Marquardt** on the exact residual least-squares (see
``jaxopt_lm_multistart.py`` for the residual derivation). Adam discovers the
basin; LM then takes damped Gauss-Newton steps from each Adam endpoint, using
the full residual Jacobian rather than a gradient-history curvature estimate.

The comparison of interest: does true second-order (LM, Jacobian) polish beat
quasi-Newton (L-BFGS) polish, given the NNLS active-set kinks in the residual?

Termination: Stage 1 user-budget (``N_STEPS``); Stage 2 self-terminating (LM
tolerance, capped at ``LM_MAXITER``).

Run on the A100:

    MULTISTART_N_STARTS=128 python -m searches_minimal.adam_then_lm

Requirements: optax + jaxopt (JAX).
"""

import os
import time

import numpy as np
import jax
import optax
from jaxopt import LevenbergMarquardt

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
LM_MAXITER = int(os.environ.get("LM_MAXITER", "100"))
LM_TOL = float(os.environ.get("LM_TOL", "1e-3"))
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
print(f"Stage 1 done: Adam best log_posterior = {adam_best_logpost:.2f}  ({adam_loop_s:.1f} s)")

# ---- Stage 2: LM polish from every Adam endpoint ---------------------------
solver = LevenbergMarquardt(residual_fun=obj.residual_z_raw, maxiter=LM_MAXITER, tol=LM_TOL)
run_batched = jax.jit(jax.vmap(lambda z: solver.run(z)))
t0 = time.time()
step = run_batched(z_after_adam)
jax.block_until_ready(step.params)
lm_cold_s = time.time() - t0
t0 = time.time()
step2 = run_batched(z_after_adam)
jax.block_until_ready(step2.params)
lm_warm_s = time.time() - t0
lm_compile_s = max(lm_cold_s - lm_warm_s, 0.0)
z_polished = np.asarray(step.params)
iter_nums = np.asarray(step.state.iter_num)
print(
    f"Stage 2 done: LM polish, iters {int(np.min(iter_nums))}..{int(np.max(iter_nums))} "
    f"(cap {LM_MAXITER})  compile {lm_compile_s:.1f}s solve {lm_warm_s:.1f}s"
)

# ---- Score polished endpoints ----------------------------------------------
phys = np.array([np.asarray(obj.physical_from_z(jax.numpy.asarray(z))) for z in z_polished])
logpost = np.array(
    [float(obj.log_likelihood(jax.numpy.asarray(p)) + obj.log_prior(jax.numpy.asarray(p))) for p in phys]
)
logpost = np.where(np.isfinite(logpost), logpost, -np.inf)
jb = int(np.argmax(logpost))
best_params = phys[jb]
r_e = np.array(
    [obj.model.instance_from_vector(vector=list(p)).galaxies.lens.mass.einstein_radius for p in phys]
)
n_in_basin = int(np.sum(np.abs(r_e - EINSTEIN_TRUTH) < EINSTEIN_TOL))
polish_gain = float(logpost[jb]) - adam_best_logpost
print(f"\n{n_in_basin}/{n_kept} polished starts in basin; polish gain over Adam = {polish_gain:+.4f} nat")

scalar_adam = n_kept * N_STEPS
scalar_lm = int(np.sum(np.clip(iter_nums, 0, None)))
write_grad_summary(
    name=f"adam_then_lm_n{n_kept}",
    title=f"Adam->LM ({n_kept}x)",
    obj=obj,
    best_params=best_params,
    log_posterior_history=best_history + [float(logpost[jb])],
    wall_s=compile_s + adam_compile_s + adam_loop_s + lm_cold_s,
    compile_s=compile_s + adam_compile_s + lm_compile_s,
    warm_ms_per_eval=adam_loop_s / N_STEPS / n_kept * 1e3,
    n_evals=scalar_adam + scalar_lm,
    n_iters=N_STEPS + int(np.max(iter_nums)),
    converged=(n_in_basin > 0),
    config_line=(
        f"Adam {n_kept}x{N_STEPS} (lr={LEARNING_RATE}) -> LM polish "
        f"(maxiter={LM_MAXITER}); {n_in_basin}/{n_kept} in basin; polish +{polish_gain:.4f} nat"
    ),
    termination="hybrid (Stage 1 user-budget; Stage 2 self-terminating)",
    accounting={
        "starts": n_kept,
        "stage1_adam_compile_s": round(compile_s + adam_compile_s, 1),
        "stage1_adam_loop_s": round(adam_loop_s, 1),
        "stage1_scalar_grad_evals": scalar_adam,
        "stage1_adam_best_logpost": round(adam_best_logpost, 4),
        "stage2_lm_compile_s": round(lm_compile_s, 1),
        "stage2_lm_solve_s": round(lm_warm_s, 1),
        "stage2_scalar_residual_evals": scalar_lm,
        "stage2_scalar_jacobian_evals": scalar_lm,
        "stage2_polish_gain_nat": round(polish_gain, 4),
        "peak_gpu_mb": _peak_gpu_mb(),
    },
)
