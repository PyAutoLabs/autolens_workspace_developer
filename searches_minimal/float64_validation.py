"""
float64 basin-recovery validation — HST MGE lens MAP
----------------------------------------------------

The A100 benchmark runs in **float32** (x64 is disabled on that GPU). This script
re-runs a small multi-start Adam in **float64** on the CPU (where x64 is
available) to confirm that basin recovery, the best log likelihood, and NaN
behaviour are not float32 artifacts — i.e. that the truth basin (einstein_radius
≈ 1.6) is found at double precision too, and that the per-start hit rate is
consistent with the float32 A100 runs.

Uses the same shared unconstrained z-parameterization and start generation as the
gradient optimizers, so the only difference from a scaled-down A100 Adam run is
the precision and the device.

Run from the workspace root (x64 forced on):

    JAX_ENABLE_X64=1 python -m searches_minimal.float64_validation

Requirements: optax (JAX).
"""

import os
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)  # belt-and-braces with JAX_ENABLE_X64=1

import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from searches_minimal._grad_setup import (  # noqa: E402
    build_map_objective,
    make_z_starts,
    time_compile,
    write_grad_summary,
    _peak_gpu_mb,
)

N_STARTS = int(os.environ.get("MULTISTART_N_STARTS", "16"))
N_STEPS = int(os.environ.get("ADAM_STEPS", "300"))
LEARNING_RATE = 1e-2
EINSTEIN_TRUTH, EINSTEIN_TOL = 1.6, 0.3

print(f"JAX backend: {jax.default_backend()}  x64_enabled: {jax.config.jax_enable_x64}")
obj = build_map_objective()
compile_s = time_compile(obj)
z_starts, n_kept = make_z_starts(obj, N_STARTS)
# confirm we really are in float64
print(f"start dtype: {z_starts.dtype}  (expect float64)")

batched_vag = jax.jit(jax.vmap(jax.value_and_grad(obj.neg_log_posterior_z_raw)))
params = z_starts
t0 = time.time()
l, g = batched_vag(params)
jax.block_until_ready(l)
jax.block_until_ready(g)
batched_compile_s = time.time() - t0

opt = optax.adam(LEARNING_RATE)
opt_state = opt.init(params)
best_history: list[float] = []
global_best_loss = np.inf
global_best_z = params[0]
n_nan_steps = 0

print(f"\nRunning {n_kept}-start Adam (float64) for {N_STEPS} steps...")
t_start = time.time()
for i in range(N_STEPS):
    losses, grads = batched_vag(params)
    losses_np = np.asarray(losses)
    if not np.all(np.isfinite(losses_np)):
        n_nan_steps += 1
    losses_np = np.where(np.isfinite(losses_np), losses_np, np.inf)
    j = int(np.argmin(losses_np))
    if losses_np[j] < global_best_loss:
        global_best_loss = float(losses_np[j])
        global_best_z = params[j]
    best_history.append(-global_best_loss)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
loop_s = time.time() - t_start

phys = np.array([np.asarray(obj.physical_from_z(jnp.asarray(z))) for z in params])
r_e = np.array(
    [obj.model.instance_from_vector(vector=list(p)).galaxies.lens.mass.einstein_radius for p in phys]
)
n_in_basin = int(np.sum(np.abs(r_e - EINSTEIN_TRUTH) < EINSTEIN_TOL))
best_params = np.asarray(obj.physical_from_z(jnp.asarray(global_best_z)))
print(f"\n{n_in_basin}/{n_kept} starts in basin (float64); NaN steps: {n_nan_steps}")

write_grad_summary(
    name=f"float64_validation_n{n_kept}",
    title=f"float64 multi-start Adam ({n_kept}x)",
    obj=obj,
    best_params=best_params,
    log_posterior_history=best_history,
    wall_s=compile_s + batched_compile_s + loop_s,
    compile_s=compile_s + batched_compile_s,
    warm_ms_per_eval=loop_s / N_STEPS / n_kept * 1e3,
    n_evals=n_kept * N_STEPS,
    n_iters=N_STEPS,
    converged=(n_in_basin > 0),
    config_line=(
        f"float64, n_starts={n_kept}, steps={N_STEPS}, device={jax.default_backend()}, "
        f"{n_in_basin}/{n_kept} in basin; NaN steps={n_nan_steps}"
    ),
    termination="user-budget (fixed step count)",
    accounting={
        "starts": n_kept,
        "batched_steps": N_STEPS,
        "scalar_grad_evals": n_kept * N_STEPS,
        "nan_steps": n_nan_steps,
        "precision": "float64",
        "peak_gpu_mb": _peak_gpu_mb(),
    },
)
