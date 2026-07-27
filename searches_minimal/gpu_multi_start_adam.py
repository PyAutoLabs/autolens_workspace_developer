"""
GPU multi-start Adam (GIGA-Lens scaling test) — HST MGE lens
------------------------------------------------------------

Multi-start Adam was the CPU winner at 12 starts (2/12 → truth). GIGA-Lens runs
*hundreds* of starts in parallel on a GPU. This script scales the start count up
(``N_STARTS`` from env, default 64) and runs on the GPU, where a large vmapped
batch is cheap — testing whether more starts push the basin-hit rate toward ~100%
(P(≥1 hit) = 1 − (1 − p)^N; at p ≈ 0.17, N = 64 → ~1.0).

Run on the laptop GPU (RTX 2060 / ~/venv/PyAutoGPU):

    JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
      MULTISTART_N_STARTS=48 ~/venv/PyAutoGPU/bin/python -m searches_minimal.gpu_multi_start_adam

Requirements: optax + a CUDA jax (PyAutoGPU).
"""

import os
import time

import numpy as np
import jax
import optax

from searches_minimal._grad_setup import (
    build_map_objective,
    time_compile,
    write_grad_summary,
)

N_STARTS = int(os.environ.get("MULTISTART_N_STARTS", "64"))
N_STEPS = 300
START_LOW, START_HIGH = 0.15, 0.85
EINSTEIN_TRUTH = 1.6
EINSTEIN_TOL = 0.3

# Local optimizer rule within multi-start (does it matter vs Adam?): set via env.
# Lion is sign-based, so it wants a ~10x smaller learning rate than Adam/ADABelief.
OPTIMIZER = os.environ.get("MULTISTART_OPTIMIZER", "adam").lower()
_OPT_FACTORY = {"adam": optax.adam, "adabelief": optax.adabelief, "lion": optax.lion}
_OPT_LR = {"adam": 1e-2, "adabelief": 1e-2, "lion": 1e-3}
LEARNING_RATE = float(os.environ.get("MULTISTART_LR", _OPT_LR[OPTIMIZER]))

print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
obj = build_map_objective()
print(
    f"Model free parameters: {obj.ndim}  |  N_STARTS = {N_STARTS}  |  OPTIMIZER = {OPTIMIZER} (lr={LEARNING_RATE})"
)

compile_s = time_compile(obj)
batched_vag = jax.jit(jax.vmap(jax.value_and_grad(obj.neg_log_posterior_raw)))
batched_log_l = jax.jit(jax.vmap(obj.log_likelihood))

# Broad finite-gradient starts.
rng = np.random.default_rng(0)
starts = []
tries = 0
while len(starts) < N_STARTS and tries < N_STARTS * 30:
    tries += 1
    u = rng.uniform(START_LOW, START_HIGH, size=obj.ndim)
    x = jax.numpy.asarray(obj.model.vector_from_unit_vector(list(u)))
    loss, grad = obj.value_and_grad(x)
    if np.isfinite(float(loss)) and np.all(np.isfinite(np.asarray(grad))):
        starts.append(x)
params = jax.numpy.stack(starts)
print(f"Collected {len(starts)} finite-gradient starts (from {tries} draws)")

t0 = time.time()
l, g = batched_vag(params)
jax.block_until_ready(l)
jax.block_until_ready(g)
batched_compile_s = time.time() - t0
print(f"Batched (vmap x{N_STARTS}) compile: {batched_compile_s:.1f} s")

opt = _OPT_FACTORY[OPTIMIZER](LEARNING_RATE)
opt_state = opt.init(params)
best_history: list[float] = []
global_best_loss = np.inf
global_best_params = params[0]

print(
    f"\nRunning {N_STARTS}-start {OPTIMIZER} for {N_STEPS} steps on {jax.default_backend()}..."
)
t_start = time.time()
for i in range(N_STEPS):
    losses, grads = batched_vag(params)
    losses_np = np.where(np.isfinite(np.asarray(losses)), np.asarray(losses), np.inf)
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

final_r_e = np.array(
    [
        obj.model.instance_from_vector(
            vector=list(np.asarray(p))
        ).galaxies.lens.mass.einstein_radius
        for p in params
    ]
)
n_in_basin = int(np.sum(np.abs(final_r_e - EINSTEIN_TRUTH) < EINSTEIN_TOL))
print(
    f"\n{n_in_basin}/{N_STARTS} starts reached the correct basin (p_hit = {n_in_basin/N_STARTS:.2f})"
)

write_grad_summary(
    name=f"gpu_multi_start_{OPTIMIZER}_n{N_STARTS}",
    title=f"GPU multi-start {OPTIMIZER} ({N_STARTS}x)",
    obj=obj,
    best_params=global_best_params,
    log_posterior_history=best_history,
    wall_s=compile_s + batched_compile_s + loop_s,
    compile_s=compile_s + batched_compile_s,
    warm_ms_per_eval=(loop_s / N_STEPS / N_STARTS * 1e3),
    n_evals=N_STARTS * N_STEPS,
    n_iters=N_STEPS,
    converged=(n_in_basin > 0),
    config_line=(
        f"optimizer={OPTIMIZER}, n_starts={N_STARTS}, lr={LEARNING_RATE}, steps={N_STEPS}, "
        f"device={jax.default_backend()}, {n_in_basin}/{N_STARTS} in basin (p_hit={n_in_basin/N_STARTS:.2f})"
    ),
)
