"""
optax ADABelief — JAX gradient MAP optimizer on the HST MGE lens likelihood
---------------------------------------------------------------------------

ADABelief (Zhuang et al. 2020) is the optimizer used by Enzi et al. 2026
(arXiv:2606.30620) for JAX strong-lens source reconstruction in Herculens
(there via NumPyro SVI). Here we run it as a straight gradient-descent MAP
point optimizer: minimise ``-(log_likelihood + log_prior)`` in the physical
parameter vector, cold-started from the prior median.

Like Adam, ADABelief normalises the step per-parameter by an exponential
moving average of the gradient's deviation from its mean, so a single global
learning rate copes with the disparate parameter scales of a lens model
(einstein_radius ~1.6 vs ell_comps ~0.05) without hand-tuned per-parameter
rates.

Run from the workspace root:

    python -m searches_minimal.optax_adabelief

Requirements: optax (JAX).
"""

import time

import numpy as np
import jax
import optax

from searches_minimal._grad_setup import (
    build_map_objective,
    robust_cold_start,
    time_compile,
    write_grad_summary,
)

# --- config (modest CPU budget; tune up for a converged run) ----------------
LEARNING_RATE = 1e-2
N_STEPS = 500
PLATEAU_PATIENCE = 60  # stop if no improvement for this many steps
MAX_GRAD_NORM = 1.0  # the raw MAP gradient norm is ~1e5; clip before the step

obj = build_map_objective()
print(f"Model free parameters: {obj.ndim}")

compile_s = time_compile(obj)
print(f"JIT compile (value_and_grad): {compile_s:.1f} s")

# Cold start off the degenerate prior median (see robust_cold_start).
params, n_try = robust_cold_start(obj)
print(f"Cold start: finite-gradient draw after {n_try} unit-space perturbation(s)")

# Post-compile fwd+grad eval timing.
t0 = time.time()
for _ in range(3):
    loss, grad = obj.value_and_grad(params)
    jax.block_until_ready(loss)
    jax.block_until_ready(grad)
warm_ms = (time.time() - t0) / 3 * 1e3
print(f"Warm fwd+grad: {warm_ms:.1f} ms/eval")

opt = optax.chain(
    optax.clip_by_global_norm(MAX_GRAD_NORM),
    optax.adabelief(learning_rate=LEARNING_RATE),
)
opt_state = opt.init(params)

history: list[float] = []
best_loss = np.inf
best_params = params
best_iter = 0

print(f"\nRunning ADABelief for up to {N_STEPS} steps (lr={LEARNING_RATE})...")
t_start = time.time()
for i in range(N_STEPS):
    loss, grads = obj.value_and_grad(params)
    loss_f = float(loss)
    history.append(-loss_f)  # log posterior
    if loss_f < best_loss:
        best_loss, best_params, best_iter = loss_f, params, i
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    if i % 50 == 0:
        print(f"  step {i:4d}: log_posterior = {-loss_f:.2f}")
    if i - best_iter >= PLATEAU_PATIENCE:
        print(f"  plateau: no improvement for {PLATEAU_PATIENCE} steps, stopping at {i}")
        break
loop_s = time.time() - t_start
n_evals = len(history)

write_grad_summary(
    name="optax_adabelief",
    title="optax ADABelief (MAP)",
    obj=obj,
    best_params=best_params,
    log_posterior_history=history,
    wall_s=compile_s + loop_s,
    compile_s=compile_s,
    warm_ms_per_eval=warm_ms,
    n_evals=n_evals,
    n_iters=best_iter + 1,
    converged=(n_evals < N_STEPS),  # stopped early = reached a plateau
    config_line=f"lr={LEARNING_RATE}, clip_norm={MAX_GRAD_NORM}, max_steps={N_STEPS}, plateau_patience={PLATEAU_PATIENCE}, cold-start (perturbed prior median)",
)
