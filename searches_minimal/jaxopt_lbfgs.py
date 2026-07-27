"""
jaxopt L-BFGS — quasi-Newton JAX gradient MAP optimizer (HST MGE lens)
----------------------------------------------------------------------

L-BFGS via jaxopt, driven by the **true** ``jax.grad`` of the MAP objective
``-(log_likelihood + log_prior)``. This is the JAX-native counterpart to the
existing ``lbfgs_simple.py``, which runs scipy L-BFGS-B on the *NumPy*
likelihood with finite-difference gradients (~1400 ms/eval). Here the gradient
is analytic (autodiff) and the likelihood is JIT-compiled, and L-BFGS builds a
curvature estimate from the gradient history — so it should need far fewer
iterations than first-order Adam/ADABelief to reach the same optimum, and
handles the disparate parameter scales via its implicit Hessian model.

Stepped manually (``solver.update``) rather than one ``solver.run`` so the
per-iteration log-posterior history feeds the evals/time-to-ML headline.

Run from the workspace root:

    python -m searches_minimal.jaxopt_lbfgs

Requirements: jaxopt (JAX).
"""

import time

import numpy as np
import jax
from jaxopt import LBFGS

from searches_minimal._grad_setup import (
    build_map_objective,
    robust_cold_start,
    time_compile,
    write_grad_summary,
)

# --- config (modest CPU budget; quasi-Newton converges in few iters) --------
MAXITER = 200
GRAD_TOL = 1e-3  # stop when the projected-gradient norm falls below this

obj = build_map_objective()
print(f"Model free parameters: {obj.ndim}")

compile_s = time_compile(obj)
print(f"JIT compile (value_and_grad): {compile_s:.1f} s")

params, n_try = robust_cold_start(obj)
print(f"Cold start: finite-gradient draw after {n_try} unit-space perturbation(s)")

t0 = time.time()
for _ in range(3):
    loss, grad = obj.value_and_grad(params)
    jax.block_until_ready(loss)
    jax.block_until_ready(grad)
warm_ms = (time.time() - t0) / 3 * 1e3
print(f"Warm fwd+grad: {warm_ms:.1f} ms/eval")

solver = LBFGS(
    fun=obj.value_and_grad, value_and_grad=True, maxiter=MAXITER, tol=GRAD_TOL
)
state = solver.init_state(params)

history: list[float] = []
best_loss = np.inf
best_params = params
best_iter = 0
converged = False

print(f"\nRunning jaxopt L-BFGS for up to {MAXITER} iters (tol={GRAD_TOL})...")
t_start = time.time()
for i in range(MAXITER):
    params, state = solver.update(params, state)
    loss_f = float(state.value)
    history.append(-loss_f)
    if loss_f < best_loss:
        best_loss, best_params, best_iter = loss_f, params, i
    err = float(state.error)
    if i % 10 == 0:
        print(f"  iter {i:4d}: log_posterior = {-loss_f:.2f}   grad_norm = {err:.3e}")
    if err < GRAD_TOL:
        print(f"  converged: grad_norm {err:.3e} < {GRAD_TOL} at iter {i}")
        converged = True
        break
loop_s = time.time() - t_start
n_evals = len(history)

write_grad_summary(
    name="jaxopt_lbfgs",
    title="jaxopt L-BFGS (MAP)",
    obj=obj,
    best_params=best_params,
    log_posterior_history=history,
    wall_s=compile_s + loop_s,
    compile_s=compile_s,
    warm_ms_per_eval=warm_ms,
    n_evals=n_evals,
    n_iters=best_iter + 1,
    converged=converged,
    config_line=f"maxiter={MAXITER}, tol={GRAD_TOL}, cold-start (perturbed prior median)",
)
