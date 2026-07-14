"""
Multi-start Nonlinear Conjugate Gradient (jaxopt) — HST MGE lens MAP
-------------------------------------------------------------------

Optional first-order-plus method: nonlinear CG builds conjugate search
directions from the gradient (Polak-Ribiere by default in jaxopt), a middle
ground between plain gradient descent and quasi-Newton L-BFGS with no
inverse-Hessian storage. Runs on the shared unconstrained z-objective as a wide
multi-start; **self-terminating** on gradient tolerance, capped at ``MAXITER``.

Included for completeness of the gradient-optimizer sweep; flagged in the
writeup if it fails to reach the basin.

Run on the A100:

    MULTISTART_N_STARTS=128 python -m searches_minimal.jaxopt_nonlinear_cg

Requirements: jaxopt (JAX).
"""

import os

import jax
from jaxopt import NonlinearCG

from searches_minimal._grad_setup import (
    build_map_objective,
    make_z_starts,
    run_vmapped_map_solver,
    time_compile,
)

N_STARTS = int(os.environ.get("MULTISTART_N_STARTS", "128"))
MAXITER = int(os.environ.get("NCG_MAXITER", "200"))
TOL = float(os.environ.get("NCG_TOL", "1e-3"))

print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
obj = build_map_objective()
print(f"Model free parameters: {obj.ndim}  |  N_STARTS = {N_STARTS}  |  NCG maxiter={MAXITER} tol={TOL}")

time_compile(obj)
z_starts, n_kept = make_z_starts(obj, N_STARTS)
print(f"Collected {n_kept} finite-gradient z-starts")

# implicit_diff=False + backtracking line search — see jaxopt_lbfgs_multistart.py.
solver = NonlinearCG(
    fun=obj.neg_log_posterior_z_raw,
    maxiter=MAXITER,
    tol=TOL,
    implicit_diff=False,
    linesearch="backtracking",
    maxls=15,
)

run_vmapped_map_solver(
    obj=obj,
    solver=solver,
    z_starts=z_starts,
    name=f"jaxopt_nonlinear_cg_n{n_kept}",
    title=f"multi-start NCG ({n_kept}x)",
    max_iters=MAXITER,
    termination=f"self-terminating (NCG grad-tol={TOL}, cap {MAXITER} iters)",
    is_residual=False,
)
