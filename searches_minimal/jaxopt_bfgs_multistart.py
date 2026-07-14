"""
Multi-start BFGS (jaxopt), properly scaled — HST MGE lens MAP
------------------------------------------------------------

Full-memory BFGS quasi-Newton multi-start. Where L-BFGS keeps a limited
gradient history, BFGS maintains the dense inverse-Hessian estimate — feasible
here because the model is only ~15-dimensional. Like the other gradient
optimizers it runs on the shared unconstrained z-parameterization
(``obj.neg_log_posterior_z_raw``) so the curvature estimate sees O(1) scaled
coordinates, and as a wide multi-start so the population supplies basin
diversity.

BFGS fills the "true second-order" slot that Gauss-Newton / Levenberg-Marquardt
**cannot** occupy on this likelihood: those least-squares methods need the
forward-mode residual Jacobian, but the positive-only (NNLS) source inversion
defines only a reverse-mode custom gradient (``custom_vjp``), so ``jacfwd``
raises ``TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp
function``. BFGS needs only the reverse-mode gradient (which is FD-verified
faithful), so it is the strongest second-order method compatible with the
production objective. **Self-terminating** on gradient tolerance, capped at
``MAXITER``.

Run on the A100:

    MULTISTART_N_STARTS=128 python -m searches_minimal.jaxopt_bfgs_multistart

Requirements: jaxopt (JAX).
"""

import os

import jax
from jaxopt import BFGS

from searches_minimal._grad_setup import (
    build_map_objective,
    make_z_starts,
    run_vmapped_map_solver,
    time_compile,
)

N_STARTS = int(os.environ.get("MULTISTART_N_STARTS", "128"))
MAXITER = int(os.environ.get("BFGS_MAXITER", "200"))
TOL = float(os.environ.get("BFGS_TOL", "1e-3"))

print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
obj = build_map_objective()
print(f"Model free parameters: {obj.ndim}  |  N_STARTS = {N_STARTS}  |  BFGS maxiter={MAXITER} tol={TOL}")

time_compile(obj)
z_starts, n_kept = make_z_starts(obj, N_STARTS)
print(f"Collected {n_kept} finite-gradient z-starts")

# implicit_diff=False + backtracking line search — see jaxopt_lbfgs_multistart.py
# (the default zoom line search + implicit diff make the vmapped compile hang).
solver = BFGS(
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
    name=f"jaxopt_bfgs_multistart_n{n_kept}",
    title=f"multi-start BFGS ({n_kept}x)",
    max_iters=MAXITER,
    termination=f"self-terminating (BFGS grad-tol={TOL}, cap {MAXITER} iters)",
    is_residual=False,
)
