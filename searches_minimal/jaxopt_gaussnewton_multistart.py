"""
Multi-start Gauss-Newton (jaxopt) — HST MGE lens MAP
----------------------------------------------------

Gauss-Newton is the undamped limit of Levenberg-Marquardt: it solves the normal
equations ``J^T J dz = -J^T r`` each iteration, taking a full second-order step
without Levenberg damping. It exploits the same exact residual least-squares
structure as ``jaxopt_lm_multistart.py`` (see that file's header for the
derivation; residual = imaging ``normalized_residual_map`` + Gaussian-prior
block, ``0.5||r||^2`` sharing its argmin with the true MAP).

Undamped GN converges faster than LM when the residual is well-behaved near the
optimum but is less robust far from it (no trust region) — so the wide
multi-start matters even more here. **Self-terminating** on tolerance, capped at
``MAXITER``. Same caveat as LM: NNLS active-set kinks make the residual only
piecewise smooth.

Run on the A100:

    MULTISTART_N_STARTS=128 python -m searches_minimal.jaxopt_gaussnewton_multistart

Requirements: jaxopt (JAX).
"""

import os

import jax
from jaxopt import GaussNewton

from searches_minimal._grad_setup import (
    build_map_objective,
    make_z_starts,
    run_vmapped_map_solver,
    time_compile,
)

N_STARTS = int(os.environ.get("MULTISTART_N_STARTS", "128"))
MAXITER = int(os.environ.get("GN_MAXITER", "100"))
TOL = float(os.environ.get("GN_TOL", "1e-3"))

print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
obj = build_map_objective()
print(f"Model free parameters: {obj.ndim}  |  N_STARTS = {N_STARTS}  |  GN maxiter={MAXITER} tol={TOL}")

time_compile(obj)
z_starts, n_kept = make_z_starts(obj, N_STARTS)
print(f"Collected {n_kept} finite-gradient z-starts")

solver = GaussNewton(residual_fun=obj.residual_z_raw, maxiter=MAXITER, tol=TOL)

run_vmapped_map_solver(
    obj=obj,
    solver=solver,
    z_starts=z_starts,
    name=f"jaxopt_gaussnewton_n{n_kept}",
    title=f"multi-start Gauss-Newton ({n_kept}x)",
    max_iters=MAXITER,
    termination=f"self-terminating (GN tol={TOL}, cap {MAXITER} iters)",
    is_residual=True,
)
