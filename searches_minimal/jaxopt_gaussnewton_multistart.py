"""
Multi-start Gauss-Newton (jaxopt) — HST MGE lens MAP
----------------------------------------------------

Gauss-Newton is the undamped limit of Levenberg-Marquardt: it solves the normal
equations ``J^T J dz = -J^T r`` each iteration, taking a full second-order step
without Levenberg damping. It exploits the same exact residual least-squares
structure as ``jaxopt_lm_multistart.py`` (see that file's header for the
derivation; residual = imaging ``normalized_residual_map`` + Gaussian-prior
block, ``0.5||r||^2`` sharing its argmin with the true MAP).

INFEASIBLE — verified outcome (kept as the documented attempt)
-------------------------------------------------------------

Gauss-Newton, like Levenberg-Marquardt, needs the **forward-mode** residual
Jacobian (``jacfwd`` / ``jvp``). The positive-only source solve
(``autoarray.util.jax_nnls.solve_nnls_primal``) is reverse-mode-only
(``custom_vjp``), so GN raises the same blocker:

    TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function

See ``jaxopt_lm_multistart.py`` for the full discussion. The strongest
second-order method compatible with this reverse-mode-only likelihood is BFGS
(``jaxopt_bfgs_multistart.py``), which needs only the gradient. **Self-
terminating** on tolerance, capped at ``MAXITER``.

Run:  MULTISTART_N_STARTS=128 python -m searches_minimal.jaxopt_gaussnewton_multistart

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
print(
    f"Model free parameters: {obj.ndim}  |  N_STARTS = {N_STARTS}  |  GN maxiter={MAXITER} tol={TOL}"
)

time_compile(obj)
z_starts, n_kept = make_z_starts(obj, N_STARTS)
print(f"Collected {n_kept} finite-gradient z-starts")

solver = GaussNewton(residual_fun=obj.residual_z_raw, maxiter=MAXITER, tol=TOL)

try:
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
except Exception as e:  # noqa: BLE001 — expected forward-mode/custom_vjp blocker
    print("\n================ GAUSS-NEWTON INFEASIBLE ================")
    print(f"{type(e).__name__}: {str(e)[:300]}")
    print(
        "Gauss-Newton needs the forward-mode residual Jacobian; the NNLS\n"
        "positive-only inversion is reverse-mode-only (custom_vjp). See\n"
        "jaxopt_lm_multistart.py header.\n"
        "========================================================"
    )
