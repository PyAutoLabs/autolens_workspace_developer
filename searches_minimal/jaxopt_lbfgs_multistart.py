"""
Multi-start L-BFGS (jaxopt), properly scaled — HST MGE lens MAP
--------------------------------------------------------------

Quasi-Newton multi-start: L-BFGS builds an implicit inverse-Hessian from the
gradient history, so each start descends its basin far faster than first-order
Adam. The "proper scaling" the method needs comes from the **shared
unconstrained z-parameterization** (``obj.neg_log_posterior_z_raw``): a sigmoid
over each prior's inverse-CDF makes every coordinate O(1), removing the disparate
physical scales (einstein_radius ~ few vs ell_comps ~ 0.1 vs centres ~ arcsec)
that otherwise wreck a raw-space quasi-Newton curvature estimate. Wide multi-
start supplies the basin diversity; L-BFGS supplies the fast in-basin descent.
**Self-terminating** on the projected-gradient tolerance, capped at ``MAXITER``.

Contrast with the single-start ``jaxopt_lbfgs.py`` (one cold start, wrong basin)
— the difference here is purely the population of starts.

Run on the A100:

    MULTISTART_N_STARTS=128 python -m searches_minimal.jaxopt_lbfgs_multistart

Requirements: jaxopt (JAX).
"""

import os

import jax
from jaxopt import LBFGS

from searches_minimal._grad_setup import (
    build_map_objective,
    make_z_starts,
    run_vmapped_map_solver,
    time_compile,
)

N_STARTS = int(os.environ.get("MULTISTART_N_STARTS", "128"))
MAXITER = int(os.environ.get("LBFGS_MAXITER", "200"))
TOL = float(os.environ.get("LBFGS_TOL", "1e-3"))

print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
obj = build_map_objective()
print(f"Model free parameters: {obj.ndim}  |  N_STARTS = {N_STARTS}  |  L-BFGS maxiter={MAXITER} tol={TOL}")

time_compile(obj)
z_starts, n_kept = make_z_starts(obj, N_STARTS)
print(f"Collected {n_kept} finite-gradient z-starts")

# implicit_diff=False: we read the solved params, never differentiate through
# the solver, and the implicit-diff machinery blows up the vmapped XLA compile.
# backtracking line search keeps the while-loop body small (the default `zoom`
# search fuses the heavy MGE grad graph many times and compiles for >15 min).
solver = LBFGS(
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
    name=f"jaxopt_lbfgs_multistart_n{n_kept}",
    title=f"multi-start L-BFGS ({n_kept}x)",
    max_iters=MAXITER,
    termination=f"self-terminating (L-BFGS grad-tol={TOL}, cap {MAXITER} iters)",
    is_residual=False,
)
