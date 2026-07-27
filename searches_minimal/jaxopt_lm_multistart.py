"""
Multi-start Levenberg-Marquardt (jaxopt) — HST MGE lens MAP
-----------------------------------------------------------

Second-order MAP optimizer exploiting the **exact residual least-squares**
structure of this objective. Verified on this model:

    analysis.log_likelihood_function == -0.5 * (chi^2 + noise_norm)

(no regularization / Occam log-det term — the MGE light is linear and there is
no pixelization). Hence

    -(log L + log prior) = 0.5*chi^2 + 0.5*noise_norm - log_prior

is an exact sum-of-squares up to theta-independent constants, with residual

    r(z) = [ (data - model_image(theta))/noise   (imaging, 15361 residuals),
             (theta_k - mu_k)/sigma_k             (one per Gaussian prior)   ]

built by ``obj.residual_z_raw`` (imaging block = ``FitImaging.normalized_
residual_map``; theta = physical params of the unconstrained coordinate z).
``0.5||r||^2`` shares its argmin with the true MAP. LM adds Levenberg damping to
Gauss-Newton, so it interpolates between GN and gradient descent — the natural
second-order method here. Run as a **wide multi-start** (same broad z-starts as
multi-start Adam) so the population still supplies basin diversity; LM supplies
fast second-order descent within each basin. **Self-terminating** on the
gradient/step tolerance, capped at ``MAXITER``.

INFEASIBLE — verified outcome (kept as the documented attempt)
-------------------------------------------------------------

Levenberg-Marquardt is **not applicable** to the production objective on this
model. jaxopt builds the residual Jacobian with **forward-mode** autodiff
(``jacfwd`` / ``jvp``), but the positive-only source solve
(``autoarray.util.jax_nnls.solve_nnls_primal``) is defined with a **reverse-mode
custom gradient** (``custom_vjp``) and has no forward rule, so LM raises:

    TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function

This is the concrete form of the "non-smoothness caused by the non-negative
linear amplitudes" the benchmark set out to identify: the NNLS active set makes
the map z -> reconstruction only reverse-mode differentiable, which rules out
every least-squares Gauss-Newton/LM method (they all need the forward Jacobian).
The gradient itself is fine (reverse mode, FD-verified), so first-order and
quasi-Newton methods (Adam, L-BFGS, BFGS, NCG) all work — see those scripts. The
strongest *compatible* second-order method is therefore BFGS, not LM.

This script attempts the run and reports the blocker cleanly (exit 0) so a batch
sweep continues. Run:  MULTISTART_N_STARTS=128 python -m searches_minimal.jaxopt_lm_multistart

Requirements: jaxopt (JAX).
"""

import os

import jax
from jaxopt import LevenbergMarquardt

from searches_minimal._grad_setup import (
    build_map_objective,
    make_z_starts,
    run_vmapped_map_solver,
    time_compile,
)

N_STARTS = int(os.environ.get("MULTISTART_N_STARTS", "128"))
MAXITER = int(os.environ.get("LM_MAXITER", "100"))
TOL = float(os.environ.get("LM_TOL", "1e-3"))

print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
obj = build_map_objective()
print(
    f"Model free parameters: {obj.ndim}  |  N_STARTS = {N_STARTS}  |  LM maxiter={MAXITER} tol={TOL}"
)

time_compile(obj)  # warm the shared value_and_grad (also used by make_z_starts)
z_starts, n_kept = make_z_starts(obj, N_STARTS)
print(f"Collected {n_kept} finite-gradient z-starts")

solver = LevenbergMarquardt(residual_fun=obj.residual_z_raw, maxiter=MAXITER, tol=TOL)

try:
    run_vmapped_map_solver(
        obj=obj,
        solver=solver,
        z_starts=z_starts,
        name=f"jaxopt_lm_n{n_kept}",
        title=f"multi-start LM ({n_kept}x)",
        max_iters=MAXITER,
        termination=f"self-terminating (LM tol={TOL}, cap {MAXITER} iters)",
        is_residual=True,
    )
except Exception as e:  # noqa: BLE001 — expected forward-mode/custom_vjp blocker
    print("\n==================== LM INFEASIBLE ====================")
    print(f"{type(e).__name__}: {str(e)[:300]}")
    print(
        "Levenberg-Marquardt needs the forward-mode residual Jacobian; the NNLS\n"
        "positive-only inversion is reverse-mode-only (custom_vjp). See header.\n"
        "======================================================="
    )
