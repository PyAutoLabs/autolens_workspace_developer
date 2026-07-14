"""
NumPyro SVI + ADABelief — paper-faithful (Enzi et al. 2026 / Herculens)
----------------------------------------------------------------------

Stochastic Variational Inference with a mean-field Gaussian guide and the
ADABelief optimizer — the exact inference recipe of Enzi et al. 2026
(arXiv:2606.30620) for JAX strong-lens source reconstruction in Herculens
(Galan et al. 2022). Unlike the point optimizers in this folder, SVI returns a
(variational) **posterior**: a Gaussian approximation whose mean is a MAP-like
point estimate and whose scale gives parameter uncertainties. It optimizes the
ELBO (a lower bound on the log evidence).

The likelihood factor is the same pure-JAX MGE ``log_likelihood`` used by the
point optimizers; the priors are expressed as the matching NumPyro
distributions (Uniform / TruncatedNormal / Normal) so SVI runs in physical
parameter space with gradients flowing end-to-end.

**Initialisation.** Enzi et al. use ``init_to_median``. On this model the prior
median is a degenerate point (ell_comps / external shear medians are exactly 0,
where the arctan2/sqrt gradients are singular — see gradient_optimizer_findings.md),
so we initialise the guide diffusely (``init_to_uniform``) to avoid the NaN.
This is the same degeneracy the Phase-1 point optimizers had to perturb around.

Run from the workspace root:

    python -m searches_minimal.svi_adabelief

Requirements: numpyro, optax (JAX).
"""

import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import optax_to_numpyro

from searches_minimal._grad_setup import build_map_objective, write_grad_summary
from searches_minimal._metrics import MLTracker

# --- config -----------------------------------------------------------------
N_STEPS = 800
LEARNING_RATE = 1e-2
INIT_SCALE = 0.05  # keep initial guide samples near the finite init location
MAX_GRAD_NORM = 1.0  # clip the large MAP gradient before the ADABelief step
EINSTEIN_TRUTH = 1.6

obj = build_map_objective()
priors = obj.model.priors_ordered_by_id
print(f"Model free parameters: {obj.ndim}")


# Raw (UN-jitted) likelihood closure. numpyro's SVI jits the whole ELBO itself,
# so we must hand it the plain function — passing the pre-jitted
# ``obj.log_likelihood`` triggers a pathological jit-inside-jit compile.
def _log_l_raw(theta):
    instance = obj.model.instance_from_vector(vector=theta, xp=jnp)
    return obj.analysis.log_likelihood_function(instance=instance)


def _numpyro_dist(p):
    """Map an autofit prior to the matching NumPyro distribution."""
    name = type(p).__name__
    if name == "UniformPrior":
        return dist.Uniform(p.lower_limit, p.upper_limit)
    if name == "TruncatedGaussianPrior":
        return dist.TruncatedNormal(
            p.mean, p.sigma, low=p.lower_limit, high=p.upper_limit
        )
    if name == "GaussianPrior":
        return dist.Normal(p.mean, p.sigma)
    # Fallback: treat as uniform over its (finite) support.
    return dist.Uniform(p.lower_limit, p.upper_limit)


_dists = [_numpyro_dist(p) for p in priors]


def numpyro_model():
    thetas = [numpyro.sample(f"p{i}", d) for i, d in enumerate(_dists)]
    theta = jnp.stack(thetas)
    numpyro.factor("loglike", _log_l_raw(theta))


# Finite physical cold start. The raw prior median is degenerate (ell_comps /
# shear at 0), and a diffuse unconstrained init (init_to_uniform) puts the
# tightly-scaled ell_comps at unphysical |x|>1 -> NaN. Instead, initialise the
# guide at a mild perturbation of the median (unit 0.53), which is finite and
# physical -- the same cold start the point optimizers use, for a fair compare.
_u0 = np.full(obj.ndim, 0.53)
x0_init = np.asarray(obj.model.vector_from_unit_vector(list(_u0)))
init_values = {f"p{i}": float(x0_init[i]) for i in range(obj.ndim)}

guide = AutoNormal(
    numpyro_model,
    init_loc_fn=init_to_value(values=init_values),
    init_scale=INIT_SCALE,
)
# ADABelief with the large MAP gradient clipped (as in the optax point runs).
optimizer = optax_to_numpyro(
    optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adabelief(LEARNING_RATE))
)
svi = SVI(numpyro_model, guide, optimizer, loss=Trace_ELBO())

print(f"\nRunning SVI (AutoNormal + ADABelief) for {N_STEPS} steps...")
print("  JIT compile happens on the first step (~minutes on CPU).")
t_start = time.time()
svi_result = svi.run(jax.random.PRNGKey(0), N_STEPS, progress_bar=False)
wall_s = time.time() - t_start

losses = np.asarray(svi_result.losses)  # ELBO loss per step (= -ELBO)
elbo_history = [-float(x) for x in losses]  # ELBO (higher is better)

# Posterior: mean-field Gaussian. median() gives the location (point estimate).
post = guide.median(svi_result.params)
theta_mean = jnp.stack([post[f"p{i}"] for i in range(obj.ndim)])

# Posterior scales (marginal std per parameter), for the uncertainty report.
quantiles = guide.quantiles(svi_result.params, [0.16, 0.84])
theta_lo = np.array([float(quantiles[f"p{i}"][0]) for i in range(obj.ndim)])
theta_hi = np.array([float(quantiles[f"p{i}"][1]) for i in range(obj.ndim)])
theta_std = (theta_hi - theta_lo) / 2.0

max_log_l = float(obj.log_likelihood(theta_mean))
best_instance = obj.model.instance_from_vector(vector=list(np.asarray(theta_mean)))
r_e = float(best_instance.galaxies.lens.mass.einstein_radius)
r_e_idx = 8  # einstein_radius index (see prior dump)

print(f"\nPosterior mean log L: {max_log_l:.2f}")
print(
    f"einstein_radius = {r_e:.3f} +/- {theta_std[r_e_idx]:.3f}  "
    f"(truth {EINSTEIN_TRUTH}; {'in basin' if abs(r_e - EINSTEIN_TRUTH) < 0.3 else 'WRONG basin'})"
)
print(f"final ELBO = {elbo_history[-1]:.2f}")

write_grad_summary(
    name="svi_adabelief",
    title="numpyro SVI ADABelief",
    obj=obj,
    best_params=theta_mean,
    log_posterior_history=elbo_history,
    wall_s=wall_s,
    compile_s=0.0,  # SVI folds compile into the first step; not separable here
    warm_ms_per_eval=wall_s / max(N_STEPS, 1) * 1e3,
    n_evals=N_STEPS,
    n_iters=N_STEPS,
    converged=(abs(r_e - EINSTEIN_TRUTH) < 0.3),
    config_line=(
        f"AutoNormal guide (init_scale={INIT_SCALE}), ADABelief lr={LEARNING_RATE} "
        f"clip={MAX_GRAD_NORM}, {N_STEPS} steps, init_to_value(unit 0.53 cold start); "
        f"posterior r_E={r_e:.3f}+/-{theta_std[r_e_idx]:.3f}"
    ),
)
