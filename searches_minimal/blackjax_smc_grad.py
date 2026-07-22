"""
BlackJAX SMC with a GRADIENT inner kernel — physical-space MGE likelihood
-------------------------------------------------------------------------

Stage (a) of the JAX-native posterior sampler wave. Upgrades the gradient-free
``blackjax_smc.py`` (adaptive-tempered SMC with a random-walk Metropolis inner
kernel) to a **gradient inner kernel** — MALA by default, HMC behind
``--kernel hmc`` — feeding the certified MGE likelihood gradient into the
sampler.

Two design changes distinguish this from ``blackjax_smc.py``:

1. **Physical-parameter space, not unit-cube space.** ``blackjax_smc.py`` samples
   in ``[0, 1]^N`` and crosses a ``jax.pure_callback`` host boundary for the
   cube -> physical inverse-CDF on every likelihood eval — that boundary is
   *not* differentiable, so only a gradient-free kernel (RWM) can run there.
   Here we sample directly in physical space, where
   ``model.instance_from_vector(vector=params, xp=jnp)`` ->
   ``analysis.log_likelihood_function`` is a pure-JAX, ``jax.grad``-able stack
   (certified ``OK_HMC_VIABLE`` by ``probe_grad.py``). No per-eval host roundtrip.

2. **JAX-native physical-space log-prior.** Because we no longer sample the
   uniform cube, the prior density must be expressed in physical space. Each
   autofit prior contributes its own log-density: Gaussian /
   TruncatedGaussian -> ``-0.5 * ((x - mu) / sigma)^2`` (normalisation dropped;
   it cancels in Metropolis-Hastings), Uniform -> flat inside its bounds. Out of
   bounds the log-prior is ``-inf``, which drives rejection.

Adaptive tempering still anneals ``lambda`` from 0 (prior) to 1 (posterior),
and the log evidence is still recovered for free as the sum of
``info.log_likelihood_increment`` across temperatures — the reason SMC leads the
sampler wave: a gradient sampler that *also* yields ``log Z``.

Gradient safety (from prior JAX campaigns): out-of-bounds proposals are made
finite-and-zero-gradient by clipping the parameter vector to the prior bounds
before the likelihood call. ``clip`` has zero gradient outside the bounds, so
``jax.grad`` of the masked likelihood is finite everywhere; the ``-inf``
log-prior (not a NaN likelihood) is what rejects the proposal. This mirrors the
``safe_cube`` clip in ``blackjax_smc.py``.

This is a wiring + first-benchmark script, not a tuned production run —
particle count, step size, and ``num_mcmc_steps`` are conservative. Compare
against ``blackjax_smc.py`` (RWM), ``nautilus_jax.py`` and the ``nss_grad`` row
in ``output/comparison.txt``.

Usage:
    python searches_minimal/blackjax_smc_grad.py                 # MALA, fixed step
    python searches_minimal/blackjax_smc_grad.py --kernel hmc    # HMC inner kernel
    python searches_minimal/blackjax_smc_grad.py --tune          # adapt step size per temperature
    python -m searches_minimal.blackjax_smc_grad --warm-start     # temper from the Prodigy-MLE Gaussian reference (representative)

Requirements:
    pip install blackjax
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import blackjax
import blackjax.smc.resampling as resampling
import blackjax.mcmc.mala as mala
import blackjax.mcmc.hmc as hmc

from searches_minimal._metrics import MLTracker
from searches_minimal._setup import (
    build_analysis,
    build_dataset,
    build_model,
    format_best_fit,
)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--kernel", choices=["mala", "hmc"], default="mala", help="Gradient inner kernel."
)
parser.add_argument("--n-particles", type=int, default=256)
parser.add_argument("--num-mcmc-steps", type=int, default=5)
parser.add_argument("--target-ess", type=float, default=0.5)
parser.add_argument(
    "--step-size",
    type=float,
    default=0.01,
    help="Inner-kernel step size (dimensionless; scaled per-parameter by the "
    "prior mass matrix). Overridden per-temperature when --tune is set.",
)
parser.add_argument(
    "--hmc-integration-steps",
    type=int,
    default=8,
    help="Leapfrog steps per HMC trajectory (ignored for MALA).",
)
parser.add_argument(
    "--tune",
    action="store_true",
    help="Adapt the inner-kernel step size per temperature toward a target "
    "acceptance rate (inner_kernel_tuning).",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--warm-start",
    action="store_true",
    help="Temper from a normalised Gaussian reference centred on the cached "
    "multi-start (Prodigy) MLE instead of from the prior. This is the "
    "representative regime — these samplers are meant to be warm-started by a "
    "JAX optimizer — and it keeps log Z valid (see _warm_start.py).",
)
parser.add_argument(
    "--ref-inflate",
    type=float,
    default=2.0,
    help="Widen the Gaussian reference by this factor. The reference must be "
    "BROADER than the posterior or the tempering path misses mass.",
)
args = parser.parse_args()


# --------------------------------------------------------------------------
# Build the standard MGE imaging problem (pure-JAX analysis).
# --------------------------------------------------------------------------

dataset = build_dataset()
model = build_model()
analysis = build_analysis(dataset, use_jax=True)

ndim = model.prior_count
print(f"Model free parameters: {ndim}")


# --------------------------------------------------------------------------
# Prior arrays for the physical-space log-prior + mass matrix.
#
# Each autofit prior contributes a per-parameter (mean, sigma, lower, upper) and
# a Gaussian/uniform flag. Gaussian & TruncatedGaussian priors expose mean/sigma;
# Uniform priors expose only bounds (sigma=None -> sentinel 1.0, gauss flag 0).
# The per-parameter "natural" scale (sigma for Gaussians, half-width for
# Uniforms) doubles as the diagonal metric that pre-conditions the inner kernel,
# so the single scalar ``step_size`` is dimensionless across heterogeneous
# parameter ranges.
# --------------------------------------------------------------------------


def _prior_arrays(model):
    means, sigmas, lowers, uppers, is_gauss, scales = [], [], [], [], [], []
    for prior in model.priors_ordered_by_id:
        mean = float(getattr(prior, "mean", 0.0) or 0.0)
        sigma = getattr(prior, "sigma", None)
        lo = float(getattr(prior, "lower_limit", -np.inf))
        hi = float(getattr(prior, "upper_limit", np.inf))
        if sigma is not None and np.isfinite(sigma) and sigma > 0:
            means.append(mean)
            sigmas.append(float(sigma))
            is_gauss.append(1.0)
            scales.append(float(sigma))
        else:
            # Uniform (or degenerate) — flat density inside bounds.
            means.append(mean)
            sigmas.append(1.0)  # sentinel; masked out by is_gauss=0
            is_gauss.append(0.0)
            width = hi - lo
            scales.append(float(width) if np.isfinite(width) and width > 0 else 1.0)
        lowers.append(lo)
        uppers.append(hi)
    return (
        jnp.asarray(means),
        jnp.asarray(sigmas),
        jnp.asarray(lowers),
        jnp.asarray(uppers),
        jnp.asarray(is_gauss),
        np.asarray(scales, dtype=np.float64),
    )


P_MEAN, P_SIGMA, P_LOWER, P_UPPER, P_IS_GAUSS, P_SCALE = _prior_arrays(model)


def _log_prior_norm(model) -> float:
    """Total log normalisation constant of the prior density.

    Dropped for plain MH (it cancels), but it does NOT cancel in the evidence:
    log Z = log integral(prior * L) requires ``prior`` to be a normalised density.
    Uniform -> -log(width); Gaussian -> -log(sigma) - 0.5*log(2*pi);
    TruncatedGaussian -> the same minus log of the retained probability mass.
    """
    total = 0.0
    for prior in model.priors_ordered_by_id:
        sigma = getattr(prior, "sigma", None)
        lo = float(getattr(prior, "lower_limit", -np.inf))
        hi = float(getattr(prior, "upper_limit", np.inf))
        if sigma is not None and np.isfinite(sigma) and sigma > 0:
            mean = float(getattr(prior, "mean", 0.0) or 0.0)
            total += -np.log(float(sigma)) - 0.5 * np.log(2.0 * np.pi)
            if np.isfinite(lo) or np.isfinite(hi):
                cdf_hi = _norm_cdf((hi - mean) / float(sigma)) if np.isfinite(hi) else 1.0
                cdf_lo = _norm_cdf((lo - mean) / float(sigma)) if np.isfinite(lo) else 0.0
                mass = max(cdf_hi - cdf_lo, 1e-300)
                total += -np.log(mass)
        else:
            total += -np.log(hi - lo)
    return float(total)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))


LOG_PRIOR_NORM = _log_prior_norm(model)

# Whitening. Parameters span very different physical scales (einstein_radius
# ~ O(1) vs a centre coordinate ~ O(0.1)); a single scalar MALA/HMC step in raw
# physical units is therefore badly conditioned. We sample in a whitened space
# ``z = params / SCALE`` where every coordinate is O(1), so one scalar step size
# behaves sensibly across all parameters. blackjax's MALA takes only a scalar
# step (no metric argument), so this whitening — not an array step or a mass
# matrix — is what preconditions it. HMC then uses an identity mass matrix.
SCALE = jnp.asarray(P_SCALE)
INV_MASS_DIAG = jnp.ones(ndim)  # identity: the space is already whitened


# --------------------------------------------------------------------------
# Physical-space log-prior and log-likelihood. Both differentiable.
# --------------------------------------------------------------------------


def log_prior(params):
    """Sum of per-parameter physical-space prior log-densities.

    Gaussian term is finite everywhere (no NaN gradient); the ``-inf`` on the
    out-of-bounds branch is a constant and contributes zero gradient.
    """
    gauss = jnp.where(
        P_IS_GAUSS > 0.5, -0.5 * ((params - P_MEAN) / P_SIGMA) ** 2, 0.0
    )
    in_bounds = (params >= P_LOWER) & (params <= P_UPPER)
    per_param = jnp.where(in_bounds, gauss, -jnp.inf)
    return jnp.sum(per_param)


def _log_likelihood_core(params):
    """MGE imaging log-likelihood in physical space.

    Clip to prior bounds before the likelihood so out-of-bounds proposals get a
    finite value with zero gradient (``clip`` has zero gradient outside bounds).
    ``log_prior`` already returns ``-inf`` out of bounds, so the tempered density
    rejects the proposal regardless of the value returned here.
    """
    in_bounds = jnp.all((params >= P_LOWER) & (params <= P_UPPER))
    safe = jnp.clip(params, P_LOWER, P_UPPER)
    instance = model.instance_from_vector(vector=safe, xp=jnp)
    log_l = analysis.log_likelihood_function(instance=instance)
    return jnp.where(in_bounds, log_l, 0.0)


@jax.custom_jvp
def log_likelihood(params):
    """Gradient-masked wrapper around the MGE log-likelihood.

    The light/mass profile gradients are singular at the profile centre
    ``(y, x) = (0, 0)`` — the polar coordinate ``r = sqrt(y^2 + x^2)`` gives a
    ``1/r`` gradient that is NaN exactly at the origin (probed: the prior-median
    point sits on it by symmetry; random particles almost never do). Per the
    campaign constraint "NaN-gradient degenerate points need masking", the custom
    JVP replaces any non-finite gradient entry with zero, so a gradient inner
    kernel (MALA/HMC) stays numerically stable if a particle wanders onto or near
    a coordinate singularity. The forward value is unchanged.
    """
    return _log_likelihood_core(params)


@log_likelihood.defjvp
def _log_likelihood_jvp(primals, tangents):
    (params,) = primals
    (tangent,) = tangents
    value, grad = jax.value_and_grad(_log_likelihood_core)(params)
    grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
    return value, jnp.vdot(grad, tangent)


# --------------------------------------------------------------------------
# Whitened-space densities the sampler actually runs on: z = params / SCALE.
# The constant Jacobian ``log|SCALE|`` cancels in Metropolis-Hastings, so it is
# dropped. Gradients w.r.t. z are O(1)-scaled, so a single scalar step size mixes
# across all parameters.
# --------------------------------------------------------------------------


def log_prior_z(z):
    return log_prior(z * SCALE)


def log_likelihood_z(z):
    return log_likelihood(z * SCALE)


# --------------------------------------------------------------------------
# Warm-start reference bridge (the representative regime).
#
# These samplers are meant to be warm-started from a JAX optimizer's MLE. A cold
# prior->posterior run wastes its whole budget crossing the ~1000x scale gap
# between prior and posterior (see smc_gradient_findings.md).
#
# Simply dropping particles at the MLE would destroy SMC's log-evidence, which
# is only valid when tempering starts from a NORMALISED distribution. So instead
# we temper geometrically from a normalised Gaussian reference g centred on the
# MLE to the true posterior:
#
#     log target_lambda = (1-l)*log g + l*(log prior + log L)
#                       = log g + l*(log prior + log L - log g)
#
# which maps onto blackjax's (logprior_fn + lambda * loglikelihood_fn) as
#
#     logprior_fn      := log g                                (normalised)
#     loglikelihood_fn := log prior + log L - log g            (bridge weight)
#
# Because g integrates to 1, the accumulated tempering increments still estimate
# log Z = log integral(prior * L) — the true evidence, directly comparable to
# Nautilus. The prior here MUST carry its normalisation (LOG_PRIOR_NORM).
# --------------------------------------------------------------------------

if args.warm_start:
    from searches_minimal._warm_start import load_warm_start

    _ws = load_warm_start()
    REF_MU_Z = jnp.asarray(_ws.mle) / SCALE
    REF_SIGMA_Z = jnp.asarray(_ws.std * args.ref_inflate) / SCALE
    _REF_LOGNORM = float(
        -jnp.sum(jnp.log(REF_SIGMA_Z)) - 0.5 * ndim * math.log(2.0 * math.pi)
    )
    print(
        f"Warm start: optimizer={_ws.optimizer} max log L={_ws.log_l:.2f} "
        f"({_ws.n_converged}/{_ws.n_starts} converged, scale via {_ws.std_source}, "
        f"inflate={args.ref_inflate})"
    )

    def log_reference_z(z):
        """Normalised Gaussian reference in whitened space."""
        return _REF_LOGNORM - 0.5 * jnp.sum(((z - REF_MU_Z) / REF_SIGMA_Z) ** 2)

    def bridge_log_weight_z(z):
        """log prior + log L - log g  (the tempered 'likelihood' of the bridge)."""
        return (
            log_prior_z(z) + LOG_PRIOR_NORM + log_likelihood_z(z) - log_reference_z(z)
        )

    smc_logprior_fn = log_reference_z
    smc_loglikelihood_fn = bridge_log_weight_z
else:
    smc_logprior_fn = log_prior_z
    smc_loglikelihood_fn = log_likelihood_z


# --------------------------------------------------------------------------
# One-shot compile of value_and_grad so the (dominant) gradient-stack compile
# time is reported separately from sampling — mirrors probe_grad.py / the other
# scripts. This is the expensive step for the gradient path.
#
# Warm up + finite-check at the physical prior MEDIAN (cube 0.5 -> physical),
# the point probe_grad.py certifies OK_HMC_VIABLE. The vector of prior *.mean*
# attributes is NOT the same point (and can land on a measure-zero degenerate
# MGE configuration where the gradient is non-finite); the median is the sound
# reference and is where the initial particle cloud is centred.
# --------------------------------------------------------------------------

Z_MEDIAN = jnp.asarray(model.vector_from_unit_vector([0.5] * ndim)) / SCALE

print("JIT-compiling value_and_grad(log_likelihood_z) (one-shot)...", flush=True)
t_jit_start = time.time()
_vag = jax.jit(jax.value_and_grad(log_likelihood_z))
_ll0, _g0 = _vag(Z_MEDIAN)
_ = float(jax.block_until_ready(_ll0))
_g0 = np.asarray(jax.block_until_ready(_g0))
t_jit = time.time() - t_jit_start
print(f"  Compiled in {t_jit:.2f} s     (log L={float(_ll0):.2f})", flush=True)
if not np.all(np.isfinite(_g0)):
    raise RuntimeError(
        "Non-finite gradient at the whitened prior median even after masking — "
        "gradient path unsound."
    )


# --------------------------------------------------------------------------
# Gradient inner kernel: MALA (default) or HMC — both in whitened space.
#
# BlackJAX's SMC machinery vmaps the inner kernel across particles and unpacks a
# leading-dim-1 entry in ``mcmc_parameters`` as a shared parameter. MALA takes a
# scalar ``step_size`` (no metric argument — whitening does the preconditioning);
# HMC additionally takes ``inverse_mass_matrix`` (identity here, space already
# whitened) and ``num_integration_steps``.
# --------------------------------------------------------------------------

if args.kernel == "mala":
    _mala_kernel = mala.build_kernel()

    def mcmc_step_fn(rng_key, state, logdensity_fn, step_size):
        return _mala_kernel(rng_key, state, logdensity_fn, step_size)

    mcmc_init_fn = mala.init

else:  # hmc
    _hmc_kernel = hmc.build_kernel()

    def mcmc_step_fn(rng_key, state, logdensity_fn, step_size):
        return _hmc_kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            INV_MASS_DIAG,
            args.hmc_integration_steps,
        )

    mcmc_init_fn = hmc.init

# Shared scalar step, broadcast across particles via the leading dim of 1.
#
# Warm-started, the correct step scale is set by the REFERENCE width (which is
# the posterior scale), not by the prior. Auto-scale to it unless the user gave
# an explicit --step-size. This is what the cold runs could not do: their cloud
# sat at prior scale, so any step large enough to move was rejected outright.
effective_step = args.step_size
if args.warm_start and args.step_size == parser.get_default("step_size"):
    effective_step = float(
        (2.38 if args.kernel == "mala" else 1.0)
        / np.sqrt(ndim)
        * float(jnp.mean(REF_SIGMA_Z))
    )
    print(f"Auto step size from reference width: {effective_step:.4g}")

mcmc_parameters = {"step_size": jnp.asarray([effective_step])}


# --------------------------------------------------------------------------
# Assemble adaptive-tempered SMC in whitened space. Optionally wrap with
# inner_kernel_tuning to adapt the step size toward a target acceptance rate.
# --------------------------------------------------------------------------

smc = blackjax.adaptive_tempered_smc(
    logprior_fn=smc_logprior_fn,
    loglikelihood_fn=smc_loglikelihood_fn,
    mcmc_step_fn=mcmc_step_fn,
    mcmc_init_fn=mcmc_init_fn,
    mcmc_parameters=mcmc_parameters,
    resampling_fn=resampling.systematic,
    target_ess=args.target_ess,
    num_mcmc_steps=args.num_mcmc_steps,
)

# Optimal MALA/HMC step scaling. A fixed step cannot work here: the fixed-step
# arm collapses to zero acceptance because the MGE likelihood is ~1000x sharper
# than the prior, so as tempering concentrates the particle cloud the step must
# shrink with it. Tie the step to the cloud's current per-dimension spread
# (diagonal-covariance MALA), scaled by the dimension-optimal constant. This is
# absolute (recomputed from state.particles each temperature), which is what
# blackjax's inner_kernel_tuning API supports — the callback receives the stepped
# SMC state, not the previous step size.
STEP_SCALE = float((2.38 if args.kernel == "mala" else 1.0) / np.sqrt(ndim))


if args.tune:
    import blackjax.smc.inner_kernel_tuning as ikt

    def mcmc_parameter_update_fn(rng_key, state, info):
        # state.particles: (n_particles, ndim) in whitened space.
        spread = jnp.mean(jnp.std(state.particles, axis=0))
        new_step = jnp.clip(STEP_SCALE * spread, 1e-6, 1.0)
        return {"step_size": jnp.asarray([new_step])}

    tuned = ikt.as_top_level_api(
        smc_algorithm=blackjax.adaptive_tempered_smc,
        logprior_fn=smc_logprior_fn,
        loglikelihood_fn=smc_loglikelihood_fn,
        mcmc_step_fn=mcmc_step_fn,
        mcmc_init_fn=mcmc_init_fn,
        resampling_fn=resampling.systematic,
        mcmc_parameter_update_fn=mcmc_parameter_update_fn,
        initial_parameter_value=mcmc_parameters,
        num_mcmc_steps=args.num_mcmc_steps,
        target_ess=args.target_ess,
    )
    init_fn, step_fn = tuned.init, tuned.step
else:
    init_fn, step_fn = smc.init, smc.step


# --------------------------------------------------------------------------
# Initial particles: draw the unit cube uniformly and map to physical space via
# the autofit inverse-CDF ONCE, host-side (init only — the per-step hot path
# stays pure-JAX and gradient-native), then whiten (divide by SCALE) since the
# sampler runs in z-space.
# --------------------------------------------------------------------------

rng_key = jax.random.PRNGKey(args.seed)
rng_key, init_key = jax.random.split(rng_key)
cube0 = np.asarray(
    jax.random.uniform(init_key, shape=(args.n_particles, ndim), minval=0.0, maxval=1.0)
)
if args.warm_start:
    # Draw the initial cloud from the normalised Gaussian reference itself —
    # required for the bridge's log Z to be valid (the particles must be
    # distributed as g at lambda=0).
    initial_particles = REF_MU_Z + REF_SIGMA_Z * jax.random.normal(
        init_key, shape=(args.n_particles, ndim)
    )
else:
    physical0 = np.stack(
        [model.vector_from_unit_vector(list(c)) for c in cube0]
    ).astype(np.float64)
    initial_particles = jnp.asarray(physical0) / SCALE  # whitened
state = init_fn(initial_particles)


print(
    f"\nRunning BlackJAX SMC (adaptive tempered + {args.kernel.upper()}"
    f"{' + tuning' if args.tune else ''}) over {ndim} whitened dims "
    f"(n_particles={args.n_particles}, num_mcmc_steps={args.num_mcmc_steps}, "
    f"target_ess={args.target_ess}, step_size={args.step_size})..."
)
print("  JIT compile of the SMC step kernel happens on the first iteration.\n")

vmapped_log_l = jax.jit(jax.vmap(log_likelihood_z))

log_l_history: list[float] = []
log_z = 0.0
n_smc_steps = 0


def _tempering_param(state):
    # inner_kernel_tuning nests the SMC state under .sampler_state.
    inner = getattr(state, "sampler_state", state)
    return float(inner.tempering_param)


def _particles(state):
    inner = getattr(state, "sampler_state", state)
    return inner.particles


t_start = time.time()
while _tempering_param(state) < 1.0:
    rng_key, sub_key = jax.random.split(rng_key)
    state, info = jax.block_until_ready(step_fn(sub_key, state))
    log_z += float(info.log_likelihood_increment)
    n_smc_steps += 1

    log_l_step = vmapped_log_l(_particles(state))
    cur_max_log_l = float(jnp.max(log_l_step))
    log_l_history.append(cur_max_log_l)

    print(
        f"  step {n_smc_steps:3d}: lambda={_tempering_param(state):.4f}  "
        f"max log L={cur_max_log_l:.2f}  log Z (running)={log_z:.4f}  "
        f"acc rate={float(jnp.mean(info.update_info.acceptance_rate)):.3f}"
    )

t_elapsed = time.time() - t_start


# --------------------------------------------------------------------------
# Results — best fit from the final particle batch.
# --------------------------------------------------------------------------

final_particles = _particles(state)  # whitened z
final_log_l = vmapped_log_l(final_particles)
best_idx = int(jnp.argmax(final_log_l))
best_physical = np.asarray(final_particles[best_idx]) * np.asarray(SCALE)  # de-whiten
best_instance = model.instance_from_vector(vector=list(best_physical))
max_logl = float(jnp.max(final_log_l))

# Likelihood eval count: per SMC step each particle runs num_mcmc_steps inner
# updates. HMC evaluates the gradient num_integration_steps times per update;
# MALA once. Plus the initial tempered weights and per-step max-tracking.
inner_evals = args.num_mcmc_steps * (
    args.hmc_integration_steps if args.kernel == "hmc" else 1
)
n_likelihood_calls = args.n_particles + n_smc_steps * args.n_particles * (
    inner_evals + 1
)

evals_to_ml, time_to_ml = MLTracker.from_log_l_history(
    log_l_history, total_sampling_time=t_elapsed, tolerance=1.0
)

kernel_label = args.kernel.upper() + (" + tuning" if args.tune else "")
summary = f"""\
--- BlackJAX SMC (adaptive tempered + {kernel_label}, whitened sampling) Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_logl:.4f}
Log evidence:    {log_z:.4f}     (sum of SMC log_likelihood_increment over temperatures)

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (excludes JIT compile, run ahead of time)
Sampling time:       {t_elapsed:.2f} s     (no separate warmup phase)
JIT compile time:    {t_jit:.2f} s     (one-shot value_and_grad warm-up)
Likelihood evals:    {n_likelihood_calls}     (upper bound; inner={inner_evals}/particle/step)
Time per eval:       {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms
ESS:                 n/a (SMC adaptive tempered targets ESS at each step)
Posterior samples:   {args.n_particles}     (final tempered particles)
Sampler config:      kernel={kernel_label}, start={'warm(Gaussian ref, inflate=' + str(args.ref_inflate) + ')' if args.warm_start else 'cold(prior)'}, n_particles={args.n_particles}, num_mcmc_steps={args.num_mcmc_steps}, target_ess={args.target_ess}, step_size={effective_step:.4g}, n_smc_steps={n_smc_steps}

--- Convergence ---
Converged:           yes (tempering reached lambda=1.0)
Evals to ML:         {evals_to_ml if evals_to_ml is not None else 'n/a'}     (SMC step index within 1 nat of max)
Time to ML:          {f'{time_to_ml:.2f} s' if time_to_ml is not None else 'n/a'}
"""

print()
print(summary)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
suffix = f"_{args.kernel}" + ("_tuned" if args.tune else "") + ("_warm" if args.warm_start else "")
summary_path = output_dir / f"{Path(__file__).stem}{suffix}_summary.txt"
summary_path.write_text(summary)
print(f"Summary written to: {summary_path}")
