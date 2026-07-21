# Gradient SMC findings — BlackJAX adaptive-tempered SMC with a MALA/HMC inner kernel

Stage (a) of the JAX-native posterior sampler wave. Script:
`searches_minimal/blackjax_smc_grad.py`. Scope: MGE parametric imaging
likelihood only (gradient-certified `OK_HMC_VIABLE` by `probe_grad.py`);
pixelized deferred.

## Goal

Upgrade the gradient-free `blackjax_smc.py` (RWM inner kernel, cube-space via a
non-differentiable `pure_callback`) to a **gradient inner kernel** (MALA / HMC)
sampling in **physical parameter space**, keeping SMC's free log-evidence (sum
of tempering `log_likelihood_increment`). Benchmark vs the Nautilus baseline
(`output/comparison.txt`: nautilus_jax ~ −169k smoke, nss_grad ~ −31).

## Status: wiring PROVEN; naive gradient SMC does NOT converge on this problem

The machinery runs end-to-end in float64 (physical/whitened-space MALA & HMC,
JAX-native log-prior, adaptive tempering, log-evidence, `--tune`). But **naive
MALA adaptive-tempered SMC fails to converge** on the MGE likelihood — see the
step-size study below. This is the informative result of stage (a).

## What was built

- **Physical-space differentiable likelihood.** `instance_from_vector(vector,
  xp=jnp)` → `analysis.log_likelihood_function` (the `probe_grad.py` form), no
  `pure_callback`, so `jax.grad` flows.
- **JAX-native physical-space log-prior** over the 15 params: Gaussian /
  TruncatedGaussian → `-0.5·((x-μ)/σ)²`, Uniform → flat-in-bounds; `-inf` out of
  bounds (drives rejection). Normalisation dropped (cancels in MH).
- **Whitening** `z = params / prior_scale` so a single scalar MALA step is
  dimensionless across heterogeneous parameter ranges; HMC uses identity mass.
- **`--tune`** wraps `blackjax.smc.inner_kernel_tuning` for per-temperature step
  adaptation; **`--kernel hmc`** swaps MALA → HMC.

## Bugs found and fixed en route

1. **Non-finite gradient at the "prior mean".** The vector of prior `.mean`
   attributes is a degenerate MGE configuration. Warm-up / finite-check moved to
   the cube-0.5 median (the `probe_grad.py`-certified point).
2. **Profile-centre coordinate singularity.** Light/mass profile gradients are
   `1/r`-singular at the centre `(y,x)=(0,0)`; the symmetric prior median sits
   exactly on it (`probe_grad.py` only ever ran on GPU, so this was latent).
   Fixed with a `custom_jvp` that **masks non-finite gradient entries to zero**
   (the campaign's "NaN-gradient degenerate points need masking" constraint) —
   the forward value is unchanged; random particles almost never hit the
   singularity, and the mask keeps the kernel stable if one wanders near.
3. **MALA step must be scalar.** blackjax MALA takes a scalar `step_size` (no
   metric arg); a per-dimension array step breaks the MH acceptance
   (`Pred must be a scalar`). Whitening (above) supplies the preconditioning.
4. **`inner_kernel_tuning` callback signature.** blackjax calls
   `mcmc_parameter_update_fn(rng_key, state, info)` (3 args) and passes the
   stepped SMC state (with `.particles`), not the previous parameter override —
   so the step must be recomputed absolutely from `state.particles`, not nudged
   multiplicatively from the previous value.

## Environment traps

- **float64 is mandatory and not automatic on RAL.** JAX defaults to float32;
  float64 needs `JAX_ENABLE_X64`, which is ambient in the local shell but **not
  inherited by an sbatch job**. A silent float32 run (only truncation warnings)
  wasted job 330958. Every RAL sbatch must `export JAX_ENABLE_X64=True`; verify
  `grep -c "truncated to dtype float32" <job>.err` == 0.
- **XLA/LLVM compile-memory wall.** The vmapped-over-particles, scanned-over-MCMC
  gradient graph is large; 128 particles × 8 MCMC steps hit
  `LLVM ERROR: Unable to allocate section memory` (compiler memory, not node
  RAM — the node has 921 GB). 64 particles × 3 MCMC steps compiles.
- **Laptop cannot host it.** ~15 GB caps the vmapped MGE-inversion gradient at
  ~4 particles (16 OOMs); convergence work must run on RAL/GPU.

## Step-size study (RAL CPU, float64, 64 particles, job 330962 + 330959)

The MGE likelihood is ~1000× sharper than the prior, so the prior→posterior
tempering path demands the inner-kernel step shrink by ~1000× along the way.
No static or naively-adaptive step tracks this:

| Step regime | Behaviour | Verdict |
|-------------|-----------|---------|
| Fixed 0.02 (whitened) | acc 0.44 → 0.000 by step ~15; λ stalls at 0.001; max-logL never improves | collapse |
| Fixed 0.001 (tiny) | acc holds ~0.8 early; **max-logL improves −165k → −121k**; then collapses by step ~17, crashes | no completion, but gradient IS useful |
| Spread-adaptive `2.38/√d · std(particles)` (`--tune`) | step too large (~0.6); acc frozen at 0.000 from step 2; tempering force-jumps λ 0.0004→1.0 → **false "converged"**, garbage logZ ≈ max-logL | collapse (false positive) |

**Key reading:** the tiny-fixed-step arm improving −165k → −121k before collapse
shows the gradient information genuinely pulls toward better fits — the method is
not dead. The blocker is **step-size scheduling across an extreme prior→posterior
concentration**, which is set by the *tempered-likelihood curvature*, not by the
prior scale or the current particle spread.

**Caution — the `Converged: yes` line is weak:** it only means tempering reached
λ=1.0. When acceptance is ~0 throughout, adaptive tempering can force a single
huge λ increment to 1.0, yielding a meaningless logZ. Judge convergence by the
acceptance trace and max-logL progression, not that line.

## Next levers (not yet tried)

1. **Warm-start particles from the multi-start Adam basin** (the prompt's stage-c
   idea): initialise near the mode so the tempering path is short and the
   prior→posterior step-scale gap never has to be crossed. Most promising.
2. **Acceptance-rate feedback step control** (dual averaging / blackjax
   `pretuning`): thread the previous step size and target ~0.574 (MALA) / ~0.65
   (HMC) — needs an outer loop or blackjax pretuning, since the standard
   `inner_kernel_tuning` callback does not expose the previous step.
3. **HMC with a posterior-matched mass matrix** (`--kernel hmc`): estimate the
   metric from a warm-start cloud rather than identity.
4. **A100 representative timing** once GPUs free (currently queued full); CPU
   ms/eval here (~190 ms at 64 particles) is not representative.
