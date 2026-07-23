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

## Headline: cold-starting was the wrong test; warm-started, the solution is found

**The representative regime is warm-started.** These samplers are meant to be
handed a starting point near the maximum-likelihood solution by a JAX optimizer,
not run cold from the prior. Cold-benchmarking them is not a fair test — and on
this problem it is not even close:

| | max log L |
|---|---|
| prior median | −159,736 |
| every cold sampler arm (all step regimes) | −158k … −180k |
| **multi-start Prodigy optimum** | **+31,788** |

The cold runs never got within **~190,000 log-units** of the solution; the
optimizer closes that entire gap in ~200 steps. The acceptance collapse
characterised below is the sampler failing to cross a gap a gradient optimizer
crosses trivially.

Warm-started (RAL job 331035, float64, 64 particles), the picture inverts:

- multi-start Prodigy: max log L **31787.93**, and the MLE's
  **einstein_radius = 1.5997 against a truth of 1.6** — essentially exact.
- Laplace reference scale **held** (no fallback): per-parameter posterior widths
  sigma ~ 2e-4 … 2e-3 (physical).
- Tempering advanced smoothly lambda 0.006 -> 1.0 in 15-16 steps (vs the cold
  runs stalling at lambda ~ 1e-3), giving log Z ~ 31701 — a sane evidence number.

### RESOLVED: it now samples

Warm-started gradient SMC works end to end:

```
Auto step size from POSTERIOR width (sigma=0.5, ref sigma=1): 0.2871
  step 1: lambda=0.108  max log L=31770.41  acc rate=0.799
  step 2: lambda=0.206  max log L=31780.07  acc rate=0.781
  step 3: lambda=0.329                      acc rate=0.754
  step 4: lambda=0.523                      acc rate=0.523
  step 5: lambda=0.664  max log L=31781.57  acc rate=0.413
  step 6: lambda=0.849                      acc rate=0.262
  step 7: lambda=1.000                      acc rate=0.166
Best fit: einstein_radius = 1.5998   (truth 1.6)
```

Acceptance is healthy, particles genuinely move, max log L climbs toward the
Prodigy MLE (31787.93). Getting there required **three compounding fixes**, each
measured rather than guessed:

1. **Prior-whitening does not whiten the posterior.** Measured **269x
   anisotropy** in prior-whitened coordinates — ``einstein_radius`` has prior
   scale 8.0 but posterior std 2e-4 (sigma_z 5.3e-5) against 1.4e-2 for the
   loosest parameter. No scalar step can serve that spread: one tuned to the mean
   is ~88x too large for the tightest parameter. Fix: whiten by the warm-start
   (posterior) scale.
2. **Diagonal whitening is not enough — the posterior is correlated.** Laplace
   covariance condition number **568**, with |r| = 0.95 between two parameters
   (two pairs above 0.9). Diagonal whitening leaves a thin *tilted* ridge that a
   spherical proposal walks off. Fix: whiten with the **Cholesky factor of the
   full covariance**, adding ``log|det L|`` to the reported evidence.
3. **The step targeted the reference width, not the posterior width.** The
   reference is deliberately inflated (``--ref-inflate``) so it covers the
   posterior; in whitened units the reference has sigma 1 but the posterior has
   sigma ~1/inflate, so scaling to sigma=1 overshoots by ``inflate^2``. Measured
   directly: ``eps=1.148 -> acc 0.00``, ``eps=0.1 -> acc 0.94``. Fix: auto step
   targets the posterior width (``eps=0.287`` at inflate=2).

**Remaining refinement:** acceptance declines 0.80 -> 0.17 as lambda -> 1, which
is exactly what a *fixed* step does as the target sharpens — this is the job of
``--tune`` (per-temperature step adaptation), still to be validated at scale.

### The earlier units error (also real, also fixed)

> **MALA units trap.** MALA proposes ``x + eps*grad + sqrt(2*eps)*xi``, so
> ``eps`` is a **squared length**; the proposal length is ``sqrt(2*eps)``, not
> ``eps``. Setting ``eps ~ sigma`` overshoots by ~1/sigma. Job 331035 ran
> ``eps = 0.0029`` against ``sigma = 0.005``, i.e. proposal length 0.076 — about
> **16x wider than the posterior** — hence zero acceptance at every temperature.
> Correct scaling: ``ell = 2.38 * d^(-1/6) * sigma``, ``eps = ell^2 / 2``
> (for HMC the step *is* a length, so it scales as sigma directly). The same
> trap applied to the ``--tune`` spread rule; both now go through
> ``auto_step_from_scale``.

Next run should show non-zero acceptance and a genuinely mixed posterior.

## Earlier cold-start study: wiring PROVEN; naive gradient SMC does NOT converge

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
