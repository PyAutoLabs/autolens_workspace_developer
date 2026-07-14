# JAX gradient-based optimizers on the MGE lens likelihood

Benchmark of JAX gradient-based **point** optimizers (maximum-a-posteriori /
MLE) against the HST MGE lens-light + source imaging likelihood used by the
samplers in this folder (`_setup.py`: 20-Gaussian MGE bulge + Isothermal mass +
external shear lens, 20-Gaussian MGE source). Where the nested / ensemble
samplers here map the whole posterior, these scripts ask a narrower question:
**which JAX gradient optimizer reaches a good fit, robustly, and fast?**

## Why gradient optimizers here

The MGE likelihood on `AnalysisImaging(use_jax=True)` is differentiable
end-to-end (`jax_profiling/gradient/imaging/mge.py`). The MGE light amplitudes
are **linear** and solved by the inversion, so the free *nonlinear* parameter
space is small (**15-D**: lens mass + shear + the shared MGE nonlinear
parameters). A small, smooth, differentiable space is exactly where
gradient-based optimization should beat gradient-free nested sampling on
time-to-solution.

## Grounding in the literature

- **Herculens** (Galan et al. 2022), the reference JAX strong-lens code, uses
  **Adam** (optax) for gradient-informed optimization.
- **Enzi et al. 2026** (arXiv:2606.30620) do JAX source reconstruction in
  Herculens via **NumPyro SVI** with the **ADABelief** optimizer, initialised
  with `init_to_median`; they deliberately skip HMC.

So Adam and ADABelief are the field-standard first-order choices; L-BFGS
(quasi-Newton) and Levenberg-Marquardt (Gauss-Newton, exploiting the
least-squares structure of chi-squared) are the natural higher-order
contrasts.

## The objective (MAP)

All point optimizers minimise the negative log-posterior in the **physical**
parameter vector:

```
-(log_likelihood(theta) + sum(log_prior(theta)))
```

`log_prior` is autofit's own JAX-traceable term
(`model.log_prior_list_from_vector(vector, xp=jnp)` — the same one `af.Fitness`
adds when `fom_is_log_likelihood=False`). Optimizing in physical space (not the
unit cube) keeps the prior explicit, so this is a genuine MAP estimate, not an
MLE.

## Key gotcha — the prior median is a degenerate start

The **raw prior median** is a NaN-gradient point: `ell_comps` and external
shear sit at exactly `(0, 0)`, where the `arctan2` / `sqrt` that map them to
(angle, magnitude) have singular gradients. The value is finite there, but the
gradient is not, so a cold-started optimizer diverges on step one. Decomposed
diagnosis (one compile):

| start point | grad(logL) | grad(log prior) |
|---|---|---|
| prior median | **NaN** | ok |
| median + unit perturbation | ok | ok |
| narrow-band random draw | ok | ok |

This is **not** NNLS gradient-poisoning — every perturbed point gives finite
gradients for both terms. `_grad_setup.robust_cold_start()` therefore perturbs
in **unit-cube** space (stays inside every prior, nudges the degenerate
parameters off zero) with a post-compile finiteness retry. This is the same
perturbation trick `jax_profiling/gradient/imaging/mge.py` uses.

## Runtime characterisation (CPU, this laptop)

- One-shot JIT compile of `value_and_grad`: **~280 s** (WSL2 GPU JAX
  unavailable; this is the CPU cost and dominates a short run).
- Post-compile forward+gradient eval: **~198 ms/eval**.
- Raw MAP gradient norms are large (~1e5), so the optax runs chain
  `optax.clip_by_global_norm` before Adam/ADABelief.

## Results

_Phase 1 (single cold-start MAP point optimizers). Modest CPU budgets — see each
`output/<name>_summary.txt` for the exact config. `einstein_radius` truth ≈ 1.6._

| Optimizer | Compile (s) | ms/eval | Iters | Wall (s) | Max log L | einstein_radius | Converged |
|-----------|------------:|--------:|------:|---------:|----------:|----------------:|:---------:|
| optax Adam (MAP)       | 310.7 | 231.7 | 187 |  374.0 | −158002.3 | 4.89 | plateau |
| optax ADABelief (MAP)  | 276.7 | 199.5 | 196 |  329.3 | −157998.7 | 5.01 | plateau (NaN @200) |
| jaxopt L-BFGS (MAP)    | 298.4 | 172.4 | 200 | 1037.3 | −158004.9 | 4.42 | no (maxiter) |

`ms/eval` is the post-compile single forward+gradient cost; note L-BFGS does a
line search per iteration, so its *wall* time per iteration (~3.7 s) is much
higher than one eval — the 200 iters cost ~12× the optax runs.

### Phase 2 — escaping the cold-start basin

_`einstein_radius` truth ≈ 1.6; a "good fit" here has **positive** log L (the
noise-normalisation term dominates when χ² ≪ N_pixels)._

| Method | Wall (s) | Compile (s) | Evals | ms/eval | Max log L | einstein_radius | Robust? |
|--------|---------:|------------:|------:|--------:|----------:|----------------:|:-------:|
| **multi-start Adam** (12×) | ~1260 | ~573† | 3600 | ~192 | **+31787.9** | **1.600** | **2/12 starts → truth** |
| numpyro SVI (ADABelief) | 107 | folded‡ | 800 | 134 | −158022.3 | 3.54 ± 0.07 | ✗ wrong basin |
| jaxopt Levenberg-Marquardt | _see note_ | — | — | — | — | — | — |

† multi-start pays **two** JIT compiles: the single-graph `value_and_grad`
(start-filtering) and the batched `vmap`-over-12 graph (the loop). Per
single-start eval is ~192 ms — the same likelihood as everything else. (The raw
`multi_start_adam_summary.txt` "ms/eval" from an early run mis-timed this by
folding the batched compile into the warm measurement; the script now times the
two compiles separately.)

**Multi-start Adam is the robust-and-fast winner.** Twelve broad random starts,
Adam, 300 steps: 2 of them land in the **correct** basin and recover
`einstein_radius = 1.600`, `centre = (0.00, 0.00)`, `shear ≈ (0.05, 0.05)` — the
simulator truth — at **log L = +31788**, roughly **190,000 nats better** than
every single-start optimizer (which sat at ≈ −158000 in the wrong basin). The
per-start success rate (~2/12 ≈ 17%) is exactly why one cold start is
unreliable and twelve are not: P(≥1 hit) ≈ 1 − 0.83¹² ≈ 89%. Three starts
diverged to NaN (re-entering the ell_comps/shear singularity), which broad
starts must tolerate — the `argmin`-over-finite bookkeeping handles it.

‡ SVI folds its JIT compile into the first optimization step (NumPyro scans the
loop), so compile is not separable — the 107 s wall includes it. It compiles
fast (~100 s) **only** when handed the *raw* likelihood; passing the pre-jitted
`obj.log_likelihood` triggers a pathological >25-min jit-inside-jit compile.

**SVI (single cold start) fails the same way — and worse, overconfidently.**
NumPyro SVI with a mean-field Gaussian guide and ADABelief (the exact Enzi et al.
2026 recipe) settles at `einstein_radius = 3.54 ± 0.07` (log L −158022) — the
wrong basin, like every other single start, but now reporting a **tight**
posterior around the wrong answer. This is the mode-seeking VI failure Enzi et
al. and Li & Turner (2016) flag (SVI underestimates uncertainty); it is why the
paper's real robustness comes from `init_to_median` chaining and the broader
pipeline, not a single cold-started SVI. Two numerical gotchas worth recording:
the diffuse `init_to_uniform` init drives the tightly-scaled ell_comps to
unphysical |x|>1 → NaN (fixed with a finite `init_to_value` cold start), and the
raw MAP gradient must be clipped as in the optax runs.

_Levenberg-Marquardt (jaxopt) note: it needs the residual **vector**
(`analysis.fit_from(instance).normalized_residual_map`) rather than the scalar
objective. As another **single cold-start** local method it is expected to fail
the same way L-BFGS did (same basin), so the higher-value form is a **multi-start
Gauss-Newton** — deferred as a follow-up rather than run as a single start._

## Takeaways

1. **Per-eval speed is excellent, and it doesn't discriminate.** The JAX MAP
   objective compiles once (~280–310 s on CPU) then runs at ~170–230 ms per
   forward+gradient eval — ~6–8× cheaper than the NumPy MGE likelihood
   (~1400 ms/eval, see the sampler table). But speed is not the deciding axis
   here: robustness is.

2. **Single cold-start gradient MAP is NOT robust — regardless of optimizer.**
   All three optimizers — first-order (Adam, ADABelief) *and* quasi-Newton
   (L-BFGS) — converge to the **same wrong basin**: log L ≈ −158000
   (reduced χ² ≈ 20) with `einstein_radius` driven to ≈ 4.4–5.0 against the
   prior's upper wall (truth ≈ 1.6). Because L-BFGS uses an independent line
   search (no learning rate, no gradient clip), this is **not** a
   hyperparameter artifact — the cold start simply sits in a basin whose
   gradient points away from the truth, and every gradient method faithfully
   follows it there.

3. **ADABelief additionally re-enters the NaN singularity mid-run.** After
   ~200 steps its log-posterior goes NaN — the optimizer wanders back through
   the `ell_comps`/shear = 0 degeneracy that also poisons the raw prior median.
   First-order steps are not protected from re-crossing it; the best (finite)
   iterate is retained by the plateau logic.

4. **This is the documented reason the field does not use a single cold
   start.** GIGA-Lens (Gu, Huang et al. 2022; 2.0 in arXiv:2606.30633) runs
   **many** gradient-descent starts in parallel and keeps the best; Herculens /
   Enzi et al. 2026 use **warm-start chaining** and `init_to_median` SVI. Our
   Phase-1 result reproduces the failure those designs are built to avoid.

## Verdict — which one is robust AND fast

**Multi-start Adam (the GIGA-Lens recipe).** The JAX MGE likelihood is fast
under every gradient method (~170–230 ms/eval, ~6–8× the NumPy path), so speed
is not the discriminator — **basin selection is**. A single cold start
(first-order *or* quasi-Newton) reliably lands in the wrong basin
(`einstein_radius` at the prior wall, log L ≈ −158000). Running a handful of
broad starts in parallel and keeping the best recovers the true model
(`einstein_radius = 1.600`, log L = +31788) at ~192 ms/eval — robust *and* fast,
exactly as GIGA-Lens (Gu/Huang et al. 2022) designed and Herculens/Enzi et al.
2026 corroborate (they warm-start / use SVI for the same reason).

Practical recommendation for a JAX gradient search on this likelihood class:
**multi-start Adam** (≥12 broad starts; scale the start count with available
GPU parallelism — this is precisely what GIGA-Lens 2.0's multi-GPU design buys),
optionally polished by L-BFGS within the winning basin, and/or handed to SVI/HMC
for the posterior (the full GIGA-Lens and Herculens pipelines).

_Caveat: robustness here = recovering the truth basin from a cold start on one
dataset, characterised at modest CPU budgets. A converged multi-seed,
multi-dataset study belongs on the A100/HPC path (`autolens_profiling`), where
GIGA-Lens-style parallel starts are cheap._

## Robustness & budget caveats

Robustness here = does the optimizer reach a consistent high log-posterior from
a cold start (reported as `Max log L` and cross-candidate agreement), not a
converged multi-seed success-rate study. These are modest CPU budgets for
runtime characterisation. A converged multi-seed robustness study belongs on
the A100/HPC path (`autolens_profiling`), per the samplers faculty's
real-likelihood promotion gate.
