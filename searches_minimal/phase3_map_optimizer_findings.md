# Phase 3 — second-order & polishing MAP optimizers vs multi-start Adam (+ Nautilus)

Follow-up to `gradient_optimizer_findings.md` (#96) and `next_wave_findings.md`
(#98). Those established, on the HST MGE lens likelihood (~15 nonlinear params;
MGE light amplitudes solved by the linear inversion), that **every single
cold-start method lands in the wrong basin**, while **wide multi-start Adam**
recovers the truth (`einstein_radius ≈ 1.600`, log L ≈ +31788); at 128 starts on
the A100, 23/128 land in the basin (p_hit ≈ 0.18 → P(≥1) ≈ 1). SVGD reaches the
truth as a mode-finder.

This phase asks the sharper production question: **can any JAX-native,
gradient-based MAP optimizer beat wide multi-start Adam on the robustness–speed
Pareto front**, and how does the best reliable MAP workflow compare in
wall-clock to one fully converged Nautilus posterior run. Scope is **MAP
optimizers only**; Nautilus is the single conventional baseline.

## The objective is an exact residual least-squares — but only reverse-mode

Verified on this model (CPU, float64):

    analysis.log_likelihood_function == fit.log_likelihood == fit.figure_of_merit
                                     == -0.5 * (chi^2 + noise_norm)      (diff ~1e-11)

There is **no regularization / Occam log-det term** — the MGE light is linear and
there is no pixelized source, so the Bayesian evidence collapses to the plain
Gaussian likelihood. Hence

    objective = -(log L + log prior) = 0.5*chi^2 + 0.5*noise_norm - log_prior

is an **exact sum of squares up to θ-independent constants**, with residual

    r(z) = [ (data - model_image(θ))/noise      (15361 imaging residuals),
             (θ_k - μ_k)/σ_k                     (one per Gaussian prior, 4 here) ]

(`0.5||r||² = objective` to a constant; the residual-identity check confirms the
difference `neg_log_posterior − 0.5||r||²` is constant to ~0.3 nat out of 40000,
the residual drift being a minor prior-normalisation detail). So on paper this is
a textbook Gauss-Newton / Levenberg-Marquardt problem.

### …and this is exactly why Gauss-Newton / Levenberg-Marquardt are INFEASIBLE here

**Key negative result.** LM and Gauss-Newton require the **forward-mode** residual
Jacobian (jaxopt builds it with `jacfwd`/`jvp`). The positive-only source solve
`autoarray.util.jax_nnls.solve_nnls_primal` is defined with a **reverse-mode
custom gradient** (`custom_vjp`) and provides **no forward rule**, so LM/GN raise:

    TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function

This is the concrete, autodiff-level form of the "non-smoothness caused by the
non-negative linear amplitudes" the benchmark set out to identify: the NNLS
active set makes `z → reconstruction` only reverse-mode differentiable, which
rules out *every* least-squares Gauss-Newton/LM method — not because the residual
formulation is wrong (it is exact), but because the inversion's gradient is
one-sided. The **gradient itself is faithful** (reverse-mode; directional
finite-difference check agrees to rel_err ~1e-9), so first-order and quasi-Newton
methods (Adam, L-BFGS, BFGS, NCG) all work. **The strongest second-order method
compatible with the production objective is therefore BFGS, not LM/GN.**

(A statistically valid LM/GN *could* be run against an **unconstrained** linear
inversion — a plain `linalg.solve` is forward-differentiable — but that targets a
different likelihood, one that allows negative source flux, so it is not the
production objective and is not pursued here.)

## Parameterization & numerical fairness (Part 3)

- **Shared unconstrained coordinate.** Every gradient optimizer runs on
  `z ∈ R^15` with `θ = vector_from_unit_vector(sigmoid(z))` — one sigmoid per
  parameter over the model's own inverse-CDF. Smooth, bijective, **no hard
  clipping**, comparable O(1) scales (proper scaling for the quasi-Newton
  curvature estimate). **No Jacobian term is added**, so the optimum is the MAP
  in *physical* coordinates — the transform is a pure preconditioner, not a
  change of target density.
- **Identical starts.** Starts are drawn in the unit cube `U(0.15, 0.85)` (as in
  the earlier campaigns) and mapped to `z = logit(u)`, so the physical start
  points are identical across every method — same RNG seed, same draws.
- **Precision.** Main runs are float32 (RAL A100; x64 disabled on that GPU). A
  float64 CPU validation run checks basin recovery is not a float32 artifact.
- **Gradient validation.** Directional finite differences vs autodiff on the
  z-objective: rel_err ~1e-9 (see `_grad_setup.fd_gradient_check`).

## Evaluation accounting (Part 5) & termination

Each summary records: batched optimizer steps, batch latency per step, and
**scalar-equivalent** objective/gradient evaluations (128 starts × K steps is K
*batched* steps, not 128·K sequential GPU calls). Every method is also tagged by
**termination discipline**, the distinction being whether the user fixes the
iteration budget up front or the method discovers convergence at run time:

| Method | Termination | Iteration count |
|--------|-------------|-----------------|
| multi-start Adam / ADABelief / Lion | **user-budget** | fixed `N_STEPS` (300) |
| SVGD | **user-budget** | fixed `N_STEPS` (300) |
| multi-start L-BFGS / BFGS / NCG | **self-terminating** | grad-tol; cap = maxiter (recorded actual vs cap) |
| Adam → L-BFGS | **hybrid** | Stage-1 budget + Stage-2 self-terminating |
| Nautilus | **self-terminating** | n_eff / f_live convergence |
| LM / Gauss-Newton | n/a — infeasible (see above) | — |

## Consolidated results (A100 80GB, float32, 128 starts, 2026-07-14)

Compile times reflect the persistent JAX compilation cache
(`JAX_COMPILATION_CACHE_DIR`): the **cold** Adam row is a true first-compile
(cache empty); every later job reused the cached `value_and_grad` executable, so
their "compile" is the warm figure (the method-specific solver graph only). Run
time is post-compile; total wall = compile + run. Batched steps ≠ scalar calls:
128 starts × K steps is K *batched* GPU steps, not 128·K sequential calls.

| Method | Starts | Iters (actual/cap) | Batched steps | Scalar-equiv f+g evals | Compile s | Run s | Total wall s | Best r_E | Best log L | Basin hits | Termination | Success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|:--:|
| **multi-start Adam** (cold) | 128 | 300/300 | 300 | 38 400 | 166 | 17 | 183 | **1.5997** | **+31 787.9** | **24/128** | user-budget | ✅ |
| **multi-start Adam** (warm) | 128 | 300/300 | 300 | 38 400 | 34 | 16 | **50** | **1.5997** | **+31 787.9** | **23/128** | user-budget | ✅ |
| multi-start ADABelief | 128 | 300/300 | 300 | 38 400 | 34 | 16 | 50 | 1.5997 | +31 787.8 | 20/128 | user-budget | ✅ |
| multi-start Lion | 128 | 300/300 | 300 | 38 400 | 33 | 16 | 50 | 1.6042 | +29 559.3 | 20/128 | user-budget | ✅ |
| SVGD (16 particles) | 16 | 300/300 | 300 | 4 800 | 82 | 98 | 180 | 1.5980 | +17 117.4 | 5/16¶ | user-budget | ◑ mode-finder |
| multi-start **L-BFGS** | 128 | 1..200/200 | 200 | 8 789 | 433 | 212 | 645 | 0.0000 | −149 525 | 3/128 | self-term | ❌ |
| multi-start **BFGS** | 128 | 1..200/200 | 200 | ~9 000 | 411 | 212 | 623 | 0.0000 | −627 634 | 2/128 | self-term | ❌ |
| multi-start **NCG** | 128 | 2..200/200 | 200 | ~9 500 | 468 | 218 | 686 | 5.6586 | −754 805 | 0/128 | self-term | ❌ |
| **Adam → L-BFGS** polish | 128 | 300 + 1..200 | 300+200 | 38 400 + 14 013 | 434 | 236 | 670 | 1.7348 | +24 881 | 3/128 | hybrid | ❌ (polish −6906 nat) |
| **LM** (Levenberg–Marquardt) | 128 | — | — | — | — | ~65 to fail | — | — | — | — | n/a | ⛔ infeasible |
| **Gauss–Newton** | 128 | — | — | — | — | ~160 to fail | — | — | — | — | n/a | ⛔ infeasible |
| float64 Adam (CPU, x64) | 16 | 300/300 | 300 | 4 800 | — | — | — | 1.5996 | +30 863 | 4/16 | user-budget | ✅ |

Peak GPU memory ≈ 5.5 GB for the jaxopt second-order runs (well within 80 GB);
the multi-start batches are lighter. LM/GN fail at trace time (before the heavy
compile) with `TypeError: can't apply forward-mode autodiff (jvp) to a
custom_vjp function`; the seconds shown are the objective build + z-start
generation before the Jacobian is attempted.

¶ SVGD is a **particle posterior approximation**, not a directly comparable point
optimizer: its *best* particle reaches the truth (r_E 1.598), but the repulsion
spreads the cloud so the best log L (+17k) is shallower than the multi-start MAP
(+31.8k) and only 5/16 particles sit inside the tight ±0.3 basin. Listed for
context, ranked separately.

### Nautilus (full converged posterior — the conventional baseline)

`af.Nautilus(n_live=100, number_of_cores=1).fit(model, analysis)` — the standard
autofit search API (as every imaging modeling example uses), JAX likelihood on
the A100, run to Nautilus's own `n_eff`/`f_live` convergence.

| Quantity | Value |
|---|---|
| Total likelihood calls | 33 000 |
| Total wall time | **523 s** (15.85 ms/eval, sequential — **not** batched) |
| Effective sample size | 1 845 |
| Posterior samples | 32 982 |
| log-evidence | **+31 690.2** (± not surfaced by autofit's `SamplesNest`; Nautilus's internal estimate is not retained after the search folder is cleaned) |
| Max-posterior einstein_radius | **1.5998** — reaches the truth basin ✅ |
| Max-posterior log L | +31 786.6 (matches the MAP optimizers' +31 787.9) |
| Other modes | none found — a single dominant mode at the truth |
| Hardware | Nautilus orchestrates on the **CPU** (1 process); each likelihood on the A100. **Not** GPU-native, **not** batched. |

## Rankings & answers

**1. By robustness (basin-hit fraction / reliability).**
Nautilus (converged to truth, evidence-backed) ≳ multi-start Adam (24/128,
P(≥1)≈1) > ADABelief (20/128) ≈ Lion (20/128) ≫ SVGD (best particle only) ≫
L-BFGS (3) > BFGS (2) > NCG (0) ; LM/GN infeasible. The first-order multi-starts
and Nautilus are the only reliable finders; every line-search method fails.

**2. By post-compile execution time (run s).**
Adam/ADABelief/Lion **16 s** < SVGD 98 s < L-BFGS 212 ≈ BFGS 212 < NCG 218 ≈
Adam→L-BFGS 236 < Nautilus 523. The multi-start first-order methods are ~13×
faster per run than any line-search method and ~30× faster than Nautilus.

**3. By total wall-clock time.**
Adam warm **50 s** ≈ ADABelief ≈ Lion < SVGD 180 ≈ Adam cold 183 < Nautilus 523 <
BFGS 623 < L-BFGS 645 < Adam→L-BFGS 670 < NCG 686. (Cold Adam 183 s is dominated
by the 166 s first-ever XLA compile; warm/cached it is 50 s.)

**4. Robustness-vs-wall-clock Pareto.**
Two points are on the frontier: **warm multi-start Adam** (50 s, reliable) and
**Nautilus** (523 s, reliable + full posterior/evidence). Everything else is
dominated — the line-search methods are both slower *and* unreliable; SVGD is
slower than Adam and only a mode-finder. For a point estimate, multi-start Adam
is the sole efficient robust choice; pay Nautilus's ~10× only when you need the
posterior and evidence.

**5. Did anything beat multi-start Adam? — No.**
No MAP optimizer beat it. The line-search quasi-Newton/CG methods (L-BFGS, BFGS,
NCG) *catastrophically* fail (0–3/128; best log L −150k to −755k, i.e. they
diverge to worse-than-cold-start regions). LM/Gauss-Newton are infeasible
(reverse-mode-only NNLS). SVGD reaches the basin but only as a mode-finder, at a
shallower optimum and 3.6× the wall time. Nautilus matches Adam's optimum but is
~10× slower. **Multi-start Adam remains the best fast, reliable MAP optimizer.**

**6. Is Adam→L-BFGS (or Adam→LM) the preferred production strategy now? — No,
emphatically.** L-BFGS polishing *destroys* Adam's solutions: starting from
Adam's basin points (log posterior +31 786) the polish ended at +24 881, a
**−6906 nat regression**, dropping in-basin starts from 23 to 3. Adam→LM is
infeasible. The recommended production strategy is **plain multi-start Adam with
no second-order polish** — the polish stage on this likelihood is actively
harmful and adds a ~400 s compile.

*Why the line-search methods fail:* the positive-only (NNLS) source solve makes
the objective **piecewise-smooth** — active-set changes (which source pixels hit
the zero bound) create gradient kinks. Line searches (backtracking/zoom) and
quasi-Newton curvature models assume local smoothness; across a kink the curvature
estimate is meaningless and the accepted step lands in a worse region. Adam,
ADABelief and Lion take **fixed-size, self-normalised steps with no line search**,
so they step over the kinks robustly. The failure is intrinsic to combining
line-search descent with the NNLS non-smoothness, not to any one library.

**7. Best reliable MAP workflow vs Nautilus wall-clock ratio.**
Warm multi-start Adam **50 s** vs Nautilus **523 s** → **≈ 10.5× faster**
(≈ 2.9× even against cold Adam's 183 s). Both reach the same truth basin
(r_E 1.5997 vs 1.5998; log L +31 787.9 vs +31 786.6); Nautilus additionally
delivers the posterior and log-evidence (+31 690), which the point optimizer does
not.

**8. Is the truth basin the global MAP / dominant posterior mode? — Yes.**
Nautilus's converged, evidence-weighted posterior (ESS 1845, 33k calls,
log Z +31 690) concentrates on a **single dominant mode at the truth**
(max-posterior r_E 1.5998) with **no other modes found**. The multi-start MAP
optimum (+31 787.9) coincides with Nautilus's max-posterior sample (+31 786.6).
So the truth basin is both the global MAP and the dominant posterior mode — the
wrong-basin attractors the single cold-starts fall into (r_E→5, r_E→8) are *not*
competitive posterior mass, they are just where naive gradient descent stalls.

**float64 check.** At double precision (CPU, x64) multi-start Adam recovers the
same basin (r_E 1.5996, 4/16 starts, log L +30 863) — the float32 A100 basin
recovery is **not** a precision artifact. (NaN losses at degenerate draws are
masked, as on the GPU.) Directional finite differences match autodiff gradients
to rel_err ~1e-9, confirming the reverse-mode gradient through the NNLS inversion.

## Reproduction

- Repo: `autolens_workspace_developer` @ branch
  `feature/jax-map-optimizer-benchmark-phase3`, commit `d406cb0` (see `git log`).
- Hardware: NVIDIA A100 80GB, RAL `gpu` partition (`euclid_jump`), float32.
- Software (base venv `/mnt/ral/jnightin/PyAuto/PyAuto`): jax/jaxlib 0.4.38
  (CUDA), jaxopt 0.8.5, optax 0.2.8, nautilus; SVGD via `scratch/bjx13`
  blackjax 1.3 overlay.
- Commands (from workspace root; `MULTISTART_N_STARTS=128`):
  - `sbatch phase3_second_order.sbatch`   (LM, GN [infeasible], L-BFGS, BFGS, NCG)
  - `sbatch phase3_polish.sbatch`         (Adam→L-BFGS)
  - `sbatch phase3_multistart.sbatch`     (Adam, Adam-warm, ADABelief, Lion)
  - `sbatch phase3_svgd.sbatch`           (SVGD)
  - `sbatch phase3_nautilus.sbatch`       (Nautilus)
- Per-run machine-readable summaries and raw SLURM logs are preserved in
  `searches_minimal/phase3_results/{summaries,logs}/` (the live `output/` dir is
  git-ignored). SLURM scripts: `phase3_*.sbatch` (base venv `activate.sh`; SVGD
  prepends the `scratch/bjx13` blackjax overlay; `JAX_COMPILATION_CACHE_DIR`
  enables the cold-vs-warm compile split). Software versions and the LM/GN
  infeasibility are as above; the float64 check ran locally on CPU with
  `JAX_ENABLE_X64=1`.

## One-line summary

On the HST MGE lens likelihood, **nothing beats wide multi-start Adam** for a
fast, reliable MAP: every JAX line-search method (L-BFGS, BFGS, NCG) fails or
diverges on the NNLS-kinked objective, Gauss-Newton/LM are autodiff-infeasible
(reverse-mode-only inversion), Adam→L-BFGS polishing is actively harmful, and a
full converged Nautilus run reaches the same truth basin (confirming it is the
dominant mode) at ~10× the wall time. Recommended production MAP: **multi-start
Adam, no second-order polish.**
