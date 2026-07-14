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

<!-- RESULTS BELOW FILLED FROM A100 RUNS -->

## Consolidated results (A100, float32, 128 starts)

_TABLE PENDING A100 RUNS._

| Method | Starts | Iters (actual/cap) | Batched steps | Scalar-equiv evals | Compile s | Run s | Total wall s | Best r_E | Best log L | Basin hits | Termination | Success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|

### Nautilus (full converged posterior)

_PENDING._ total likelihood calls · ESS · log-evidence ± σ · max-posterior r_E +
log L · other modes · total wall · setup time. Nautilus orchestrates on the CPU
while each PyAutoLens/JAX likelihood evaluation runs on the A100 — it is **not**
GPU-native, and it does not batch its likelihood calls.

## Rankings & answers

_PENDING A100 RUNS._

1. By robustness (basin-hit fraction): …
2. By post-compile execution time: …
3. By total wall-clock time: …
4. Robustness-vs-wall-clock Pareto: …
5. **Did anything beat multi-start Adam?** …
6. **Is Adam→L-BFGS the preferred production strategy now?** …
7. **Best reliable MAP workflow vs Nautilus wall-clock ratio:** …
8. **Is the truth basin the global MAP / dominant mode?** …

## Reproduction

- Repo: `autolens_workspace_developer` @ branch
  `feature/jax-map-optimizer-benchmark-phase3`, commit `<HASH>`.
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
- Per-run machine-readable summaries: `searches_minimal/output/*_summary.txt`;
  raw logs: `/mnt/ral/jnightin/phase3_logs/`.
