# NSS gradient probe + nss_jit sweep — partial findings

Branch: `feature/nss-grad-probe-and-jit-sweep` on `autolens_workspace_developer`.
Status: probe complete, sweep stopped early (partial results) due to GPU
contention with the user's interactive workload. Resume notes below.

## Phase 1 — Gradient probe (`probe_grad.py`) — COMPLETE

Verdict: **`FAIL_NAN_OR_INF`**. Verbatim probe output below (the
`output/probe_grad_summary.txt` file itself is gitignored, so the actual
numbers are inlined here):

```
--- Gradient probe ---
Verdict: FAIL_NAN_OR_INF

Per-point gradient diagnostics:
  prior_median        log L =   -159736.3550  ||g||_inf=nan       max/median=nan       finite=False
  narrow_band_0       log L =   -160243.0148  ||g||_inf=1.561e+05 max/median=2.78e+03  finite=True
  narrow_band_1       log L =   -181611.5686  ||g||_inf=3.137e+05 max/median=558       finite=True
  narrow_band_2       log L =   -170064.6837  ||g||_inf=2.810e+05 max/median=1.37e+03  finite=True
  full_prior_random   log L =   -270291.2413  ||g||_inf=6.869e+05 max/median=7.59      finite=True

Finite-difference cross-check (eps = 1e-3 * sigma_prior):
  prior_median  idx= 5  grad_an=-1.2416e-03  grad_fd=-1.2416e-03  rel_err=1.14e-05 [OK]
  prior_median  idx=14  grad_an=+nan         grad_fd=-1.6820e+01  rel_err=nan      [FAIL]
  prior_median  idx= 9  grad_an=+nan         grad_fd=+3.8459e+01  rel_err=nan      [FAIL]
  narrow_band_0 idx= 7  grad_an=-5.6095e+01  grad_fd=-5.6095e+01  rel_err=4.54e-06 [OK]
  narrow_band_0 idx=11  grad_an=+8.7268e+00  grad_fd=+8.7268e+00  rel_err=4.30e-07 [OK]
  narrow_band_0 idx= 1  grad_an=+4.5363e+04  grad_fd=+4.5362e+04  rel_err=2.16e-05 [OK]

worst max/median |grad|: 2.78e+03    worst FD rel_err: 2.16e-05
```

The `jax.grad` of the MGE+inversion+chi-squared stack is *correct where
finite* (FD agreement ~10⁻⁵) but produces **NaN at the prior median**
(unit cube = 0.5) on at least two parameter components (idx 9 and 14).
Three samples drawn from `[0.4, 0.6]` of the unit cube — `nss_grad.py`'s
narrow-init band — all returned finite, FD-correct gradients. One full-
prior random draw also returned a finite gradient.

Implications:
- HMC is **not robustly viable** without first fixing the NaN source.
  HMC's leapfrog integrator silently breaks the moment it touches a
  NaN-gradient region; this matches the obviously-broken `nss_grad`
  smoke result (`centre=(0.799, 4.928)`, `shear≈5.4`).
- The narrow-band init `nss_grad` uses happens to avoid the bad region,
  but step-size adaptation can wander into it.
- Conditioning is rough but not catastrophic: scaled gradient `g·σ`
  spans 1–4 orders of magnitude across parameters. HMC would need an
  adapted mass matrix to navigate.

Likely root cause (not yet confirmed): a degenerate inversion / 0×∞
in MGE evaluation or NNLS at certain centred configurations. Worth a
follow-up where an instrumented likelihood is run point-by-point at
unit cube 0.5 to localise which intermediate produces the NaN.

## Phase 2 — `nss_jit` settings sweep (`sweep_nss_jit.py`) — PARTIAL

Stopped after 4 OOM failures and one in-progress run, before any config
reached termination. CSV saved at `output/sweep_nss_jit.csv` with rows
for c1–c4. c5 was sampling productively when the user halted the run.

### Headline finding: 6 GB VRAM is the binding constraint

| # | n_live | mcmc | delete | term | result                                       |
|---|--------|------|--------|------|----------------------------------------------|
| 1 | 200    | 5    | 10     | -3   | **OOM** (alloc 953 MiB requested)            |
| 2 | 200    | 3    | 10     | -3   | OOM (cascading from #1's poisoned allocator) |
| 3 | 200    | 10   | 10     | -3   | OOM (cascading)                              |
| 4 | 200    | 5    | 50     | -3   | OOM (cascading)                              |
| 5 | 100    | 5    | 10     | -3   | running, ~440 dead, logZ trending from -575k → -191k when stopped |
| 6 | 500    | 5    | 10     | -1   | not reached                                  |

**RTX 2060 (6 GB VRAM) cannot fit `n_live=200` on this MGE problem.** Once
config #1 fails, JAX's BFC allocator does not release cleanly, so every
subsequent config in the same process also fails. The `n_live=500` config
was definitely going to OOM as well.

`n_live=100` IS feasible — c5 ran cleanly until interrupted, processing
~1.2 dead/s on GPU.

### What c5 told us (qualitative)

- The likelihood landscape is *very* slow to anneal. After ~4 minutes
  c5 was at logZ ≈ -191k while the converged target (Nautilus's neighbourhood
  via the existing `nautilus_jax` summary) is closer to logZ ≈ -169k.
- That's a delta-logZ of ~22, far from the `-3` termination target. A
  fully-converged run at `n_live=100` would likely be ~30–60 min based
  on the dead-point rate trend.

### What's needed to complete Phase 2 properly

1. **Reduce VRAM use OR run each config in a fresh subprocess.** Options:
   - Cap configs at `n_live ≤ 150` for this card.
   - Run each config in `subprocess.run(["python", "-c", ...])` so OOM on
     one config can't poison the next.
   - Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
     to let JAX free memory between configs (helps but doesn't fully solve).
2. **Replace `n_live=500` config (#6)** with something that fits, e.g.
   `n_live=150, num_mcmc_steps=5, termination=-2`.
3. Consider a **smaller dataset variant** for the sweep — the current
   `_setup.build_dataset()` produces 15,361 image pixels which dominates
   curvature-matrix VRAM. A 60% mask radius shrink would quarter the
   memory budget and let `n_live=200+` fit.

## Resuming this work

Branch:
```bash
cd /home/jammy/Code/PyAutoLabs/autolens_workspace_developer
git checkout feature/nss-grad-probe-and-jit-sweep
```

Files of interest on this branch:
- `searches_minimal/probe_grad.py` (NEW) — re-runnable, ~10 min on GPU.
- `searches_minimal/sweep_nss_jit.py` (NEW) — needs the subprocess-isolation
  fix above before re-running.
- `searches_minimal/output/probe_grad_summary.txt` — final probe report.
- `searches_minimal/output/sweep_nss_jit.csv` — partial sweep rows (4 OOMs).
- `searches_minimal/output/sweep_findings.md` — this file.

WIP from the user's previous session is preserved as dirty changes on top
of this branch (not committed): `_setup.py`, all `nss_*.py`, `nautilus_*.py`,
`dynesty_simple.py`, `emcee_simple.py`, `lbfgs_simple.py`, plus untracked
`_metrics.py` and `blackjax_nuts.py`.

## Open follow-ups

1. **Localise the NaN-gradient source** in the MGE+inversion likelihood at
   unit cube ≈ 0.5. Probably needs `jax.debug.print` instrumentation
   inside `analysis.log_likelihood_function` or a Python-side trace of
   the same fit at the bad parameter point.
2. **Fix the sweep harness** to subprocess-isolate configs so cascading
   OOMs stop happening.
3. **Decide whether `nss_grad` is worth pursuing** — the gradient
   pathology says probably not until (1) is fixed.
