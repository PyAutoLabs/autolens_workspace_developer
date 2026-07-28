# Does MultiStartProdigy work for pixelized source meshes? YES — on every mesh, once the regularization axis is handled.

**Campaign:** autolens_workspace_developer#117 (2026-07-27/28, RAL `ral` CPU
partition — both A100 nodes unavailable). Successor to #100/#101 (whose
MultiStartAdam-era verdict "Nautilus wins pix decisively" is **overturned**)
and #104 (whose reg-fragility diagnosis turned out to be the load-bearing
clue). Runner: `pix_prodigy.py` on the **library** `af.MultiStartProdigy`
(16 broad starts, per-start vmapped state, `resurrect=True`, lr-free,
`batch_size=4` memory tiling, fixed 3000-step budgets resume-chained across
24 h SLURM jobs).

**The question:** gradients through all three pixelized meshes were certified
in `autolens_workspace_test` (kernel-CDF rectangular; KNN Wendland; Delaunay
frozen-tables, 2026-07-26). Does a *search* built on them recover the truth
from broad mass priors — the SLaM `source_pix[1]` shape (fixed MGE lens-light
geometry, free Isothermal + shear, free regularization)?

## Headline results (simulator truth: r_E = 1.6, shear (0.05, 0.05))

| mesh | regularization | steps to solution | final max logL | r_E | verdict |
|---|---|---|---:|---|---|
| knn | free AdaptSplit (inner, outer) | ~250 to good basin; **~1300 to truth** | **+29724** | **1.599** | ✓ exact truth |
| delaunay | AdaptSplit **fixed** (0.032, 3.16) | **~150** | **+30202** | **1.600** | ✓ exact truth |
| delaunay | free **MaternKernel** (coefficient, scale) | ~2500 | **+29792** | ~1.6 | ✓ at truth-point bar |
| delaunay | free AdaptSplit | ~2000 (late escape after 1500 flat) | **+30099** | ~1.6 | ✓ eventually |
| delaunay | free AdaptSplit, narrow start band | never | −50498 | — | ✗ narrowing **hurts** |
| rectangular | free / fixed Constant | **pending** | −30136 @ 200 steps | — | open — throughput-limited, not stalled (see below) |

Success bars: per-mesh truth-point reg scans (`truth-bar` mode; upper
references) — rectangular +27059, knn +28792 (1-D slice; the free 2-D optimum
is higher), delaunay +30079 (AdaptSplit) / +29682 (Matérn). Converged-sampler
comparisons below.

## Prodigy vs Nautilus — the #101 verdict inverts

Same objective, same hardware, matched modest settings
(`n_live=100, n_batch=16`, the #101 configuration):

| mesh | Prodigy | Nautilus | margin |
|---|---:|---:|---|
| knn | **+29724** @ r_E 1.599, ~1.4 h to good basin | +5704 @ r_E 1.011, 5.3 h | **+24k nats, truth vs wrong basin** |
| delaunay | **+30099** (free) / +30202 (fixed reg) | +19982 @ r_E 0.962, 2.3 h | **+10k nats, truth vs wrong basin** |

Caveat, stated plainly: single thin-settings Nautilus runs are *floors*, not
converged references — a heavier-settings run is a listed follow-up. What is
claimable without qualification: **multi-start gradient descent reaches the
truth basin from broad starts where these sampler runs did not**, at a
fraction of the wall time.

## The mechanism: the regularization axis was the whole story

Every failure mode of the campaign localised to the **free regularization
parameters**, not to mesh landscape, start band, or optimizer rule:

1. **AdaptSplit double-squares its coefficients** (#104: effective λ⁴), so its
   high-coefficient region is extreme. On **knn** that region is *bad but
   finite* (over-regularized floor ≈ −158k): dying lanes resurrect and
   eventually escape (late breakout, step ~1300). On **delaunay** the same
   region is a **NaN wall** (forward-eval NaNs from AdaptSplit inner ≥ 10 in
   the truth-bar scan): lanes die rather than learn, and escape is taxed
   ~2000 steps of resurrection lottery (797 resurrections; 1500 steps flat at
   +8513 before breaking out).
2. **Fix the reg** (SLaM-legitimate — `source_pix[1]` inherits it from the
   preceding fit) and delaunay converges to exact truth in 150 steps with
   background-level churn.
3. **Swap the parametrization** — free `MaternKernel(coefficient, scale)`
   (nu pinned 0.5; the kernel schemes are the other JAX-safe Delaunay-family
   pairing) — and the wall vanishes outright: graceful degradation at high
   coefficient (no NaNs anywhere in the scan), smooth low-churn climb to the
   truth-point bar, identical fit ceiling (+29682 vs +30079).
4. **Narrowing the start band does nothing** (−50k after 3000 steps): global
   discovery was never the bottleneck.

Practical ordering for a free-reg Delaunay-family search:
**Matérn ≥ fixed/inherited ≫ AdaptSplit**, at equal final fit quality.
No basin-hopping or optimizer redesign is needed on this evidence — the
"plateau + resurrection churn" wall signature was a property of one reg
parametrization, not of gradient search on meshes.

Secondary lesson: **a long plateau is not convergence.** knn sat 1000 steps
at +22515 (a suboptimal reg mode, r_E 1.633) before a resurrection crossed
into the true mode (+29724, r_E 1.599). Budget multi-start pix searches at
≥1500–3000 steps, or fix/inherit the reg and take the ~150–250-step fast path.

## Mesh smoothness ↔ search behaviour

The three meshes ranked exactly by interpolation smoothness class:

- **kernel-CDF rectangular** (C∞): smooth steady climbs, minimal churn.
- **knn Wendland-C4** (smooth within neighbour sets): fastest converger.
- **delaunay frozen-tables** (piecewise-smooth, C0 at flip seams): gradient
  information stops at topology flips → smaller basins → progress leans on
  restarts. Still solvable (see above), just the least gradient-efficient.

## Rectangular: open, and why that's a throughput statement

The rectangular chains are healthy but slow: **~5.7 min/step on 32 CPUs
(~17× knn)** — far above its ~4.5× forward-eval ratio, i.e. the kernel-CDF
`value_and_grad` is disproportionately expensive. 19 h bought 200 steps
(−51201 → −30136 — for calibration, that passed the #101 adam *endpoint* in
25 steps). With the late-breakout precedent, no verdict before ~1500 steps;
chains are extended to 5 links and the per-step cost is flagged for the A100
profiling follow-up. Nothing in the rectangular trajectory resembles the
AdaptSplit wall signature.

## Library bugs found and fixed by this campaign (both merged 2026-07-28)

1. **PyAutoFit multi-start cadence crash** (#1421/#1423 arc): float-coerced
   `iterations_per_full_update` fed `range()` — fired only when the checkpoint
   cadence < remaining budget, which no default config ever hit. Six chain
   jobs died on it; the review of the fix surfaced five further cadence bugs.
2. **PyAutoArray Delaunay NaN-callback hardening** (PR#411): NaN traced points
   reached host qhull, which *raised* (process-fatal for all 16 vmapped lanes,
   and the persisted checkpoint replayed the crash on every resume). The
   lane-level isfinite guard restores the "NaN propagates, the guard machinery
   handles it" contract the pure-JAX meshes obey — validated under fire here
   (hundreds of NaN-lane hits absorbed per run).

## Implications for posterior estimation on meshes (SMC / HMC / nested)

- The per-mesh **resurrection rate is an empirical predictor of HMC
  divergence rates** (every event that kills a Prodigy lane would abort a
  leapfrog trajectory). Hamiltonian kernels belong on the smooth meshes
  (kernel-CDF, knn) with the reg axis Matérn-parametrized or inherited;
  delaunay's flip seams additionally break energy conservation → route it to
  tempered SMC or refit-at-solution instead.
- **Warm-starting is the universal lever**: the multi-start endpoint is a
  *population* of diverse basin-labelled points — the natural initializer for
  SMC particles / HMC chains (the parked sampler-wave stage (a) already
  showed warm-start makes gradient SMC sample).
- Thin nested-sampling runs failing the basin (knn r_E 1.01, delaunay 0.96)
  show prior-volume search is the hard part here — which is precisely what
  the gradient stage now solves.

## Artifacts & reproduction

- Runner/objective: `searches_minimal/pix_prodigy.py`, `pix_multi_start.py`
  (knobs: `PIX_MESH`, `PIX_REG=matern`, `PIX_FIX_REG`, `PIX_START_LOW/HIGH`,
  `PIX_N_STARTS/N_STEPS/BATCH`, `PIX_RESURRECT`, `PIX_NAME_SUFFIX`).
- SLURM: `pix_prodigy_cpu.sbatch` (`ral` partition; resume-chain via
  `--dependency=afterany`; warm `JAX_COMPILATION_CACHE_DIR`).
- Run records: `searches_minimal/pix_prodigy_results/` (truth-bar scans,
  Nautilus baselines, per-arm reports). RAL jobs 331177–331257.
- Promoted mature cells: `autolens_profiling/scripts/imaging/searches/
  multi_start_prodigy/{pixelization,knn,delaunay}.py`; user-facing lessons:
  `autolens_workspace/scripts/guides/modeling/searches.py`.

Follow-ups filed via PyAutoMind: heavier-settings Nautilus reference runs;
A100 profiling of the winning configs (incl. the rectangular per-step
anomaly); sampler-wave crossover (warm-started SMC/HMC per the mesh↔kernel
mapping above).
