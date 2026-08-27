# MultiStartProdigy on pixelized meshes: laptop-GPU findings

**Campaign:** PyAutoLabs/autolens_workspace_developer#125, 2026-08-11.
**Hardware:** NVIDIA RTX 2060 Max-Q (6 GiB), JAX 0.10.2, CUDA backend,
64-bit enabled. Source revisions: PyAutoFit
`08f73c315e344b0ed2d9e90d72c7ed4580abe315`, PyAutoArray
`5dedb5e90c3ac5957f4288f1968d34f9f4618a95`, and PyAutoLens
`13a4655c73c50d7b056939afb359b41cd77ed7f4`.

The objective is the existing SLaM `source_pix[1]`-style benchmark: 15,361
masked image pixels, fixed MGE lens-light geometry, seven free mass/shear
parameters, broad starts over 0.15--0.85 of the unit cube, and
`resurrect=True`. The Delaunay-family fixed-regularization arms use the
truth-scan optimum scale 0.316227766 (AdaptSplit inner approximately 0.0316,
outer approximately 3.162). KNN uses scale 0.1. The exact JSON artifacts are
under `pix_prodigy_results/laptop_gpu/`.

## Result across all four meshes

| mesh | evidence | max logL / r_E | verdict |
|---|---|---|---|
| RectangularRTUAdaptDensity | RTX 2060, 4 starts, batch 1 (minimum), FP64 | compile failed: XLA requested one 10.90-GiB allocation on a 6-GiB card | **Does not fit this laptop GPU** at production size. This is a VRAM limit, not a Prodigy or gradient failure. The prior CPU bandwidth-0.1 arm was still improving at step 200 (logL -30136; about 5.7 min/step); bandwidth 1.0 remains the preferred smoother follow-up, with truth-point reference +25659.5. |
| KNearestNeighbor | RTX 2060, fixed reg, 8 starts / batch 2 / 300 steps | +2725.067 / 1.0593; 1036.6 s; 5 resurrections | Search and gradients run cleanly, but **8 starts / 300 steps misses the truth basin**. Compatibility is independently confirmed by the earlier 16-start CPU free-AdaptSplit run: +29724 / 1.599, exact recovery near step 1300. |
| Delaunay | RTX 2060, fixed reg, 8 starts / batch 2 / 300 steps | -3664.124 / 1.0163; 1141.1 s; 4 resurrections | Search and gradients run cleanly, but **8 starts misses the truth basin**. The earlier 16-start CPU fixed-reg run reached +30202 / 1.600 near step 150, confirming compatibility and showing that start count, rather than the optimizer rule, explains this miss. |
| DelaunayNN | RTX 2060, fixed reg, 8 starts / batch 4 / 300 steps | **+30374.791 / 1.59979**; truth reference crossed at **step 175**; 1524.2 s total | **Works and recovers the maximum-likelihood basin.** No capacity overflow or degenerate rows were observed. The objective stopped improving after step 200. |

The truth-point values are recovery references, not mathematical upper bounds:
the fitted mass and shear may improve on a likelihood evaluated with all
simulator parameters held at truth. DelaunayNN's fitted +30374.8 therefore
legitimately exceeds its +30304.0 truth-point reference.

Step-to-reference values use the search's recorded best-posterior history
(`-0.5 * figure_of_merit`) as a proxy because per-step likelihood alone is not
stored; the final maximum log likelihood quoted for every completed run is the
exact likelihood. The prior offset is small at the winning DelaunayNN fit.

## DelaunayNN settings experiment

| starts / batch / steps | max logL | r_E | first truth-reference crossing | total wall | interpretation |
|---|---:|---:|---:|---:|---|
| 4 / 1 / 200 | -25625.106 | 1.4010 | not reached | 1169.1 s | Too few starts; improves but remains in the wrong basin. |
| 8 / 2 / 300 | +30374.293 | 1.59868 | 269 | 3957.6 s | Exact recovery, but poor GPU utilization. |
| 8 / 4 / 300 | **+30374.791** | **1.59979** | **175** | **1524.2 s** | Winner: 2.60x lower end-to-end wall time and 94 fewer steps to the reference. Stable by step 200. |

Batching is intended to tile independent starts, but the batch-2 and batch-4
histories diverge after small floating-point differences accumulate near
narrow basin boundaries. Thus the step-count improvement is an empirical
result for this deterministic benchmark, not a guarantee that batch size
changes the underlying optimization problem. The wall-time improvement is
robust: batch 4 keeps more of this GPU occupied and still fits in 6 GiB.

At the batch-4 maximum-likelihood point the recovered mass centre is
(0.00020, -0.00194), ellipticity components are (0.04965, -0.00216),
Einstein radius is 1.59979, and shear is (0.04964, 0.04853). The DelaunayNN
main mapper used at most cavity size 10 and 12 neighbours; its split mapper
used at most cavity size 8 and 10 neighbours. All main/split overflow and
degeneracy counts are zero, comfortably inside the production cap of 32.

## Recommended settings

- **DelaunayNN on this laptop:** 8 starts, batch 4, inherited/fixed
  AdaptSplit regularization, resurrection enabled, and a 200-step cap. Use a
  final-only full-update cadence while benchmarking. This reaches the
  reference at step 175 and is already at its final maximum by step 200.
- **Delaunay:** use 16 starts, batch 4, inherited/fixed regularization, and
  about 200 steps. Eight starts are not reliable; the 16-start evidence
  reaches the solution near step 150.
- **KNN:** retain 16 starts. With free AdaptSplit, budget at least 1500 steps
  because the demonstrated exact recovery occurs near step 1300 after a long
  plateau. The 8-start fixed-reg 300-step arm is not a safe substitute.
- **Rectangular:** do not run this production FP64 objective on a 6-GiB GPU.
  Use CPU/HPC or first validate a reduced-size/precision variant as a distinct
  experiment. Continue the bandwidth-1.0 hypothesis on hardware with enough
  memory; the laptop OOM says nothing about its gradient quality.

The central compatibility conclusion is therefore: MultiStartProdigy works
with KNN, Delaunay, and the new smoother-gradient DelaunayNN. Rectangular is
not testable at production size on this particular GPU because its compiled
objective exceeds VRAM. Across the meshes that fit, too few starts manifests
as a finite wrong-basin result, whereas an implementation incompatibility
would produce compile, non-finite, or capacity failures; none were seen for
the three Delaunay-family meshes.

---

# Follow-up: n_starts control and phase-2 settings (2026-08-11, evening)

Same hardware, same objective, and the **same three source revisions** as the
section above (PyAutoFit `08f73c31`, PyAutoArray `5dedb5e9`, PyAutoLens
`13a4655c`) — verified unchanged before the runs, so these numbers compose
directly with the earlier table. The environment was validated first by
reproducing the earlier delaunay 4-start/batch-1 compile cell bit-identically
(-59873.326 at r_E 1.2153).

All cells below use batch 4, 300 steps, fixed regularization, `resurrect=True`
and the broad 0.15--0.85 start band unless stated. Only the named variable
moves.

## 1. The 16-start claim, now measured on GPU

The section above explained the Delaunay and KNN 8-start misses by start count,
citing CPU evidence. Held at fixed batch/steps/regularization, that is correct:

| mesh | starts | max logL | r_E | truth bar | verdict |
|---|---:|---:|---:|---:|---|
| delaunay | 8 | -9314.8 | 0.9691 | 30078.7 | wrong basin (Nautilus mode 0.962) |
| delaunay | **16** | **+24581.8** | **1.6314** | 30078.7 | truth basin, 5497 short of bar |
| knn | 8 | +1548.2 | 0.9958 | 28791.5 | wrong basin (Nautilus mode 1.011) |
| knn | **16** | **+28693.8** | **1.6007** | 28791.5 | truth basin, **99.7% of bar** |

Doubling starts alone moves Delaunay ~34,000 nats and KNN ~27,000 nats, from
the sampler's wrong mode into the truth basin. **The start-count explanation
holds.**

Two riders that the earlier text does not cover:

- **KNN needs far fewer steps than "≥1500" with fixed reg.** That budget was
  extrapolated from the free-AdaptSplit CPU run breaking out near step 1300.
  With fixed reg, KNN reaches r_E 1.6007 and 99.7% of its truth bar inside
  **300** steps (90% of bar at step 271) and was still climbing at the cap
  (+587 over the last 10 steps). The long budget is a property of free
  AdaptSplit, not of the mesh.
- **Delaunay at 16 starts / batch 4 converges to a local maximum.** It
  plateaued at +24581.8 / r_E 1.6314 — flat for the last 10 steps, so it
  stopped rather than ran out of budget — against +30202 / r_E 1.600 near step
  150 for the 16-start CPU run at the same fixed reg (inner ~0.0316, outer
  ~3.162). **Resolved in §6: the batch size is the cause.** At batch 2 the same
  16-start configuration reproduces the CPU result to 1.3 nats.

## 2. VRAM ceiling (6 GiB RTX 2060 Max-Q)

| mesh | batch 4 | batch 8 |
|---|---|---|
| delaunay_nn | fits | **OOM** — single 5.64 GiB allocation |
| delaunay | fits | **OOM** — single 3.93 GiB allocation |
| rectangular | OOM at batch **1** (10.90 GiB) | — |

**Batch 4 is a hardware ceiling for the whole Delaunay family on this card**,
not a tuning preference — the batch-4 winner sits directly on it. Both failures
are recorded as `failure_kind: vram` artifacts.

## 3. DelaunayNN starts curve (batch 4, fixed reg)

| starts | max logL | r_E | steps to bar | wall | wall to bar | s/step | resurrections |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 30147.9 | 1.5972 | never (still climbing) | 815 s | — | 2.72 | 7 |
| **8** | 30374.8 | 1.5998 | 175 | 1524 s | **889 s** | 5.08 | 10 |
| 16 | **30388.2** | 1.5992 | **162** | 2827 s | 1527 s | 9.42 | 31 |

The three phase-2 criteria split, and the split is lopsided:

- **Highest likelihood:** 16 starts, by 13 nats — noise.
- **Fewest steps to bar:** 16 starts, by 13 steps — noise.
- **Shortest wall to bar:** **8 starts, by 1.7x.** 8 -> 16 buys 13 steps while
  paying 1.85x per step, so it is a net loss. Resurrections tripling (10 -> 31)
  shows the extra lanes mostly die in the bad-regularization region.

**8 starts / batch 4 is confirmed the wall-time optimum**, now from a measured
curve rather than from being the only batch-4 configuration tried.

### Correction: 4 starts is not "too few"

The phase-1 settings table reads *"4 / 1 / 200 — too few starts; improves but
remains in the wrong basin."* That cell moved three variables at once (4 starts
**and** batch 1 **and** 200 steps) relative to the 8-start winner. Holding batch
and steps fixed, **4 starts does reach the truth basin** (+30147.9 at r_E
1.5972). Its history shows why the earlier run missed:

```
step 200-230   -34518.6   flat — a regularization mode, not convergence
step 240       -24837.2   late breakout
step 290       +30127.1
step 299       +30147.9   still climbing (+20.7 over last 10), 156 nats short of bar
```

So 4 starts neither failed nor converged — it escaped late and ran out of
budget mid-climb. The failure in the earlier cell was the batch-1/200-step
budget, not the start count. The defensible statement is that **4 starts works
but gambles on a late resurrection breakout, so its timing is luck; 8 starts
breaks out early and reliably.** This is the same "long plateau is a reg mode,
not convergence" lesson as #117, reappearing at low start counts.

## 4. Recommended settings (revised)

- **DelaunayNN** — 8 starts, batch 4, fixed/inherited AdaptSplit, resurrection
  on, 200-step cap. Unchanged, now backed by the curve above. Batch 4 is the
  VRAM ceiling, not a choice.
- **Delaunay** — 16 starts, **batch 2**, fixed reg, ~300 steps. Batch 4 is
  actively wrong on this mesh: it converges to a local maximum 5622 nats low at
  r_E 1.6314. Batch 2 costs ~30% more wall and recovers truth exactly (§6).
- **KNN** — 16 starts, batch 4, fixed reg, ~300 steps (not 1500). Reserve the
  ≥1500-step budget for the free-AdaptSplit parametrization.
- **Rectangular** — unchanged: not runnable at production size on this card.

## 5. Not covered

- **KNN free-AdaptSplit on GPU** was deliberately not run: it is the one arm
  needing ~1500 steps, costing more than the rest of this matrix combined.
  Its CPU recovery (+29724 / 1.599 near step 1300) stands as the evidence.
- **DelaunayNN free-AdaptSplit beyond 300 steps** — see §6.2; it was still
  creeping upward at the cap, and the KNN precedent suggests a ~1500-step
  budget would be needed to test escape properly.

## 6. Batch size and regularization (2026-08-13)

The two cells that were still running when §1--§5 were written.

### 6.1 Batch size decides whether Delaunay finds truth

| mesh | starts | batch | max logL | r_E | steps to bar | wall | resurrections |
|---|---:|---:|---:|---:|---:|---:|---:|
| delaunay | 16 | **2** | **+30203.3** | **1.6001** | **253** | 2080 s | 9 |
| delaunay | 16 | 4 | +24581.8 | 1.6314 | never | 1598 s | 6 |

At batch 2 the 16-start Delaunay run reproduces the CPU reference (+30202 @
r_E 1.600) to **1.3 nats**, and recovers the full truth model — r_E 1.6001
against truth 1.6, shear (0.0512, 0.0502) against (0.05, 0.05), centre
(0.0024, -0.0029) against (0, 0). The batch-4 local maximum of §1 is therefore
**caused by the batching**, at a cost of ~30% wall time to avoid.

Crucially this is **mesh-specific**, and it splits along the line the mesh
design predicts:

| mesh | batch 2 | batch 4 | agree? |
|---|---:|---:|---|
| delaunay | +30203.3 @ 1.6001 | +24581.8 @ 1.6314 | **no — 5622 nats apart** |
| delaunay_nn | +30374.3 @ 1.59868 | +30374.8 @ 1.59979 | yes — 0.5 nats |

Plain Delaunay's mapper is discontinuous through triangulation flips, so the
small floating-point differences batching introduces can tip a trajectory
across a flip into a different basin. DelaunayNN's Sibson natural-neighbour
interpolation is continuous through those flips and lands in the same place
either way. **The smoother gradient buys reproducibility under numerical
perturbation, not just faster convergence** — a property worth more than the
raw step count, and one that shows up only in a batch comparison.

Consequence for §3: "batch 4 is the winner" was established on DelaunayNN and
**does not generalise**. Batch 4 is the DelaunayNN optimum and the VRAM
ceiling; on plain Delaunay it is a correctness hazard.

### 6.2 Free AdaptSplit does not recover within 300 steps

DelaunayNN, 8 starts, batch 4, 300 steps, free AdaptSplit (inner, outer):

| regularization | max logL | r_E | steps to bar | resurrections |
|---|---:|---:|---:|---:|
| fixed (0.316227766) | **+30374.8** | **1.5998** | 175 | 10 |
| **free AdaptSplit** | -790.5 | 1.0452 | never | **109** |

Free reg is catastrophic here: the wrong basin entirely (recovered shear
(0.1359, 0.0914) against truth (0.05, 0.05)) and a **10.9x resurrection rate**,
the #104/#117 AdaptSplit signature of lanes dying in the double-squared
high-coefficient region. Mapper capacity was never the problem — all overflow
and degeneracy counts stayed zero.

The trace sat at -1142 from step ~120 to ~280 then crept to -794.9, gaining
+347 over the last 50 steps, i.e. still climbing at the cap. That is the "long
plateau is a reg mode, not convergence" pattern again.

**One thing this artifact cannot settle:** whether the 109 lane deaths are NaN
deaths (as the truth-bar scan showed for plain Delaunay at inner >= 10) or
survivable over-regularized-floor deaths (as on KNN). The recorded FoM history
holds no non-finite entries, but it tracks the *best surviving* lane, so it
would not show a dead lane's NaN either way. Distinguishing the two needs a
DelaunayNN truth-bar reg scan at high coefficients, which was not run.

Either way the practical recommendation is unchanged and now evidenced: **use
fixed/inherited regularization**. Free AdaptSplit costs at least 5x the step
budget for no benefit, and per #117 free Matern is the better free
parametrization if a free reg is genuinely required.
