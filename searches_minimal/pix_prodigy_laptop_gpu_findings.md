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
| RectangularAdaptDensity | RTX 2060, 4 starts, batch 1 (minimum), FP64 | compile failed: XLA requested one 10.90-GiB allocation on a 6-GiB card | **Does not fit this laptop GPU** at production size. This is a VRAM limit, not a Prodigy or gradient failure. The prior CPU bandwidth-0.1 arm was still improving at step 200 (logL -30136; about 5.7 min/step); bandwidth 1.0 remains the preferred smoother follow-up, with truth-point reference +25659.5. |
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
