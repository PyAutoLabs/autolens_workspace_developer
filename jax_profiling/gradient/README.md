# JAX autodiff gradient audit

Status of `jax.grad` / `jax.value_and_grad` through the PyAutoLens likelihoods,
maintained by the gradient probes in this folder and the finite-difference (FD)
correctness tests in `autolens_workspace_test/scripts/jax_grad/`.

Audit issue: https://github.com/PyAutoLabs/autolens_workspace_developer/issues/87
(2026-07). "FD-validated" means the autodiff gradient agrees with central finite
differences parameter-by-parameter (`jax_grad/util.py`), not merely that it is
finite and non-zero.

## Status by likelihood

| Likelihood | Autodiff status | FD-validated | Notes |
|---|---|---|---|
| Imaging, standard `lp.Sersic` | **works** | **yes** (2026-07-09, rel err ≤ 1e-5 over 22 params) | pure profile evaluation; no inversion; `jax_grad/imaging_lp.py` |
| Imaging, `lp_linear.Sersic` | **works** | **yes** (2026-07-09, rel err ≤ 3e-9 over 20 params) | linear intensities via positive-only NNLS solve; gradient flows through the solve, including source-shape params |
| Imaging, MGE (`lp_linear.Gaussian`) | **works** | **yes** (2026-07-09, rel err ≤ 5e-7 over 10 params) | probe `imaging/mge.py` 9/9 PASS on current mains; `lmp_linear.GaussianGradient` (light+mass) has known unresolved autodiff issues — excluded from models here |
| Interferometer, MGE | **works** (re-confirmed 2026-07-09: probe 9/9 PASS) | not yet | probe: `interferometer/mge.py` (NUFFT/DFT visibility space) |
| Interferometer, standard `lp.Sersic` | **works** | **yes** (2026-07-09, rel err ≤ ~1e-7 over 14 params; `jax_grad/interferometer.py` variant A) | gradient flows through the DFT visibility transform |
| Interferometer, `lp_linear.Sersic` | **works** | **yes** (2026-07-09, rel err ≤ ~1e-6 over 13 params) | linear source through NNLS in visibility space |
| Interferometer, `RectangularUniform` (sparse operator) | **works** | **yes** (2026-07-09, strict FD match, all 7 mass/shear params live; `jax_grad/interferometer.py` variant C) | `TransformerDFT` + `apply_sparse_operator(use_jax=True)`; the gradient-capable mesh for interferometer |
| Interferometer, `RectangularAdaptDensity` + `reg.Adapt` (sparse operator — **the production config**) | **autodiff correct — but zero everywhere: the staircase, with no escape hatch** | **yes** (2026-07-09; `jax_grad/interferometer.py` variant B) | interferometer pixelization has **no over-sampling**, so mesh queries always coincide with the rank-transform knots — the imaging os_pix=1 staircase applies in full. With no lens light in the model, *every* parameter's gradient is (correctly) ~zero: no usable gradients at all. Fix requires `RectangularUniform` or a smooth-density transform (see below) |
| Imaging, `RectangularUniform` | **works** | **yes** (2026-07-09: AD = FD to 7 s.f., FD-step-stable, all 14 params; `jax_grad/imaging_pixelization.py` variant A) | non-adaptive mesh: fully smooth likelihood, ready for gradient-based inference |
| Imaging, `RectangularAdaptDensity` (pixelization over-sampling 1) | **autodiff correct — likelihood is a staircase in mass/shear** | **yes** (see below) | lens-light params FD-matched (≤2e-8); mass/shear: LL is *bit-identical* under ≤1e-6 parameter shifts, so AD's ~zero is the true a.e. derivative and larger-step FD only measures discrete rank-reordering jumps. Gradient-based **mass** inference impossible in this config — a likelihood-design property, not an AD bug (see "Rectangular adaptive mesh" section) |
| Imaging, `RectangularAdaptDensity` (pixelization over-sampling 4) | **works** | **yes** (2026-07-09: all 14 params live, AD ≈ FD(h=1e-7) ≤ ~3%, FD converges to AD as h→0; variant D of `jax_grad/imaging_pixelization.py`) | sub-pixel strain between interp queries and knots carries smooth mass information; residual AD-FD gap is micro-staircase contamination of FD, not AD error |
| Imaging, `RectangularAdaptImage` + `reg.Adapt` + adapt images + border relocator (pixelization over-sampling 4 — **the production config**) | **works** | **yes** (2026-07-09: all 14 params live, AD ≈ FD(h=1e-7) ≤ ~1%, lens light to 6 digits; variant C of `jax_grad/imaging_pixelization.py`) | full production shape incl. `AdaptImages` weight map and border relocator; mixed precision off in the test (FD needs float64) |
| Imaging, `RectangularKernelAdaptDensity` (os_pix=1, bandwidth=0.1) | **works — the os_pix=1 staircase corner is fixed** | **yes** (2026-07-10: all 14 params incl. mass/shear at STRICT tolerances via FD-step-sweep; variant E of `jax_grad/imaging_pixelization.py`) | kernel-density CDF mesh (PyAutoArray#374): no ranks/sorts, C^∞ transform; FoM *beats* the linear `AdaptDensity` by 4.1e-2 relative at bandwidth 0.1 (value-parity is undefined at os_pix=1 — the linear mesh there is the staircase) |
| Imaging, `RectangularKernelAdaptDensity` (os_pix=4) | **works** | **yes** (2026-07-10: strict FD-step-sweep, all 14 params; variant F) | FoM parity with linear `AdaptDensity` = 2.7e-5 relative at default bandwidth |
| Imaging, `RectangularKernelAdaptImage` + `reg.Adapt` + adapt images + border relocator (os_pix=4, bandwidth=0.1) | **works** | **yes** (2026-07-10: strict FD-step-sweep, all 14 params; variant G) | full production shape; FoM parity floor 6.3e-4 relative (intrinsic: the kernel smooths the adapt-image weights over its bandwidth; swept bandwidth×n_knots 2026-07-10) |
| Interferometer, `RectangularKernelAdaptDensity` + `reg.Adapt` (sparse operator) | **works — the interferometer staircase now has its escape hatch** | **yes** (2026-07-10: strict FD-step-sweep, all 7 mass/shear params live; variant D of `jax_grad/interferometer.py`) | the production-shape adaptive mesh with usable gradients on the no-over-sampling path; FoM parity 3.9e-5 relative at default bandwidth |
| Imaging, Delaunay pixelization | **works — frozen-tables a.e.-exact gradient** (SHIPPED 2026-07-26: `stop_gradient` on the tables callback input) | **yes** (2026-07-26: median 9.6e-6, max 2.1e-3 over 14 params at documented rtol=1e-2; `jax_grad/delaunay.py`) | everything differentiable runs in-graph (visibility-walk point location, barycentric weights, dual areas, split points); the callback returns only int32 tables, piecewise-constant in the vertices, so freezing them under differentiation drops nothing — the exact a.e. derivative. Residual FD scatter on mass/shear = FD steps crossing triangle-flip events (measure-zero likelihood discontinuities). Batched caveat: callback is `vmap_method="sequential"` — KNN meshes remain the vmap-throughput option. See "Delaunay gradients: SHIPPED" section below |
| Imaging, `KNearestNeighbor` (Wendland kNN) + `reg.ConstantSplit` / `reg.AdaptSplit` | **works** | **yes** (2026-07-26: strict FD-step-sweep, all 14 params, rel err ≤ 3.3e-8; `jax_grad/knn.py` variants A/B) | the JAX-native Delaunay-family mesh: Hilbert image mesh + edge zeroing, no scipy callback anywhere in the graph — gradients flow through traced query points AND traced mesh vertices. **Split-family regularization only**: `reg.Constant`/`ConstantZeroth`/`Adapt` need `MeshGeometryDelaunay.neighbors` (a direct scipy call on the traced mesh grid) and raise `TracerArrayConversionError` under `jax.grad` — pinned as a negative test in the script. Science caveat: Wendland kNN historically underperforms Delaunay (kernel knobs, caustic smearing — see PyAutoArray#317 background) |
| Imaging, `KNNBarycentric` + `reg.ConstantSplit` | **works** (gradients only) | **yes** (2026-07-26: strict FD-step-sweep, all 14 params, rel err ≤ 4.1e-7; `jax_grad/knn.py` variant C) | 3-nearest barycentric weights; slightly noisier FD than Wendland (3-NN-set swaps move weights discontinuously — measure-zero jump sites). **Mesh failed its science gate as a Delaunay replacement** (PyAutoArray#317: 2.2% log-evidence drift, ~5% of vertices unreachable) — certified for gradient correctness, not for production science |
| Point source, source-plane χ² (`FitPositionsSource`) | **works** (probe 4/4 PASS; forward `jax.jit` still blocked by the `Grid2DIrregular` xp gap) | **yes** (2026-07-09, rel err ≤ 5e-6; `jax_grad/point_source.py`) | includes magnification-via-Hessian term (3rd derivatives of the potential); flux/H0 legitimately zero in positions-only fits |
| Point source, image-plane (`FitPositionsImagePairAll`) | prior probe: **not differentiable** | n/a | `PointSolver` triangle-tiling forward solve uses `jnp.where` masking + integer neighbour lookups |
| Weak lensing (`FitWeak`, `xp=jnp`) | **works** | **yes** (2026-07-09, rel err ≤ 3e-9, plain + redshift-scaled; `jax_grad/weak.py`) | gradients through the deflection-Hessian shear derivation are correct; no step-by-step probe needed — full pipeline validated first try |

## Reading the table

- **works** — `value_and_grad` returns finite, non-zero gradients end-to-end.
- **works with intentional approximation** — gradients flow, but documented
  `stop_gradient` (or equivalent) drops specific derivative terms; the FD tests
  measure the deviation instead of hiding it under a loose tolerance.
- **hard error / not differentiable** — autodiff raises or is structurally
  meaningless; the per-likelihood section documents the root cause.

## Rectangular adaptive mesh: the staircase verdict (phase 2b, 2026-07-09)

The adaptive mesh (`RectangularAdaptDensity`) maps ray-traced source-plane points to
rank space via `create_transforms` (`PyAutoArray/autoarray/inversion/mesh/interpolator/rectangular.py`):
a per-axis sort + `jnp.interp` CDF transform — an implementation of the "ray-guided
transformed uniform grid" of **arXiv:2606.30620** (Enzi, Krawczyk, Li & Collett).

**History**: ray-traced grids contain ~50% exactly-duplicate per-axis coordinates,
which made `jnp.interp`'s vjp explode to O(1e24) (PyAutoArray PR **#281**, fix via
`stop_gradient` on the sort-knots — closed **unmerged** 2026-06-01, staled by the
interpolator refactor). On the refactored main there is no explosion and no
`stop_gradient`; the failure mode changed shape entirely:

**Measured verdict (FD step-size scan, jax_test HST-like config, 14-param model):**

| config | einstein_radius: AD | FD h=1e-7 | FD h=1e-6 | FD h=1e-5 |
|---|---|---|---|---|
| `RectangularUniform`, os_pix=1 | −5.868261e5 | −5.868261e5 | −5.868261e5 | −5.868261e5 |
| `AdaptDensity`, os_pix=1 | 0.0 | **0.0 (LL bit-identical)** | **0.0 (bit-identical)** | +1.7e4 (jump artifact) |
| `AdaptDensity`, os_pix=4 | +7.33e4 | +7.57e4 | +1.9e4 (jumps) | −2.2e5 (jumps) |

- With over-sampling 1 the interp queries coincide with the knots, so the mapping is
  **exactly invariant** under any order-preserving deformation of the traced grid
  (rigid translations exactly; general smooth mass changes below the first rank
  re-ordering). The likelihood is piecewise-constant in the mass/shear directions:
  autodiff's zero is the *correct* almost-everywhere derivative, and naive FD
  "gradients" are pure discontinuity artifacts (ΔLL jumps of O(1–1000) at the
  re-ordering scale). Smooth mass information is *destroyed by the discretisation*,
  not lost by autodiff.
- With over-sampling > 1 the sub-pixel queries sit between knots and pick up local
  strain, restoring a genuine smooth mass gradient that autodiff tracks. Full
  14-parameter FD sweeps at os_pix=4 (2026-07-09): `RectangularAdaptImage` in the
  production shape (`reg.Adapt`, `AdaptImages`, border relocator) agrees with
  FD(h=1e-7) to ≤ ~1% on mass/shear and 6 digits on lens light;
  `RectangularAdaptDensity` to ≤ ~3% (worst: einstein_radius). In both cases the
  FD values drift with step size while AD is h-consistent — the residual gap is
  micro-staircase contamination of the finite differences, not an autodiff error.
- The uniform mesh has no transform and is exactly smooth-differentiable.

**Implications** *(updated 2026-07-10 — the continuous transform now exists)*:
(1) HMC/NUTS over mass parameters works with `RectangularUniform`, with either
linear adaptive mesh at pixelization over-sampling > 1 (imaging production
config validated ≤ ~1%), **and now with the kernel-CDF meshes
(`RectangularKernelAdaptDensity` / `RectangularKernelAdaptImage`,
PyAutoArray#374) in every configuration** — the continuous density transform
anticipated below shipped as opt-in mesh classes and is FD-certified at strict
tolerances at os_pix=1, os_pix=4 and on the interferometer sparse path
(variants E/F/G of `imaging_pixelization.py`, variant D of `interferometer.py`).
The linear meshes' staircase documentation above remains accurate — they are
unchanged. (2) even non-gradient samplers see a micro-staircase surface with
the *linear* meshes at os_pix=1 — worth keeping in mind for evidence estimates;
the kernel meshes remove this too. (3) PR #281's fix is moot on the refactored
code — do not re-land it.

**Kernel-mesh caveats** (2026-07-10): (a) *resolved same day* (PyAutoArray#376):
the exact kernel forward previously broadcast O(M×N) memory (~60 GB at the
production imaging scale of M ≈ 246k over-sampled queries × N ≈ 15.4k traced
points — observed OOM); it now evaluates in fixed 512-query blocks (`lax.map`
under jax, block loop under numpy) with float-identical values — the same
scale runs at ~1.1 GB peak RSS and the certification re-passes unchanged. CPU
wall-time at that scale is ~10 min/eval (2×10⁹ erf evaluations — the
arithmetic, not the blocking); GPU remains the production target for kernel
meshes at scale. (b) FD probing of any pixelized-source likelihood is
poisoned pseudo-randomly by measure-thin branch flips (width < 1e-15 in the
parameter, ΔLL ~1.6e-3 on the 8×8 interferometer config up to ~14 on 28×28
imaging; present under `reg.Constant` and `reg.Adapt`; pre-existing, exposed —
not caused — by the kernel meshes making mass/shear FD certifiable at all).
Investigated 2026-07-10 (PyAutoArray#377): the flips are **JIT-only** (eager is
clean — an XLA-fusion ulp crossing a discrete threshold), the positive-only
solver is **exonerated** (flips persist unconstrained and are
solver-tolerance-invariant), and one amplifier is confirmed: the linear
rank-CDF forward transform is genuinely discontinuous at the data
bounding-box edge (U jumps by 1/(N+1) crossing the max point). Fix candidate
and the remaining kernel-config localization live on #377.
`jax_grad/util.py` therefore runs an FD-step-sweep for the kernel variants:
per parameter, FD at rel steps {1e-8, 1e-7, 1e-6}, compared at the step closest
to autodiff — clean steps converge to AD at 1e-6..1e-9 relative, so a wrong AD
fails every step. The flips deserve their own investigation (likely
positive-only-solver / PDIP tie-breaks — NNLS-ledger territory).

## Delaunay gradients: SHIPPED via frozen tables (2026-07-26)

The section below this one records the 2026-07-09 verdict and is kept as
history; its premises are superseded. The in-graph visibility walk moved
every differentiable quantity (point location, barycentric weights, dual
areas, split points) inside the JIT program, leaving the host qhull
``pure_callback`` returning only int32 connectivity tables. Those tables are
piecewise-constant in the vertex positions — their true derivative is zero
between re-wiring events — so wrapping the callback input in
``stop_gradient`` (shipped in ``_jax_delaunay_tables``) yields the EXACT
almost-everywhere derivative, not an approximation (the audit's point 3
below, now realised; points 1–2 described the pre-walk architecture).

Certification: `autolens_workspace_test/scripts/imaging/jax_grad/delaunay.py`
(production shape: Hilbert + edge zeroing + AdaptSplit) — all 14 params live,
lens light at 1e-8..1e-10, mass/shear at 1e-5..2e-3 (FD steps straddling
triangle-flip events — measure-zero likelihood discontinuities where no
method has a gradient; AD differentiates the branch the point is on), at a
documented rtol=1e-2. Batched-sampler caveat: the callback remains
``vmap_method="sequential"`` (one host qhull call per vmap lane) — the KNN
meshes stay the batched-throughput option.

## Why Delaunay gradients were infeasible before the walk (phase 2a, 2026-07-09 — superseded, see above)

Re-confirmed on current mains via `imaging/delaunay.py`: **3 PASS / 8 ERROR**. The
pre-inversion stages (ray-trace, blurred lens light, profile subtraction) are fully
differentiable; every stage from the mapping matrix onward hard-errors with

```
ValueError: Pure callbacks do not support JVP. Please use `jax.custom_jvp` ...
```

raised from `_jax_delaunay_tables` (`PyAutoArray/autoarray/inversion/mesh/interpolator/delaunay.py:126`),
where `scipy.spatial.Delaunay` is host-called via `jax.pure_callback` to build the
triangulation tables (simplices, neighbours, vertex→simplex). `pure_callback` has no
JVP rule, so `jax.value_and_grad` through any Delaunay-mesh likelihood raises rather
than silently zeroing.

Three structurally distinct obstacles, in increasing depth:

1. **Mechanical**: the callback needs a `jax.custom_jvp` wrapper before *anything*
   downstream can differentiate. The natural rule is a zero-JVP (treat the
   triangulation tables as locally constant), which is also mathematically honest —
   the tables are integer-valued and piecewise-constant in the vertex positions.
2. **Frozen-triangulation approximation**: with a zero-JVP rule, gradients flow
   through the barycentric weights *within* the frozen triangulation but ignore how
   the triangulation itself re-wires as source-plane vertices move. This is the
   Delaunay analogue of the rectangular mesh's dropped bin-boundary terms — usable
   for samplers, but an approximation whose error grows near mesh re-wiring events.
3. **Fundamental**: connectivity changes are discrete. Exact end-to-end gradients of
   a Delaunay-mesh likelihood do not exist at re-wiring boundaries, and between them
   the "gradient of the triangulation" is exactly zero — so the frozen-triangulation
   gradient is in fact the correct one almost everywhere in parameter space. The
   practical question is only whether the sampler tolerates the (measure-zero)
   non-smooth set; the literature answer (cf. arXiv:2606.30620, which abandons
   tessellation for a continuous CDF-transformed uniform grid precisely for this
   reason) is that a continuous formulation is preferable when gradients matter.

**Follow-up filed** (not done here — PyAutoArray is claimed by another task): wrap
`_jax_delaunay_tables` in `jax.custom_jvp` with a zero rule, then FD-validate the
frozen-triangulation gradients the same way as the rectangular mesh.

## Final assessment: rectangular-mesh pixelized-source gradients (2026-07-26)

Post-consolidation (PyAutoArray#403 — the kernel-CDF meshes now ARE
`RectangularAdaptDensity` / `RectangularAdaptImage`; the FD tests are variants
A–D of `autolens_workspace_test/scripts/imaging/jax_grad/pixelization.py`), a
final certification re-run plus a linear-algebra precision probe (fresh
environment, jax 0.10.2 CPU, float64) settles the two standing questions.

**Do the gradients work?** Yes — re-certified. All four variants pass strict
FD on all 14 parameters: `RectangularUniform` rel err ≤ 2e-7,
`RectangularAdaptDensity` os_pix=4 ≤ 1.7e-7, and the full production shape
(`RectangularAdaptImage` + `reg.Adapt` + `AdaptImages` + border relocator,
os_pix=4) ≤ 6.2e-8, with typical parameters at 1e-9–1e-11. The one standing
exclusion (os_pix=1 `einstein_radius`) reproduces exactly as documented: all
three FD steps land on branch flips (FD values 5.4e8 / −5.4e7 / −5.3e6 against
a self-consistent AD of +1.5e5) while every other parameter matches at ≤ 4e-7.

**Does the linear algebra need reformulating (à la the slogdet option)?**
No — measured, each candidate in turn on the os_pix=4 production-adjacent
config:

- **`log_det_method="slogdet"` vs `"cholesky"`** (PyAutoArray#391/#392): LL
  differs by 6.4e-10 and the 14-parameter AD gradients agree to **2.1e-15**
  max relative — in the positive-definite regime the two formulations are
  gradient-equivalent to machine precision. `slogdet` buys *robustness*
  (finite + differentiable where the Cholesky NaNs, i.e. extreme
  regularization coefficients during exploration), not precision; keeping it
  opt-in for gradient-based searches is the right call.
- **Relaxed-KKT NNLS backward pass** (jaxnnls implicit differentiation):
  sweeping `target_kappa` 1e-9 → 1e-13 around the 1e-11 default moves the
  gradient by ≤ **3.1e-10** max relative — the backward-pass relaxation error
  is 2–3 orders of magnitude below the FD certification floor. Tightening
  `nnls_solver_tol` to 1e-12 moves gradients by 7.5e-15 (the forward solve is
  already converged).
- **NNLS custom-VJP vs exact unconstrained solve**: with
  `use_positive_only_solver=False` (plain `linalg.solve`, exact built-in
  implicit autodiff) the AD-vs-FD floor is the same (max 1.1e-7 vs 1.7e-7) —
  the positive-only solver's custom VJP loses nothing measurable against the
  analytically exact reference.
- **os_pix=1 branch-flip cross-check**: the `einstein_radius` FD poisoning
  persists unchanged under the unconstrained solver — re-confirming #377's
  solver exoneration on the imaging config (the flips live in the fused JIT
  graph, not the NNLS solve). AD remains the trustworthy value there.

The AD-vs-FD residual (~1e-7 worst-case) is FD noise (step quantization +
micro-flips), not autodiff error: FD converges toward AD as steps shrink at
every clean step. **Verdict: gradients for rectangular-mesh pixelized sources
are correct and already at the measurable precision floor; no linear-algebra
reformulation is warranted.** What remains open is orthogonal to gradient
precision: the JIT-only branch-flip localization (#377 follow-up — an XLA
fusion ulp threshold, measure-thin, documented LL accuracy floor under jit),
and the bandwidth-default quality question
(`PyAutoMind/draft/research/autoarray/rectangular_kernel_bandwidth_defaults.md`),
which affects reconstruction quality, not differentiability.

## Regularization × mesh gradient matrix (2026-07-26 sweep)

Every ``al.reg`` scheme swept against the gradient-capable meshes
(``RectangularAdaptDensity`` os_pix=4; ``KNearestNeighbor`` /
``KNNBarycentric`` with Hilbert image mesh + edge zeroing, os_pix=1), on the
jax_test 14-parameter fiducial. Positive results are pinned by
``autolens_workspace_test/scripts/imaging/jax_grad/regularization.py``;
mesh-family negatives by ``jax_grad/knn.py``.

| Regularization | Rectangular (kernel-CDF) | KNN meshes | Notes |
|---|---|---|---|
| `Constant` | **FD-certified** (jax_grad/pixelization.py) | **hard error** | rectangular neighbors are analytic/static; Delaunay-family neighbors call scipy on the traced mesh grid (`MeshGeometryDelaunay.neighbors`) → `TracerArrayConversionError` |
| `Adapt` | **FD-certified** (production config, pixelization.py variant D) | **hard error** | same neighbors split as `Constant` |
| `ConstantSplit` / `AdaptSplit` | **incompatible** (shape error: split machinery expects 4-cross-per-pixel, rectangular's shared 4-corner mappings are per-query) | **FD-certified** (jax_grad/knn.py, ≤ 3.3e-8) | the split family is the KNN/Delaunay-family production pairing; note `AdaptSplit()` at default inner==outer==1.0 ≡ `ConstantSplit(1.0)` |
| `Zeroth` | **FD-certified** (1.4e-7) | **FD-certified** (7.5e-8 / 2.9e-7) | neighbour-free, pure xp |
| `MaternKernel(nu=2.5)` | **FD-certified strict** (2.2e-4) | **works — FD-limited** (~2e-3, at the FD noise floor) | **the tfp question: YES, gradients flow** — `tfp.substrates.jax.math.bessel_kve` (tfp-nightly) ships a registered gradient w.r.t. its argument (`nu` is static), and the dense-covariance Cholesky inverse differentiates. See conditioning note below |
| `MaternKernel(nu=0.5)` / `MaternAdaptKernel` / `GaussianKernel` | **works — FD-limited** (1.4e-2 / 9.4e-2, at each variant's FD noise floor) | **works — FD-limited** (~2e-3 / 6.7e-3) | gradients finite + live everywhere; the likelihood itself carries a 1e-6..4e-5 absolute numerical noise floor (dense kernel inverse / bessel lowering), which central FD divides by the step — the "mismatch" equals that floor in every case, i.e. no evidence of wrong AD. `MaternAdaptKernel()` at default coefficients ≡ `MaternKernel` (uniform weights) |
| `BrightnessZeroth` | **hard error** | **hard error** | not xp-ported: numpy boolean ops on the traced pixel-signals array (`TracerArrayConversionError`) |
| `ExponentialKernel` | **works — FD-limited** (kernel-family noise floor, as `Matérn ν=0.5` — the exponential kernel *is* Matérn ν=0.5) | **works — FD-limited** | xp-ported 2026-07-26 (was: numpy (N,N,2) pairwise-diff build + un-threaded `xp`; also switched to the NaN-safe `sqrt(d²+1e-20)` distance form — `linalg.norm`'s derivative is NaN at the zero diagonal) |
| `ConstantZeroth` | **broken** (numpy too) | **broken** | known dead code — missing `neighbors_sizes` argument (filed: `draft/bug/autoarray/constant_zeroth_broken_dead_code.md`) |
| `CurvatureMask` / `FourthOrderMask` | **incompatible** | **incompatible** | dpsi (potential-correction) schemes sized to the data grid (952), not source-mesh schemes (784/330) — shape error by construction |
| `AdaptSplitZeroth` | **incompatible** (split shape) | **hard error** (neighbors via its zeroth/adapt leg) | |

**Kernel-scheme conditioning note.** The kernel regularizations form
``coefficient * C^-1`` explicitly (``inv_via_cholesky``). On the rectangular
mesh's well-spaced vertices cond(C) ≈ 3e5 (nu=2.5) and the scheme is
strict-FD-certifiable; on the KNN meshes' TRACED (clustered) vertices
cond(C) ≈ 1.4e9 (min pairwise separation 7e-3 vs median 9e-2), which puts a
~1e-6 absolute noise floor on the likelihood — the one genuine
linear-algebra reformulation candidate this sweep surfaced (avoid the
explicit inverse, e.g. keep H implicit through the Cholesky of C for the
``s^T H s`` and ``log det H`` terms, and/or scale the 1e-8 diagonal jitter
with the kernel's dynamic range).

## Findings log

- **2026-07-26** (Delaunay frozen-tables gradient probe): `stop_gradient` on
  the tables `pure_callback` input unlocks `jax.grad` through the full
  Delaunay likelihood — the in-graph visibility walk means only the int32
  tables (true derivative: zero between re-wirings) are frozen, so this is
  the exact a.e. gradient, not an approximation. Probe on the production
  shape: 14/14 params live, FD median 9.6e-6 / max 2.1e-3, eager==jit. Ship
  task filed to PyAutoMind. Remaining Delaunay-vs-KNN trade: the callback is
  `vmap_method="sequential"` — one host qhull call per vmap lane — so the
  KNN meshes stay the batched-throughput option.
- **2026-07-26** (regularization × mesh sweep, table above +
  `jax_grad/regularization.py` — new): every `al.reg` scheme swept on both
  gradient-capable mesh families. New FD certifications: `Zeroth` (both
  families) and `MaternKernel(nu=2.5)` on rectangular — settling the
  tfp/Bessel question: **Matérn gradients work** (tfp-nightly `bessel_kve`
  has a registered gradient). Kernel schemes elsewhere are AD-live but
  FD-limited by the likelihood's own noise floor (dense `C^-1` at
  cond ~1e9 on clustered traced vertices) — reformulation candidate noted.
  xp-port gaps found: `BrightnessZeroth`, `ExponentialKernel` (hard error
  under trace); split-family structurally incompatible with rectangular;
  `CurvatureMask`/`FourthOrderMask` are dpsi-only.
- **2026-07-26** (KNN mesh certification, `jax_grad/knn.py` — new): both
  k-nearest-neighbour meshes FD-certified strict on all 14 params
  (`KNearestNeighbor` + ConstantSplit/AdaptSplit ≤ 3.3e-8 rel err;
  `KNNBarycentric` ≤ 4.1e-7). Pure JAX end to end — no scipy callback,
  gradients flow through traced mesh vertices as well as queries. Boundary
  pinned: neighbour-based regularizations (`Constant`/`ConstantZeroth`/
  `Adapt`) raise `TracerArrayConversionError` (scipy `Delaunay` on the
  traced mesh grid in `MeshGeometryDelaunay.neighbors`) — split-family and
  kernel schemes are the JAX pairings. `AdaptSplit()` at default
  coefficients (inner == outer == 1.0) is numerically identical to
  `ConstantSplit(1.0)` — use asymmetric coefficients to exercise the
  adaptive path.
- **2026-07-26** (final assessment, this section above): certification
  re-passes post-consolidation; slogdet/cholesky gradient-equivalent to 2e-15;
  relaxed-KKT backward error ≤ 3e-10; solver re-exonerated at os_pix=1 on the
  imaging config. No linear-algebra reformulation warranted.
- **2026-07-10** (kernel-CDF certification, PyAutoArray#373/#374): the
  kernel-density CDF meshes pass strict FD on ALL parameters in every
  configuration, including the two previously-dead corners (imaging os_pix=1,
  interferometer sparse). Two new cross-cutting findings: (i) measure-thin
  solver branch flips poison single-step FD pseudo-randomly (see "Kernel-mesh
  caveats" above; `util.compare_gradients` grew an FD-step-sweep mode);
  (ii) the exact kernel forward is O(M×N) memory — production-scale imaging
  needs chunking before the kernel meshes leave opt-in status.
- **2026-07-09** (phase 1, setup): the `jax_grad` FD test scripts previously
  pointed at datasets that no longer exist (`imaging/simple`,
  `imaging/source_complex`) while their auto-simulate fallback writes
  `imaging/jax_test` — i.e. the finiteness tests were broken on a clean
  checkout. Re-pointed at `jax_test` and guarded with
  `al.util.dataset.should_simulate`.

(Sections per likelihood are appended as each phase lands.)
