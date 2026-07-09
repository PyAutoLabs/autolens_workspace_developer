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
| Imaging, `RectangularUniform` | **works** | **yes** (2026-07-09: AD = FD to 7 s.f., FD-step-stable, all 14 params; `jax_grad/imaging_pixelization.py` variant A) | non-adaptive mesh: fully smooth likelihood, ready for gradient-based inference |
| Imaging, `RectangularAdaptDensity` (pixelization over-sampling 1) | **autodiff correct — likelihood is a staircase in mass/shear** | **yes** (see below) | lens-light params FD-matched (≤2e-8); mass/shear: LL is *bit-identical* under ≤1e-6 parameter shifts, so AD's ~zero is the true a.e. derivative and larger-step FD only measures discrete rank-reordering jumps. Gradient-based **mass** inference impossible in this config — a likelihood-design property, not an AD bug (see "Rectangular adaptive mesh" section) |
| Imaging, `RectangularAdaptDensity` (pixelization over-sampling 4) | **works** | **yes** (2026-07-09: all 14 params live, AD ≈ FD(h=1e-7) ≤ ~3%, FD converges to AD as h→0; variant D of `jax_grad/imaging_pixelization.py`) | sub-pixel strain between interp queries and knots carries smooth mass information; residual AD-FD gap is micro-staircase contamination of FD, not AD error |
| Imaging, `RectangularAdaptImage` + `reg.Adapt` + adapt images + border relocator (pixelization over-sampling 4 — **the production config**) | **works** | **yes** (2026-07-09: all 14 params live, AD ≈ FD(h=1e-7) ≤ ~1%, lens light to 6 digits; variant C of `jax_grad/imaging_pixelization.py`) | full production shape incl. `AdaptImages` weight map and border relocator; mixed precision off in the test (FD needs float64) |
| Imaging, Delaunay pixelization | **hard error** (re-confirmed 2026-07-09: 3 PASS / 8 ERROR) | n/a | `jax.pure_callback` → `scipy.spatial.Delaunay` has no JVP rule (`PyAutoArray .../interpolator/delaunay.py:126`); see "Why Delaunay gradients are infeasible today" below (phase 2) |
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

**Implications**: (1) HMC/NUTS over mass parameters works with `RectangularUniform`
or with either adaptive mesh at pixelization over-sampling > 1 — which is the
production configuration (`RectangularAdaptImage`, os_pix=4, validated ≤ ~1%); only
the os_pix=1 adaptive corner is unusable. A continuous density transform (the
paper's actual formulation — CDF from a *smooth* GP-weighted density rather than
empirical point ranks) would remove the micro-staircase entirely;
(2) even non-gradient samplers see a micro-staircase surface with os_pix=1 —
worth keeping in mind for evidence estimates; (3) PR #281's fix is moot on the
refactored code — do not re-land it.

## Why Delaunay gradients are infeasible today (phase 2a, 2026-07-09)

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

## Findings log

- **2026-07-09** (phase 1, setup): the `jax_grad` FD test scripts previously
  pointed at datasets that no longer exist (`imaging/simple`,
  `imaging/source_complex`) while their auto-simulate fallback writes
  `imaging/jax_test` — i.e. the finiteness tests were broken on a clean
  checkout. Re-pointed at `jax_test` and guarded with
  `al.util.dataset.should_simulate`.

(Sections per likelihood are appended as each phase lands.)
