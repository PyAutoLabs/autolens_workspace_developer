# Can multi-start gradient MAP optimizers work for a pixelized source?

**Question (autolens_workspace_developer#100):** the benchmark that promoted
`af.MultiStartAdam` / `MultiStartADABelief` / `MultiStartLion` (PyAutoFit#1369)
used an **MGE** source. Do the same `jax.grad`-driven optimizers work when the
source is a **pixelization**?

## Answer: YES — pixelized likelihoods are gradient-differentiable, given the right config.

This is already **certified** in
`autolens_workspace_test/scripts/jax_grad/imaging_pixelization.py` (commits
`45ce603`, `be8ce41`, `540e093`), which asserts strict / FD-matched gradients on
ALL parameters (incl. lens mass + shear). Re-run 2026-07-14 confirms:

- `RectangularUniform` (os_pix=1): autodiff = FD on all params (7 sig figs).
- `RectangularAdaptImage` + `reg.Adapt` + adapt images + border relocator
  (**os_pix=4**): all gradients live and FD-matched (5%).
- `RectangularAdaptDensity` (**os_pix=4**): all gradients live and FD-matched.
- `RectangularKernelAdaptDensity` / `RectangularKernelAdaptImage`
  (`bandwidth=0.1`): C^inf continuous-density transform, **strict FD on ALL
  params including mass/shear even at os_pix=1** — the meshes built precisely for
  gradient inference.

## Why my first probe said "no" (it was wrong — a methodology error)

The initial `probe_grad_pix*.py` runs reported FAIL. Every cause was in the
probe, not the library:

1. **over_sample_size_pixelization = 1** (never set → default). For the *adaptive*
   meshes the mass gradient is an exact **staircase (zero/flat)** at os_pix=1 —
   the documented "gradient-based mass inference is impossible in THIS
   configuration". The fix is **os_pix=4** (adaptive) or a **kernel-CDF mesh**
   (differentiable at os_pix=1).
2. **Wrong mesh.** I tested `SplineAdaptDensity` / `SplineAdaptImage`. The
   certified differentiable meshes are the **kernel-CDF** ones
   (`RectangularKernelAdapt*`) and adaptive-at-os_pix=4.
3. **Garbage evaluation points.** Broad `UniformPrior`s → random near-median
   points with logL ~ -5e5 (source arcs miss the mesh). The certified harness
   uses **truth-centred `GaussianPrior`s** so the arcs land on the mesh and every
   param has real sensitivity.
4. **FD step too large.** I used a single `eps=1e-4`; the certified harness sweeps
   small steps `1e-8..1e-6` (`util.compare_gradients`) because the likelihood is
   steep, and excludes measure-thin solver-branch-flip steps by name.

## Correct recipe for the experiment (resume tomorrow)

Build the pixelized objective mirroring the certified harness:

- Mesh: **`RectangularKernelAdaptDensity(shape, bandwidth=0.1)`** (simplest,
  differentiable at os_pix=1, no adapt image) — or the production
  `RectangularKernelAdaptImage` + `reg.Adapt` + `AdaptImages` + os_pix=4 +
  `al.Settings(use_border_relocator=True, use_positive_only_solver=True)`.
- **`dataset.apply_over_sampling(over_sample_size_lp=4, over_sample_size_pixelization=…)`**
  (1 for kernel-CDF, 4 for adaptive).
- Model per SLaM pix-1: **fixed** MGE lens light geometry, **free** Isothermal +
  shear mass, free regularization; truth-centred priors so starts are in the
  physical basin.
- Adapt-image trap (if using AdaptImage): `AdaptImages(galaxy_name_image_dict=…)`
  keys must be the **stringified** path tuple `str(("galaxies","source"))`.

Then: `probe_grad` FD-check (small step sweep, near-truth) → local
`af.MultiStartAdam/ADABelief/Lion` + `af.Nautilus` baseline → A100 on RAL. The
open scientific question is not "is the gradient correct" (it is) but "can
**multi-start** gradient descent from broad starts recover the mass basin with a
pixelized source, vs Nautilus" — which the samplers now answer.
