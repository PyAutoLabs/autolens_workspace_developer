# Can multi-start gradient MAP optimizers work for a pixelized source?

**Question (autolens_workspace_developer#100):** the benchmark that promoted
`af.MultiStartAdam` / `MultiStartADABelief` / `MultiStartLion` (PyAutoFit#1369)
used an **MGE** source. Do the same `jax.grad`-driven optimizers work when the
source is a **pixelization** (SLaM SOURCE_PIX-style: fixed MGE lens light, free
Isothermal + shear mass, pixelized source)?

**Answer so far: no, not out of the box.** The FD feasibility gate
(`probe_grad_pix.py`, `probe_grad_pix_adapt_image.py`) found a gradient
obstruction for every pixelized mesh tried — so no A100 sampler runs were
launched (the gate did its job).

## Results (HST imaging, x64, CPU, mass ~7–9 free params)

| Mesh (source) | Forward `logL` | `jax.grad` | Verdict |
|---|---|---|---|
| `RectangularAdaptDensity` + `Constant` | finite | finite but **FD-mismatched ~40–100%** | FAIL_FD_MISMATCH |
| `RectangularSplineAdaptDensity` + `Constant` | finite | finite but **FD-mismatched ~40–100%** | FAIL_FD_MISMATCH |
| `RectangularSplineAdaptImage` + `Adapt` (fixed adapt image) | finite | **NaN (7/9 params)** | FAIL_NAN_OR_INF |

Two distinct obstructions:

1. **Mass-adaptive meshes** (`*AdaptDensity`) build the source-plane mesh from
   the *mass* density. When the mass params change the mesh re-bins — a
   non-smooth operation — so reverse-mode autodiff returns a *wrong* gradient
   (finite, but ~100% off finite-difference). The spline variant
   (`SplineAdaptDensity`) does not fix this: the mesh still adapts to the mass.

2. **Fixed-adapt-image mesh** (`SplineAdaptImage` + `Adapt` regularization,
   adapt image fixed from a bootstrap `AdaptDensity` inversion) removes the
   mass→mesh dependence — the forward `logL` is finite — but `jax.grad` is
   **NaN** for most parameters, at random *and* near-truth points. A non-finite
   reverse-mode gradient somewhere in the spline interpolation / adaptive
   regularization / regularized-inversion (log-det evidence) backward pass.

The MGE benchmark's probe was FD-faithful to ~1e-9 at the same kind of points,
so the probe method is sound; the pixelized likelihood genuinely differs.

## Implication for gradient sampling

`MultiStartAdam` (and ADABelief / Lion) descend on `jax.grad`. A wrong gradient
(case 1) sends them the wrong way; a NaN gradient (case 2) breaks the optimiser
outright. So the gradient MAP optimizers **cannot drive a pixelized source
reconstruction as the code stands** — neither with mass-adaptive nor
fixed-adapt-image meshes. A Nautilus baseline was therefore not run: there is no
working gradient to compare against.

## Caveats / what would change the answer

- The `SplineAdaptImage` fits here were poor even near the assumed truth
  (`logL ~ -6e5`); the bootstrap adapt image + reg were not tuned, and the exact
  truth mass for this dataset was not pinned. Fit *quality* is secondary — the
  gradient is NaN regardless — but a well-tuned fit is worth confirming.
- Case 2 (NaN gradient) is the *potentially fixable* one: like the Delaunay
  gradient fix in the JAX gradient audit (a `custom_jvp` on the offending op),
  the NaN could be traced to a specific non-differentiable operation and
  hardened. That is a **library-level gradient-hardening project**, not this
  experiment. Case 1 (mass-adaptive → wrong gradient) is more fundamental.

## Reproduce

    python searches_minimal/probe_grad_pix.py              # mass-adaptive meshes (edit MESH_CLS)
    python searches_minimal/probe_grad_pix_adapt_image.py  # SplineAdaptImage + fixed adapt image

## Traps recorded

- `AdaptImages(galaxy_name_image_dict=...)` keys must be the **stringified** path
  tuple (`str(("galaxies","source"))`), not the tuple — the binding
  (`updated_via_instance_from`) looks up `str(galaxy_name)`.
- Adapt-image bootstrap: a numpy `fit_from(instance)` with `AdaptDensity` +
  `Constant`, then `fit.galaxy_model_image_dict[source]` → `AdaptImages`.
- Always x64 for standalone gradient parity.
