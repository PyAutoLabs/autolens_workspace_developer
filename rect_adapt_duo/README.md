# rect_adapt_duo

Visual demonstration of the multi-source ghost-peak failure in the existing
adaptive rectangular pixelization (`RectangularSplineAdaptImage`) and the
fix delivered by the PCA-rotated variant (`RectangularRotatedAdaptImage`).

Companion to [PyAutoArray #322](https://github.com/PyAutoLabs/PyAutoArray/issues/322).
Background:

- `PyAutoArray/files/cdf_audit.md` — the existing CDF mechanics.
- `PyAutoArray/files/ghost_peak_findings.md` — empirical confirmation of
  the separable-CDF failure and the three possible fixes.

## Scenario

Two compact Sérsic sources at source-plane positions `(+0.5, +0.5)` and
`(−0.5, −0.5)` arcsec, behind an isothermal lens at the origin
(Einstein radius 1.2"). The diagonal peak placement is the textbook case
where the separable per-axis CDF produces a 4-quadrant outer-product
mesh — two pixel zones on the real source peaks, two on the empty
`(±0.5, ∓0.5)` ghost cross-products. Wasted half the pixel budget.

## Run order

```bash
source ~/Code/PyAutoLabs-wt/rectangular-adapt-cdf/activate.sh
cd rect_adapt_duo

python simulator.py        # → dataset/{data,noise_map,psf}.fits
python compare_meshes.py   # → output/comparison.png
```

`compare_meshes.py` reconstructs the truth tracer (via
`simulator.build_tracer`) and computes the adapt image inline from its
noiseless lensed image, so there's no separate adapt-image step.

## What you should see

`output/comparison.png` is a 2×3 grid. **The headline panels are the
source-plane meshes (left column).**

| Row             | Source mesh (LEFT — headline)               | Image-plane model        | Residuals                              |
| --------------- | ------------------------------------------- | ------------------------ | -------------------------------------- |
| Baseline        | **4-quadrant ghost grid** at (±0.5, ±0.5)   | Einstein ring, 4 images  | mostly noise (good fit)                |
| PCA-rotated     | **Clean diagonal band** through real peaks  | Sharper image peaks      | One sharp residual blob (see below)    |

The source-mesh panels are the proof-of-concept: baseline's separable CDF
concentrates pixels at all four `(±0.5, ±0.5)` cross-products even though
only two of those positions have source brightness. The rotated mesh
aligns to the +45° principal axis and places pixels only on the real
peaks. Green circles mark real source positions; red ×'s mark the ghost
cross-products.

## A note on χ²

The rotated case usually shows a **higher** χ² than baseline at fixed
regularization coefficient. This is **not a flaw** — it's the
predictable consequence of the rotated mesh giving ~2× higher effective
pixel resolution per real peak (no ghost waste). At the same
`Constant(coefficient=1.0)` the rotated reconstruction under-smooths the
peak and overshoots the data.

A real lens-modelling pipeline tunes regularization per mesh class with
a non-linear search. For the apples-to-apples comparison this demo
intentionally skips, both meshes need different coefficients to land
their best χ². The mesh-layout panels are the regularization-independent
proof.

## Design notes

- **No non-linear search.** The comparison is about MESH behaviour at the
  true mass model. Running a search would add hours of compute and
  confound the mesh-class effect with mass-model variance. The lens
  isothermal parameters are pinned to truth in `compare_meshes.py`.
- **Adapt image is truth-derived.** We render the noiseless source-plane
  image from the true tracer and use that as the adapt image. This is
  the "best possible" prior — fair to both mesh classes; isolates the
  question of "what does each mesh do with a perfect adapt image".
- **Mesh shape:** `(40, 40)` → 38×38 ≈ 1444 effective interior pixels,
  matching the diagnostic experiments in
  `PyAutoArray/files/ghost_peak_experiment.py`.
