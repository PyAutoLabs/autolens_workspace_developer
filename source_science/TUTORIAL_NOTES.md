# Source Science Tutorial Assessment

Audit of the two existing workspace tutorial scripts that describe source-flux
and magnification calculations. The goal is to spot anything that is
numerically wrong, misleading, or missing before we use them as the public
reference for source-science work on Euclid.

Files reviewed:

- `autolens_workspace/scripts/imaging/source_science.py`
- `autolens_workspace/scripts/imaging/features/pixelization/source_science.py`

Both scripts use the `simple__no_lens_light` dataset. The developer-workspace
sibling here (`source_science/simulator.py`) intentionally uses the `simple`
dataset *with* lens light, because lens-light contamination is part of the
Euclid problem we are trying to reproduce.

## `imaging/source_science.py` (parametric)

### Numerics

- The flux integral uses a 500×500 grid at 0.02"/pix, and the magnification
  ratio uses 1000×1000 at 0.03"/pix. Both choices are defensible but the
  script asserts the result is "stable" without showing it. For a tutorial
  reader who later switches to a higher Sersic index or a non-cored profile,
  there is no signal that they should re-check convergence. **Suggested:**
  add a convergence check (sum at 250×250 vs 500×500) and one printed
  comparison so the reader sees the size of the discretisation error.
- Uses `SersicCore` rather than `Sersic`, which avoids the central
  singularity but is never explained as a numerical-stability choice. A
  reader copying the pattern to a non-cored Sersic will get a very different
  flux. **Suggested:** one sentence flagging this and pointing at the
  cored-vs-uncored discussion.
- The pixel-area cancellation explanation (lines 212–216) is correct and
  clearly written — keep as-is.

### Pedagogy

- The repeated "tracer reproduces the calculation" block at lines 231–245
  prints the same numbers a second time. Useful as a fallback recipe for
  when you only have a tracer (the real-data case), but **without an
  explicit `np.allclose` or printed delta** the reader cannot see the two
  paths actually agree. **Suggested:** print the delta between the
  `source_galaxy.bulge.image_2d_from(...)` and `tracer.planes[1]...` paths
  to show the equivalence.
- The script never converts flux to magnitude. The header docstring mentions
  magnitudes but defers to `guides/units/flux`. For a "source science"
  reference, **at least one worked magnitude with a quoted zero-point is
  warranted** — it is the quantity most papers actually report.
- The script uses `max_log_likelihood_tracer` as the implied source of
  inferred values when discussing real fits. It does **not** mention that
  for any real Euclid analysis the user must propagate posterior
  uncertainty into the derived flux/magnification, not just take the MLE.
  This is the central point of the Euclid investigation in this developer
  workspace, so the tutorial should at least signpost it.
- No mention that PSF convolution is *not* applied to the magnification
  ratio. For a parametric tracer this is exactly right (magnification is a
  property of the lens model, not the detector). The tutorial should still
  state it explicitly, because Euclid users routinely conflate the
  image-plane lensed model with the data and expect the script to convolve.

### Wording / typos

- Line 198: "To calculation this" → "To calculate this".
- Line 138: "the 2D grid of (y,x) coordinates which simulate the dataset" →
  "which were used to simulate the dataset".
- Several `__Contents__` lines are sentence fragments ending in trailing
  full stops; not strictly wrong but reads like cut-off prose.

## `imaging/features/pixelization/source_science.py`

### Real bugs

- **Lines 279–281: noise-map interpolation passes the wrong array.**
  ```python
  reconstruction_noise_map = inversion.reconstruction_noise_map
  interpolated_noise_map = griddata(
      points=source_plane_mesh_grid, values=reconstruction, xi=interpolation_grid
  )
  ```
  `values=reconstruction` should be `values=reconstruction_noise_map`. As
  written, `interpolated_noise_map` is a second copy of the interpolated
  reconstruction, not the noise. Anything downstream that uses
  `interpolated_noise_map` for an SNR cut or for source-plane model
  fitting is silently wrong.
- **Line 156: `total_flux = np.sum(reconstruction)` is missing the pixel
  area factor.** For a rectangular mesh the reconstruction stores intensity
  (surface brightness) per cell; the flux integral must be
  `np.sum(reconstruction * mesh_areas)`. The same omission appears at
  line 235 for the interpolated array (no `* pixel_area`). The magnification
  block (lines 309–311) does include `pixel_area`, so the two halves of the
  script are inconsistent in their units.
- Both bugs combine to make the printed "Total Source Flux via
  Pixelization" not directly comparable to the parametric tutorial's
  flux, which is what a reader would naively assume from the matching
  print labels.

### Numerics / accuracy

- `griddata` with the default `linear` method returns `NaN` outside the
  convex hull of the source-plane mesh. The current 200×200 @ 0.05" /
  401×401 @ 0.005" extents are large enough to leave a NaN halo. Any
  `np.sum` on the array silently propagates NaN to the total. **Suggested:**
  `np.nansum`, or zero-fill the outside-hull pixels explicitly so the
  reader sees they are being dropped.
- Line 366–370 gives the "magnification via mesh" calculation, which is the
  more accurate method (no interpolation). The script computes it but never
  prints it, so the reader cannot compare against the interpolated-grid
  value at line 311. **Suggested:** add the print.
- The error-bar section (lines 273–292) treats `griddata` interpolation of
  the noise map as if it preserves error statistics. It does not — the
  reconstruction noise map is regularization-correlated, so naive linear
  interpolation under-estimates the true uncertainty by an amount that
  depends on the regularization coefficient. This is a deep issue worth
  flagging even if the fix is "use the unsmoothed mesh values for
  propagation".

### Structure

- The script performs **two** fits: a non-search `FitImaging` early on
  (line 122) and a Nautilus search at the end (line 426). The two fits use
  different mass priors (`Isothermal` vs `PowerLaw`) and slightly
  different model definitions, but produce ostensibly comparable
  reconstructions. This dual-fit structure is confusing for a "source
  science" reference. **Suggested:** run a single fit upfront and use its
  result throughout.
- Comparison against truth is absent. The script computes flux estimates
  by four different paths (raw mesh sum, interpolated, zoomed interpolated,
  S/N-masked) and never tabulates them against either each other or the
  simulator truth flux. **Suggested:** end with a small table.

### Wording / typos

- Line 11: "speciifc" → "specific". Line 11 also has "required" where
  "require" is intended.
- Line 23 in `__Contents__`: "the simplest way ... is to." (truncated).

## Recommended follow-up

This developer-workspace task **does not edit the tutorial scripts** —
`autolens_workspace` is held by another active task. Specific actions to file
as a follow-up issue once that workspace clears:

1. Fix the `values=reconstruction` → `values=reconstruction_noise_map` bug
   on line 280 of the pixelization tutorial. This is the only outright
   defect; everything else is pedagogy.
2. Fix the missing pixel-area factor on line 156 (and line 235) of the
   pixelization tutorial, or rename `total_flux` to `total_surface_brightness`
   to make the units explicit.
3. Add a printed `Magnification via Mesh` comparison alongside the
   interpolated value.
4. Replace `np.sum` with `np.nansum` (or zero-fill) wherever the result of
   `griddata` is summed.
5. In `imaging/source_science.py`, add a one-paragraph magnitude calculation
   with a worked zero-point so the tutorial actually shows the unit users
   ultimately report.
6. In both scripts, replace the implicit
   `max_log_likelihood_tracer` framing with explicit posterior propagation,
   matching the approach used in `developer/source_science/fit_compare.py`.

For now these notes feed into the developer-workspace fit analysis and the
`RESULTS.md` synthesis. They are not yet a tutorial-script PR.
