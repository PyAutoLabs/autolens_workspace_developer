# Test Run 4 — MGE source truth, NO lens light

Date: 2026-05-19
Sibling: [`../3_mge_truth_with_lens_light/RESULTS.md`](../3_mge_truth_with_lens_light/RESULTS.md)
Predecessor: [`../2_no_lens_light/RESULTS.md`](../2_no_lens_light/RESULTS.md) (Sersic-truth equivalent)
Follow-up of: [PyAutoLabs/autolens_workspace_developer#72](https://github.com/PyAutoLabs/autolens_workspace_developer/issues/72)

## Hypothesis tested

> If the truth source is an MGE (specifically, the well-recovered MGE
> from test 2's mge_source MLE) and there is no lens light to confuse
> the fit, does MGE source recover truth cleanly? This is the
> *matched* case — same model class on both sides — with the cleanest
> possible setup (no lens-light degree of freedom).

The naive expectation: MGE fit should recover MGE truth within 1σ on
every quantity. Sersic fit will have a profile-shape bias because the
truth is multi-scale and Sersic can't represent that.

## Setup

- Simulator: same as test 2 (SIE + shear, no lens bulge) with source
  loaded from `source_science/results/mge_truth_source.json`. Writes to
  `dataset/imaging/mge_truth_no_lens_light/`.
- Truth source-science quantities (same as test 3, because source and
  lens mass+shear are identical):

| Quantity              | Truth   |
|-----------------------|---------|
| image-plane flux      | 5910.37 |
| source-plane flux     | 167.63  |
| source magnification  | 35.26   |
| source magnitude (zp=25) | 19.44 |

- Two fits, matching test 2: `sersic_source` (Sersic source profile),
  `mge_source` (40-Gaussian MGE source). `unique_tag=mge_truth_no_lens_light_v1`.
- Posterior expansion: 50 draws with per-draw inversion for MGE.

## Bottom line — magnification is "easy" without lens light; MGE source flux has its own internal bias

The headline finding from tests 1-4 together:

> Without lens light, **magnification** is recovered at ~4-5% bias
> across every combination of (source truth class × source fit class).
> The model mismatch effect that drives the test-3 disaster only
> manifests when lens light is co-fit.

Test 4 is the cleanest demonstration: both Sersic source fit and MGE
source fit on MGE source truth give **magnification within -4.9% to
-5.3%** of truth, essentially identical to test 2's Sersic-truth result
(-3.6% for both fits). So magnification is robust to source-class
mismatch when there is no lens light to absorb residuals.

**But** there's a wrinkle on source flux specifically. The MGE source
fit on MGE truth gives **source flux +14.8% high** and image-plane flux
**+9.5% high** — the in-mask data isn't even being matched perfectly.
The Sersic source fit on the same data gives source flux +3.6% and
image-plane flux **-1.8% off** in the opposite direction. This is a
*basis-internal* MGE degeneracy: the 40-Gaussian basis has many
near-degenerate intensity solutions for the in-mask data, and the MLE
doesn't pick truth's solution even though truth is in the basis space.

| Quantity | Truth | Sersic source fit | MGE source fit |
|---|---|---|---|
| magnification | 35.26 | 33.40 ±0.16 (-5.3%, z=-11.9σ) | 33.52 ±0.27 (-4.9%, z=-5.4σ) |
| source flux | 167.6 | 173.7 ±0.9 (+3.6%, z=+6.8σ) | 192.4 +5/-3 (**+14.8%**, z=+5.9σ) |
| image-plane flux | 5910 | 5802 ±13 (**-1.8%**, z=-7.6σ) | 6469 ±175 (**+9.5%**, z=+3.7σ) |
| magnitude bias | — | -0.039 mag | **-0.18 mag** |

For test 2 (Sersic truth on same lens setup), both fits had image-plane
flux within 1σ of truth. Here, both fits have image-plane flux **biased
at the multi-σ level** — meaning even with no lens light to fight over,
the fits are not matching the in-mask data faithfully.

## What the numbers tell us

### 1. The MGE basis has internal degeneracies that bite even on matched truth

This is the headline result. The truth source IS a 40-Gaussian MGE basis,
saved with specific `(σ, intensity, ell_comps)` values from test-2's
MLE. The fit MGE has the same basis structure (40 Gaussians, same σ
range, same `centre_prior_is_uniform=False`). And yet the fit recovers
source flux 14.8% too high and image-plane flux 9.5% too high.

The mechanism: `al.model_util.mge_model_from` puts Gaussian sigmas on a
log-spaced grid in the prior; the truth's MLE sigmas were chosen from
the *posterior* of that prior so they may not exactly align with where
the new fit's MLE lands. With 40 Gaussians the basis is highly
overcomplete relative to the data's degrees of freedom, so many distinct
intensity vectors fit the in-mask data near-equally well. The MLE
picks one that doesn't match truth's intensity vector — and that
mismatch becomes visible when integrating to the source-plane flux on
the large science grid.

This is a different mechanism from the test-1 lens-light absorption:
*the MGE basis has enough internal degeneracy that even matched-truth,
clean-data fits show 10-15% systematic biases.* No lens light is needed
to trigger it.

### 2. The Sersic source fit has a different signature

Sersic on MGE truth: source flux **only +3.6% high** but image-plane flux
**1.8% LOW** — the Sersic profile fits the in-mask data with less total
flux than truth has, then over-extrapolates outside the mask. This is the
classic profile-shape bias: a single Sersic can't capture multi-scale
truth, so it fits the brightest part well and underestimates the wings.

This is *not* the same direction as the MGE bias (which puts +9.5% extra
flux into the image-plane integration). The two fits fail in *different*
directions on the same dataset — strong evidence that the bias mechanism
depends on the basis class even in the matched-no-lens-light setting.

### 3. Posterior under-coverage continues to be severe

- Sersic source magnification z-score: **-11.9σ** (1σ width 0.16, abs
  deviation 1.86)
- MGE source magnification z-score: **-5.4σ** (1σ width 0.27, abs
  deviation 1.74)

The Sersic fit's z-score is the worst across all four tests in this
series — 12σ off truth despite tight posterior. The MGE fit's z is
roughly the same as test 2's MGE fit (-5σ on Sersic truth), which is
suggestive of a regime where both source classes show ~5% magnification
bias regardless of which is matched to truth.

### 4. Practical implication: even with no lens light, source flux is unreliable

The first key result from PR #74 was that lens-light removal gets MGE
out of catastrophe. Test 4 shows the residual after that improvement
is still **15% on source flux for MGE** and **3.6% for Sersic**, with
posteriors that exclude truth at ~6σ.

For Euclid scientists: even in the idealised case of perfect lens-light
subtraction, MGE source fits over-estimate flux by ~15%; Sersic source
fits under-estimate image-plane flux by ~2% (so over-estimate source
plane flux as a consequence). Neither posterior reflects this.

## Compared to test 3 (with lens light)

| Quantity | Test 3 (with lens light) | Test 4 (no lens light) | Improvement |
|---|---|---|---|
| MGE source flux bias | +604% (~7× truth) | **+15%** | 40× |
| MGE source magnification bias | -77% | **-4.9%** | 16× |
| MGE magnitude bias | -2.12 mag | **-0.18 mag** | 12× |
| Sersic source flux bias | +20% | +3.6% | 5× |
| Sersic source magnification bias | -13.3% | -5.3% | 2.5× |

Lens-light removal still produces dramatic improvement (the conclusion
of PR #74 stands). But the absolute floor after removal is now ~5-15%
bias, not ~1-4% as test 2 suggested for Sersic truth.

## Decisions for the next experiment

Same as test 3's recommendations, with one addition:

1. **σ-cap test on test-4 MGE source.** Re-run the MGE-source fit on
   the no-lens-light data with σ_max capped at, say, 0.3 arcsec. If
   the MGE source flux bias drops from +15% to ~+3%, we have a clear
   mechanism (wide-σ Gaussians soaking up out-of-mask flux even
   without lens light).
2. **n_gaussians = 10 fit instead of 40.** Reduces basis flexibility.
   Should sharpen the matched-truth recovery if the issue is over-
   parameterisation.
3. **A controlled MGE-truth where the truth's sigmas exactly match a
   pre-defined prior grid.** Removes the prior-vs-truth-sigma-grid
   alignment issue. If matched recovery becomes near-perfect under
   this synthetic case, the test-4 bias is just sample-realisation
   mismatch.

## Diagnostic plots

- `fits/source_1d_profile.png` — radial source-plane brightness, both
  fits + truth + posterior 1σ band.
- `fits/source_cumulative_flux.png` — cumulative integrated flux vs
  aperture radius.
- `fits/source_2d_brightness_panel.png` — 2D source-plane side-by-side.

Cross-experiment plot: `../test3_vs_test4_mge_source.png`.
