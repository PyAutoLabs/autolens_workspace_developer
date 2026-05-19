# Test Run 3 — MGE source truth, WITH lens light

Date: 2026-05-19
Sibling: [`../4_mge_truth_no_lens_light/RESULTS.md`](../4_mge_truth_no_lens_light/RESULTS.md)
Predecessor: [`../1_with_lens_light/RESULTS.md`](../1_with_lens_light/RESULTS.md) (Sersic-truth equivalent)
Follow-up of: [PyAutoLabs/autolens_workspace_developer#72](https://github.com/PyAutoLabs/autolens_workspace_developer/issues/72)

## Hypothesis tested

PR #74 (test 1 vs test 2) argued the MGE-source catastrophe was driven
by lens-light/source-light degeneracy under a *Sersic source truth*.
The natural follow-up question:

> If the truth source is itself an MGE (specifically, the MGE we infer
> from the test-2 mge_source MLE), do MGE fits recover well? Does the
> "MGE catastrophe under lens light" persist when MGE is the matched
> model class?

## Setup

- Simulator: same as test 1 (`Sersic` lens bulge + SIE + shear) but the
  source is loaded from `source_science/results/mge_truth_source.json`,
  produced by `source_science/extract_mge_truth.py` from the test-2
  mge_source MLE. Saves to `dataset/imaging/mge_truth_with_lens_light/`.
- Truth source-science quantities:

| Quantity              | Truth   |
|-----------------------|---------|
| image-plane flux      | 5910.37 |
| source-plane flux     | 167.63  |
| source magnification  | 35.26   |
| source magnitude (zp=25) | 19.44 |

These are ~1% off the test-1/2 truth values because the MGE source is a
slightly different profile shape from the original SersicCore truth.

- Three fits, matching test 1: `sersic__sersic`, `mge_lens__sersic_source`,
  `mge_lens__mge_source`. `unique_tag=mge_truth_with_lens_light_v1`.
- Posterior expansion: 50 draws with per-draw inversion for the MGE fits.

## Bottom line — model mismatch only bites when lens light is present

The cleanest way to read tests 1-4 together:

> Without lens light, magnification is recovered at ~4-5% across every
> combination of (source truth class × source fit class). When lens
> light is co-fit, source-model mismatch on top of lens-light/source-
> light degeneracy creates dramatic bias.

Test 3 is the worst end of this pattern: MGE truth + lens light + MGE
source fit gives **-77% magnification, +604% source flux, -2.12 mag**.
But even the Sersic source on MGE truth + lens light is **-13% to
-26%** biased — far worse than the same Sersic source on Sersic truth
+ lens light (-4.4%) or on MGE truth without lens light (-5.3%).

### Side-by-side with Sersic-truth test 1

| Quantity | Test 1 (Sersic truth) | Test 3 (MGE truth) | Change |
|---|---|---|---|
| sersic+sersic magnification bias | -4.4% (-5.5σ) | **-13.3% (-14σ)** | 3× worse |
| sersic+sersic source-flux bias | +6.4% | **+20%** | 3× worse |
| MGE-lens + Sersic-source mag bias | -4.6% | **-26% (-21σ)** | 6× worse |
| MGE-lens + MGE-source mag bias | -44% | **-77%** | 1.8× worse |
| MGE-lens + MGE-source flux bias | +103% (~2× truth) | **+604% (~7× truth)** | 3× worse |
| MGE-lens + MGE-source magnitude bias | -0.77 mag | **-2.12 mag** | 2.8× worse |

## Results

| Model | log L | Magnification (median±1σ) | Mag z | Mag in 3σ? | Magnitude (median±1σ) |
|---|---|---|---|---|---|
| sersic__sersic | 4834.2 | 30.53 +0.40/-0.28 | **-14.2σ** | NO | 19.23 ±0.02 |
| mge_lens__sersic_source | 4896.4 | 26.11 +0.37/-0.44 | **-21.3σ** | NO | 18.98 ±0.025 |
| mge_lens__mge_source | 5418.2 | 10.23 +7.0/-2.1 | **-4.7σ** | NO | 17.69 +0.75/-0.34 |
| Truth | — | 35.26 | — | — | 19.44 |

Detailed numbers in `dataset/imaging/mge_truth_with_lens_light/fit_comparison.{json,md}`;
per-fit subplots in `dataset/imaging/mge_truth_with_lens_light/fits/`.

## What the numbers tell us

### 1. The MGE+MGE catastrophe is worse on MGE truth, not better

This was the most surprising prediction this test could have produced.
The simple lens-light-degeneracy story from PR #74 implicitly assumed
the MGE+MGE failure came from the MGE source absorbing residuals
peculiar to *Sersic-source* truth (where the MGE basis is over-flexible
and ends up parameterising the wrong shape). Under that story, MGE
fitting MGE truth should remove that source of degeneracy.

Instead, MGE+MGE on MGE truth gives source flux **7× truth** versus
test-1's "merely" 2× truth, and magnification **22% of truth** versus
test-1's 56%. The MGE+MGE catastrophe is therefore *not* about the fit's
source basis being unable to represent truth — it's about the lens-light
and source-light MGE bases having *more* room to swap flux when both are
flexible. When truth itself has more diffuse extended structure (which
an MGE can have more easily than a SersicCore), the inversion finds
basis solutions even further from truth.

### 2. Sersic source is NOT robust to lens light when truth is MGE

Test 1 showed Sersic-source fits were ~5% biased on magnification
regardless of whether the lens light model was Sersic or MGE — they
were "robust" to lens-light parameterisation. Tests 3 inverts this:
the MGE-lens + Sersic-source fit on MGE truth gives **-26% magnification
bias**, far worse than the matched Sersic+Sersic fit on the same data
(-13.3%). And even Sersic+Sersic is **3× worse** than test 1.

So the "Sersic source is robust to lens-light" finding from PR #74 was a
coincidence of having a Sersic truth where the Sersic fit naturally
landed near truth. With a more complex (MGE) truth, the Sersic profile-
shape mismatch becomes the dominant bias.

### 3. Higher log-likelihood still picks the wrong answer

Log-likelihood ordering: MGE+MGE (5418) > MGE-lens + Sersic-source (4896)
> Sersic+Sersic (4834). The model with the best evidence is the
catastrophically-biased one. This was already noted in PR #73 for
Sersic truth; tests 3+4 confirm it's general. **For source-science
recovery, log-evidence is anti-correlated with truth recovery.**

### 4. Posterior under-coverage is even more severe here

z-scores in test 3 range from -3.4σ to -21.3σ — the most extreme are
the Sersic-source fits with their very tight 1σ widths. The MGE+MGE
fit's posterior is wider (mag std 7) and yet still excludes truth at
-4.7σ. The "MGE gives less-misleading error bars" framing from PR #73
doesn't help when the median itself is 22% of truth.

### 5. Where do the catastrophic biases come from?

The image-plane flux is too high by 10-60% in every fit — meaning
the fits are not even matching the in-mask data well any more. With
the truth source being a 40-Gaussian MGE that doesn't admit a clean
Sersic or compact-MGE-fit representation, the fits choose source
configurations that approximate the *masked* image plane data but
predict much more lensed flux than the data implies. Specifically:

- Sersic+Sersic: image flux +5% high. The Sersic source can't fit the
  fine structure of the MGE truth's image-plane signal, so the lens
  model shifts to compensate (-13% on Einstein-radius-derived
  magnification). The source absorbs the residual.
- MGE-lens + Sersic-source: image flux +13% high. Same story, worse —
  the now-flexible MGE lens light has even more room to siphon flux off
  the source.
- MGE+MGE: image flux **+59% high**. Both MGE bases compete for in-mask
  flux, find a configuration that produces wildly more lensed flux but
  is mathematically permitted by the basis flexibility.

## What this means for Euclid source-flux science

The story we told in PR #74 — "use parametric Sersic sources to be safe
against lens-light degeneracy" — was based on a Sersic-truth controlled
experiment. The real world is closer to MGE-truth: galaxy sources have
complex multi-scale structure that no Sersic can perfectly capture.
Test 3 says that in this realistic regime:

- Sersic-source fits are biased by **10-25%** on magnification when
  lens light is present
- MGE-source fits with co-fitted MGE lens light fail
  **catastrophically** (factor 7× source flux errors)
- Even posterior 3σ bands do not contain truth in any case

**There is currently no source-light parameterisation we have tested
that gives reliable Euclid source flux when the lens light is also being
fit and the source has non-Sersic structure.** This is a much more
serious finding than tests 1+2 implied.

## Sanity checks

- All three fits completed cleanly with `n_draws_failed=0` for posterior
  expansion.
- log-likelihood values are positive and ordering is consistent with
  parameter count (more flexible model → higher log L).
- Truth tracer round-trips through `al.from_json` and gives the same
  source-science quantities as the simulator wrote.

## Decisions for the next experiment

The catastrophic test-3 results put the simple "lens-light degeneracy"
framing under pressure. The follow-ups in priority order:

1. **σ-cap on the MGE source.** The test-3 MGE+MGE source MLE almost
   certainly puts intensity into wide Gaussians (σ ~ 1-3 arcsec). Cap
   the basis at σ_max = 0.3 arcsec and re-run test 3. If the
   catastrophe collapses, the mechanism is "fit MGE basis has Gaussians
   wider than the source needs, those wide Gaussians are responsible".
2. **MGE truth + reduced n_gaussians fit.** Instead of capping σ, reduce
   the fit MGE from 40 Gaussians to 10. Tests whether the catastrophe
   needs MGE flexibility on the fit side.
3. **Sersic+Sersic + sersic_index in [0.5, 2.0] prior tightening.** Sees
   whether even the matched-model case improves when we constrain the
   profile shape.

All of these can run on the existing `dataset/imaging/mge_truth_with_lens_light/`
dataset — no re-simulation needed.

## Diagnostic plots

- `fits/source_1d_profile.png` — radial source-plane brightness, all
  three fits + truth + posterior 1σ band.
- `fits/source_cumulative_flux.png` — cumulative integrated flux vs
  aperture radius.
- `fits/source_2d_brightness_panel.png` — 2D source-plane side-by-side.

Cross-experiment plots:

- `../test3_vs_test4_mge_source.png` — MGE-source fits in tests 3 vs
  4 on MGE truth, analogous to `test1_vs_test2_mge_source.png` from
  PR #74.
- `../matched_vs_mismatched_2x2.png` — 2×2 grid showing source radial
  profile for every (truth class × lens-light condition) combination,
  with both Sersic and MGE source fits overlaid per cell. This is the
  single image that distinguishes the lens-light degeneracy effect
  from the MGE-basis-internal degeneracy.
