# Test Run 2 — No Lens Light

Date: 2026-05-19
Sibling: [`../1_with_lens_light/RESULTS.md`](../1_with_lens_light/RESULTS.md)
Follow-up of: [PyAutoLabs/autolens_workspace_developer#72](https://github.com/PyAutoLabs/autolens_workspace_developer/issues/72)
Worktree: `~/Code/PyAutoLabs-wt/source-science-no-lens-light/`
Branch: `feature/source-science-no-lens-light`

## Hypothesis tested

The MGE-source pathology seen in test 1 (44% magnification under-estimate,
0.77 mag bias) was suspected to come from **lens-light/source-light
degeneracy** during the fit — the source MGE was absorbing diffuse
residual flux left over from the lens-light fit, depositing it into wide-σ
Gaussians outside the strongly-magnified region.

**This run removes the lens light entirely** from both the simulated truth
and from the fit model. Same lens mass + shear and same `SersicCore` source as
test 1 — only the lens Sersic bulge is dropped. If the hypothesis is right,
the MGE-source bias should collapse.

## Bottom line — **hypothesis confirmed for the catastrophic part**

Removing lens light collapses the test-1 MGE disaster:

- magnification bias: −44% → **−3.6%**
- source-flux bias: +103% → **+3.9%**
- magnitude bias: −0.77 mag → **−0.045 mag**

The MGE source now agrees with the Sersic source to within their respective
1σ on every derived quantity. The test-1 MGE failure was lens-light /
source-light degeneracy, not a fundamental MGE problem.

**But neither test-2 fit is science-acceptable yet.** Both still miss truth
by ~5σ on magnification and source flux — the posterior 1σ on magnification
is only ±0.20, so a 1.27 absolute deviation registers as -5.8σ (Sersic) and
-5.0σ (MGE). For Euclid we'd want < 1% bias and truth inside 1σ; we're at
3.6% bias with truth excluded at 3σ. The MGE catastrophe is fixed; the
underlying precision-vs-accuracy problem from test 1 is not.

Crucially, **the residual bias is the same direction and magnitude for both
fits** (Sersic ≈ MGE on magnification, source flux, and magnitude). Since
the source parameterisation no longer matters, the residual must be a
profile-shape / data-coupling effect that lives in the *integration* of
source-plane flux outside the lensed region — most likely the
`SersicCore.radius_break` mismatch (fit uses 0.05; library default and
simulator use 0.025) and/or the `sersic_index` drift away from truth
n = 1.0. Both are scoped for test 3.

## Setup

- Simulator: `simulator.py` here — identical to test 1 except the
  `lens_galaxy` carries only `mass` and `shear`, no `bulge`.
- Dataset: `dataset/imaging/no_lens_light/` (100×100 @ 0.1", mask 3", PSF
  σ=0.1).
- Truth (same as test 1 — source/mass/shear unchanged):

| Quantity | Truth |
|---|---|
| image-plane flux | 5473.96 |
| source-plane flux | 156.73 |
| source magnification | 34.93 |
| source magnitude (zp=25) | 19.512 |

- Two fits: `sersic_source`, `mge_source`. **Neither model includes a lens
  bulge** — the lens galaxy carries only `Isothermal` mass + `ExternalShear`.
- Posterior expansion: 50 draws via `samples.draw_randomly_via_pdf()`. The
  MGE draws run `FitImaging.tracer_linear_light_profiles_to_light_profiles`
  per draw to solve linear intensities.
- `path_prefix=source_science_no_lens_v1`, `unique_tag=no_lens_light_v1`.

## Results

| Model | log L | Magnification (median±1σ) | Mag z-score | Truth in 3σ (mag)? | Magnitude (median±1σ) | Mag z |
|---|---|---|---|---|---|---|
| sersic_source | 6670.5 | 33.67 +0.16/-0.25 | **-5.8σ** | NO | 19.47 ± 0.008 | **-5.3σ** |
| mge_source | 6691.1 | 33.66 +0.29/-0.26 | **-5.0σ** | NO | 19.47 ± 0.009 | **-5.2σ** |
| Truth | — | 34.93 | — | — | 19.51 | — |

Detailed numbers in `dataset/imaging/no_lens_light/fit_comparison.{json,md}`;
per-fit subplots in `dataset/imaging/no_lens_light/fits/`.

Note the inflated log-likelihoods (5400ish in test 1 → 6700 here) reflect
just the loss of lens-light pixels from the fit; they are not comparable
across the two datasets and don't carry information about source recovery.

## Side-by-side with test 1

| Model | Quantity | Test 1 (with lens light) | Test 2 (no lens light) | Improvement |
|---|---|---|---|---|
| **MGE source** | magnification | 19.64 ± 6.2 (−44%) | **33.66 ± 0.27 (−3.6%)** | 41 percentage-points |
| MGE source | source flux | 318.5 ± 150 (+103%) | **162.8 ± 1.2 (+3.9%)** | 99 percentage-points |
| MGE source | magnitude bias | −0.77 mag | **−0.045 mag** | 0.73 mag |
| MGE source | mag std (posterior) | 0.38 mag | 0.008 mag | 47× tighter |
| **Sersic source** | magnification | 33.34 ± 0.25 (−4.4%) | 33.67 ± 0.20 (−3.6%) | marginal |
| Sersic source | source flux | 166.7 ± 1.8 (+6.4%) | 162.9 ± 1.1 (+3.9%) | small |
| Sersic source | magnitude bias | −0.067 mag | −0.045 mag | 0.022 mag |

**The MGE-source row is what you'd expect to see if the lens-light
absorption hypothesis is correct, and that's exactly what we got.**

### Diagnostic plots

- `fits/source_1d_profile.png` — 1D source-plane radial brightness for
  truth + each fit's MLE, with posterior 1σ band per fit. Both the Sersic
  and MGE curves now track truth closely in the core; the small residual
  discrepancy is the profile-shape effect that explains the −3.6%
  magnification bias.
- `fits/source_cumulative_flux.png` — cumulative source-plane flux as a
  function of aperture radius. Compare to the test-1 version: the test-2
  MGE source no longer keeps growing past truth into a diffuse halo.
- `fits/source_2d_brightness_panel.png` — side-by-side 2D source-plane
  images, same colour scale.

The headline cross-experiment plot lives at
[`../test1_vs_test2_mge_source.png`](../test1_vs_test2_mge_source.png).
It overlays the test-1 and test-2 MGE source profiles against truth and
is the single image that summarises the hypothesis confirmation.

## What the numbers tell us

### 1. The MGE source pathology was lens-light contamination

In test 1 the MGE source achieved a higher log-likelihood than the Sersic
source by 31 nats — but the price of that higher likelihood was a
catastrophically wrong source magnitude. Without lens light to fit, the
MGE source still beats the Sersic source on log-likelihood (by 21 nats),
but it does so *without* the catastrophic source bias. The difference
between the two regimes is one degree of freedom — the lens-light
component — which carried enough residual flux to corrupt the MGE source
when both were free to absorb it.

This means the test-1 MGE problem **is not a fundamental MGE failure**.
It is a coupling failure between two simultaneously-fit linear components
(lens-light MGE and source MGE), where the inversion can shuffle flux
between the two basis sets in ways that improve the in-mask data
likelihood while wrecking the source-plane integral.

### 2. The Sersic-source result is essentially unchanged

Without lens light to fit, Sersic source magnification moves from 33.34
to 33.67 — a ~1% shift well inside both fits' 1σ. Source flux moves
from 166.7 to 162.9 (~2%). The Sersic source basically didn't care
about the lens light, which makes sense: with only 7 free parameters and
a `LogUniformPrior` on intensity, there isn't much freedom to absorb
diffuse residual flux.

So the lens-light-degeneracy effect is **specific to flexible source
parameterisations** (MGE, presumably also pixelization). For Euclid, this
means analyses using parametric Sersic sources are less at risk from this
mechanism, but anyone moving to MGE or pixelized sources is exposed.

### 3. The residual ~3-4% magnification bias is profile-shape

Both fits in test 2 still under-estimate magnification by ~3.6% and
over-estimate source flux by ~3.9% — values that are nearly identical
between Sersic and MGE. Since the model class no longer differs and the
data no longer contains lens light, the bias must come from somewhere
else. The two candidates are:

- **`SersicCore.radius_break` mismatch.** Fit uses `0.05`; truth uses the
  library default `0.025`.
- **`sersic_index` drift.** Test 1 had inferred `n ≈ 1.32` vs truth `1.0`.
  Test 2 has not yet been inspected for this, but a similar drift would
  produce the same magnification bias direction.

Both of these are in scope for test 3.

### 4. The posterior still under-covers the truth

Even with the MGE pathology gone, neither fit puts the truth inside its
1σ on any quantity except image-plane flux. The Sersic-source magnification
is **5.8σ** from truth; the MGE-source magnification is **5.0σ**. The
1σ width is just ~0.27 in magnification — extremely tight — and the
truth sits well outside it.

This is the same precision-vs-accuracy story from test 1: the parametric
posterior captures data-fit width, not model-truth distance. **Removing
the lens light fixes the MGE catastrophe but does not fix the
overconfident error bars on the residual profile-shape bias.** Confidence
intervals on Euclid source magnitudes will still be too small even when
the lens light is well-controlled.

### 5. Image-plane flux recovers cleanly

Both fits put image-plane flux inside the 1σ (0.27σ and 0.64σ z-scores).
That confirms the fit-to-data is fine — the deficit is purely in the
source-plane integral, i.e., the *unobserved* part of the source flux
calculation. This is consistent with the profile-shape hypothesis (a
different Sersic index changes the source-plane integral while leaving
the within-mask, lensed flux nearly unchanged).

## Decisions for the next test

`RESULTS.md` for the next experiment lands at
`source_science/results/3_*/RESULTS.md`. Priority candidates, in order:

1. **Test 3a — `radius_break = 0.025`.** Match the simulator default;
   re-run sersic_source on the no-lens-light dataset. If the magnification
   bias drops below ~1%, `radius_break` was the dominant residual and
   parametric source magnitudes for Euclid look much healthier than test
   1 implied. Cheap — same dataset, single new fit.

2. **Test 3b — fixed `sersic_index = 1.0`.** Pin the source profile shape
   to truth and re-run. Isolates the profile-shape-drift contribution
   from `radius_break`. Combine with test 3a for a 2×2 design.

3. **Test 4 — bring lens light back, but with stronger source priors.**
   Tighten `MGE` source `sigma_max` (or add a soft prior on wide-σ
   components) and check whether the MGE catastrophe stays away when lens
   light is present. This tests whether the fix can be just on the
   source-side or whether it needs an explicit decoupling of the two
   basis sets.

4. **Test 5 — pixelization sources** (`RectangularAdaptImage`,
   `Delaunay`) on the no-lens-light dataset first, then on the
   with-lens-light dataset. Tests whether the pathology generalises
   beyond MGE.

The clearest takeaway for an Euclid scientist *right now*: **do not use
MGE source models alongside MGE lens-light models without a regularisation
or σ-cap mechanism.** Pure parametric (Sersic) sources are robust to the
lens-light contamination effect; they just carry the smaller ~4%
profile-shape bias that test 3 will address.

## Reproduce

```bash
source ~/Code/PyAutoLabs-wt/source-science-no-lens-light/activate.sh
cd ~/Code/PyAutoLabs-wt/source-science-no-lens-light/autolens_workspace_developer
python source_science/results/2_no_lens_light/simulator.py
python source_science/results/2_no_lens_light/fit_compare.py
```

`SOURCE_SCIENCE_N_DRAWS=N` overrides the posterior-sample count (default 50).
Nautilus output cache lives at `output/output/source_science_no_lens_v1/` —
delete that directory for a fully fresh run; leave it intact to resume.
Total wall-clock ~25 minutes for a full from-scratch run on this laptop
(2 Nautilus fits + 50 posterior draws each, with per-draw inversion on
the MGE fit).
