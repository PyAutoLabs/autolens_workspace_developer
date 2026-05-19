# Parametric Source-Science Recovery — Results

## Bottom line

**No parametric model recovers the source-plane flux or magnification within
its quoted posterior 1σ**, and only one of twelve (3 models × 4 quantities)
has the truth inside its 3σ band. The MGE source is *catastrophically*
biased: source flux off by a factor of 2, magnification off by 44%, magnitude
off by 0.77 mag. Higher Nautilus log-evidence does **not** correlate with
better source-flux recovery — in fact the highest-evidence fit (MGE+MGE) has
the worst recovery.

For Euclid: parametric Sersic sources are precise but biased at the
~0.06 mag level with quoted errors that *do not cover this bias*. MGE
sources are wider in their error bars but median-biased at the ~0.8 mag
level. Both will mislead a science analysis that consumes the quoted
posterior at face value.

## Setup

Simulated lens dataset `dataset/imaging/simple` (Sersic lens bulge + SIE +
shear, SersicCore source). The truth values come from integrating the truth
tracer on a 400×400 @ 0.03 "/px source-plane grid:

| Quantity                | Truth   |
|-------------------------|---------|
| image-plane flux        | 5473.96 |
| source-plane flux       | 156.73  |
| source magnification    | 34.93   |
| source magnitude (zp=25)| 19.512  |

All three fits use `af.Nautilus(n_live=75, n_batch=25)` with `use_jax=True`.
Posterior uncertainty is from 50 random draws via `samples.draw_randomly_via_pdf()`,
with `FitImaging.tracer_linear_light_profiles_to_light_profiles` per draw
for the two MGE fits (so the inversion-solved intensities are included).

## Results summary

| Model            | log L   | Mag (median ± 1σ) | Mag z-score | Mag truth in 3σ? | Magnitude (median ± 1σ) | Magnitude z-score |
|------------------|---------|---|---|---|---|---|
| sersic+sersic    | 5381.5  | 33.34 +0.24/-0.27 | **-5.5σ**  | NO | 19.45 ± 0.012 | **-5.7σ** |
| MGE+sersic source| 5395.3  | 33.38 +0.52/-0.43 | **-3.4σ**  | NO | 19.45 ± 0.023 | **-2.7σ** |
| MGE+MGE source   | 5412.9  | 19.64 +6.3/-6.1   | **-2.85σ** | NO | 18.74 ± 0.38  | **-2.0σ** |

See `fit_comparison.md` for the full table including flux quantities; raw
JSON in `fit_comparison.json`; per-fit subplots in `fits/<model_name>.png`.

## What the numbers say

### 1. Sersic source ≠ truth, even when the lens is also Sersic

The controlled-case fit (sersic+sersic) recovers magnification 33.34 ± 0.25
versus truth 34.93 — a 5% bias at **5.5σ significance**. Source magnitude
is similarly biased (-0.067 mag at 5.7σ). The posterior 1σ band is so tight
(±0.012 mag) that the systematic shift is many standard deviations from the
truth.

**Likely contributors:**

- The fit places `radius_break=0.05` on `SersicCore`, but the simulator
  uses the default `radius_break=0.025`. The 2× mismatch in the cored
  inner region biases the integrated source flux by a few percent. (Easy
  to retest: re-run with `radius_break=0.025` and check whether the bias
  drops.)
- Inferred `sersic_index ≈ 1.32` versus truth `1.0`. Higher index means
  more concentrated profile. The fit prefers this presumably because of
  small lens-light residuals or PSF-convolution interactions; the change
  in profile shape between truth and fit translates to a small but
  significant flux-integral discrepancy.

The headline lesson is that **parametric fits give error bars that
characterise the data-likelihood width, not the model-truth distance.**
The "truth within posterior" criterion fails comprehensively here even
though the model class is correct.

### 2. Lens-light model has almost no effect on Sersic-source recovery

Swapping the Sersic lens bulge for a 40-Gaussian MGE-lens (40 linear
intensities solved per draw) leaves the source magnification essentially
unchanged: 33.34 → 33.38, well inside both fits' 1σ. So the parametric
bias on the source is driven by the *source* model, not the lens light.
This matters for Euclid: switching the lens light to MGE will not save
you from the source-side bias.

### 3. The MGE source is the danger zone

Swapping the source to MGE drives source-plane flux from 167 to 318 — a
**2× over-estimate**. Magnification drops from 33.4 to 19.6 — a **44% under-
estimate**. Source magnitude shifts by **0.77 mag** (from 19.45 to 18.74).

The posterior on the MGE source is much wider (std 0.38 mag vs 0.012 mag for
parametric), so the bias only registers as **-2.0σ** on magnitude. An MGE
source thus gives *less misleading* error bars than parametric, but the
median itself is far wrong.

**Mechanism (working hypothesis from the inferred fit images at
`fits/mge_lens__mge_source.png`):** the MGE has 20 logarithmically-spaced
Gaussians with σ from 10⁻⁴ to 3.0 arcsec. Wide Gaussians (σ ~ 1 arcsec)
can absorb diffuse residual flux outside the highly-magnified region of
the source plane. That flux integrates over a much larger area than the
compact truth source, so:

- source-plane flux integral is inflated (it picks up the diffuse halo)
- image-plane flux integral grows only modestly (the halo's lensed image
  is faint and spread out)
- magnification = image / source therefore drops dramatically

This is a known pathology of MGE source models when the linear inversion
is unregularised in the source plane — there is nothing in the
likelihood preventing the inversion from pushing flux into wide-σ
Gaussians as long as it improves the in-mask data fit at the few-σ level.

### 4. Highest log-evidence ≠ best science recovery

Ranking by Nautilus log-evidence: MGE+MGE (5412.9) > MGE+Sersic (5395.3)
> Sersic+Sersic (5381.5). Ranking by magnification recovery: Sersic+Sersic
(-5%) ≈ MGE+Sersic (-4%) ≫ MGE+MGE (-44%). The model the data prefers is
the model the *science* most distrusts.

This is the central message for Euclid: **maximising Bayesian evidence is
the wrong objective if you actually care about magnitude/magnification
recovery.** Source magnitude is a derived quantity; the posterior on
derived quantities can be very different from the posterior on the data
fit.

## Decisions

- The bias is real and reproducible at known significance. The next step
  in the source-science investigation is **not** "fit more carefully" but
  to add a controlled diagnostic of *why* the MGE source pulls flux into
  wide-σ Gaussians. Candidates to try (in priority order):

  1. **MGE with σ capped below 1.0 arcsec** — does forbidding the
     widest Gaussians remove the bias?
  2. **MGE with a positive-flux prior penalty on wide-σ components** —
     does a soft prior keep the source compact?
  3. **Stronger source-plane regularisation** when running through the
     SLaM/inversion stack (this experiment ran the MGE as a direct
     linear inversion with no regularisation hyperparameter).
  4. **Smaller mask radius** — `mask_radius=2.0` reduces the available
     halo budget and constrains the source more.

- Before any of those, the trivial test: re-run sersic+sersic with
  `radius_break=0.025` to match truth, and re-check the 5% magnification
  bias. If the bias persists, the issue is profile-shape; if it drops to
  <1%, the radius_break was the dominant cause and the Euclid science is
  more robust than this comparison suggests.

- The next major experiment, per the original prompt, is the
  `RectangularAdaptImage` and `Delaunay` pixelization fits. Those should
  resolve the diffuse-halo question: if they recover the truth flux within
  uncertainty, the MGE source pathology is specifically an artefact of
  the linear-Gaussian basis. If they show the same bias, the problem is
  more general.

## Reproducing

From the worktree root, with the env active:

```bash
cd autolens_workspace_developer
python source_science/fit_compare.py
```

The script resumes from cached Nautilus runs at
`output/output/source_science_v3/`, then redoes the posterior expansion
and writes:

- `dataset/imaging/simple/fit_comparison.json`
- `dataset/imaging/simple/fit_comparison.md`
- `dataset/imaging/simple/fits/<model_name>.png`

Override `SOURCE_SCIENCE_N_DRAWS=N` to change the posterior-sample count
(default 50). MGE runs add ~0.5 s per draw because each draw requires a
linear-intensity inversion via FitImaging; Sersic+Sersic runs at full
speed because the intensities come straight from the posterior.
