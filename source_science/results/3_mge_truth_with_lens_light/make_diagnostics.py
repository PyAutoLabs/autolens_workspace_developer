"""
Diagnostic plots for test 1 (with lens light).

Resumes the three cached Nautilus fits at
`output/output/source_science_v3/simple_v3/<model>/` and produces:

- `fits/source_1d_profile.png`        — 1D source-plane radial brightness, truth + each fit (MLE + posterior 1σ band)
- `fits/source_cumulative_flux.png`   — cumulative integrated flux vs aperture radius
- `fits/source_2d_brightness_panel.png` — N+1 panel side-by-side, same colour scale

Run from `autolens_workspace_developer/` with the worktree env activated.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `source_science/visualize.py` importable from this nested location.
_DEV_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_DEV_ROOT))

from autoconf import jax_wrapper  # Sets JAX environment before other imports
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import autofit as af
import autolens as al

from source_science.visualize import (
    posterior_radial_profile_band,
    source_2d_image,
    source_cumulative_flux,
    source_radial_profile,
    solved_tracer_from_instance,
)


DATASET_NAME = "mge_truth_with_lens_light"
DATASET_PATH = Path("dataset") / "imaging" / DATASET_NAME
MASK_RADIUS = 3.0
POSTERIOR_BAND_N_DRAWS = int(os.environ.get("DIAGNOSTICS_BAND_DRAWS", "20"))
PROFILE_R_MAX = 0.5
PROFILE_N = 600
CUMULATIVE_R_MAX = 3.0
CUMULATIVE_N = 80

MODEL_NAMES = (
    "sersic__sersic",
    "mge_lens__sersic_source",
    "mge_lens__mge_source",
)
MODELS_WITH_LINEAR_LP = {"mge_lens__sersic_source", "mge_lens__mge_source"}
MODEL_COLORS = {
    "sersic__sersic": "tab:blue",
    "mge_lens__sersic_source": "tab:orange",
    "mge_lens__mge_source": "tab:red",
}
MODEL_LABELS = {
    "sersic__sersic": "Sersic lens + Sersic source",
    "mge_lens__sersic_source": "MGE lens + Sersic source",
    "mge_lens__mge_source": "MGE lens + MGE source",
}

# Must match fit_compare.py for the cache to resume.
PATH_PREFIX = Path("output") / "source_science_mge_truth_with_lens_v1"
UNIQUE_TAG = f"{DATASET_NAME}_v1"


def load_dataset() -> al.Imaging:
    dataset = al.Imaging.from_fits(
        data_path=DATASET_PATH / "data.fits",
        psf_path=DATASET_PATH / "psf.fits",
        noise_map_path=DATASET_PATH / "noise_map.fits",
        pixel_scales=0.1,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=MASK_RADIUS,
    )
    dataset = dataset.apply_mask(mask=mask)
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    return dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)


def mass_model():
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.05)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
    mass.einstein_radius = af.UniformPrior(lower_limit=1.2, upper_limit=2.0)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.2)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.1, sigma=0.2)
    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.05)
    shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.05)
    return mass, shear


def lens_sersic_model():
    bulge = af.Model(al.lp.Sersic)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.05)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
    bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.2)
    bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.05, sigma=0.2)
    bulge.intensity = af.LogUniformPrior(lower_limit=0.1, upper_limit=10.0)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.2, upper_limit=1.2)
    bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=6.0)
    return bulge


def source_sersic_model():
    bulge = af.Model(al.lp.SersicCore)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.1, sigma=0.3)
    bulge.intensity = af.LogUniformPrior(lower_limit=0.1, upper_limit=20.0)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.02, upper_limit=0.5)
    bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)
    bulge.radius_break = 0.05
    return bulge


def mge_model(centre_prior_is_uniform: bool) -> af.Model:
    return al.model_util.mge_model_from(
        mask_radius=MASK_RADIUS,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=centre_prior_is_uniform,
    )


def make_model(model_name: str) -> af.Collection:
    mass, shear = mass_model()
    if model_name == "sersic__sersic":
        lens_bulge = lens_sersic_model()
        source_bulge = source_sersic_model()
    elif model_name == "mge_lens__sersic_source":
        lens_bulge = mge_model(centre_prior_is_uniform=True)
        source_bulge = source_sersic_model()
    elif model_name == "mge_lens__mge_source":
        lens_bulge = mge_model(centre_prior_is_uniform=True)
        source_bulge = mge_model(centre_prior_is_uniform=False)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)
    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def resume_result(model_name: str, dataset: al.Imaging) -> af.Result:
    search = af.Nautilus(
        path_prefix=PATH_PREFIX,
        name=model_name,
        unique_tag=UNIQUE_TAG,
        n_live=75,
        n_batch=25,
        iterations_per_quick_update=2000000,
    )
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)
    return search.fit(model=make_model(model_name=model_name), analysis=analysis)


def main():
    dataset = load_dataset()

    out_dir = DATASET_PATH / "fits"
    out_dir.mkdir(parents=True, exist_ok=True)

    truth_tracer = al.from_json(file_path=DATASET_PATH / "tracer.json")
    truth_radii, truth_brightness = source_radial_profile(
        tracer=truth_tracer, r_max=PROFILE_R_MAX, n=PROFILE_N
    )
    truth_cum_radii, truth_cum_flux = source_cumulative_flux(
        tracer=truth_tracer, r_max=CUMULATIVE_R_MAX, n=CUMULATIVE_N
    )
    truth_2d = source_2d_image(tracer=truth_tracer)

    fit_profiles = {}
    fit_cumulative = {}
    fit_2d = {}

    for model_name in MODEL_NAMES:
        print(f"loading fit: {model_name}")
        result = resume_result(model_name=model_name, dataset=dataset)
        has_linear_lp = model_name in MODELS_WITH_LINEAR_LP

        # MLE tracer — already has solved intensities for MGE.
        mle_tracer = result.max_log_likelihood_tracer
        radii, brightness = source_radial_profile(
            tracer=mle_tracer, r_max=PROFILE_R_MAX, n=PROFILE_N
        )
        _, cum_flux = source_cumulative_flux(
            tracer=mle_tracer, r_max=CUMULATIVE_R_MAX, n=CUMULATIVE_N
        )

        print(
            f"  posterior band ({POSTERIOR_BAND_N_DRAWS} draws,"
            f" has_linear_lp={has_linear_lp})..."
        )
        band_radii, band_median, band_lower, band_upper = posterior_radial_profile_band(
            samples=result.samples,
            dataset=dataset,
            has_linear_lp=has_linear_lp,
            r_max=PROFILE_R_MAX,
            n=PROFILE_N,
            n_draws=POSTERIOR_BAND_N_DRAWS,
        )

        fit_profiles[model_name] = {
            "radii": radii,
            "mle": np.asarray(brightness),
            "band_radii": band_radii,
            "band_lower": band_lower,
            "band_upper": band_upper,
        }
        fit_cumulative[model_name] = cum_flux
        fit_2d[model_name] = source_2d_image(tracer=mle_tracer)

    # --- Plot 1: 1D radial source-plane brightness ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        truth_radii, truth_brightness, "k-", lw=2.0, label="Truth (MGE)"
    )
    for model_name in MODEL_NAMES:
        p = fit_profiles[model_name]
        ax.fill_between(
            p["band_radii"],
            p["band_lower"],
            p["band_upper"],
            color=MODEL_COLORS[model_name],
            alpha=0.18,
        )
        ax.plot(
            p["radii"],
            p["mle"],
            color=MODEL_COLORS[model_name],
            lw=1.5,
            label=MODEL_LABELS[model_name] + " (MLE)",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Source-plane radius (arcsec, along x-axis)")
    ax.set_ylabel("Source surface brightness (e⁻ s⁻¹ arcsec⁻²)")
    ax.set_title("Test 3 — Source radial profile (MGE truth, with lens light)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, PROFILE_R_MAX)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "source_1d_profile.png", dpi=140)
    plt.close(fig)

    # --- Plot 2: cumulative source flux vs aperture radius ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(truth_cum_radii, truth_cum_flux, "k-", lw=2.0, label="Truth")
    for model_name in MODEL_NAMES:
        ax.plot(
            truth_cum_radii,
            fit_cumulative[model_name],
            color=MODEL_COLORS[model_name],
            lw=1.5,
            label=MODEL_LABELS[model_name] + " (MLE)",
        )
    ax.set_xlabel("Aperture radius (arcsec)")
    ax.set_ylabel("Cumulative source-plane flux inside aperture")
    ax.set_title("Test 3 — Cumulative source flux (MGE truth, with lens light)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, CUMULATIVE_R_MAX)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "source_cumulative_flux.png", dpi=140)
    plt.close(fig)

    # --- Plot 3: 2D source-plane panel ---
    n_panels = 1 + len(MODEL_NAMES)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
    panels = [("Truth", np.asarray(truth_2d.native))] + [
        (MODEL_LABELS[m], np.asarray(fit_2d[m].native)) for m in MODEL_NAMES
    ]
    vmin = float(min(p[1].min() for p in panels))
    vmax = float(max(p[1].max() for p in panels))
    extent = 0.5 * panels[0][1].shape[0] * 0.005  # half-extent in arcsec
    for ax, (title, arr) in zip(axes, panels):
        im = ax.imshow(
            arr,
            origin="lower",
            extent=(-extent, extent, -extent, extent),
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (arcsec)")
    axes[0].set_ylabel("y (arcsec)")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle(
        "Test 3 — Source-plane brightness (MGE truth, with lens light)", fontsize=12
    )
    fig.savefig(out_dir / "source_2d_brightness_panel.png", dpi=140)
    plt.close(fig)

    print(f"\nWrote diagnostics to {out_dir}/")


if __name__ == "__main__":
    main()
