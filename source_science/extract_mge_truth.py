"""
Extract the MGE-source truth used by tests 3 and 4.

Runs (or resumes) the test-2 `mge_source` fit on `dataset/imaging/no_lens_light/`,
takes its MLE tracer, runs the inversion to populate the per-Gaussian
intensities, then saves *just the source galaxy* to
`source_science/results/mge_truth_source.json`. Tests 3 and 4 simulators
load this file as their source truth and compose it with their own lens
setup (Sersic bulge + SIE + shear for test 3, SIE + shear only for test 4).

Also saves a sanity-check 1D radial profile of the MGE truth source.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEV_ROOT = _HERE.parent
sys.path.insert(0, str(_DEV_ROOT))

from autoconf import jax_wrapper  # Sets JAX environment before other imports
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import autofit as af
import autolens as al

from source_science.visualize import source_radial_profile


DATASET_NAME = "no_lens_light"
DATASET_PATH = Path("dataset") / "imaging" / DATASET_NAME
MASK_RADIUS = 3.0
PATH_PREFIX = Path("output") / "source_science_no_lens_v1"
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


def mge_source_model() -> af.Model:
    return al.model_util.mge_model_from(
        mask_radius=MASK_RADIUS,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=False,
    )


def make_model() -> af.Collection:
    mass, shear = mass_model()
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
    source = af.Model(al.Galaxy, redshift=1.0, bulge=mge_source_model())
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def main():
    out_path = Path("source_science") / "results" / "mge_truth_source.json"
    sanity_plot = Path("source_science") / "results" / "mge_truth_source_profile.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    search = af.Nautilus(
        path_prefix=PATH_PREFIX,
        name="mge_source",
        unique_tag=UNIQUE_TAG,
        n_live=75,
        n_batch=25,
        iterations_per_quick_update=2000000,
    )
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)
    result = search.fit(model=make_model(), analysis=analysis)

    fit = al.FitImaging(
        dataset=dataset, tracer=result.max_log_likelihood_tracer
    )
    solved_tracer = fit.tracer_linear_light_profiles_to_light_profiles
    source_galaxy = solved_tracer.galaxies[1]

    al.output_to_json(obj=source_galaxy, file_path=out_path)
    print(f"\nSaved MGE truth source to {out_path}")

    # Sanity check — load back and compare to in-memory.
    reloaded = al.from_json(file_path=out_path)
    print(f"Reloaded source type: {type(reloaded).__name__}")

    # Brightness check: compose a trivial tracer to use source_radial_profile.
    fake_lens = al.Galaxy(redshift=0.5)
    sanity_tracer = al.Tracer(galaxies=[fake_lens, reloaded])
    radii, brightness = source_radial_profile(
        tracer=sanity_tracer, r_max=0.5, n=600
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(radii, brightness, color="tab:red", lw=1.5, label="MGE truth source")
    ax.set_yscale("log")
    ax.set_xlabel("Source-plane radius (arcsec, along x-axis)")
    ax.set_ylabel("Surface brightness (e⁻ s⁻¹ arcsec⁻²)")
    ax.set_title("Sanity check: MGE truth source 1D radial profile")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0, 0.5)
    fig.tight_layout()
    fig.savefig(sanity_plot, dpi=140)
    plt.close(fig)
    print(f"Wrote sanity profile to {sanity_plot}")


if __name__ == "__main__":
    main()
