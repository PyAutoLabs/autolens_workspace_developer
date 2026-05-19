"""
Cross-experiment headline plot: MGE source in test 1 vs test 2 vs truth.

The single image that visualises the lens-light-degeneracy hypothesis.
Two panels:

- LEFT: 1D source-plane radial profile — truth, test-1 MGE-source, test-2
  MGE-source.
- RIGHT: cumulative source flux vs aperture radius — same three curves.

In test 1, the MGE source absorbs diffuse lens-light residuals into wide-σ
Gaussians; its profile has a clear excess at large radii and its cumulative
flux blows past truth. In test 2, with no lens light to absorb, the MGE
source tracks the Sersic recovery and the cumulative flux nearly matches
truth — except for the residual ~4% bias which is profile-shape, not source-
flexibility.

Output: `source_science/results/test1_vs_test2_mge_source.png`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEV_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DEV_ROOT))

from autoconf import jax_wrapper  # Sets JAX environment before other imports
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import autofit as af
import autolens as al

from source_science.visualize import (
    source_2d_image,
    source_cumulative_flux,
    source_radial_profile,
)


PROFILE_R_MAX = 0.5
PROFILE_N = 600
CUMULATIVE_R_MAX = 3.0
CUMULATIVE_N = 80
MASK_RADIUS = 3.0


# --- Helpers to load the test-1 and test-2 MGE-source fits ---


def _mass_model():
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


def _mge_model(centre_prior_is_uniform: bool) -> af.Model:
    return al.model_util.mge_model_from(
        mask_radius=MASK_RADIUS,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=centre_prior_is_uniform,
    )


def _make_test1_mge_source_model() -> af.Collection:
    """test 1 used MGE lens + MGE source."""
    mass, shear = _mass_model()
    lens_bulge = _mge_model(centre_prior_is_uniform=True)
    source_bulge = _mge_model(centre_prior_is_uniform=False)
    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)
    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _make_test2_mge_source_model() -> af.Collection:
    """test 2 used no lens bulge + MGE source."""
    mass, shear = _mass_model()
    source_bulge = _mge_model(centre_prior_is_uniform=False)
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _load_dataset(dataset_name: str) -> al.Imaging:
    dataset_path = Path("dataset") / "imaging" / dataset_name
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
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


def _resume_test1() -> al.Tracer:
    dataset = _load_dataset("simple")
    search = af.Nautilus(
        path_prefix=Path("output") / "source_science_v3",
        name="mge_lens__mge_source",
        unique_tag="simple_v3",
        n_live=75,
        n_batch=25,
        iterations_per_quick_update=2000000,
    )
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)
    result = search.fit(model=_make_test1_mge_source_model(), analysis=analysis)
    return result.max_log_likelihood_tracer


def _resume_test2() -> al.Tracer:
    dataset = _load_dataset("no_lens_light")
    search = af.Nautilus(
        path_prefix=Path("output") / "source_science_no_lens_v1",
        name="mge_source",
        unique_tag="no_lens_light_v1",
        n_live=75,
        n_batch=25,
        iterations_per_quick_update=2000000,
    )
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)
    result = search.fit(model=_make_test2_mge_source_model(), analysis=analysis)
    return result.max_log_likelihood_tracer


# --- Plot ---


def main():
    truth_tracer = al.from_json(
        file_path=Path("dataset") / "imaging" / "simple" / "tracer.json"
    )

    print("loading test 1 MGE-source MLE tracer...")
    test1_tracer = _resume_test1()
    print("loading test 2 MGE-source MLE tracer...")
    test2_tracer = _resume_test2()

    radii, truth_b = source_radial_profile(
        tracer=truth_tracer, r_max=PROFILE_R_MAX, n=PROFILE_N
    )
    _, test1_b = source_radial_profile(
        tracer=test1_tracer, r_max=PROFILE_R_MAX, n=PROFILE_N
    )
    _, test2_b = source_radial_profile(
        tracer=test2_tracer, r_max=PROFILE_R_MAX, n=PROFILE_N
    )

    cum_radii, truth_c = source_cumulative_flux(
        tracer=truth_tracer, r_max=CUMULATIVE_R_MAX, n=CUMULATIVE_N
    )
    _, test1_c = source_cumulative_flux(
        tracer=test1_tracer, r_max=CUMULATIVE_R_MAX, n=CUMULATIVE_N
    )
    _, test2_c = source_cumulative_flux(
        tracer=test2_tracer, r_max=CUMULATIVE_R_MAX, n=CUMULATIVE_N
    )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))

    # LEFT — 1D radial profile (log y)
    ax_l.plot(radii, truth_b, "k-", lw=2.5, label="Truth (SersicCore)")
    ax_l.plot(
        radii, test1_b, color="tab:red", lw=1.7,
        label="Test 1 — MGE source (WITH lens light)",
    )
    ax_l.plot(
        radii, test2_b, color="tab:blue", lw=1.7,
        label="Test 2 — MGE source (NO lens light)",
    )
    ax_l.set_yscale("log")
    ax_l.set_xlabel("Source-plane radius (arcsec, along x-axis)")
    ax_l.set_ylabel("Source surface brightness (e⁻ s⁻¹ arcsec⁻²)")
    ax_l.set_title("MGE source radial profile — diffuse halo collapses without lens light")
    ax_l.legend(loc="upper right", fontsize=10)
    ax_l.set_xlim(0, PROFILE_R_MAX)
    ax_l.grid(True, alpha=0.3, which="both")

    # RIGHT — cumulative flux
    ax_r.axhline(
        y=float(truth_c[-1]), color="k", linestyle="--", lw=1.0, alpha=0.4
    )
    ax_r.plot(cum_radii, truth_c, "k-", lw=2.5, label="Truth")
    ax_r.plot(
        cum_radii, test1_c, color="tab:red", lw=1.7,
        label="Test 1 — MGE source (WITH lens light)",
    )
    ax_r.plot(
        cum_radii, test2_c, color="tab:blue", lw=1.7,
        label="Test 2 — MGE source (NO lens light)",
    )
    ax_r.set_xlabel("Aperture radius (arcsec)")
    ax_r.set_ylabel("Cumulative source-plane flux inside aperture")
    ax_r.set_title("Cumulative flux — test-1 MGE keeps growing past truth into the halo")
    ax_r.legend(loc="lower right", fontsize=10)
    ax_r.set_xlim(0, CUMULATIVE_R_MAX)
    ax_r.grid(True, alpha=0.3)

    fig.suptitle(
        "Hypothesis confirmed: MGE-source bias is driven by lens-light/source-light degeneracy",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out = Path("source_science") / "results" / "test1_vs_test2_mge_source.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote cross-experiment plot to {out}")


if __name__ == "__main__":
    main()
