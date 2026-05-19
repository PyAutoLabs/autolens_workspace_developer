"""
Headline cross-experiment plot for tests 3 + 4 (MGE source truth).

Mirrors `make_cross_experiment_plot.py` for tests 1+2 but for the MGE-truth
case. Two panels:

- LEFT: 1D source-plane radial profile — MGE truth, test-3 MGE+MGE MLE,
  test-4 MGE-source MLE.
- RIGHT: cumulative source flux inside circular apertures — same three.

If the lens-light/source-light degeneracy mechanism is general, test 3
(MGE truth + with lens light) should show some bias and test 4 (MGE truth
+ no lens light) should recover tightly. If the lens-light effect only
hits the Sersic-truth case, test 3 might recover cleanly too — that would
be a surprise.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEV_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_DEV_ROOT))

from autoconf import jax_wrapper  # Sets JAX environment before other imports
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import autofit as af
import autolens as al

from source_science.visualize import (
    source_cumulative_flux,
    source_radial_profile,
)


PROFILE_R_MAX = 0.5
PROFILE_N = 600
CUMULATIVE_R_MAX = 3.0
CUMULATIVE_N = 80
MASK_RADIUS = 3.0


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


def _model_test3_mge_mge() -> af.Collection:
    mass, shear = _mass_model()
    lens = af.Model(
        al.Galaxy, redshift=0.5, bulge=_mge_model(centre_prior_is_uniform=True),
        mass=mass, shear=shear,
    )
    source = af.Model(
        al.Galaxy, redshift=1.0,
        bulge=_mge_model(centre_prior_is_uniform=False),
    )
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _model_test4_mge() -> af.Collection:
    mass, shear = _mass_model()
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
    source = af.Model(
        al.Galaxy, redshift=1.0,
        bulge=_mge_model(centre_prior_is_uniform=False),
    )
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


def _resume_test3() -> al.Tracer:
    dataset = _load_dataset("mge_truth_with_lens_light")
    search = af.Nautilus(
        path_prefix=Path("output") / "source_science_mge_truth_with_lens_v1",
        name="mge_lens__mge_source",
        unique_tag="mge_truth_with_lens_light_v1",
        n_live=75,
        n_batch=25,
        iterations_per_quick_update=2000000,
    )
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)
    result = search.fit(model=_model_test3_mge_mge(), analysis=analysis)
    return result.max_log_likelihood_tracer


def _resume_test4() -> al.Tracer:
    dataset = _load_dataset("mge_truth_no_lens_light")
    search = af.Nautilus(
        path_prefix=Path("output") / "source_science_mge_truth_no_lens_v1",
        name="mge_source",
        unique_tag="mge_truth_no_lens_light_v1",
        n_live=75,
        n_batch=25,
        iterations_per_quick_update=2000000,
    )
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)
    result = search.fit(model=_model_test4_mge(), analysis=analysis)
    return result.max_log_likelihood_tracer


def main():
    # Truth — same MGE source either way, but load from each dataset's tracer.json
    # to ensure correct lens for the lensed comparison; for the source-plane
    # radial profile we just need the source galaxy which is identical.
    truth_tracer = al.from_json(
        file_path=Path("dataset") / "imaging" / "mge_truth_no_lens_light"
        / "tracer.json"
    )

    print("loading test 3 (with lens light) MGE+MGE MLE...")
    test3_tracer = _resume_test3()
    print("loading test 4 (no lens light) MGE-source MLE...")
    test4_tracer = _resume_test4()

    radii, truth_b = source_radial_profile(
        tracer=truth_tracer, r_max=PROFILE_R_MAX, n=PROFILE_N
    )
    _, test3_b = source_radial_profile(
        tracer=test3_tracer, r_max=PROFILE_R_MAX, n=PROFILE_N
    )
    _, test4_b = source_radial_profile(
        tracer=test4_tracer, r_max=PROFILE_R_MAX, n=PROFILE_N
    )

    cum_radii, truth_c = source_cumulative_flux(
        tracer=truth_tracer, r_max=CUMULATIVE_R_MAX, n=CUMULATIVE_N
    )
    _, test3_c = source_cumulative_flux(
        tracer=test3_tracer, r_max=CUMULATIVE_R_MAX, n=CUMULATIVE_N
    )
    _, test4_c = source_cumulative_flux(
        tracer=test4_tracer, r_max=CUMULATIVE_R_MAX, n=CUMULATIVE_N
    )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))

    ax_l.plot(radii, truth_b, "k-", lw=2.5, label="Truth (MGE source)")
    ax_l.plot(
        radii, test3_b, color="tab:red", lw=1.7,
        label="Test 3 — MGE+MGE fit (WITH lens light)",
    )
    ax_l.plot(
        radii, test4_b, color="tab:blue", lw=1.7,
        label="Test 4 — MGE source fit (NO lens light)",
    )
    ax_l.set_yscale("log")
    ax_l.set_xlabel("Source-plane radius (arcsec, along x-axis)")
    ax_l.set_ylabel("Source surface brightness (e⁻ s⁻¹ arcsec⁻²)")
    ax_l.set_title("MGE source radial profile — does the halo come back when truth is MGE?")
    ax_l.legend(loc="upper right", fontsize=10)
    ax_l.set_xlim(0, PROFILE_R_MAX)
    ax_l.grid(True, alpha=0.3, which="both")

    ax_r.axhline(
        y=float(truth_c[-1]), color="k", linestyle="--", lw=1.0, alpha=0.4
    )
    ax_r.plot(cum_radii, truth_c, "k-", lw=2.5, label="Truth")
    ax_r.plot(
        cum_radii, test3_c, color="tab:red", lw=1.7,
        label="Test 3 — MGE+MGE fit (WITH lens light)",
    )
    ax_r.plot(
        cum_radii, test4_c, color="tab:blue", lw=1.7,
        label="Test 4 — MGE source fit (NO lens light)",
    )
    ax_r.set_xlabel("Aperture radius (arcsec)")
    ax_r.set_ylabel("Cumulative source-plane flux inside aperture")
    ax_r.set_title("Cumulative flux — MGE truth")
    ax_r.legend(loc="lower right", fontsize=10)
    ax_r.set_xlim(0, CUMULATIVE_R_MAX)
    ax_r.grid(True, alpha=0.3)

    fig.suptitle(
        "Tests 3+4 (MGE source truth): is the lens-light degeneracy general or Sersic-truth-specific?",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = Path("source_science") / "results" / "test3_vs_test4_mge_source.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote cross-experiment plot to {out}")


if __name__ == "__main__":
    main()
