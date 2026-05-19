"""
Cross-experiment 2x2 plot: matched vs mismatched source-truth/source-fit.

Rows = truth class (top: Sersic source truth; bottom: MGE source truth).
Cols = lens-light condition (left: no lens light; right: with lens light).
Each cell overlays 1D source-plane radial profile of truth + Sersic-source-fit
MLE + MGE-source-fit MLE.

This makes obvious whether the bias in each cell is driven by fit-vs-truth
mismatch (when one of the fit profiles diverges from truth) or by lens-light
contamination (when the same fit class differs across columns).

Cells:
  top-left:    test 2, Sersic-source truth, no lens light  — fits = sersic_source, mge_source
  top-right:   test 1, Sersic-source truth, with lens light — fits = mge_lens__sersic_source, mge_lens__mge_source
  bottom-left: test 4, MGE-source truth,   no lens light    — fits = sersic_source, mge_source
  bottom-right:test 3, MGE-source truth,   with lens light  — fits = mge_lens__sersic_source, mge_lens__mge_source

For the with-lens-light row, the *sersic+sersic* test-1 fit is not included
because the lens-light parameterisation also differs there; including it
would conflate two effects. The MGE-lens variants give the cleanest pairwise
comparison (lens-light parameterisation held constant; only source class
varies).

Output: `source_science/results/matched_vs_mismatched_2x2.png`.
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

from source_science.visualize import source_radial_profile


PROFILE_R_MAX = 0.5
PROFILE_N = 600
MASK_RADIUS = 3.0


# --- Shared model components ---


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


def _source_sersic_model():
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


def _mge_model(centre_prior_is_uniform: bool) -> af.Model:
    return al.model_util.mge_model_from(
        mask_radius=MASK_RADIUS,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=centre_prior_is_uniform,
    )


# --- Per-cell model builders ---


def _model_no_lens_sersic_source() -> af.Collection:
    mass, shear = _mass_model()
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
    source = af.Model(al.Galaxy, redshift=1.0, bulge=_source_sersic_model())
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _model_no_lens_mge_source() -> af.Collection:
    mass, shear = _mass_model()
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
    source = af.Model(
        al.Galaxy, redshift=1.0, bulge=_mge_model(centre_prior_is_uniform=False)
    )
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _model_with_lens_sersic_source() -> af.Collection:
    mass, shear = _mass_model()
    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=_mge_model(centre_prior_is_uniform=True),
        mass=mass,
        shear=shear,
    )
    source = af.Model(al.Galaxy, redshift=1.0, bulge=_source_sersic_model())
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _model_with_lens_mge_source() -> af.Collection:
    mass, shear = _mass_model()
    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=_mge_model(centre_prior_is_uniform=True),
        mass=mass,
        shear=shear,
    )
    source = af.Model(
        al.Galaxy, redshift=1.0, bulge=_mge_model(centre_prior_is_uniform=False)
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


def _resume(
    dataset_name: str, path_prefix_dir: str, unique_tag: str,
    fit_name: str, model_factory,
) -> al.Tracer:
    dataset = _load_dataset(dataset_name)
    search = af.Nautilus(
        path_prefix=Path("output") / path_prefix_dir,
        name=fit_name,
        unique_tag=unique_tag,
        n_live=75,
        n_batch=25,
        iterations_per_quick_update=2000000,
    )
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)
    result = search.fit(model=model_factory(), analysis=analysis)
    return result.max_log_likelihood_tracer


CELLS = [
    # (row, col, title, truth_dataset, fits[list of (label, dataset, path_prefix, unique_tag, name, model_factory)])
    (
        0, 0, "Sersic truth · no lens light",
        "no_lens_light",
        [
            ("Sersic source fit", "no_lens_light", "source_science_no_lens_v1",
             "no_lens_light_v1", "sersic_source", _model_no_lens_sersic_source),
            ("MGE source fit", "no_lens_light", "source_science_no_lens_v1",
             "no_lens_light_v1", "mge_source", _model_no_lens_mge_source),
        ],
    ),
    (
        0, 1, "Sersic truth · with lens light",
        "simple",
        [
            ("Sersic source fit", "simple", "source_science_v3",
             "simple_v3", "mge_lens__sersic_source", _model_with_lens_sersic_source),
            ("MGE source fit", "simple", "source_science_v3",
             "simple_v3", "mge_lens__mge_source", _model_with_lens_mge_source),
        ],
    ),
    (
        1, 0, "MGE truth · no lens light",
        "mge_truth_no_lens_light",
        [
            ("Sersic source fit", "mge_truth_no_lens_light",
             "source_science_mge_truth_no_lens_v1",
             "mge_truth_no_lens_light_v1", "sersic_source",
             _model_no_lens_sersic_source),
            ("MGE source fit", "mge_truth_no_lens_light",
             "source_science_mge_truth_no_lens_v1",
             "mge_truth_no_lens_light_v1", "mge_source",
             _model_no_lens_mge_source),
        ],
    ),
    (
        1, 1, "MGE truth · with lens light",
        "mge_truth_with_lens_light",
        [
            ("Sersic source fit", "mge_truth_with_lens_light",
             "source_science_mge_truth_with_lens_v1",
             "mge_truth_with_lens_light_v1", "mge_lens__sersic_source",
             _model_with_lens_sersic_source),
            ("MGE source fit", "mge_truth_with_lens_light",
             "source_science_mge_truth_with_lens_v1",
             "mge_truth_with_lens_light_v1", "mge_lens__mge_source",
             _model_with_lens_mge_source),
        ],
    ),
]
FIT_COLORS = {"Sersic source fit": "tab:blue", "MGE source fit": "tab:red"}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)

    for row, col, title, truth_dataset, fits in CELLS:
        ax = axes[row, col]
        truth_tracer = al.from_json(
            file_path=Path("dataset") / "imaging" / truth_dataset / "tracer.json"
        )
        radii, truth_b = source_radial_profile(
            tracer=truth_tracer, r_max=PROFILE_R_MAX, n=PROFILE_N
        )
        ax.plot(radii, truth_b, "k-", lw=2.5, label="Truth", zorder=3)

        for label, dname, pp, ut, fit_name, model_factory in fits:
            print(f"[{title}] loading {label} ...")
            mle_tracer = _resume(
                dataset_name=dname, path_prefix_dir=pp,
                unique_tag=ut, fit_name=fit_name, model_factory=model_factory,
            )
            _, brightness = source_radial_profile(
                tracer=mle_tracer, r_max=PROFILE_R_MAX, n=PROFILE_N
            )
            ax.plot(
                radii, brightness, color=FIT_COLORS[label], lw=1.6,
                label=label,
            )

        ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        if row == 1:
            ax.set_xlabel("Source-plane radius (arcsec)")
        if col == 0:
            ax.set_ylabel("Source surface brightness (e⁻ s⁻¹ arcsec⁻²)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim(0, PROFILE_R_MAX)

    fig.suptitle(
        "Matched vs mismatched: source-fit class × source-truth class × lens-light condition",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    out = Path("source_science") / "results" / "matched_vs_mismatched_2x2.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote 2x2 to {out}")


if __name__ == "__main__":
    main()
