from autoconf import jax_wrapper  # Sets JAX environment before other imports

import json
from pathlib import Path

import autofit as af
import autolens as al
import numpy as np


DATASET_NAME = "simple"
DATASET_PATH = Path("dataset") / "imaging" / DATASET_NAME
MASK_RADIUS = 3.0
ZERO_POINT = 25.0
SCIENCE_GRID_SHAPE = (400, 400)
SCIENCE_PIXEL_SCALE = 0.03


def load_dataset():
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


def magnitude_from_flux(flux: float) -> float:
    return ZERO_POINT - 2.5 * np.log10(flux)


def source_science_from(tracer: al.Tracer) -> dict:
    grid = al.Grid2D.uniform(
        shape_native=SCIENCE_GRID_SHAPE, pixel_scales=SCIENCE_PIXEL_SCALE
    )

    source_plane_image = tracer.planes[1].image_2d_from(grid=grid)
    source_plane_flux = float(np.sum(source_plane_image))

    traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)
    lensed_source_image = tracer.planes[1].image_2d_from(grid=traced_grid_list[1])
    image_plane_flux = float(np.sum(lensed_source_image))

    source_magnification = image_plane_flux / source_plane_flux

    return {
        "image_plane_flux": image_plane_flux,
        "source_plane_flux": source_plane_flux,
        "source_magnification": float(source_magnification),
        "source_magnitude_zp_25": float(magnitude_from_flux(source_plane_flux)),
    }


def with_comparison(values: dict, truth: dict) -> dict:
    compared = dict(values)

    compared["delta_source_plane_flux"] = (
        values["source_plane_flux"] - truth["source_plane_flux"]
    )
    compared["frac_source_plane_flux"] = (
        values["source_plane_flux"] / truth["source_plane_flux"]
    )
    compared["delta_image_plane_flux"] = (
        values["image_plane_flux"] - truth["image_plane_flux"]
    )
    compared["frac_image_plane_flux"] = (
        values["image_plane_flux"] / truth["image_plane_flux"]
    )
    compared["delta_source_magnification"] = (
        values["source_magnification"] - truth["source_magnification"]
    )
    compared["frac_source_magnification"] = (
        values["source_magnification"] / truth["source_magnification"]
    )
    compared["delta_source_magnitude_zp_25"] = (
        values["source_magnitude_zp_25"] - truth["source_magnitude_zp_25"]
    )

    return compared


def truth_values_from() -> tuple[al.Tracer, dict]:
    tracer = al.from_json(file_path=DATASET_PATH / "tracer.json")
    values = source_science_from(tracer=tracer)
    return tracer, values


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


def fit_model(dataset: al.Imaging, model_name: str) -> tuple[af.Result, dict]:
    model = make_model(model_name=model_name)

    search = af.Nautilus(
        path_prefix=Path("output") / "source_science_v2",
        name=model_name,
        unique_tag=f"{DATASET_NAME}_v2",
        n_live=75,
        n_batch=25,
        iterations_per_quick_update=2000000,
    )

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)
    result = search.fit(model=model, analysis=analysis)
    values = source_science_from(tracer=result.max_log_likelihood_tracer)

    return result, values


def main():
    dataset = load_dataset()
    truth_tracer, truth = truth_values_from()

    if (DATASET_PATH / "source_science.json").exists():
        with open(DATASET_PATH / "source_science.json") as f:
            simulator_values = json.load(f)
        print("Simulator source_science.json:")
        print(json.dumps(simulator_values, indent=4))

    print("Truth from tracer:")
    print(json.dumps(truth, indent=4))

    summary = {
        "dataset_path": str(DATASET_PATH),
        "zero_point_assumption": ZERO_POINT,
        "truth": truth,
        "fits": {},
    }

    for model_name in [
        "sersic__sersic",
        "mge_lens__sersic_source",
        "mge_lens__mge_source",
    ]:
        print(f"Running fit: {model_name}")
        result, values = fit_model(dataset=dataset, model_name=model_name)
        compared = with_comparison(values=values, truth=truth)
        summary["fits"][model_name] = compared

        print(f"Completed fit: {model_name}")
        print(json.dumps(compared, indent=4))
        print(f"Max log likelihood: {result.max_log_likelihood_fit.log_likelihood}")

    output_path = DATASET_PATH / "fit_comparison.json"

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Wrote comparison summary to {output_path}")


if __name__ == "__main__":
    main()
