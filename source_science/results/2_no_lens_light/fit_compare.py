"""
Fit Compare: No Lens Light
==========================

Same posterior-aware comparison as experiment 1 but fitting the
no-lens-light dataset (`dataset/imaging/no_lens_light/`). Two models only:

- `sersic_source`  — Isothermal+shear lens + SersicCore source
- `mge_source`     — Isothermal+shear lens + MGE source (40 Gaussians)

Hypothesis under test: the MGE-source magnification/magnitude bias seen in
experiment 1 was driven by lens-light/source-light degeneracy during the fit.
With no lens light to fit, the MGE source should recover truth far better.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import autofit as af
import autolens as al
import autolens.plot as aplt
import numpy as np


DATASET_NAME = "no_lens_light"
DATASET_PATH = Path("dataset") / "imaging" / DATASET_NAME
MASK_RADIUS = 3.0
ZERO_POINT = 25.0
SCIENCE_GRID_SHAPE = (400, 400)
SCIENCE_PIXEL_SCALE = 0.03
N_POSTERIOR_DRAWS = int(os.environ.get("SOURCE_SCIENCE_N_DRAWS", "50"))

QUANTITY_KEYS = (
    "image_plane_flux",
    "source_plane_flux",
    "source_magnification",
    "source_magnitude_zp_25",
)
MODEL_NAMES = (
    "sersic_source",
    "mge_source",
)
MODELS_WITH_LINEAR_LP = {"mge_source"}


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
    for key in QUANTITY_KEYS[:-1]:  # magnitude has no frac/delta convention
        compared[f"delta_{key}"] = values[key] - truth[key]
        compared[f"frac_{key}"] = values[key] / truth[key]
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


def mge_source_model() -> af.Model:
    return al.model_util.mge_model_from(
        mask_radius=MASK_RADIUS,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=False,
    )


def make_model(model_name: str) -> af.Collection:
    mass, shear = mass_model()

    if model_name == "sersic_source":
        source_bulge = source_sersic_model()
    elif model_name == "mge_source":
        source_bulge = mge_source_model()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # No lens bulge — the lens galaxy carries only mass and shear. This is the
    # whole point of experiment 2: remove the lens-light degree of freedom.
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def fit_model(dataset: al.Imaging, model_name: str) -> af.Result:
    model = make_model(model_name=model_name)
    search = af.Nautilus(
        path_prefix=Path("output") / "source_science_no_lens_v1",
        name=model_name,
        unique_tag=f"{DATASET_NAME}_v1",
        n_live=75,
        n_batch=25,
        iterations_per_quick_update=2000000,
    )
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)
    return search.fit(model=model, analysis=analysis)


def _solved_tracer_from_instance(
    instance, dataset: al.Imaging, has_linear_lp: bool
) -> al.Tracer:
    tracer = al.Tracer(galaxies=instance.galaxies)
    if not has_linear_lp:
        return tracer
    fit = al.FitImaging(dataset=dataset, tracer=tracer)
    return fit.tracer_linear_light_profiles_to_light_profiles


def posterior_source_science_from(
    samples,
    dataset: al.Imaging,
    has_linear_lp: bool,
    n_draws: int = N_POSTERIOR_DRAWS,
) -> dict:
    draws_by_quantity = {key: [] for key in QUANTITY_KEYS}
    n_failed = 0
    for _ in range(n_draws):
        instance = samples.draw_randomly_via_pdf()
        try:
            tracer = _solved_tracer_from_instance(
                instance=instance, dataset=dataset, has_linear_lp=has_linear_lp
            )
            values = source_science_from(tracer=tracer)
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            print(f"  posterior draw failed: {e}")
            continue
        for key in QUANTITY_KEYS:
            draws_by_quantity[key].append(values[key])

    summary = {"n_draws_requested": n_draws, "n_draws_failed": n_failed}
    for key, draws in draws_by_quantity.items():
        if not draws:
            summary[key] = None
            continue
        arr = np.asarray(draws)
        summary[key] = {
            "median": float(np.median(arr)),
            "lower_1sigma": float(np.percentile(arr, 15.865)),
            "upper_1sigma": float(np.percentile(arr, 84.135)),
            "lower_3sigma": float(np.percentile(arr, 0.135)),
            "upper_3sigma": float(np.percentile(arr, 99.865)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        }
    return summary


def with_pdf_comparison(pdf_summary: dict, truth: dict) -> dict:
    compared = {
        "n_draws_requested": pdf_summary.get("n_draws_requested"),
        "n_draws_failed": pdf_summary.get("n_draws_failed"),
    }
    for key in QUANTITY_KEYS:
        entry = pdf_summary.get(key)
        if entry is None:
            compared[key] = None
            continue
        truth_value = truth[key]
        compared[key] = {
            **entry,
            "truth": truth_value,
            "delta_median": entry["median"] - truth_value,
            "z_score": (
                (entry["median"] - truth_value) / entry["std"]
                if entry["std"] > 0
                else None
            ),
            "truth_within_1sigma": bool(
                entry["lower_1sigma"] <= truth_value <= entry["upper_1sigma"]
            ),
            "truth_within_3sigma": bool(
                entry["lower_3sigma"] <= truth_value <= entry["upper_3sigma"]
            ),
        }
    return compared


def save_fit_subplot(result: af.Result, model_name: str) -> Path:
    fit_image_dir = DATASET_PATH / "fits"
    fit_image_dir.mkdir(parents=True, exist_ok=True)
    output_path = fit_image_dir / f"{model_name}.png"
    src = Path(result.paths.image_path) / "fit.png"
    if src.exists():
        shutil.copyfile(src, output_path)
    else:
        raise FileNotFoundError(f"Expected {src} from search output")
    return output_path


def _fmt(value, precision: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "**NO**"
    return f"{value:.{precision}g}"


def write_markdown_summary(summary: dict, output_path: Path) -> None:
    truth = summary["truth"]
    lines = [
        "# Source-Science Fit Comparison — No Lens Light",
        "",
        f"Dataset: `{summary['dataset_path']}` (zero-point = {summary['zero_point_assumption']})",
        f"Posterior draws per fit: {N_POSTERIOR_DRAWS}",
        "",
        "## Truth (from tracer)",
        "",
        f"- image-plane flux:    {truth['image_plane_flux']:.4f}",
        f"- source-plane flux:   {truth['source_plane_flux']:.4f}",
        f"- magnification:       {truth['source_magnification']:.4f}",
        f"- magnitude (zp=25):   {truth['source_magnitude_zp_25']:.4f}",
        "",
    ]
    for model_name, fit_data in summary["fits"].items():
        mle = fit_data["mle"]
        pdf = fit_data["pdf"]
        lines += [
            f"## {model_name}",
            "",
            f"max log likelihood: {fit_data['log_likelihood']:.4f}",
            "",
            "| Quantity | Truth | MLE | MLE / truth | PDF median | PDF ±1σ | within 1σ? | within 3σ? | z-score |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for key, label in [
            ("source_plane_flux", "source flux"),
            ("image_plane_flux", "image flux"),
            ("source_magnification", "magnification"),
            ("source_magnitude_zp_25", "magnitude"),
        ]:
            truth_value = truth[key]
            mle_value = mle[key]
            mle_frac = mle.get(f"frac_{key}")
            pdf_entry = pdf.get(key)
            if pdf_entry is None:
                lines.append(
                    f"| {label} | {_fmt(truth_value)} | {_fmt(mle_value)} | "
                    f"{_fmt(mle_frac)} | — | — | — | — | — |"
                )
                continue
            sigma_str = (
                f"+{pdf_entry['upper_1sigma'] - pdf_entry['median']:.4g} / "
                f"-{pdf_entry['median'] - pdf_entry['lower_1sigma']:.4g}"
            )
            lines.append(
                f"| {label} | {_fmt(truth_value)} | {_fmt(mle_value)} | {_fmt(mle_frac)} | "
                f"{_fmt(pdf_entry['median'])} | {sigma_str} | "
                f"{_fmt(pdf_entry['truth_within_1sigma'])} | "
                f"{_fmt(pdf_entry['truth_within_3sigma'])} | "
                f"{_fmt(pdf_entry['z_score'])} |"
            )
        lines += ["", ""]
    output_path.write_text("\n".join(lines))


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
        "n_posterior_draws": N_POSTERIOR_DRAWS,
        "truth": truth,
        "fits": {},
    }

    for model_name in MODEL_NAMES:
        print(f"Running fit: {model_name}")
        result = fit_model(dataset=dataset, model_name=model_name)
        log_likelihood = float(result.max_log_likelihood_fit.log_likelihood)
        print(f"Completed fit: {model_name}  log_likelihood={log_likelihood}")

        mle_values = source_science_from(tracer=result.max_log_likelihood_tracer)
        mle_compared = with_comparison(values=mle_values, truth=truth)

        has_linear_lp = model_name in MODELS_WITH_LINEAR_LP
        print(
            f"  posterior expansion (n_draws={N_POSTERIOR_DRAWS}, has_linear_lp={has_linear_lp})..."
        )
        pdf_summary = posterior_source_science_from(
            samples=result.samples, dataset=dataset, has_linear_lp=has_linear_lp
        )
        pdf_compared = with_pdf_comparison(pdf_summary=pdf_summary, truth=truth)

        try:
            fit_image_path = save_fit_subplot(result=result, model_name=model_name)
            print(f"  wrote fit subplot: {fit_image_path}")
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING failed to save fit subplot for {model_name}: {e}")

        summary["fits"][model_name] = {
            "log_likelihood": log_likelihood,
            "mle": mle_compared,
            "pdf": pdf_compared,
        }

        print(f"  MLE comparison:\n{json.dumps(mle_compared, indent=4)}")
        print(f"  posterior comparison:\n{json.dumps(pdf_compared, indent=4)}")

    json_path = DATASET_PATH / "fit_comparison.json"
    md_path = DATASET_PATH / "fit_comparison.md"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
    write_markdown_summary(summary=summary, output_path=md_path)

    print(f"Wrote comparison JSON to {json_path}")
    print(f"Wrote comparison Markdown to {md_path}")


if __name__ == "__main__":
    main()
