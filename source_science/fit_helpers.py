"""
Shared fit + posterior-expansion helpers for the lens-config robustness study.

Each per-test driver script under `results/<config>/<test>/` builds a list
of `(model_name, model_factory, has_linear_lp)` tuples and calls
`run_fits_and_compare()` here.
"""

from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import numpy as np
import autofit as af
import autolens as al


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


def load_dataset(dataset_path: Path) -> al.Imaging:
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


# --- Model component factories ---


def mass_model() -> Tuple[af.Model, af.Model]:
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.05)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=2.5)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.1)
    return mass, shear


def lens_sersic_bulge_model() -> af.Model:
    bulge = af.Model(al.lp.Sersic)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.05)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
    bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge.intensity = af.LogUniformPrior(lower_limit=0.05, upper_limit=20.0)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.1, upper_limit=2.0)
    bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=6.0)
    return bulge


def source_sersic_bulge_model() -> af.Model:
    bulge = af.Model(al.lp.SersicCore)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
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


def make_collection(*, source_class: str, lens_light_class: str) -> af.Collection:
    """Compose an `af.Collection` lens-source model.

    Parameters
    ----------
    source_class : "sersic" | "mge"
    lens_light_class : "none" | "sersic" | "mge"
    """
    mass, shear = mass_model()
    if source_class == "sersic":
        source_bulge = source_sersic_bulge_model()
    elif source_class == "mge":
        source_bulge = mge_model(centre_prior_is_uniform=False)
    else:
        raise ValueError(source_class)

    if lens_light_class == "none":
        lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
    elif lens_light_class == "sersic":
        lens = af.Model(
            al.Galaxy,
            redshift=0.5,
            bulge=lens_sersic_bulge_model(),
            mass=mass,
            shear=shear,
        )
    elif lens_light_class == "mge":
        lens = af.Model(
            al.Galaxy,
            redshift=0.5,
            bulge=mge_model(centre_prior_is_uniform=True),
            mass=mass,
            shear=shear,
        )
    else:
        raise ValueError(lens_light_class)

    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def fit_has_linear_lp(*, source_class: str, lens_light_class: str) -> bool:
    """Both lens-light MGE and source MGE use linear intensities."""
    return source_class == "mge" or lens_light_class == "mge"


# --- Source-science quantities ---


def magnitude_from_flux(flux: float) -> float:
    return ZERO_POINT - 2.5 * np.log10(flux)


def source_science_from(tracer: al.Tracer) -> Dict[str, float]:
    grid = al.Grid2D.uniform(
        shape_native=SCIENCE_GRID_SHAPE, pixel_scales=SCIENCE_PIXEL_SCALE
    )
    source_plane_image = tracer.planes[1].image_2d_from(grid=grid)
    source_plane_flux = float(np.sum(source_plane_image))
    traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)
    lensed_source_image = tracer.planes[1].image_2d_from(grid=traced_grid_list[1])
    image_plane_flux = float(np.sum(lensed_source_image))
    return {
        "image_plane_flux": image_plane_flux,
        "source_plane_flux": source_plane_flux,
        "source_magnification": float(image_plane_flux / source_plane_flux),
        "source_magnitude_zp_25": float(magnitude_from_flux(source_plane_flux)),
    }


def with_comparison(
    values: Dict[str, float], truth: Dict[str, float]
) -> Dict[str, float]:
    out = dict(values)
    for k in QUANTITY_KEYS[:-1]:
        out[f"delta_{k}"] = values[k] - truth[k]
        out[f"frac_{k}"] = values[k] / truth[k]
    out["delta_source_magnitude_zp_25"] = (
        values["source_magnitude_zp_25"] - truth["source_magnitude_zp_25"]
    )
    return out


# --- Posterior expansion ---


def _draw_indices_from_pdf(samples, n_draws: int, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    weights = np.asarray(samples.weight_list, dtype=np.float64)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError(f"weights sum is {total}")
    return rng.choice(len(weights), size=n_draws, p=weights / total)


def _solved_tracer(instance, dataset: al.Imaging, has_linear_lp: bool) -> al.Tracer:
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
) -> Dict:
    draws = {k: [] for k in QUANTITY_KEYS}
    n_failed = 0
    indices = _draw_indices_from_pdf(samples=samples, n_draws=n_draws)
    for idx in indices:
        vector = samples.parameter_lists[idx]
        try:
            instance = samples.model.instance_from_vector(
                vector=vector, ignore_assertions=True
            )
            tracer = _solved_tracer(
                instance=instance,
                dataset=dataset,
                has_linear_lp=has_linear_lp,
            )
            values = source_science_from(tracer=tracer)
        except Exception as e:  # noqa: BLE001 - skip bad draws
            n_failed += 1
            print(f"  posterior draw failed: {e}")
            continue
        for k in QUANTITY_KEYS:
            draws[k].append(values[k])

    summary = {"n_draws_requested": n_draws, "n_draws_failed": n_failed}
    for k, lst in draws.items():
        if not lst:
            summary[k] = None
            continue
        arr = np.asarray(lst)
        summary[k] = {
            "median": float(np.median(arr)),
            "lower_1sigma": float(np.percentile(arr, 15.865)),
            "upper_1sigma": float(np.percentile(arr, 84.135)),
            "lower_3sigma": float(np.percentile(arr, 0.135)),
            "upper_3sigma": float(np.percentile(arr, 99.865)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        }
    return summary


def with_pdf_comparison(pdf_summary: Dict, truth: Dict[str, float]) -> Dict:
    out = {
        "n_draws_requested": pdf_summary.get("n_draws_requested"),
        "n_draws_failed": pdf_summary.get("n_draws_failed"),
    }
    for k in QUANTITY_KEYS:
        entry = pdf_summary.get(k)
        if entry is None:
            out[k] = None
            continue
        truth_value = truth[k]
        out[k] = {
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
    return out


# --- The fit driver ---


def _save_subplot(result: af.Result, model_name: str, fits_dir: Path) -> Optional[Path]:
    fits_dir.mkdir(parents=True, exist_ok=True)
    src = Path(result.paths.image_path) / "fit.png"
    if not src.exists():
        return None
    dst = fits_dir / f"{model_name}.png"
    shutil.copyfile(src, dst)
    return dst


def _fmt(v, precision: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "**NO**"
    return f"{v:.{precision}g}"


def write_markdown(summary: Dict, output_path: Path) -> None:
    truth = summary["truth"]
    lines = [
        f"# Source-Science Fit Comparison — {summary['name']}",
        "",
        f"Dataset: `{summary['dataset_path']}` (zero-point = {summary['zero_point_assumption']})",
        f"Posterior draws per fit: {summary['n_posterior_draws']}",
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


def run_fits_and_compare(
    *,
    name: str,
    dataset_path: Path,
    fits_dir: Path,
    truth: Dict[str, float],
    fit_list: Sequence[
        Tuple[str, str, str]
    ],  # (model_name, source_class, lens_light_class)
    path_prefix: Path,
    unique_tag: str,
) -> Dict:
    """Run a list of fits, do posterior expansion, write JSON + Markdown comparison.

    Returns the full summary dictionary.
    """
    dataset = load_dataset(dataset_path)
    truth_full = {
        **truth,
        "source_magnitude_zp_25": magnitude_from_flux(truth["source_plane_flux"]),
    }

    summary = {
        "name": name,
        "dataset_path": str(dataset_path),
        "zero_point_assumption": ZERO_POINT,
        "n_posterior_draws": N_POSTERIOR_DRAWS,
        "truth": truth_full,
        "fits": {},
    }

    for model_name, source_class, lens_light_class in fit_list:
        print(
            f"Running fit: {model_name}  (source={source_class}, lens_light={lens_light_class})"
        )
        model = make_collection(
            source_class=source_class,
            lens_light_class=lens_light_class,
        )
        search = af.Nautilus(
            path_prefix=path_prefix,
            name=model_name,
            unique_tag=unique_tag,
            n_live=75,
            n_batch=25,
            iterations_per_quick_update=2000000,
        )
        analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)
        result = search.fit(model=model, analysis=analysis)
        log_likelihood = float(result.max_log_likelihood_fit.log_likelihood)
        print(f"  log_likelihood={log_likelihood}")

        mle_values = source_science_from(tracer=result.max_log_likelihood_tracer)
        mle_compared = with_comparison(values=mle_values, truth=truth_full)
        has_linear_lp = fit_has_linear_lp(
            source_class=source_class,
            lens_light_class=lens_light_class,
        )
        pdf_summary = posterior_source_science_from(
            samples=result.samples,
            dataset=dataset,
            has_linear_lp=has_linear_lp,
        )
        pdf_compared = with_pdf_comparison(pdf_summary=pdf_summary, truth=truth_full)
        _save_subplot(result=result, model_name=model_name, fits_dir=fits_dir)
        summary["fits"][model_name] = {
            "log_likelihood": log_likelihood,
            "mle": mle_compared,
            "pdf": pdf_compared,
        }

    json_path = dataset_path / "fit_comparison.json"
    md_path = dataset_path / "fit_comparison.md"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
    write_markdown(summary=summary, output_path=md_path)
    print(f"\nWrote {json_path}\nWrote {md_path}")
    return summary
