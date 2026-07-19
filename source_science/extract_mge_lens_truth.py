"""
Extract MGE lens-light truth for a given config.

For each config we need an MGE lens light to serve as truth in tests
that use `lens_light_truth="mge"`. The natural choice is the MLE of an
MGE-lens-light fit against the Sersic-lens-light dataset of the same
config. So:

1. Make sure the Sersic-lens-light dataset exists (`<dataset_path>/data.fits`).
2. Run (or resume) an MGE-lens + Sersic-source fit on it.
3. Take the MLE tracer, run the inversion to populate intensities, take
   the lens galaxy (with its bulge as a solved Basis), save as JSON.

Usage (from `autolens_workspace_developer/`):

    python source_science/extract_mge_lens_truth.py --config config_0 \\
        --dataset-name simple --output-name mge_lens_truth_config_0.json
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEV_ROOT = _HERE.parent
sys.path.insert(0, str(_DEV_ROOT))

from autolens import jax_wrapper  # Sets JAX environment before other imports

import autofit as af
import autolens as al

from source_science.fit_helpers import (
    load_dataset,
    make_collection,
)


def extract(config_name: str, dataset_name: str, output_name: str) -> Path:
    dataset_path = Path("dataset") / "imaging" / dataset_name
    out_path = Path("source_science") / "results" / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    model = make_collection(source_class="sersic", lens_light_class="mge")
    search = af.Nautilus(
        path_prefix=Path("output") / "lens_config_robustness" / config_name / "mge_lens_extractor",
        name="mge_lens__sersic_source",
        unique_tag=f"{config_name}_extractor_v1",
        n_live=75,
        n_batch=25,
        iterations_per_quick_update=2000000,
    )
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)
    result = search.fit(model=model, analysis=analysis)

    fit = al.FitImaging(dataset=dataset, tracer=result.max_log_likelihood_tracer)
    solved_tracer = fit.tracer_linear_light_profiles_to_light_profiles
    lens_galaxy = solved_tracer.galaxies[0]  # lens at z=0.5

    al.output_to_json(obj=lens_galaxy, file_path=out_path)
    print(f"Saved MGE lens light truth to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config name e.g. config_0")
    parser.add_argument(
        "--dataset-name", required=True,
        help="Dataset folder under dataset/imaging/ to use as the source data",
    )
    parser.add_argument(
        "--output-name", required=True,
        help="Output JSON filename relative to source_science/results/",
    )
    args = parser.parse_args()
    extract(
        config_name=args.config,
        dataset_name=args.dataset_name,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
