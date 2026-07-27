"""
Master orchestrator: run every (config × truth combination) test.

Each "test" is a (config_name, source_truth, lens_light_truth) tuple. The
test list covers:

  - Sersic source × {no lens light, Sersic lens light, MGE lens light}
  - MGE source × {no lens light, Sersic lens light, MGE lens light}

For each of the 6 truth combinations, at each lens config, this script:

  1. Ensures the MGE source truth JSON exists (run extract_mge_truth.py
     if not — re-uses existing `source_science/results/mge_truth_source.json`).
  2. Ensures the MGE lens light truth JSON for the config exists (runs a
     one-time MGE-lens fit to extract it if not).
  3. Simulates the dataset if not already on disk.
  4. Runs the fits for that test, with posterior expansion.
  5. Writes `fit_comparison.{json,md}` and per-fit subplots.

Idempotent at fit-cache level: re-running just confirms `Fit Already
Completed` for cached fits. Re-runs of simulation are skipped if the data
file is present.

Config 0 tests 1-4 are deliberately skipped here (they shipped in PRs
#73/#74/#75 with their own folder layout). Config 0 tests 5+6 are
included.

Usage (from `autolens_workspace_developer/`):

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python source_science/run_all_tests.py
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python source_science/run_all_tests.py --only config_1
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_DEV_ROOT = _HERE.parent
sys.path.insert(0, str(_DEV_ROOT))

from autolens import jax_wrapper  # Sets JAX environment before other imports

import autofit as af
import autolens as al

from source_science.lens_configs import CONFIGS, get_config
from source_science.sim_helpers import simulate_and_save
from source_science.fit_helpers import (
    fit_has_linear_lp,
    load_dataset,
    make_collection,
    run_fits_and_compare,
)


SOURCE_TRUTH_JSON = Path("source_science") / "results" / "mge_truth_source.json"


# Fit list per lens_light_truth condition.
# Tuples: (model_name, source_class_for_fit, lens_light_class_for_fit)
FIT_LIST_NO_LENS_LIGHT = [
    ("sersic_source", "sersic", "none"),
    ("mge_source", "mge", "none"),
]
FIT_LIST_WITH_LENS_LIGHT = [
    ("sersic__sersic", "sersic", "sersic"),
    ("mge_lens__sersic_source", "sersic", "mge"),
    ("mge_lens__mge_source", "mge", "mge"),
]


def fit_list_for(lens_light_truth: str):
    if lens_light_truth == "none":
        return FIT_LIST_NO_LENS_LIGHT
    return FIT_LIST_WITH_LENS_LIGHT


# Test definitions.
# Each tuple: (test_short_name, source_truth, lens_light_truth)
TESTS_PER_CONFIG = [
    ("1_sersic_truth_sersic_lens", "sersic", "sersic"),
    ("2_sersic_truth_no_lens", "sersic", "none"),
    ("3_mge_truth_sersic_lens", "mge", "sersic"),
    ("4_mge_truth_no_lens", "mge", "none"),
    ("5_sersic_truth_mge_lens", "sersic", "mge"),
    ("6_mge_truth_mge_lens", "mge", "mge"),
]


def dataset_path_for(config_name: str, test_name: str) -> Path:
    return Path("dataset") / "imaging" / f"{config_name}_{test_name}"


def mge_lens_truth_path_for(config_name: str) -> Path:
    return Path("source_science") / "results" / f"mge_lens_truth_{config_name}.json"


def path_prefix_for(config_name: str, test_name: str) -> Path:
    return Path("output") / "lens_config_robustness" / config_name / test_name


def ensure_mge_source_truth_exists() -> Path:
    if not SOURCE_TRUTH_JSON.exists():
        raise FileNotFoundError(
            f"{SOURCE_TRUTH_JSON} missing — run source_science/extract_mge_truth.py"
            " first."
        )
    return SOURCE_TRUTH_JSON


def ensure_mge_lens_truth_exists(config_name: str) -> Path:
    """For tests using MGE lens light truth, we need an MGE lens galaxy.

    We extract it from the same config's Sersic-truth+Sersic-lens dataset
    by running a one-time MGE-lens fit. The dataset for this config's
    Sersic-truth+Sersic-lens must already have been simulated.
    """
    target = mge_lens_truth_path_for(config_name)
    if target.exists():
        return target

    src_dataset_path = dataset_path_for(config_name, "1_sersic_truth_sersic_lens")
    if not (src_dataset_path / "data.fits").exists():
        raise FileNotFoundError(
            f"Cannot extract MGE lens light for {config_name}: "
            f"{src_dataset_path/'data.fits'} missing."
            " Simulate the sersic-lens dataset first."
        )

    print(f"\n=== Extracting MGE lens light truth for {config_name} ===")
    dataset = load_dataset(src_dataset_path)
    model = make_collection(source_class="sersic", lens_light_class="mge")
    search = af.Nautilus(
        path_prefix=Path("output")
        / "lens_config_robustness"
        / config_name
        / "mge_lens_extractor",
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
    lens_galaxy = solved_tracer.galaxies[0]
    al.output_to_json(obj=lens_galaxy, file_path=target)
    print(f"Saved MGE lens light truth to {target}")
    return target


def run_one_test(
    *,
    config_name: str,
    test_name: str,
    source_truth: str,
    lens_light_truth: str,
    skip_existing_results: bool = False,
):
    print(f"\n========================================")
    print(f"  TEST: {config_name} / {test_name}")
    print(f"  source_truth={source_truth}, lens_light_truth={lens_light_truth}")
    print(f"========================================\n")

    config = get_config(config_name)
    dataset_path = dataset_path_for(config_name, test_name)

    if skip_existing_results and (dataset_path / "fit_comparison.json").exists():
        print(f"  [skip] fit_comparison.json already present at {dataset_path}")
        return

    # Truth sources
    mge_source_truth_path = None
    mge_lens_truth_path = None
    if source_truth == "mge":
        mge_source_truth_path = ensure_mge_source_truth_exists()
    if lens_light_truth == "mge":
        mge_lens_truth_path = ensure_mge_lens_truth_exists(config_name)

    # Simulator
    if (dataset_path / "data.fits").exists():
        print(f"  [skip simulator] dataset already at {dataset_path}")
        truth = al.from_json(file_path=dataset_path / "source_science.json")
    else:
        print(f"  Simulating to {dataset_path} ...")
        truth = simulate_and_save(
            config=config,
            source_truth=source_truth,
            lens_light_truth=lens_light_truth,
            dataset_path=dataset_path,
            mge_source_truth_path=mge_source_truth_path,
            mge_lens_truth_path=mge_lens_truth_path,
        )
        print(f"  Truth: {truth}")

    # Fits
    fits_dir = dataset_path / "fits"
    fit_list = fit_list_for(lens_light_truth)
    print(f"  Running {len(fit_list)} fits ...")
    run_fits_and_compare(
        name=f"{config_name} / {test_name}",
        dataset_path=dataset_path,
        fits_dir=fits_dir,
        truth=truth,
        fit_list=fit_list,
        path_prefix=path_prefix_for(config_name, test_name),
        unique_tag=f"{config_name}_{test_name}_v1",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        default=None,
        help="If set, only run tests for this config (e.g. config_1).",
    )
    parser.add_argument(
        "--skip",
        default=None,
        help="Comma-separated list of test_names to skip globally (e.g. 1_sersic_truth_sersic_lens,2_sersic_truth_no_lens).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tests whose fit_comparison.json already exists.",
    )
    args = parser.parse_args()

    skip_names = set((args.skip or "").split(",")) - {""}

    configs_to_run = [args.only] if args.only else list(CONFIGS.keys())

    for config_name in configs_to_run:
        for test_name, source_truth, lens_light_truth in TESTS_PER_CONFIG:
            if test_name in skip_names:
                continue
            run_one_test(
                config_name=config_name,
                test_name=test_name,
                source_truth=source_truth,
                lens_light_truth=lens_light_truth,
                skip_existing_results=args.skip_existing,
            )


if __name__ == "__main__":
    main()
