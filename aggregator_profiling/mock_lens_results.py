"""
Profiling: Mock Lens Aggregator Results
=======================================

Fabricates lens search-output directories that the real `af.Aggregator` and the
`al.agg` wrappers (`TracerAgg`, `FitImagingAgg`, ...) accept, without running a
non-linear search.

One template result is written by a real search `fit` under the test-mode sampler
bypass (`PYAUTO_TEST_MODE=2` + `PYAUTO_TEST_MODE_SAMPLES=N`, PyAutoFit#1381), so
every file — including a `samples.csv` whose row count and byte size are
representative of a production run — is produced by the canonical library write
path. The template is then stamped `n_results` times with `shutil.copytree`,
giving each copy a unique `dataset_name` metadata entry and `unique_tag` in its
`search.json` — mirroring the autofit harness
(`autofit_workspace_test/scripts/profiling/aggregator/mock_results.py`).

Run from the `autolens_workspace_developer` root, e.g.:

    python aggregator_profiling/mock_lens_results.py --n-results 50 --n-samples 500
"""

import argparse
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

from autolens import conf
import autofit as af
import autolens as al
from autolens import fixtures

DEFAULT_ROOT = Path("output") / "profiling_aggregator" / "mock"


def model_from() -> af.Collection:
    """
    A lens + source model matching the fixture analysis (the shape used across this
    repo's aggregator integration scripts).
    """
    dataset_model = af.Model(al.DatasetModel)
    dataset_model.background_sky_level = af.UniformPrior(
        lower_limit=0.5, upper_limit=1.5
    )
    return af.Collection(
        dataset_model=dataset_model,
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, light=al.lp.Sersic),
            source=af.Model(al.Galaxy, redshift=1.0, light=al.lp.Sersic),
        ),
    )


@contextmanager
def test_mode_bypass(n_samples: int):
    """
    Run a real search `fit` as an instant sampler bypass writing `n_samples`
    size-realistic samples (PYAUTO_TEST_MODE_SAMPLES, minimum 4).
    """
    previous = {
        key: os.environ.get(key)
        for key in ("PYAUTO_TEST_MODE", "PYAUTO_TEST_MODE_SAMPLES")
    }
    os.environ["PYAUTO_TEST_MODE"] = "2"
    os.environ["PYAUTO_TEST_MODE_SAMPLES"] = str(max(n_samples, 4))
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def write_template(root: Path, n_samples: int) -> Path:
    """
    Write one full lens search-output directory via a test-mode bypass fit and
    return the leaf directory (the one containing the `metadata` file).
    """
    conf.instance.push(new_path="config", output_path=str(root))

    # This workspace's output.yaml disables the samples table; the harness needs it
    # (set after the push, which would clobber a decorator-level override).
    conf.instance["general"]["output"]["samples_to_csv"] = True
    conf.instance["output"]["samples"] = True

    model = model_from()
    analysis = fixtures.make_analysis_imaging_7x7()

    # A stale template would append duplicate metadata lines (the file is opened in
    # append mode), so remove any previous template for this config first.
    for candidate in (
        root / "test_mode" / "_template" / f"s{n_samples}",
        root / "_template" / f"s{n_samples}",
    ):
        if candidate.exists():
            shutil.rmtree(candidate)

    with test_mode_bypass(n_samples=n_samples):
        search = af.DynestyStatic(
            name="fit", path_prefix=f"_template/s{n_samples}", number_of_cores=1
        )
        search.fit(model=model, analysis=analysis)
        analysis.modify_before_fit(paths=search.paths, model=model)
        analysis.visualize_before_fit(paths=search.paths, model=model)

        return Path(search.paths.output_path)


def stamp_results(template_leaf: Path, results_root: Path, n_results: int):
    """
    Copy the template `n_results` times, each copy distinguishable by metadata
    `dataset_name` and by `unique_tag` in its search.json (the identifier input).
    """
    search_json_path = template_leaf / "files" / "search.json"
    search_dict = json.loads(search_json_path.read_text())

    for i in range(n_results):
        dataset_name = f"dataset_{i:04d}"
        destination = results_root / dataset_name / "fit"
        shutil.copytree(template_leaf, destination)
        with open(destination / "metadata", "a") as f:
            f.write(f"\ndataset_name={dataset_name}")
        search_dict["arguments"]["unique_tag"] = dataset_name
        (destination / "files" / "search.json").write_text(json.dumps(search_dict))


def generate_mock_lens_results(
    root: Path = DEFAULT_ROOT,
    n_results: int = 50,
    n_samples: int = 500,
    fresh: bool = False,
) -> Path:
    """
    Generate a directory of `n_results` mock lens results and return it, ready for
    `Aggregator.from_directory` and the `al.agg` wrappers. Skips regeneration when
    an identical set already exists (recorded in a manifest file).
    """
    root = Path(root)
    config = {"n_results": n_results, "n_samples": n_samples}
    tag = f"r{n_results}_s{n_samples}"
    results_root = root / tag
    manifest_path = root / f"{tag}.manifest.json"

    if not fresh and manifest_path.exists() and results_root.exists():
        if json.loads(manifest_path.read_text()) == config:
            return results_root

    if results_root.exists():
        shutil.rmtree(results_root)

    template_leaf = write_template(root=root, n_samples=n_samples)
    stamp_results(
        template_leaf=template_leaf, results_root=results_root, n_results=n_results
    )

    manifest_path.write_text(json.dumps(config))
    return results_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--n-results", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    results_root = generate_mock_lens_results(
        root=args.root,
        n_results=args.n_results,
        n_samples=args.n_samples,
        fresh=args.fresh,
    )
    print(
        f"{args.n_results} mock lens results at {results_root} "
        f"in {time.perf_counter() - start:.2f}s"
    )
