"""
Profiling: Lens Aggregator Loading Pathways
===========================================

Times the lens-level aggregator loading pathways over mock result sets — answering
what `TracerAgg` / `FitImagingAgg` object reconstruction adds on top of the generic
loading costs measured by the autofit harness.

Stages timed per grid cell:

- `generate`: building the mock lens results (context, not an aggregator cost).
- `from_directory`: the `af` directory Aggregator scan.
- `values_samples_summary`: the lightweight summary path.
- `tracer_max_lh`: `TracerAgg.max_log_likelihood_gen_from` consumed — tracer
  reconstruction per result.
- `fit_max_lh`: `FitImagingAgg.max_log_likelihood_gen_from` consumed — dataset load +
  tracer + likelihood re-evaluation per result (the csv/fits/png workflow cost).
- `aggregate_csv`: an `af.AggregateCSV` catalogue over lens model paths.

Run from the `autolens_workspace_developer` root:

    python aggregator_profiling/profile_lens_aggregator.py --quick

Results print as a table and are written as JSON under
`output/profiling_aggregator/results/`.
"""

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import autofit as af
import autolens as al
from autofit.aggregator.aggregator import Aggregator
from mock_lens_results import generate_mock_lens_results

RESULTS_PATH = Path("output") / "profiling_aggregator" / "results"

BASE = {"n_results": 50, "n_samples": 500}
AXES = {
    "n_results": [10, 50, 200],
    "n_samples": [100, 500, 2000],
}
AXES_QUICK = {
    "n_results": [10, 50],
    "n_samples": [100, 500],
}


def grid_cells(axes: dict, base: dict) -> list:
    cells = []
    for axis, values in axes.items():
        for value in values:
            cell = dict(base)
            cell[axis] = value
            if cell not in cells:
                cells.append(cell)
    return cells


def timed(func) -> float:
    with contextlib.redirect_stdout(io.StringIO()):
        start = time.perf_counter()
        func()
        return time.perf_counter() - start


def fresh_agg(results_root: Path, n_results: int) -> Aggregator:
    """
    A new Aggregator per stage, so each stage is timed cold (SearchOutput caches
    model/samples/summaries on the instance).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        agg = Aggregator.from_directory(results_root)
    assert len(agg) == n_results, f"found {len(agg)}, expected {n_results}"
    return agg


def profile_cell(cell: dict, keep: bool) -> dict:
    import shutil

    timings = {}

    generate_start = time.perf_counter()
    results_root = generate_mock_lens_results(**cell)
    timings["generate"] = time.perf_counter() - generate_start

    # Warm up one-off import/deserialization costs so they don't pollute stage 1.
    warm = fresh_agg(results_root, cell["n_results"])[0]
    _ = warm.samples_summary

    timings["from_directory"] = timed(lambda: Aggregator.from_directory(results_root))

    agg = fresh_agg(results_root, cell["n_results"])
    timings["values_samples_summary"] = timed(
        lambda: list(agg.values("samples_summary"))
    )

    agg = fresh_agg(results_root, cell["n_results"])
    tracer_agg = al.agg.TracerAgg(aggregator=agg)
    timings["tracer_max_lh"] = timed(
        lambda: deque(tracer_agg.max_log_likelihood_gen_from(), maxlen=0)
    )

    agg = fresh_agg(results_root, cell["n_results"])
    fit_agg = al.agg.FitImagingAgg(aggregator=agg)
    timings["fit_max_lh"] = timed(
        lambda: deque(fit_agg.max_log_likelihood_gen_from(), maxlen=0)
    )

    agg = fresh_agg(results_root, cell["n_results"])

    def aggregate_csv():
        agg_csv = af.AggregateCSV(agg)
        agg_csv.add_variable("galaxies.lens.light.effective_radius")
        agg_csv.add_variable("galaxies.source.light.effective_radius")
        agg_csv.save(results_root.parent / "aggregate.csv")

    timings["aggregate_csv"] = timed(aggregate_csv)

    if not keep:
        shutil.rmtree(results_root.parent)

    return {**cell, "timings": timings}


def print_table(rows: list):
    stages = list(rows[0]["timings"])
    header = f"{'n_results':>9} {'n_samples':>9} " + " ".join(
        f"{stage:>24}" for stage in stages
    )
    print("\nTimings in seconds (per-result milliseconds in brackets):\n")
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for stage in stages:
            seconds = row["timings"][stage]
            per_result_ms = 1000.0 * seconds / row["n_results"]
            cells.append(f"{seconds:>14.3f} ({per_result_ms:>7.2f})")
        print(f"{row['n_results']:>9} {row['n_samples']:>9} " + " ".join(cells))


if __name__ == "__main__":
    logging.disable(logging.WARNING)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    axes = dict(AXES_QUICK if args.quick else AXES)
    base = dict(BASE)
    if os.environ.get("PYAUTO_TEST_MODE"):
        # Validation sweeps only smoke-check the harness — one tiny cell.
        axes = {"n_results": [4]}
        base = {"n_results": 4, "n_samples": 20}

    rows = []
    for cell in grid_cells(axes=axes, base=base):
        print(f"profiling {cell} ...", flush=True)
        rows.append(profile_cell(cell, keep=args.keep))

    print_table(rows)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    label = f"_{args.label}" if args.label else ""
    output_file = (
        RESULTS_PATH
        / f"profile_lens_{datetime.now().strftime('%Y%m%d_%H%M%S')}{label}.json"
    )
    output_file.write_text(
        json.dumps(
            {
                "autofit_version": af.__version__,
                "autolens_version": al.__version__,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nJSON written to {output_file}")
