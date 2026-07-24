# Lens aggregator profiling

Lens-level leg of the aggregator profiling harness (the generic leg lives in
`autofit_workspace_test/scripts/profiling/aggregator/`): measures what
`TracerAgg`/`FitImagingAgg` object reconstruction adds on top of generic result
loading, using mock lens output folders written through `al.m.MockSearch` +
`al.fixtures` — no sampler or real model-fit ever runs.

- `mock_lens_results.py` — generate a mock lens result set (`--n-results/--n-samples`).
- `profile_lens_aggregator.py` — time `from_directory`, summaries, `TracerAgg`/
  `FitImagingAgg` max-likelihood generators and an `AggregateCSV` catalogue over a
  one-axis grid (`--quick` for a fast pass, `--label` to tag the JSON); table + JSON
  under `output/profiling_aggregator/results/`.

Run both from the `autolens_workspace_developer` root. Everything lands under `output/`
(gitignored). Profiling tools, not smoke tests — do not add to `smoke_tests.txt`.
