# Quantity package — archived 2026-05-22

This directory is a frozen copy of the `quantity` fitting module that used to live
inside the `PyAutoGalaxy` and `PyAutoLens` libraries.

## What the quantity package was

`FitQuantity` / `AnalysisQuantity` / `DatasetQuantity` let you fit a galaxy or
lens model **directly to a derived quantity** — e.g. a convergence map, a pair of
deflection-angle maps, or a potential map — instead of to imaging data. The
intended use case was profile-family comparison: given a convergence map produced
by one mass profile, fit it with a different profile family and compare.

## Why it was archived

- Nobody was actually using it. The user maintained it for years, but it never
  picked up real users in any of the workspaces.
- It was blocking the JAX migration: every JAX porting step had to keep
  `FitQuantity` working, which slowed unrelated work.
- It carried test surface area, configs, and re-exports that had to stay green
  in CI for no real-world benefit.

The code is preserved here so it can be restored if the user ever wants to
revisit the feature. **It is not maintained.** Imports and tests will rot over
time as `autogalaxy` and `autolens` evolve.

## Source of this snapshot

Copied from the following `main` SHAs on 2026-05-22:

| Repository                | Commit                                     |
|---------------------------|--------------------------------------------|
| PyAutoGalaxy              | `2547ca175a82f365a64af261923e0ac7232655ac` |
| PyAutoLens                | `a91febcb1aa12797f9d5ece54c1cbbac528cd087` |
| autogalaxy_workspace_test | `314be475298f588c8c0f9af979fa3c4f9e36f0b5` |

The branch `feature/archive-quantity-package` (across all 7 affected repos)
contains the matching removal commits.

## Layout

```
legacy/quantity/
├── README.md                ← you are here
├── autogalaxy/quantity/     ← copy of PyAutoGalaxy/autogalaxy/quantity/
├── autolens/quantity/       ← copy of PyAutoLens/autolens/quantity/
├── tests/
│   ├── autogalaxy/          ← copy of PyAutoGalaxy/test_autogalaxy/quantity/
│   └── autolens/            ← copy of PyAutoLens/test_autolens/quantity/
├── scripts/                 ← copy of autogalaxy_workspace_test/scripts/quantity/
└── config_snippets/         ← the YAML stanzas removed from library + workspace configs
    └── plots_yaml_excerpt.md
```

## How to restore

If you ever want quantity back in active development:

1. **Source files**
   ```
   cp -r legacy/quantity/autogalaxy/quantity  PyAutoGalaxy/autogalaxy/
   cp -r legacy/quantity/autolens/quantity    PyAutoLens/autolens/
   cp -r legacy/quantity/tests/autogalaxy/*   PyAutoGalaxy/test_autogalaxy/quantity/
   cp -r legacy/quantity/tests/autolens/*     PyAutoLens/test_autolens/quantity/
   ```
2. **Re-exports** — re-add to `PyAutoGalaxy/autogalaxy/__init__.py`:
   ```python
   from .quantity.fit_quantity import FitQuantity
   from .quantity.model.analysis import AnalysisQuantity
   from .quantity.dataset_quantity import DatasetQuantity
   ```
   and to `PyAutoGalaxy/autogalaxy/plot/__init__.py`:
   ```python
   from autogalaxy.quantity.plot.fit_quantity_plots import (
       subplot_fit as subplot_fit_quantity,
   )
   ```
   and to `PyAutoLens/autolens/__init__.py`:
   ```python
   from autogalaxy.quantity.dataset_quantity import DatasetQuantity
   from .quantity.fit_quantity import FitQuantity
   from .quantity.model.analysis import AnalysisQuantity
   ```
   and re-add `subplot_fit_quantity` to the `from autogalaxy.plot import (...)`
   block in `PyAutoLens/autolens/plot/__init__.py`.
3. **Configs** — re-add the `fit_quantity:` stanzas listed in
   `config_snippets/plots_yaml_excerpt.md` to each YAML file noted there.
4. **Workspace examples** — copy `scripts/*.py` back into
   `autogalaxy_workspace_test/scripts/quantity/`.
5. **Verify**: run `pytest test_autogalaxy/quantity/` and
   `pytest test_autolens/quantity/`. These tests are frozen against the
   2026-05-22 library state, so non-trivial repairs may be needed if the
   surrounding APIs (especially `FitImaging` and `Visualizer`) have moved.

## What lives in the active libraries instead

After archival, the active libraries expose:

- `FitImaging`, `FitInterferometer` — fit a model to a real dataset (the
  workhorse path).
- `FitEllipse` — fit ellipses to isophotes (`autogalaxy` only).
- `FitWeak`, `FitPointDataset` — `autolens`-specific specialised fits.

No equivalent of `FitQuantity` (fit-to-derived-map) exists. If you need that
behaviour, reach for the archive above.
