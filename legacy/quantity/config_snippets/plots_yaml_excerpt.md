# Removed `fit_quantity:` YAML stanzas

These are the exact YAML fragments that were deleted from each `plots.yaml` /
`visualize.yaml` file when the quantity package was archived on 2026-05-22.
To restore, paste each fragment back into the matching file at the same
location (typically just before the `galaxies:` or `fit_ellipse:` block).

## PyAutoGalaxy

### `PyAutoGalaxy/autogalaxy/config/visualize/plots.yaml`
```yaml
fit_quantity: {}                           # Settings for plots of fit quantities (e.g. FitQuantity).
```

### `PyAutoGalaxy/test_autogalaxy/config/visualize.yaml`
```yaml
  fit_quantity:
    subplot_fit: false
```
(inside the `plots:` block)

## PyAutoLens

### `PyAutoLens/autolens/config/visualize/plots.yaml`
```yaml
fit_quantity:                              # Settings for plots of fit quantities (e.g. FitQuantityPlotter).
  subplot_fit: true
```

### `PyAutoLens/test_autolens/config/visualize.yaml`
```yaml
  fit_quantity:
    subplot_fit: true
```
(inside the `plots:` block)

## Workspaces

### `autogalaxy_workspace/config/visualize/plots.yaml`
```yaml
fit_quantity: {}                           # Settings for plots of fit quantities (e.g. FitQuantityPlotter).
```

### `autolens_workspace/config/visualize/plots.yaml`
```yaml
fit_quantity:                              # Settings for plots of fit quantities (e.g. FitQuantityPlotter).
  subplot_fit: true
```

### `autolens_workspace_test/config/visualize/plots.yaml`
```yaml
fit_quantity:                              # Settings for plots of fit quantities (e.g. FitQuantity).
  subplot_fit: true
```

### `autolens_workspace_test/scripts/aggregator/config/visualize.yaml`
```yaml
  fit_quantity:
    subplot_fit: true
```
(inside the `plots:` block)

### `autolens_workspace_test/scripts/imaging/config/visualize/plots.yaml`
```yaml
fit_quantity:
  subplot_fit: false
```

### `autolens_workspace_test/scripts/imaging/config_source/visualize/plots.yaml`
```yaml
fit_quantity:
  subplot_fit: true
```

### `autolens_workspace_test/scripts/interferometer/config/visualize/plots.yaml`
```yaml
fit_quantity:
  subplot_fit: false
```

### `autolens_workspace_test/scripts/interferometer/config_source/visualize/plots.yaml`
```yaml
fit_quantity:
  subplot_fit: true
```
