"""
Visualization JAX Pilot: Quantity Analysis (autogalaxy)
=======================================================

Tests that ``VisualizerQuantity.visualize`` with ``use_jax_for_visualization=True``
dispatches through the JIT-cached ``fit_for_visualization`` path wired in
PyAutoGalaxy #404.

Dataset is synthesized inline from the convergence of a known ``IsothermalSph``
mass profile so that the model can recover a meaningful signal.  No pre-canned
FITS fixture is required.

Scope
-----
- Parametric single-galaxy model with ``IsothermalSph`` mass profile.
- Calls ``VisualizerQuantity.visualize`` only (not ``visualize_before_fit``).
- Synthesizes the dataset inline — no external FITS files.
- ``use_jax_for_visualization=True`` exercises the JIT path added in PR #404.
"""

import shutil
from os import path
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import autofit as af
import autogalaxy as ag
from autofit.jax.pytrees import enable_pytrees, register_model
from autogalaxy.quantity.dataset_quantity import DatasetQuantity
from autogalaxy.quantity.model.visualizer import VisualizerQuantity

enable_pytrees()


"""
__Dataset__

Synthesize a convergence map from a known IsothermalSph mass profile.
The mask covers a 30x30 field at 0.2 arcsec/pixel; the convergence of
einstein_radius=1.5 gives a signal strong enough for a meaningful fit.
"""
mask = ag.Mask2D.circular(
    shape_native=(30, 30),
    pixel_scales=0.2,
    radius=2.5,
)

grid = ag.Grid2D.from_mask(mask=mask)

target_convergence = ag.mp.IsothermalSph(
    centre=(0.0, 0.0), einstein_radius=1.5
).convergence_2d_from(grid=grid)

data = target_convergence
noise_map = ag.Array2D(
    values=np.ones(target_convergence.shape_native) * 0.05,
    mask=mask,
)

dataset = DatasetQuantity(data=data, noise_map=noise_map)


"""
__Model__

Single galaxy with an IsothermalSph mass profile.  Tight priors so the
model can recover the known einstein_radius=1.5 quickly.
"""
mass = af.Model(ag.mp.IsothermalSph)
mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.einstein_radius = af.UniformPrior(lower_limit=1.3, upper_limit=1.7)

galaxy = af.Model(ag.Galaxy, redshift=0.5, mass=mass)
model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

register_model(model)


"""
__Analysis__

``use_jax=True`` turns on the JAX ``_xp`` path; ``use_jax_for_visualization=True``
tells the search-level visualization path to wrap ``fit_from`` in ``jax.jit``
via the ``Analysis.fit_for_visualization`` helper (wired in PyAutoGalaxy #404).
"""
analysis = ag.AnalysisQuantity(
    dataset=dataset,
    func_str="convergence_2d_from",
    use_jax=True,
    use_jax_for_visualization=True,
    title_prefix="JAX_PILOT",
)


"""
__Paths__
"""
image_path = Path("scripts") / "quantity" / "images" / "visualization_jax"
if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)
output_path = image_path / "output"
output_path.mkdir(parents=True)
paths = SimpleNamespace(image_path=image_path, output_path=output_path)


"""
__Run visualize on the eager-JAX fit__
"""
instance = model.instance_from_prior_medians()

print("Running VisualizerQuantity.visualize with use_jax_for_visualization=True ...")
VisualizerQuantity.visualize(
    analysis=analysis,
    paths=paths,
    instance=instance,
    during_analysis=False,
)

assert (image_path / "fit.png").exists(), "fit.png was not produced"
print("PILOT SUCCEEDED — JAX-backed quantity visualization produced fit.png.")


"""
__Visualization Sanity__

Phase D.2.a rollout — autogalaxy quantity variant (no Tracer, no
lensing). `FitQuantity.model_data` is the model field on the dataset
grid; the analogous failure mode is "JAX trace through the quantity
helpers loses tracer values → model field collapses to zero/NaN".
"""
import numpy as _sanity_np

_fit_for_vis = analysis.fit_from(instance=instance)
_model_field = _sanity_np.asarray(_fit_for_vis.model_data)
assert _sanity_np.isfinite(_model_field).all(), (
    "fit.model_data has nan/inf — JAX-trace mismatch on quantity helpers"
)
assert float(_sanity_np.abs(_model_field).sum()) > 0.0, (
    "fit.model_data all-zero — quantity model field collapsed"
)
_fom = float(_fit_for_vis.figure_of_merit)
assert _sanity_np.isfinite(_fom), (
    f"figure_of_merit = {_fom} — chi² nan/inf, fit collapsed"
)
print(
    f"  PASS Visualization Sanity (autogalaxy quantity): "
    f"|model_data|.sum() = {float(_sanity_np.abs(_model_field).sum()):.4f}, "
    f"figure_of_merit = {_fom:.4f}"
)
