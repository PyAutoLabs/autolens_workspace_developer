"""
Visualization: Quantity Analysis (autogalaxy)
=============================================

Tests that ``VisualizerQuantity.visualize`` outputs the expected ``fit.png``
to disk on the NumPy (non-JAX) code path.

Dataset is synthesized inline from the convergence of a known ``IsothermalSph``
mass profile so that the model can recover a meaningful signal.  No pre-canned
FITS fixture is required.
"""

import shutil
from os import path
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import autofit as af
import autogalaxy as ag
from autogalaxy.quantity.dataset_quantity import DatasetQuantity
from autogalaxy.quantity.model.visualizer import VisualizerQuantity


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


"""
__Analysis__

Explicit NumPy path (use_jax=False).  This exercises the ``VisualizerQuantity``
call chain without JIT compilation, confirming the baseline doesn't regress.
"""
analysis = ag.AnalysisQuantity(
    dataset=dataset,
    func_str="convergence_2d_from",
    use_jax=False,
)


"""
__Paths__
"""
image_path = Path("scripts") / "quantity" / "images" / "visualization"
if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)
output_path = image_path / "output"
output_path.mkdir(parents=True)
paths = SimpleNamespace(image_path=image_path, output_path=output_path)


"""
__Visualize__
"""
instance = model.instance_from_prior_medians()

print("Running VisualizerQuantity.visualize (NumPy path) ...")
VisualizerQuantity.visualize(
    analysis=analysis,
    paths=paths,
    instance=instance,
    during_analysis=False,
)

assert (image_path / "fit.png").exists(), "fit.png was not produced"
print("NumPy quantity visualization produced fit.png.")
