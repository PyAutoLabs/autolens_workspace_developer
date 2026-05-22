"""
End-to-end test: jit-cached visualization for AnalysisQuantity.
===============================================================

Single-galaxy autogalaxy quantity port of the autolens
``scripts/imaging/modeling_visualization_jit.py`` end-to-end test —
**reduced to Part 1 + Sanity only**.

The imaging variant runs a Part-2 live Nautilus quick-update to confirm
``fit.png`` lands on disk during a real search. The quantity variant
cannot run Part 2 today because ``AnalysisQuantity + use_jax=True``
hits a pre-existing library limitation under ``jax.vmap`` (Nautilus's
fitness path): at ``autogalaxy/profiles/geometry_profiles.py:168``,
``xp.array(self.centre)`` calls ``__array__()`` on a tuple of traced
scalars and raises ``jax.errors.TracerArrayConversionError``. The fix
would replace the call with ``jnp.stack([self.centre[0], self.centre[1]])``
(or equivalent xp-safe construction) — but that's a library change,
out of scope for this workspace_test script. See follow-up prompt
``autogalaxy/geometry_profiles_centre_jax_traceable.md`` for the fix.

The sibling ``visualization_jax.py`` script doesn't hit this because
it calls ``analysis.fit_from(instance=...)`` once (no vmap). This
script's Part 1 (caching probe) and Sanity block use the same
single-shot path, so they exercise everything that ``fit_for_visualization``
needs without triggering the vmap-only library bug.

When the library limitation is fixed, restore Part 2 here by mirroring
the imaging variant's Nautilus quick-update block.

This script deliberately opts in with
``AnalysisQuantity(use_jax=True, use_jax_for_visualization=True)``.
Default model-fit scripts elsewhere in the workspace leave both flags at
``False`` and are therefore untouched by this change.
"""

import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import autofit as af
import autogalaxy as ag
from autofit.jax.pytrees import enable_pytrees, register_model
from autogalaxy.quantity.dataset_quantity import DatasetQuantity

enable_pytrees()


"""
__Dataset__

Synthesize a convergence map from a known ``IsothermalSph`` mass profile
(matching the sibling ``visualization_jax.py`` setup). 30x30 field at
0.2 arcsec/pixel; einstein_radius=1.5 gives a signal strong enough for
a meaningful fit.
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
============================================================================
Part 1 — Caching probe
============================================================================

Model: parametric ``IsothermalSph`` mass profile with tight priors so the
prior median sits near the truth (einstein_radius=1.5).
"""
print("\n" + "=" * 72)
print("Part 1: Quantity caching probe")
print("=" * 72)

mass = af.Model(ag.mp.IsothermalSph)
mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.einstein_radius = af.UniformPrior(lower_limit=1.3, upper_limit=1.7)

galaxy_mge = af.Model(ag.Galaxy, redshift=0.5, mass=mass)

model_mge = af.Collection(galaxies=af.Collection(galaxy=galaxy_mge))

register_model(model_mge)

analysis_mge = ag.AnalysisQuantity(
    dataset=dataset,
    func_str="convergence_2d_from",
    use_jax=True,
    use_jax_for_visualization=True,
)

instance_mge = model_mge.instance_from_prior_medians()

t0 = time.perf_counter()
fit_1 = analysis_mge.fit_for_visualization(instance_mge)
jax.block_until_ready(fit_1.log_likelihood)
t1 = time.perf_counter()
compile_time = t1 - t0
print(f"First call (compile + run): {compile_time:.3f}s")
print(f"  log_likelihood leaf type: {type(fit_1.log_likelihood).__name__}")
assert isinstance(
    fit_1.log_likelihood, jnp.ndarray
), f"expected jax.Array, got {type(fit_1.log_likelihood)}"

t0 = time.perf_counter()
fit_2 = analysis_mge.fit_for_visualization(instance_mge)
jax.block_until_ready(fit_2.log_likelihood)
t1 = time.perf_counter()
cached_time = t1 - t0
print(f"Second call (cached):       {cached_time:.3f}s")
print(f"Speedup:                    {compile_time / max(cached_time, 1e-9):.1f}x")

assert cached_time < compile_time * 0.5, (
    f"Cached call ({cached_time:.3f}s) not faster than compile "
    f"({compile_time:.3f}s) — JIT cache is not being hit."
)
assert (
    analysis_mge._jitted_fit_from is not None
), "expected _jitted_fit_from to be cached on the analysis instance after first call"
print("PASS: Quantity jit-cached fit_for_visualization works and is reused.")


"""
__Visualization Sanity__

Phase D.2.b.i — autogalaxy quantity variant (no Tracer / no lensing
latents). `FitQuantity.model_data` is the model field on the dataset
grid; the analogous failure mode is "JAX trace through the quantity
helpers loses tracer values → model field collapses to zero/NaN". The
asserts run on the cached `fit_2` from Part 1 so the warm JIT path is
exercised (the first-call compile is already paid above).
"""

_model_field = np.asarray(fit_2.model_data)
assert np.isfinite(_model_field).all(), (
    "fit.model_data has nan/inf — JAX-trace mismatch on quantity helpers"
)
assert float(np.abs(_model_field).sum()) > 0.0, (
    "fit.model_data all-zero — quantity model field collapsed"
)
_fom = float(fit_2.figure_of_merit)
assert np.isfinite(_fom), (
    f"figure_of_merit = {_fom} — chi² nan/inf, fit collapsed"
)
print(
    f"  PASS Visualization Sanity (autogalaxy quantity): "
    f"|model_data|.sum() = {float(np.abs(_model_field).sum()):.4f}, "
    f"figure_of_merit = {_fom:.4f}"
)


# Part 2 (live Nautilus quick-update) is intentionally absent — see the
# module docstring for the upstream library limitation that blocks it.
# When the geometry_profiles.py:168 fix lands, restore the Part 2 block
# by mirroring the imaging variant's structure.
print(
    "\nPASS: Quantity Part 1 + Sanity exercised the jit-cached "
    "fit_for_visualization path. Part 2 (Nautilus) skipped pending "
    "library fix for AnalysisQuantity + use_jax=True vmap."
)
