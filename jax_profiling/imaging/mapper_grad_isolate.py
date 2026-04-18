"""
Isolate which intermediate in the rectangular interpolator has the exploding
gradient. We re-run the same probe but stop the forward pass at successive
intermediates and take sum-of-squares to get a scalar, then compare JAX grad
vs finite-difference grad.

Interpretation:
- If JAX/FD ratio is O(1) at stage X but blows up at stage X+1, the bug is
  in the step between them.
- ``sum(M**2)`` earlier showed JAX=-2.3e25, FD=1643 for the full mapping
  matrix. We drill in from there.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import subprocess
import sys

from functools import partial

import autolens as al
import autoarray as aa
from autoarray.inversion.mesh.interpolator.rectangular import (
    create_transforms,
    adaptive_rectangular_mappings_weights_via_interpolation_from,
)

instrument = "hst"
_script_dir = Path(__file__).resolve().parent
dataset_path = Path("jax_profiling") / "imaging" / "dataset" / "imaging" / instrument
if al.util.dataset.should_simulate(str(dataset_path)):
    subprocess.run(
        [sys.executable, str(_script_dir / "simulators" / "imaging.py"),
         "--instrument", instrument],
        cwd=str(_script_dir), check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.5,
)
dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=4, over_sample_size_pixelization=1,
)

mass = al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0)
lens = al.Galaxy(redshift=0.5, mass=mass)
source = al.Galaxy(redshift=1.0)
tracer = al.Tracer(galaxies=[lens, source])
traced = tracer.traced_grid_2d_list_from(grid=dataset.grids.pixelization, xp=jnp)
src_grid = traced[-1]
data_grid0 = jnp.array(src_grid.array)
over_grid0 = jnp.array(src_grid.over_sampled.array)

# -- Check for near-duplicates in the sorted source grid --
y_sorted = np.sort(np.array(data_grid0)[:, 0])
x_sorted = np.sort(np.array(data_grid0)[:, 1])
dy = np.diff(y_sorted)
dx = np.diff(x_sorted)
print(f"data_grid N                 = {data_grid0.shape[0]}")
print(f"y sort gap min / median     = {dy.min():.3e} / {np.median(dy):.3e}")
print(f"x sort gap min / median     = {dx.min():.3e} / {np.median(dx):.3e}")
print(f"# y gaps <= 1e-12           = {int(np.sum(dy <= 1e-12))}")
print(f"# x gaps <= 1e-12           = {int(np.sum(dx <= 1e-12))}")
print(f"# y gaps <= 1e-8            = {int(np.sum(dy <= 1e-8))}")
print(f"# x gaps <= 1e-8            = {int(np.sum(dx <= 1e-8))}")


np.random.seed(0)
pert_data = jnp.array(
    (np.random.randn(*data_grid0.shape) / np.linalg.norm(np.random.randn(*data_grid0.shape))).astype(np.float64)
)
pert_over = jnp.array(
    (np.random.randn(*over_grid0.shape) / np.linalg.norm(np.random.randn(*over_grid0.shape))).astype(np.float64)
)


SRC_GRID_SIZE = 28


def _stage(eps, which):
    """Build forward chain up to `which` and return sum-of-squares scalar."""
    data_grid = data_grid0 + eps * pert_data
    data_grid_over = over_grid0 + eps * pert_over

    mu = data_grid.mean(axis=0)
    scale = data_grid.std(axis=0).min()
    source_grid_scaled = (data_grid - mu) / scale
    if which == "source_grid_scaled":
        return jnp.sum(source_grid_scaled ** 2)

    transform, _ = create_transforms(source_grid_scaled, mesh_weight_map=None, xp=jnp)

    if which == "sort_points":
        # internal access: replicate what create_transforms does
        sort_points = jnp.sort(source_grid_scaled, axis=0)
        return jnp.sum(sort_points ** 2)

    grid_over_scaled = (data_grid_over - mu) / scale
    if which == "grid_over_scaled":
        return jnp.sum(grid_over_scaled ** 2)

    grid_over_transformed = transform(grid_over_scaled)
    if which == "grid_over_transformed":
        return jnp.sum(grid_over_transformed ** 2)

    grid_over_index = (SRC_GRID_SIZE - 3) * grid_over_transformed + 1
    if which == "grid_over_index":
        return jnp.sum(grid_over_index ** 2)

    ix_down = jnp.floor(grid_over_index[:, 0])
    ix_up = jnp.ceil(grid_over_index[:, 0])
    iy_down = jnp.floor(grid_over_index[:, 1])
    iy_up = jnp.ceil(grid_over_index[:, 1])

    t_row = (grid_over_index[:, 0] - ix_down) / (ix_up - ix_down + 1e-12)
    t_col = (grid_over_index[:, 1] - iy_down) / (iy_up - iy_down + 1e-12)
    if which == "t_row_t_col":
        return jnp.sum(t_row ** 2) + jnp.sum(t_col ** 2)

    w_tl = (1 - t_row) * (1 - t_col)
    w_tr = (1 - t_row) * t_col
    w_bl = t_row * (1 - t_col)
    w_br = t_row * t_col
    weights = jnp.stack([w_tl, w_tr, w_bl, w_br], axis=1)
    if which == "weights":
        return jnp.sum(weights ** 2)

    # Full mapping-matrix loss -- use the shared utility to be safe
    flat_idx, weights = adaptive_rectangular_mappings_weights_via_interpolation_from(
        source_grid_size=SRC_GRID_SIZE,
        data_grid=data_grid,
        data_grid_over_sampled=data_grid_over,
        mesh_weight_map=None,
        xp=jnp,
    )
    if which == "weights_util":
        return jnp.sum(weights ** 2)

    raise ValueError(which)


STAGES = [
    "source_grid_scaled",
    "sort_points",
    "grid_over_scaled",
    "grid_over_transformed",
    "grid_over_index",
    "t_row_t_col",
    "weights",
    "weights_util",
]

print("\n{:>24s}   {:>14s}   {:>14s}   {:>10s}".format("stage", "JAX grad", "FD grad", "ratio"))
print("-" * 70)

h = 1e-5
for stage in STAGES:
    fn = partial(_stage, which=stage)
    val_plus = float(fn(jnp.float64(h)))
    val_minus = float(fn(jnp.float64(-h)))
    fd = (val_plus - val_minus) / (2 * h)
    val, g = jax.value_and_grad(fn)(jnp.float64(0.0))
    g = float(g)
    ratio = g / fd if abs(fd) > 1e-20 else float('nan')
    tag = "  <-- BLOWUP" if abs(ratio) > 1e6 or (np.isnan(ratio) and abs(g) > 1e6) else ""
    print(f"{stage:>24s}   {g:>14.4e}   {fd:>14.4e}   {ratio:>10.3e}{tag}")
