"""
Minimal probe: does the mapping matrix of a RectangularAdaptDensity mapper
carry gradient w.r.t. a perturbation of the source-plane data grid?

We bypass the FitImaging / Tracer / Inversion wrappers entirely. A synthetic
source-plane grid is built as `grid0 + eps * unit_vec`; we then construct the
interpolator + mapper from that grid, build the mapping matrix, sum it, and
differentiate w.r.t. `eps`. If `d(sum(M)) / d(eps)` is of order ~1 (as it
should be for bilinear interpolation weights with O(1) geometry), the
interpolator is differentiable. If it is ~1e-20, the gradient chain is
broken somewhere inside the mapper/interpolator path.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import subprocess
import sys

import autolens as al
import autoarray as aa

# Build a realistic source-plane data grid from the HST dataset's lens-plane
# grid ray-traced through a trivial mass model — this guarantees the grid is
# the same shape / range the real pipeline uses.
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

mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)
shear = al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
lens = al.Galaxy(redshift=0.5, mass=mass, shear=shear)
source = al.Galaxy(redshift=1.0)
tracer = al.Tracer(galaxies=[lens, source])

traced = tracer.traced_grid_2d_list_from(grid=dataset.grids.pixelization, xp=jnp)
src_grid = traced[-1]
src_grid_raw = jnp.array(src_grid.array)
src_over_raw = jnp.array(src_grid.over_sampled.array)

print(f"src_grid_raw shape           = {src_grid_raw.shape}")
print(f"src_grid_raw range           = [{jnp.min(src_grid_raw):.4f}, {jnp.max(src_grid_raw):.4f}]")
print(f"src_over_raw shape           = {src_over_raw.shape}")

# Fixed perturbation direction (unit vector in (N, 2) space)
np.random.seed(0)
pert_dir = np.random.randn(*src_grid_raw.shape).astype(np.float64)
pert_dir /= np.linalg.norm(pert_dir)
pert_dir = jnp.array(pert_dir)

pert_over_dir = np.random.randn(*src_over_raw.shape).astype(np.float64)
pert_over_dir /= np.linalg.norm(pert_over_dir)
pert_over_dir = jnp.array(pert_over_dir)

mesh_shape = (28, 28)
mesh = al.mesh.RectangularAdaptDensity(shape=mesh_shape)
# Build mesh_grid once from eager grid — not critical for gradient test,
# because RectangularAdaptDensity rebuilds it inside interpolator_from.
mesh_grid_dummy = al.Grid2DIrregular(values=src_grid_raw, xp=jnp)


def loss(eps):
    g_raw = src_grid_raw + eps * pert_dir
    g_over_raw = src_over_raw + eps * pert_over_dir

    # Build a Grid2D-like container directly — bypass border_relocator + Tracer.
    # We instantiate Grid2D with a mask so that .over_sampled is reachable.
    over_sampled = aa.Grid2DIrregular(values=g_over_raw, xp=jnp)
    g = aa.Grid2D(
        values=g_raw,
        mask=src_grid.mask,
        over_sample_size=src_grid.over_sample_size,
        over_sampled=over_sampled,
        over_sampler=src_grid.over_sampler,
        xp=jnp,
    )
    interp = mesh.interpolator_from(
        source_plane_data_grid=g,
        source_plane_mesh_grid=mesh_grid_dummy,
        border_relocator=None,
        adapt_data=None,
        xp=jnp,
    )
    mapper = aa.Mapper(interpolator=interp, xp=jnp)
    M = mapper.mapping_matrix
    # Non-degenerate loss: bilinear sum-of-row is always 1, so sum(M) is constant.
    # Use sum of squares to actually see row-to-row weight movements.
    return jnp.sum(M ** 2)


eps0 = jnp.float64(0.0)
val, grad = jax.value_and_grad(loss)(eps0)
print(f"\nloss(eps=0)                  = {float(val):.6g}")
print(f"d loss / d eps (JAX grad)    = {float(grad):.6g}")

# Finite-difference cross-check
h = 1e-4
fd = (float(loss(jnp.float64(h))) - float(loss(jnp.float64(-h)))) / (2 * h)
print(f"d loss / d eps (finite diff) = {fd:.6g}")

print(f"\nratio JAX/FD                 = {float(grad) / fd if fd != 0 else float('nan'):.6g}")
if abs(float(grad)) < 1e-10 and abs(fd) > 1e-6:
    print("=> JAX grad is dead while FD is alive: gradient chain broken inside mapper.")
elif abs(float(grad) - fd) / max(abs(fd), 1e-30) > 1e-2:
    print("=> JAX grad disagrees with FD — partial gradient path only.")
else:
    print("=> JAX grad matches FD: interpolator IS differentiable.")
