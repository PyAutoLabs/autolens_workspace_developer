"""
NNLS preconditioning benchmark.

Compares the wall-clock time of `jaxnnls.solve_nnls_primal` on the raw MGE
curvature matrix vs. the Jacobi-preconditioned equivalent. Both variants are
JIT-compiled and warmed up before timing. Reports (a) single-solve time,
(b) vmap-batched time, and (c) `value_and_grad` time through the full
`_build_Q_q` pipeline.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import jaxnnls

import autofit as af
import autolens as al
import autoarray as aa
from pathlib import Path
import subprocess
import sys


# ---------------------------------------------------------------------------
# Build the exact same Q, q as mge_gradients.py
# ---------------------------------------------------------------------------

instrument = "hst"
INSTRUMENTS = {"hst": {"pixel_scale": 0.05}}
_script_dir = Path(__file__).resolve().parent
_workspace_root = _script_dir.parents[1]
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
dataset_path = Path("jax_profiling") / "dataset" / "imaging" / instrument

if al.util.dataset.should_simulate(str(dataset_path)):
    subprocess.run(
        [sys.executable, str(_workspace_root / "jax_profiling" / "dataset_setup" / "imaging.py"),
         "--instrument", instrument],
        cwd=str(_workspace_root), check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
)

mask_radius = 3.5
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)
dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(over_sample_size_lp=4)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

# GaussianPrior(mean=truth, sigma=small) centres prior-median at the
# simulator truth while keeping params free so gradient diagnostics
# have dimensionality.
lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)

mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
_lens_mass_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=_lens_mass_ell[0], sigma=0.01)
mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=_lens_mass_ell[1], sigma=0.01)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

jnp_params = jnp.array(model.physical_values_from_prior_medians)
key = jax.random.PRNGKey(42)
perturbation = jax.random.uniform(key, shape=jnp_params.shape, minval=0.01, maxval=0.05)
jnp_params = jnp_params + perturbation

instance = model.instance_from_vector(vector=jnp_params)
tracer = al.Tracer(galaxies=list(instance.galaxies))
fit = al.FitImaging(
    dataset=dataset, tracer=tracer,
    settings=al.Settings(use_border_relocator=True),
)

data_array = jnp.array(dataset.data.array)
noise_map_array = jnp.array(dataset.noise_map.array)


def _build_Q_q(params):
    inst = model.instance_from_vector(vector=params, xp=jnp)
    t = al.Tracer(galaxies=list(inst.galaxies))
    tti = al.TracerToInversion(
        dataset=aa.DatasetInterface(
            data=fit.profile_subtracted_image,
            noise_map=dataset.noise_map,
            grids=dataset.grids,
            psf=dataset.psf,
            sparse_operator=dataset.sparse_operator,
        ),
        tracer=t,
        settings=al.Settings(use_border_relocator=True),
        xp=jnp,
    )
    funcs = list(tti.lp_linear_func_list_galaxy_dict.keys())
    matrices = [f.operated_mapping_matrix_override for f in funcs]
    bmm = jnp.hstack(matrices) if len(matrices) > 1 else matrices[0]
    q = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=bmm, image=data_array, noise_map=noise_map_array,
    )
    n_linear = bmm.shape[1]
    Q = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=bmm, noise_map=noise_map_array, add_to_curvature_diag=True,
        no_regularization_index_list=list(range(n_linear)), xp=jnp,
    )
    return Q, q


Q_eval, q_eval = _build_Q_q(jnp_params)
print(f"Q shape = {Q_eval.shape}, dtype = {Q_eval.dtype}")
print(f"cond(Q)  = {np.linalg.cond(np.array(Q_eval)):.3g}")

# ---------------------------------------------------------------------------
# Solver variants
# ---------------------------------------------------------------------------

def solve_raw(Q, q):
    return jaxnnls.solve_nnls_primal(Q, q)


def solve_pc(Q, q):
    d = jnp.sqrt(jnp.diag(Q))
    D = 1.0 / d
    Q_pc = (Q * D[:, None]) * D[None, :]
    q_pc = q * D
    y = jaxnnls.solve_nnls_primal(Q_pc, q_pc)
    return y * D


solve_raw_jit = jax.jit(solve_raw)
solve_pc_jit = jax.jit(solve_pc)


def time_fn(label, fn, args, n_warmup=3, n_iters=50):
    for _ in range(n_warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = fn(*args)
        jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / n_iters
    print(f"  {label:<35s} {dt*1e3:8.3f} ms / call   result sum = {float(jnp.sum(out)):.6g}")
    return dt


print("\n--- single solve (JIT) ---")
t_raw = time_fn("raw solve_nnls_primal        ", solve_raw_jit, (Q_eval, q_eval))
t_pc = time_fn("Jacobi preconditioned         ", solve_pc_jit, (Q_eval, q_eval))
print(f"  speedup (raw / pc) = {t_raw/t_pc:.2f}x")

# ---------------------------------------------------------------------------
# vmap-batched (simulates what Fitness._vmap does)
# ---------------------------------------------------------------------------

batch_size = 8
# Batch the same (Q, q) to remove build overhead and isolate solver cost.
Q_batch = jnp.broadcast_to(Q_eval, (batch_size,) + Q_eval.shape)
q_batch = jnp.broadcast_to(q_eval, (batch_size,) + q_eval.shape)

solve_raw_vmap = jax.jit(jax.vmap(solve_raw))
solve_pc_vmap = jax.jit(jax.vmap(solve_pc))

print(f"\n--- vmap (batch={batch_size}, same Q,q) ---")
t_raw_v = time_fn("raw vmap                      ", solve_raw_vmap, (Q_batch, q_batch))
t_pc_v = time_fn("Jacobi preconditioned vmap    ", solve_pc_vmap, (Q_batch, q_batch))
print(f"  speedup (raw / pc) = {t_raw_v/t_pc_v:.2f}x")

# ---------------------------------------------------------------------------
# value_and_grad on the NNLS solver alone (both versions JIT-compiled).
# Note: raw version returns NaN gradients; we still time it to report cost.
# ---------------------------------------------------------------------------

def solve_loss_raw(Q, q):
    return jnp.sum(jaxnnls.solve_nnls_primal(Q, q))


def solve_loss_pc(Q, q):
    d = jnp.sqrt(jnp.diag(Q))
    D = 1.0 / d
    Q_pc = (Q * D[:, None]) * D[None, :]
    q_pc = q * D
    y = jaxnnls.solve_nnls_primal(Q_pc, q_pc)
    return jnp.sum(y * D)


vg_raw = jax.jit(jax.value_and_grad(solve_loss_raw, argnums=(0, 1)))
vg_pc = jax.jit(jax.value_and_grad(solve_loss_pc, argnums=(0, 1)))

print("\n--- solver value_and_grad (JIT) ---")
for _ in range(3):
    v, g = vg_raw(Q_eval, q_eval); jax.block_until_ready(v)
for _ in range(3):
    v, g = vg_pc(Q_eval, q_eval); jax.block_until_ready(v)


def time_vg(label, fn, args, n_iters=30):
    t0 = time.perf_counter()
    for _ in range(n_iters):
        v, g = fn(*args)
        jax.block_until_ready(v)
        for gi in g:
            jax.block_until_ready(gi)
    dt = (time.perf_counter() - t0) / n_iters
    dQ = np.array(g[0])
    n_nan = int(np.sum(~np.isfinite(dQ)))
    print(f"  {label:<35s} {dt*1e3:8.3f} ms / call   dQ NaN entries = {n_nan}/{dQ.size}")
    return dt


t_vg_raw = time_vg("raw value_and_grad            ", vg_raw, (Q_eval, q_eval))
t_vg_pc = time_vg("preconditioned value_and_grad ", vg_pc, (Q_eval, q_eval))
print(f"  speedup (raw / pc) = {t_vg_raw/t_vg_pc:.2f}x")
