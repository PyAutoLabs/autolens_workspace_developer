"""
Spline-CDF vs linear-CDF rectangular-pixelization comparison.
=============================================================

Compares four rectangular mesh variants on the same HST imaging dataset,
with identical lens parameters, mask, and over-sampling.

Meshes compared
---------------
1. ``RectangularUniform``
2. ``RectangularAdaptDensity``                        — linear CDF
3. ``RectangularSplineAdaptDensity``                  — spline CDF
4. ``RectangularSplineAdaptImage``                    — spline CDF + adapt image

Metrics reported per mesh
-------------------------
- Eager ``FitImaging.figure_of_merit`` (reference log-evidence).
- JIT-compiled ``log_likelihood_function(instance)`` value + steady-state time.
- JIT-compiled ``jax.grad(log_L)(instance)`` at a single point.  At the
  prior-median operating point the autograd gradient is NaN for every mesh
  via a downstream pipeline issue unrelated to the CDF choice — so we
  measure smoothness via a different route.
- **Likelihood-smoothness sweep**: vary ``mass.einstein_radius`` ±5% at 41
  points, compute ``log_L`` under each mesh.  From the 1-D curve we compute
  the sup-norm of the discrete second difference — a direct proxy for
  curvature noise that HMC's symplectic integrator would see.  Spline meshes
  should give smaller second-difference norms than linear meshes if the
  spline CDF really does remove piecewise-linear kinks.

Outputs
-------
- ``results/spline_vs_linear_<instrument>_v<version>.json``
- ``results/spline_vs_linear_<instrument>_v<version>_sweep.png``
- Console table summary.
"""

import copy
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import autofit as af
import autolens as al
import autoarray as aa
from autofit.jax import register_model as _register_model_pytrees


# Force unbuffered stdout so progress is visible when piped to a file.
def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INSTRUMENTS = {
    "euclid": {"pixel_scale": 0.1},
    "hst": {"pixel_scale": 0.05},
    "jwst": {"pixel_scale": 0.03},
    "ao": {"pixel_scale": 0.01},
}

instrument = "hst"
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)
mask_radius = 3.5

SWEEP_N = 41
SWEEP_REL = 0.05
N_JIT_REPEATS = 3

# ---------------------------------------------------------------------------
# Dataset + model setup
# ---------------------------------------------------------------------------

_script_dir = Path(__file__).resolve().parent
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
dataset_path = Path("jax_profiling") / "imaging" / "dataset" / "imaging" / instrument

_print(f"--- Dataset load ({instrument}, {pixel_scale}\"/px) ---")
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess

    subprocess.run(
        [
            sys.executable,
            str(_script_dir / "simulators" / "imaging.py"),
            "--instrument",
            instrument,
        ],
        cwd=str(_script_dir),
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)
dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=4, over_sample_size_pixelization=1
)
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size, over_sample_size_pixelization=1
)

_print(f"  image pixels (masked): {dataset.data.shape[0]}")
_print(f"  over-sampled pixels:   {dataset.grids.lp.over_sampled.shape[0]}")
_print(f"  mesh shape:            {mesh_shape}")


# ---------------------------------------------------------------------------
# Adapt image from a first-pass RectangularAdaptDensity fit, used by the
# SplineAdaptImage mesh below.
# ---------------------------------------------------------------------------

_print("\n--- Building adapt image via RectangularAdaptDensity first-pass fit ---")
t0 = time.time()

adapt_pix = al.Pixelization(
    mesh=aa.mesh.RectangularAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)
# GaussianPrior(mean=truth, sigma=small) centres prior-median at the
# simulator truth while keeping params free so gradient diagnostics
# have dimensionality.
_adapt_lens_bulge = af.Model(al.lp.Sersic)
_adapt_lens_bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
_adapt_lens_bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
_adapt_lens_bulge_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
_adapt_lens_bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=_adapt_lens_bulge_ell[0], sigma=0.01)
_adapt_lens_bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=_adapt_lens_bulge_ell[1], sigma=0.01)
_adapt_lens_bulge.intensity = af.GaussianPrior(mean=2.0, sigma=0.1)
_adapt_lens_bulge.effective_radius = af.GaussianPrior(mean=0.6, sigma=0.05)
_adapt_lens_bulge.sersic_index = af.GaussianPrior(mean=3.0, sigma=0.2)
_adapt_mass = af.Model(al.mp.Isothermal)
_adapt_mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
_adapt_mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
_adapt_mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
_adapt_mass_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
_adapt_mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=_adapt_mass_ell[0], sigma=0.01)
_adapt_mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=_adapt_mass_ell[1], sigma=0.01)
_adapt_shear = af.Model(al.mp.ExternalShear)
_adapt_shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
_adapt_shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)
_adapt_lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=_adapt_lens_bulge,
    mass=_adapt_mass,
    shear=_adapt_shear,
)
_adapt_source = af.Model(al.Galaxy, redshift=1.0, pixelization=adapt_pix)
_adapt_model = af.Collection(galaxies=af.Collection(lens=_adapt_lens, source=_adapt_source))
_adapt_inst = _adapt_model.instance_from_vector(
    vector=_adapt_model.physical_values_from_prior_medians
)
_adapt_tracer = al.Tracer(galaxies=list(_adapt_inst.galaxies))
_adapt_fit = al.FitImaging(
    dataset=dataset,
    tracer=_adapt_tracer,
    settings=al.Settings(use_border_relocator=True),
    xp=np,
)
adapt_image_array = np.abs(np.asarray(_adapt_fit.model_data.array))
_print(
    f"  adapt image extracted: shape={adapt_image_array.shape}, "
    f"sum={adapt_image_array.sum():.3e} in {time.time()-t0:.1f}s"
)


# ---------------------------------------------------------------------------
# Helper: build (model, instance, tracer, analysis, params_tree) for a mesh.
# ---------------------------------------------------------------------------


def build_model(mesh_factory, adapt_image_array=None):
    # GaussianPrior(mean=truth, sigma=small) centres prior-median at the
    # simulator truth while keeping params free so gradient diagnostics
    # have dimensionality.
    lens_bulge = af.Model(al.lp.Sersic)
    lens_bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    lens_bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    _lens_bulge_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
    lens_bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=_lens_bulge_ell[0], sigma=0.01)
    lens_bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=_lens_bulge_ell[1], sigma=0.01)
    lens_bulge.intensity = af.GaussianPrior(mean=2.0, sigma=0.1)
    lens_bulge.effective_radius = af.GaussianPrior(mean=0.6, sigma=0.05)
    lens_bulge.sersic_index = af.GaussianPrior(mean=3.0, sigma=0.2)

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

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=lens_bulge,
        mass=mass,
        shear=shear,
    )
    pixelization = al.Pixelization(
        mesh=mesh_factory(),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    instance = model.instance_from_vector(
        vector=model.physical_values_from_prior_medians
    )
    _register_model_pytrees(model)
    params_tree = jax.tree_util.tree_map(jnp.asarray, instance)
    tracer = al.Tracer(galaxies=list(instance.galaxies))

    # ``use_positive_only_solver=False`` routes through unconstrained solve
    # (``jnp.linalg.solve``) rather than NNLS.  NNLS' relaxed-KKT backward pass
    # produces NaN gradients at the prior-median operating point for all
    # meshes tried here, which would swamp the CDF-smoothness signal.  For
    # gradient-sampler use cases the positive-only constraint is typically
    # replaced by a prior on reconstruction sign, so this isn't a limitation.
    analysis_settings = al.Settings(
        use_border_relocator=True, use_positive_only_solver=False
    )
    adapt_images = None
    if adapt_image_array is not None:
        # Path-keyed so Analysis.adapt_images_via_instance_from rebuilds the
        # instance-keyed dict for every JIT evaluation via
        # AdaptImages.updated_via_instance_from.  Instance-keying here would
        # survive eager but lose the lookup under JIT (new traced galaxy
        # identity).
        from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages

        adapt_arr = al.Array2D(values=adapt_image_array, mask=dataset.mask)
        # Key format matches str(path) as produced by
        # path_instance_tuples_for_class inside
        # AdaptImages.updated_via_instance_from.
        adapt_images = AdaptImages(
            galaxy_name_image_dict={str(("galaxies", "source")): adapt_arr}
        )
    analysis = al.AnalysisImaging(
        dataset=dataset,
        settings=analysis_settings,
        adapt_images=adapt_images,
        use_jax=True,
    )
    return model, instance, params_tree, tracer, analysis


# ---------------------------------------------------------------------------
# Mesh configurations to compare.
# ---------------------------------------------------------------------------

MESH_CONFIGS = [
    (
        "RectangularUniform",
        lambda: aa.mesh.RectangularUniform(shape=mesh_shape),
        None,
    ),
    (
        "RectangularAdaptDensity",
        lambda: aa.mesh.RectangularAdaptDensity(shape=mesh_shape),
        None,
    ),
    (
        "RectangularSplineAdaptDensity",
        lambda: aa.mesh.RectangularSplineAdaptDensity(shape=mesh_shape),
        None,
    ),
    (
        "RectangularSplineAdaptImage",
        lambda: aa.mesh.RectangularSplineAdaptImage(shape=mesh_shape),
        adapt_image_array,
    ),
]


# ---------------------------------------------------------------------------
# Per-mesh benchmark
# ---------------------------------------------------------------------------


def bench_mesh(name: str, mesh_factory, adapt_image):
    _print(f"\n{'=' * 70}")
    _print(f"  Mesh: {name}")
    _print("=" * 70)

    model, instance, params_tree, tracer, analysis = build_model(
        mesh_factory, adapt_image_array=adapt_image
    )

    # For AdaptImage meshes, eager FitImaging reads
    # adapt_images.galaxy_image_dict[galaxy] directly (see
    # PyAutoGalaxy/autogalaxy/galaxy/to_inversion.py:555), so resolve the
    # path-keyed analysis.adapt_images against the current instance to
    # produce an instance-keyed dict for the eager call.
    eager_adapt_images = None
    if analysis.adapt_images is not None:
        eager_adapt_images = analysis.adapt_images.updated_via_instance_from(
            instance=instance
        )

    # --- Eager FitImaging reference ---------------------------------------
    t0 = time.time()
    fit_eager = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=eager_adapt_images,
        settings=al.Settings(use_border_relocator=True),
        xp=np,
    )
    log_evidence_eager = float(fit_eager.figure_of_merit)
    _print(
        f"  eager figure_of_merit = {log_evidence_eager:.4f} ({time.time()-t0:.1f}s)"
    )

    # --- JIT log_L + single-point grad ------------------------------------
    log_L = lambda p: analysis.log_likelihood_function(instance=p)
    log_L_jit = jax.jit(log_L)
    grad_fn = jax.jit(jax.grad(log_L))

    t0 = time.time()
    ll_val = float(log_L_jit(params_tree))
    _print(f"  JIT log_L first call = {ll_val:.4f} ({time.time()-t0:.1f}s)")

    steady = []
    for _ in range(N_JIT_REPEATS):
        t0 = time.time()
        _ = float(log_L_jit(params_tree))
        steady.append(time.time() - t0)
    t_steady = float(np.median(steady))
    _print(f"  JIT log_L steady-state = {t_steady*1000:.2f} ms")

    t0 = time.time()
    grad_tree = grad_fn(params_tree)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        grad_tree,
    )
    grad_flat = np.concatenate(
        [np.asarray(g).ravel() for g in jax.tree_util.tree_leaves(grad_tree)]
    )
    _print(
        f"  JIT grad first call: {time.time()-t0:.1f}s, dim={grad_flat.size}, "
        f"|grad|_1={np.abs(grad_flat).sum():.4e}, "
        f"finite={bool(np.isfinite(grad_flat).all())}"
    )
    grad_steady = []
    for _ in range(N_JIT_REPEATS):
        t0 = time.time()
        g = grad_fn(params_tree)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            g,
        )
        grad_steady.append(time.time() - t0)
    t_grad_steady = float(np.median(grad_steady))
    _print(f"  JIT grad steady-state = {t_grad_steady*1000:.2f} ms")

    # --- Likelihood-smoothness sweep via lax.map --------------------------
    einstein_base = float(params_tree.galaxies.lens.mass.einstein_radius)
    sweep_thetas = np.linspace(
        einstein_base * (1 - SWEEP_REL),
        einstein_base * (1 + SWEEP_REL),
        SWEEP_N,
    )
    sweep_thetas_jnp = jnp.asarray(sweep_thetas)

    params_template = copy.deepcopy(params_tree)

    def at_einstein(einstein):
        t = copy.deepcopy(params_template)
        t.galaxies.lens.mass.einstein_radius = einstein
        return log_L(t)

    # lax.map sequences 41 calls through one compiled function — avoids the
    # O(20GB) memory spike a vmap'd inversion would need on this dataset.
    sweep_ll_fn = jax.jit(lambda arr: jax.lax.map(at_einstein, arr))

    t0 = time.time()
    ll_sweep = np.asarray(sweep_ll_fn(sweep_thetas_jnp))
    _print(f"  sweep log_L (lax.map, {SWEEP_N} pts): {time.time()-t0:.1f}s")

    dtheta = sweep_thetas[1] - sweep_thetas[0]
    # Discrete first and second differences of log_L(θ). The second difference
    # is the discrete analogue of d²L/dθ², which HMC's symplectic integrator
    # consumes via the momentum update. A kinky linear-CDF produces visible
    # piecewise-constant structure in the first difference (the first step of
    # integrated gradient descent for HMC) and large variance in the second
    # difference. The spline should give a smoother curve of both.
    fd1 = (ll_sweep[2:] - ll_sweep[:-2]) / (2 * dtheta)
    fd2 = (ll_sweep[2:] - 2 * ll_sweep[1:-1] + ll_sweep[:-2]) / (dtheta ** 2)
    # Some meshes (notably AdaptImage at large θ perturbations) produce an
    # isolated NaN log_L where the weighted inversion degenerates. Use
    # nan-safe reductions so a single bad sweep point doesn't collapse the
    # whole row to NaN — and detrend fd1 using only the finite entries.
    n_nan_sweep = int((~np.isfinite(ll_sweep)).sum())
    theta_mid = sweep_thetas[1:-1]
    fd1_valid = np.isfinite(fd1)
    if fd1_valid.sum() > 2:
        p = np.polyfit(theta_mid[fd1_valid], fd1[fd1_valid], 1)
        fd1_detrended = fd1 - np.polyval(p, theta_mid)
        fd1_roughness = float(np.nanstd(fd1_detrended))
    else:
        fd1_roughness = float("nan")
    fd2_sup = float(np.nanmax(np.abs(fd2))) if np.isfinite(fd2).any() else float("nan")
    fd2_std = float(np.nanstd(fd2))

    _print(
        f"  likelihood-smoothness ({SWEEP_N} pts ±{SWEEP_REL*100:.0f}%, "
        f"ein_base={einstein_base:.4f}):"
    )
    if n_nan_sweep:
        _print(f"    sweep NaN count            = {n_nan_sweep}/{SWEEP_N}")
    _print(f"    std(first-diff, detrended) = {fd1_roughness:.4e}")
    _print(f"    sup|second-diff|           = {fd2_sup:.4e}")
    _print(f"    std(second-diff)           = {fd2_std:.4e}")

    return {
        "name": name,
        "log_evidence_eager": log_evidence_eager,
        "log_likelihood_jit": ll_val,
        "log_likelihood_jit_steady_ms": t_steady * 1000,
        "grad_jit_steady_ms": t_grad_steady * 1000,
        "grad_abs_sum": float(np.abs(grad_flat).sum()),
        "grad_finite": bool(np.isfinite(grad_flat).all()),
        "sweep_theta": sweep_thetas.tolist(),
        "sweep_log_L": ll_sweep.tolist(),
        "fd1_roughness": fd1_roughness,
        "fd2_sup": fd2_sup,
        "fd2_std": fd2_std,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

results = []
for name, factory, adapt in MESH_CONFIGS:
    try:
        results.append(bench_mesh(name, factory, adapt))
    except Exception as e:
        import traceback

        _print(f"\n  !!! {name} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=5)
        results.append({"name": name, "error": f"{type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# Summary + save
# ---------------------------------------------------------------------------

_print("\n" + "=" * 100)
_print("SUMMARY — spline vs linear rectangular meshes")
_print("=" * 100)
header = (
    f"{'mesh':<34} "
    f"{'log_L_jit':>16} "
    f"{'steady_ms':>10} "
    f"{'fd1_rough':>14} "
    f"{'fd2_sup':>14}"
)
_print(header)
_print("-" * 100)
for r in results:
    if "error" in r:
        _print(f"{r['name']:<34} ERROR: {r['error']}")
        continue
    _print(
        f"{r['name']:<34} "
        f"{r['log_likelihood_jit']:>16.4f} "
        f"{r['log_likelihood_jit_steady_ms']:>10.2f} "
        f"{r['fd1_roughness']:>14.4e} "
        f"{r['fd2_sup']:>14.4e}"
    )
_print("=" * 100)
_print(
    "  fd1_rough = std(first difference of log_L vs θ, linear-detrended) "
    "— lower is smoother gradient"
)
_print(
    "  fd2_sup   = sup|second difference| — lower means the curvature of "
    "log_L has fewer kinks across the sweep"
)

al_version = al.__version__
results_dir = _script_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)
json_path = results_dir / f"spline_vs_linear_{instrument}_v{al_version}.json"
json_path.write_text(
    json.dumps(
        {
            "instrument": instrument,
            "pixel_scale": pixel_scale,
            "mesh_shape": list(mesh_shape),
            "sweep_n": SWEEP_N,
            "sweep_rel": SWEEP_REL,
            "results": results,
        },
        indent=2,
    )
)
_print(f"  JSON results: {json_path}")

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
colors = {
    "RectangularUniform": "#888888",
    "RectangularAdaptDensity": "#C44E52",
    "RectangularSplineAdaptDensity": "#4C72B0",
    "RectangularSplineAdaptImage": "#55A868",
}

for r in results:
    if "error" in r:
        continue
    theta = np.asarray(r["sweep_theta"])
    ll = np.asarray(r["sweep_log_L"])
    color = colors.get(r["name"], None)
    axes[0].plot(theta, ll, "-", label=r["name"], color=color)
    dtheta = theta[1] - theta[0]
    fd1 = (ll[2:] - ll[:-2]) / (2 * dtheta)
    fd2 = (ll[2:] - 2 * ll[1:-1] + ll[:-2]) / (dtheta ** 2)
    axes[1].plot(theta[1:-1], fd1, "-", label=r["name"], color=color)
    axes[2].plot(theta[1:-1], fd2, "-", label=r["name"], color=color)

axes[0].set_ylabel("log likelihood")
axes[0].legend(fontsize=9, loc="best")
axes[0].set_title(
    f"Pixelization mesh comparison — {instrument.upper()}, "
    f"{mesh_shape[0]}x{mesh_shape[1]} mesh"
)

axes[1].set_ylabel(r"$dL/d\theta$ (finite diff)")
axes[1].grid(True, which="both", alpha=0.3)

axes[2].set_xlabel("mass.einstein_radius (arcsec)")
axes[2].set_ylabel(r"$d^2L/d\theta^2$ (finite diff)")
axes[2].grid(True, which="both", alpha=0.3)
axes[2].set_title("Smoother (flatter / less kinky) = better for gradient samplers")

fig.tight_layout()
png_path = results_dir / f"spline_vs_linear_{instrument}_v{al_version}_sweep.png"
fig.savefig(png_path, dpi=150)
plt.close(fig)
_print(f"  sweep chart : {png_path}")
