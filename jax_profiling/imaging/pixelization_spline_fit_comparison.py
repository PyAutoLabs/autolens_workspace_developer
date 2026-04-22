"""
Direct fit comparison: RectangularAdaptDensity vs RectangularSplineAdaptDensity
===============================================================================

Fits the same HST dataset with each of the two adaptive rectangular meshes
and saves a ``subplot_fit`` PNG for each. Prints the eager
``FitImaging.figure_of_merit`` (reference log-evidence) for both so the
numbers can be compared side-by-side.

Uses the **true lens mass + light** from the simulator
(``simulators/imaging.py`` stored as ``tracer.json``) so the inversion's
source reconstruction is well-behaved.  Replaces the simulated source
light profile with a pixelization mesh (density-adaptive or spline-density
-adaptive).  Nothing here runs under JAX — results reproduce run-to-run.
"""

import time
from pathlib import Path

import numpy as np

import autolens as al
import autolens.plot as aplt
import autoarray as aa


INSTRUMENT = "hst"
PIXEL_SCALE = 0.05
MASK_RADIUS = 3.5
MESH_SHAPE = (28, 28)

_script_dir = Path(__file__).resolve().parent
_dataset_path = Path("jax_profiling") / "imaging" / "dataset" / "imaging" / INSTRUMENT
_results_dir = _script_dir / "results" / "spline_vs_linear_fit"
_results_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

print(f"--- loading {INSTRUMENT} dataset ---")
dataset = al.Imaging.from_fits(
    data_path=_dataset_path / "data.fits",
    psf_path=_dataset_path / "psf.fits",
    noise_map_path=_dataset_path / "noise_map.fits",
    pixel_scales=PIXEL_SCALE,
)
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=MASK_RADIUS,
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
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=1,
)

# ---------------------------------------------------------------------------
# True lens galaxy from the simulator.
# ---------------------------------------------------------------------------

# Simulator constants — replicated exactly from simulators/imaging.py so the
# fit sees the same mass, light, and external shear the data was generated
# with. Only the source light profile is replaced with a pixelization mesh.

_LENS_BULGE = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=2.0,
    effective_radius=0.6,
    sersic_index=3.0,
)
_LENS_MASS = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)
_LENS_SHEAR = al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)


def build_tracer(mesh):
    lens_galaxy = al.Galaxy(
        redshift=0.5, bulge=_LENS_BULGE, mass=_LENS_MASS, shear=_LENS_SHEAR
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=al.Pixelization(
            mesh=mesh, regularization=al.reg.Constant(coefficient=1.0)
        ),
    )
    return al.Tracer(galaxies=[lens_galaxy, source_galaxy])


# ---------------------------------------------------------------------------
# Fit + subplot_fit for each mesh
# ---------------------------------------------------------------------------

mesh_configs = [
    ("RectangularAdaptDensity", aa.mesh.RectangularAdaptDensity(shape=MESH_SHAPE)),
    (
        "RectangularSplineAdaptDensity",
        aa.mesh.RectangularSplineAdaptDensity(shape=MESH_SHAPE),
    ),
]

results = []
for name, mesh in mesh_configs:
    print(f"\n--- fitting {name} ---")
    tracer = build_tracer(mesh)
    t0 = time.perf_counter()
    fit = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        settings=al.Settings(use_border_relocator=True),
        xp=np,
    )
    fom = float(fit.figure_of_merit)
    loglike = float(fit.log_likelihood)
    elapsed = time.perf_counter() - t0

    print(f"  figure_of_merit (log-evidence)   = {fom:.4f}")
    print(f"  log_likelihood (chi² + noise)    = {loglike:.4f}")
    print(f"  fit time                         = {elapsed:.2f} s")

    png_path = _results_dir / f"subplot_fit_{name}.png"
    aplt.subplot_fit_imaging(
        fit=fit,
        output_path=str(_results_dir),
        output_format="png",
        title_prefix=name,
    )
    # ``subplot_fit_imaging`` writes to ``fit.png`` by default — rename so
    # successive calls don't overwrite.
    for candidate in ("fit.png", "subplot_fit.png"):
        src = _results_dir / candidate
        if src.exists():
            src.replace(png_path)
            break
    print(f"  subplot saved to {png_path}")

    results.append(
        {"name": name, "figure_of_merit": fom, "log_likelihood": loglike, "time_s": elapsed}
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\n" + "=" * 84)
print(f"COMPARISON — {INSTRUMENT.upper()} HST, {MESH_SHAPE[0]}×{MESH_SHAPE[1]} mesh, prior-median model")
print("=" * 84)
print(f"{'mesh':<34} {'figure_of_merit':>18} {'log_likelihood':>18} {'time_s':>8}")
print("-" * 84)
for r in results:
    print(
        f"{r['name']:<34} "
        f"{r['figure_of_merit']:>18.4f} "
        f"{r['log_likelihood']:>18.4f} "
        f"{r['time_s']:>8.2f}"
    )

d_fom = results[1]["figure_of_merit"] - results[0]["figure_of_merit"]
d_ll = results[1]["log_likelihood"] - results[0]["log_likelihood"]
print("-" * 84)
print(f"{'Δ (Spline − Linear)':<34} {d_fom:>18.4f} {d_ll:>18.4f}")
print("=" * 84)
print(f"  subplot_fit PNGs in {_results_dir}")
