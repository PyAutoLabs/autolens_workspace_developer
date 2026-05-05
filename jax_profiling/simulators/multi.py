"""
Simulator Profiling: Multi-Wavelength Imaging
==============================================

Profiles `autolens_workspace/scripts/multi/simulator.py` to pinpoint where
time goes when simulating two-waveband imaging (g-band + r-band). Times:

- Per-band grid setup with adaptive over-sampling
- Per-band PSF and simulator construction
- Galaxy and tracer construction (two tracers, one per waveband)
- Per-band `tracer.image_2d_from` (eager + JIT)
- Per-band `simulator.via_tracer_from` (convolution path)
- FITS + JSON output per band

Run from any path:
    python jax_profiling/simulators/multi.py
"""

from autoconf import jax_wrapper  # noqa: F401 — must be first

import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import autolens as al
import autolens.plot as aplt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Timer:
    def __init__(self):
        self.records: list[tuple[str, float]] = []

    @contextmanager
    def section(self, label: str):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.records.append((label, elapsed))
        print(f"  [{label}] {elapsed:.4f} s")

    def summary(self):
        print("\n" + "=" * 70)
        print("PROFILING SUMMARY")
        print("=" * 70)
        max_label = max(len(r[0]) for r in self.records)
        total = 0.0
        for label, elapsed in self.records:
            print(f"  {label:<{max_label}}  {elapsed:>10.4f} s")
            total += elapsed
        print("-" * 70)
        print(f"  {'TOTAL':<{max_label}}  {total:>10.4f} s")
        print("=" * 70)


def block(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def jit_profile(func, label, *args, n_repeats=10):
    jitted = jax.jit(func)

    with timer.section(f"{label}_lower"):
        lowered = jitted.lower(*args)

    with timer.section(f"{label}_compile"):
        compiled = lowered.compile()

    with timer.section(f"{label}_first_call"):
        result = compiled(*args)
        block(result)

    with timer.section(f"{label}_steady_x{n_repeats}"):
        for _ in range(n_repeats):
            result = compiled(*args)
            block(result)

    per_call = timer.records[-1][1] / n_repeats
    print(f"    -> per-call avg: {per_call:.6f} s")
    return compiled, result


timer = Timer()

_script_dir = Path(__file__).resolve().parent
_workspace_root = _script_dir.parents[1]

dataset_name = "simple"
dataset_path = (
    _workspace_root / "jax_profiling" / "dataset" / "multi" / "imaging" / "lens_sersic"
)
dataset_path.mkdir(parents=True, exist_ok=True)

waveband_list = ["g", "r"]
pixel_scales_list = [0.08, 0.12]
sigma_list = [0.1, 0.2]
background_sky_level_list = [0.1, 0.15]


# === PART 1 — Setup ===

print("\n--- PART 1: Setup ---")

with timer.section("setup_grids"):
    grid_list = []
    for pixel_scales in pixel_scales_list:
        grid = al.Grid2D.uniform(shape_native=(150, 150), pixel_scales=pixel_scales)
        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=grid,
            sub_size_list=[32, 8, 2],
            radial_list=[0.3, 0.6],
            centre_list=[(0.0, 0.0)],
        )
        grid_list.append(grid.apply_over_sampling(over_sample_size=over_sample_size))

with timer.section("setup_psf_simulators"):
    psf_list = [
        al.Convolver.from_gaussian(
            shape_native=(11, 11), sigma=sigma, pixel_scales=grid.pixel_scales
        )
        for grid, sigma in zip(grid_list, sigma_list)
    ]
    simulator_list = [
        al.SimulatorImaging(
            exposure_time=300.0,
            psf=psf,
            background_sky_level=background_sky_level,
            add_poisson_noise_to_data=True,
        )
        for psf, background_sky_level in zip(psf_list, background_sky_level_list)
    ]

with timer.section("setup_galaxies"):
    # Lens intensities differ per band; mass is shared
    intensity_list_lens = [0.05, 1.5]
    mass = al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    )
    lens_galaxy_list = [
        al.Galaxy(
            redshift=0.5,
            bulge=al.lp.Sersic(
                centre=(0.0, 0.0),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
                intensity=intensity,
                effective_radius=0.8,
                sersic_index=4.0,
            ),
            mass=mass,
            shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
        )
        for intensity in intensity_list_lens
    ]

    intensity_list_source = [0.5, 0.7]
    source_galaxy_list = [
        al.Galaxy(
            redshift=1.0,
            bulge=al.lp.SersicCore(
                centre=(0.0, 0.0),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
                intensity=intensity,
                effective_radius=0.1,
                sersic_index=1.0,
            ),
        )
        for intensity in intensity_list_source
    ]

with timer.section("setup_tracers"):
    tracer_list = [
        al.Tracer(galaxies=[lens_galaxy, source_galaxy])
        for lens_galaxy, source_galaxy in zip(lens_galaxy_list, source_galaxy_list)
    ]


# === PART 2 — image_2d_from per band: eager + JIT ===

print("\n--- PART 2: tracer.image_2d_from per band (eager + JIT) ---")

for band, tracer, grid in zip(waveband_list, tracer_list, grid_list):
    with timer.section(f"image_2d_eager_{band}"):
        image_eager = tracer.image_2d_from(grid=grid)

    _tracer = tracer
    _grid = grid

    def _image_fn(grid_array, _tracer=_tracer, _grid=_grid):
        return _tracer.image_2d_from(grid=_grid, xp=jnp).array

    jnp_grid = jnp.asarray(grid.array)
    _, image_jit = jit_profile(_image_fn, f"image_2d_jit_{band}", jnp_grid)

    np.testing.assert_allclose(
        np.asarray(image_eager.array), np.asarray(image_jit), rtol=1e-4,
        err_msg=f"multi/{band}: eager vs JIT image_2d_from mismatch",
    )
    print(f"  {band}-band: eager ≡ JIT assertion PASSED")


# === PART 3 — via_tracer_from per band ===

print("\n--- PART 3: simulator.via_tracer_from per band ---")

np.random.seed(1)
dataset_list = []
for band, simulator, tracer, grid in zip(
    waveband_list, simulator_list, tracer_list, grid_list
):
    with timer.section(f"via_tracer_from_{band}"):
        dataset_list.append(simulator.via_tracer_from(tracer=tracer, grid=grid))


# === PART 4 — outputs ===

print("\n--- PART 4: outputs ---")

with timer.section("output_fits"):
    for band, dataset in zip(waveband_list, dataset_list):
        aplt.fits_imaging(
            dataset=dataset,
            data_path=dataset_path / f"{band}_data.fits",
            psf_path=dataset_path / f"{band}_psf.fits",
            noise_map_path=dataset_path / f"{band}_noise_map.fits",
            overwrite=True,
        )

with timer.section("output_json"):
    for band, tracer in zip(waveband_list, tracer_list):
        al.output_to_json(
            obj=tracer, file_path=dataset_path / f"{band}_tracer.json"
        )


# === Summary ===

al_version = al.__version__
results_dir = _workspace_root / "jax_profiling" / "results" / "simulators"
results_dir.mkdir(parents=True, exist_ok=True)

phases = dict(timer.records)

results_summary = {
    "autolens_version": al_version,
    "type": "multi",
    "configuration": {
        "bands": waveband_list,
        "grid_shape": [150, 150],
        "pixel_scales": pixel_scales_list,
        "psf_sigmas": sigma_list,
    },
    "phases": phases,
    "key_timings": {
        "via_tracer_from_g_s": phases.get("via_tracer_from_g"),
        "via_tracer_from_r_s": phases.get("via_tracer_from_r"),
    },
}

json_path = results_dir / f"multi_summary_v{al_version}.json"
json_path.write_text(json.dumps(results_summary, indent=2))
print(f"\n  Results saved to: {json_path}")

labels = [r[0] for r in timer.records]
times = [r[1] for r in timer.records]
colors = plt.cm.tab20.colors[: len(labels)]

fig, ax = plt.subplots(figsize=(12, max(4.0, len(labels) * 0.45)))
y_pos = range(len(labels))
bars = ax.barh(y_pos, times, color=colors, edgecolor="white", height=0.6)
for bar, t in zip(bars, times):
    ax.text(
        bar.get_width() + max(times) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.4f} s",
        va="center",
        fontsize=8,
    )
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Time (s)", fontsize=11)
fig.suptitle("Simulator Profiling: Multi-Wavelength Imaging", fontsize=12, fontweight="bold")
ax.set_title(f"AutoLens v{al_version}  |  150×150  |  g+r bands", fontsize=9)
ax.margins(x=0.22)
fig.tight_layout()
chart_path = results_dir / f"multi_summary_v{al_version}.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to: {chart_path}")

timer.summary()
