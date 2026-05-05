"""
Simulator Profiling: Interferometer
=====================================

Profiles `autolens_workspace/scripts/interferometer/simulator.py` to pinpoint
where time goes during simulation. Times the following phases:

- Grid setup (256×256 @ 0.1"/px; no over-sampling for interferometry)
- Synthetic uv_wavelengths construction (100 baselines inline — no disk file)
- SimulatorInterferometer + TransformerDFT setup
- Galaxy and tracer construction
- `tracer.image_2d_from` (eager + JIT)
- `simulator.via_tracer_from` (DFT to visibilities — expected bottleneck)
- `solver.solve` (eager)
- FITS + JSON output

Run from any path:
    python jax_profiling/simulators/interferometer.py
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
    _workspace_root / "jax_profiling" / "dataset" / "interferometer" / dataset_name
)
dataset_path.mkdir(parents=True, exist_ok=True)


# === PART 1 — Setup ===

print("\n--- PART 1: Setup ---")

with timer.section("setup_grids"):
    # Interferometer does not use over-sampling
    grid = al.Grid2D.uniform(shape_native=(256, 256), pixel_scales=0.1)

with timer.section("setup_uv_wavelengths"):
    # Inline synthetic baselines — 100 visibilities covering ~10–500 klambda
    np.random.seed(42)
    n_vis = 100
    uv_wavelengths = np.random.uniform(low=10_000, high=500_000, size=(n_vis, 2))

with timer.section("setup_simulator"):
    simulator = al.SimulatorInterferometer(
        uv_wavelengths=uv_wavelengths,
        exposure_time=300.0,
        noise_sigma=1000.0,
        transformer_class=al.TransformerDFT,
    )

with timer.section("setup_galaxies"):
    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.6,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        ),
        shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.SersicCore(
            centre=(0.0, 0.0),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=0.3,
            effective_radius=1.0,
            sersic_index=2.5,
        ),
    )

with timer.section("setup_tracer"):
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])


# === PART 2 — image_2d_from: eager + JIT ===

print("\n--- PART 2: tracer.image_2d_from (eager + JIT) ---")

with timer.section("image_2d_eager"):
    image_eager = tracer.image_2d_from(grid=grid)

def _image_fn(grid_array):
    return tracer.image_2d_from(grid=grid, xp=jnp).array

jnp_grid = jnp.asarray(grid.array)
_, image_jit = jit_profile(_image_fn, "image_2d_jit", jnp_grid)

np.testing.assert_allclose(
    np.asarray(image_eager.array), np.asarray(image_jit), rtol=1e-4,
    err_msg="interferometer: eager vs JIT image_2d_from mismatch",
)
print("  eager ≡ JIT assertion PASSED")


# === PART 3 — via_tracer_from (DFT bottleneck) ===

print("\n--- PART 3: simulator.via_tracer_from (DFT) ---")

np.random.seed(1)
with timer.section("via_tracer_from"):
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)


# === PART 4 — solver.solve ===

print("\n--- PART 4: solver.solve ---")

with timer.section("solver_build"):
    solver = al.PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
    )

with timer.section("solver_solve_eager"):
    positions = solver.solve(
        tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre
    )


# === PART 5 — outputs ===

print("\n--- PART 5: outputs ---")

with timer.section("output_fits"):
    aplt.fits_interferometer(
        dataset=dataset,
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
        overwrite=True,
    )

with timer.section("output_json"):
    al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")


# === Summary ===

al_version = al.__version__
results_dir = _workspace_root / "jax_profiling" / "results" / "simulators"
results_dir.mkdir(parents=True, exist_ok=True)

phases = dict(timer.records)

results_summary = {
    "autolens_version": al_version,
    "type": "interferometer",
    "configuration": {
        "grid_shape": [256, 256],
        "pixel_scales": 0.1,
        "n_visibilities": n_vis,
        "transformer": "TransformerDFT",
    },
    "phases": phases,
    "key_timings": {
        "image_2d_eager_s": phases.get("image_2d_eager"),
        "via_tracer_from_s": phases.get("via_tracer_from"),
        "solver_solve_eager_s": phases.get("solver_solve_eager"),
    },
}

json_path = results_dir / f"interferometer_summary_v{al_version}.json"
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
fig.suptitle("Simulator Profiling: Interferometer", fontsize=12, fontweight="bold")
ax.set_title(
    f"AutoLens v{al_version}  |  256×256 @ 0.1\"/px  |  {n_vis} visibilities (DFT)",
    fontsize=9,
)
ax.margins(x=0.22)
fig.tight_layout()
chart_path = results_dir / f"interferometer_summary_v{al_version}.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to: {chart_path}")

timer.summary()
