"""
Simulator Profiling: Point Source
===================================

Profiles `autolens_workspace/scripts/point_source/simulator.py` to pinpoint
where time goes during simulation. Times the following phases:

- Grid setup (200×200 @ 0.05"/px)
- Lens + source galaxy and tracer construction
- `solver.solve` eager
- JIT-compiled `solver.solve` (xp=jnp, remove_infinities=False)
- `tracer.time_delays_from` on solved positions
- `simulator.via_tracer_from` (imaging side — PSF convolution)
- Point dataset + CSV + FITS + JSON output

Run from any path:
    python jax_profiling/simulators/point_source.py
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
    _workspace_root / "jax_profiling" / "dataset" / "point_source" / dataset_name
)
dataset_path.mkdir(parents=True, exist_ok=True)


# === PART 1 — Setup ===

print("\n--- PART 1: Setup ---")

with timer.section("setup_grids"):
    grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05)

with timer.section("setup_galaxies"):
    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.6,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        ),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.ExponentialCore(
            centre=(0.07, 0.07),
            intensity=0.1,
            effective_radius=0.02,
            radius_break=0.025,
        ),
        point_0=al.ps.Point(centre=(0.07, 0.07)),
    )

with timer.section("setup_tracer"):
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

with timer.section("solver_build"):
    solver = al.PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
    )


# === PART 2 — solver.solve (eager) ===

print("\n--- PART 2: solver.solve (eager) ---")

with timer.section("solver_solve_eager"):
    positions = solver.solve(
        tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
    )

print(f"  Found {len(positions)} image positions")


# === PART 3 — solver.solve (JIT) ===

print("\n--- PART 3: solver.solve (JIT) ---")

# Close over `tracer` so it does not cross the JIT boundary — avoids needing
# pytree registration for a one-tracer profiler.
@jax.jit
def jitted_solve(source_plane_coordinate):
    return solver.solve(
        tracer=tracer,
        source_plane_coordinate=source_plane_coordinate,
        xp=jnp,
        remove_infinities=False,
    ).array

src_coord = jnp.asarray(source_galaxy.point_0.centre)

_, raw_jit = jit_profile(jitted_solve, "solver_jit", src_coord, n_repeats=10)

# Strip infinities and compare to eager
raw_np = np.asarray(raw_jit)
finite_mask = ~(np.isinf(raw_np).any(axis=1) | np.isnan(raw_np).any(axis=1))
positions_jit = al.Grid2DIrregular(raw_np[finite_mask])

np.testing.assert_allclose(
    np.sort(np.asarray(positions), axis=0),
    np.sort(np.asarray(positions_jit), axis=0),
    rtol=1e-4,
    err_msg="point_source: eager vs JIT solver.solve positions mismatch",
)
print("  eager ≡ JIT solver assertion PASSED")


# === PART 4 — time_delays_from ===

print("\n--- PART 4: tracer.time_delays_from ---")

with timer.section("time_delays_from"):
    time_delays = tracer.time_delays_from(grid=positions)


# === PART 5 — imaging via_tracer_from ===

print("\n--- PART 5: simulator.via_tracer_from (imaging) ---")

with timer.section("setup_psf_simulator"):
    psf = al.Convolver.from_gaussian(
        shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
    )
    imaging_simulator = al.SimulatorImaging(
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_poisson_noise_to_data=True,
    )

np.random.seed(1)
with timer.section("via_tracer_from"):
    imaging = imaging_simulator.via_tracer_from(tracer=tracer, grid=grid)


# === PART 6 — outputs ===

print("\n--- PART 6: outputs ---")

with timer.section("output_point_datasets"):
    positions_with_noise = positions + np.random.normal(
        loc=0.0, scale=grid.pixel_scale, size=positions.shape
    )
    positions_with_noise = al.Grid2DIrregular(values=positions_with_noise)
    dataset = al.PointDataset(
        name="point_0",
        positions=positions_with_noise,
        positions_noise_map=grid.pixel_scale,
    )
    al.output_to_json(
        obj=dataset,
        file_path=dataset_path / "point_dataset_positions_only.json",
    )
    dataset.to_csv(
        file_path=dataset_path / "point_dataset_positions_only.csv",
    )

with timer.section("output_fits"):
    aplt.fits_imaging(
        dataset=imaging,
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
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
    "type": "point_source",
    "configuration": {
        "grid_shape": [200, 200],
        "pixel_scales": 0.05,
        "pixel_scale_precision": 0.001,
        "magnification_threshold": 0.1,
        "source_centre": list(source_galaxy.point_0.centre),
    },
    "phases": phases,
    "key_timings": {
        "solver_solve_eager_s": phases.get("solver_solve_eager"),
        "via_tracer_from_s": phases.get("via_tracer_from"),
        "time_delays_from_s": phases.get("time_delays_from"),
        "n_positions_found": len(positions),
    },
}

json_path = results_dir / f"point_source_summary_v{al_version}.json"
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
fig.suptitle("Simulator Profiling: Point Source", fontsize=12, fontweight="bold")
ax.set_title(
    f"AutoLens v{al_version}  |  200×200 @ 0.05\"/px  |  {len(positions)} images found",
    fontsize=9,
)
ax.margins(x=0.22)
fig.tight_layout()
chart_path = results_dir / f"point_source_summary_v{al_version}.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to: {chart_path}")

timer.summary()
