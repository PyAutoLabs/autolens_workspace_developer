"""
JAX Profiling: Point-Source Likelihood — Source-Plane Chi-Squared
==================================================================

Profiles ``AnalysisPoint.log_likelihood_function`` for a lensed point-source
``PointDataset`` using the **source-plane** chi-squared
(``al.FitPositionsSource``).

Source-plane fitting traces each *observed* image-plane position back to the
source plane via the lens model, then computes a chi-squared between the
ray-traced positions and the model source position.  No image-plane solver
is required.

Pytree-native parameter inputs
------------------------------

This script uses ``af.ModelInstance`` as the JIT input via PyAutoFit's opt-in
pytree registration (``autofit.jax.register_model``, PR #1220 / #1221 / #1222).
The JIT'd closure consumes the registered instance directly.
"""

import json
import time
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import autofit as af
import autoarray as aa
import autolens as al
from autofit.jax import register_model as _register_model_pytrees


# ---------------------------------------------------------------------------
# Profiling helpers (mirrors imaging/mge.py)
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
dataset_name = "simple"


# ===================================================================
# PART A — Setup
# ===================================================================

print(f"\n--- Dataset loading [{dataset_name}] ---")

_script_dir = Path(__file__).resolve().parent
dataset_path = (
    Path("jax_profiling") / "point_source" / "dataset" / "point_source" / dataset_name
)

if al.util.dataset.should_simulate(str(dataset_path)):
    print(f"  Simulating {dataset_name} dataset...")
    subprocess.run(
        [
            sys.executable,
            str(_script_dir / "simulators" / "point_source.py"),
            "--name",
            dataset_name,
        ],
        cwd=str(_script_dir),
        check=True,
    )

with timer.section("dataset_load"):
    dataset = al.from_json(
        file_path=dataset_path / "point_dataset_positions_only.json",
    )

n_observed_positions = dataset.positions.shape[0]
positions_noise_sigma = float(dataset.positions_noise_map[0])

print("\n--- Point solver ---")

with timer.section("solver_build"):
    grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.2)
    solver = al.PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.001,
        magnification_threshold=0.1,
        xp=jnp,
    )

print("\n--- Model construction ---")

with timer.section("model_build"):
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
    mass.centre.centre_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
    mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
    mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
    mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.8)
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    point_0 = af.Model(al.ps.PointFlux)
    point_0.centre.centre_0 = af.UniformPrior(lower_limit=0.06, upper_limit=0.08)
    point_0.centre.centre_1 = af.UniformPrior(lower_limit=0.06, upper_limit=0.08)
    source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")

print("\n--- Instantiate concrete model ---")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

with timer.section("register_pytrees"):
    _register_model_pytrees(model)

params_tree = jax.tree_util.tree_map(jnp.asarray, instance)


# ---------------------------------------------------------------------------
# Eager baseline — full FitPointDataset (source-plane chi-squared)
# ---------------------------------------------------------------------------

print("\n--- Eager FitPointDataset (source-plane) ---")

analysis_eager = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSource,
    use_jax=False,
)

with timer.section("fit_eager"):
    fit_eager = analysis_eager.fit_from(instance=instance)
    log_likelihood_ref = float(fit_eager.log_likelihood)
    figure_of_merit_ref = float(fit_eager.figure_of_merit)

n_eager_repeats = 100
with timer.section(f"eager_log_likelihood_x{n_eager_repeats}"):
    for _ in range(n_eager_repeats):
        analysis_eager.log_likelihood_function(instance=instance)
eager_per_call = timer.records[-1][1] / n_eager_repeats

print(f"  log_likelihood   = {log_likelihood_ref}")
print(f"  figure_of_merit  = {figure_of_merit_ref}")
print(f"  eager per-call   = {eager_per_call:.6f} s")


# ===================================================================
# PART B — Full-pipeline JIT (expected to fail — see module docstring)
# ===================================================================

print("\n" + "=" * 70)
print("FULL-PIPELINE JIT (source-plane)")
print("=" * 70)

analysis_jax = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSource,
    use_jax=True,
)


def full_pipeline_from_params(params_tree):
    return analysis_jax.log_likelihood_function(instance=params_tree)


full_pipeline_jits = False
full_pipeline_per_call = None
full_result = None
full_pipeline_blocker = None

try:
    _, full_result = jit_profile(
        full_pipeline_from_params, "full_pipeline", params_tree
    )
    full_pipeline_per_call = timer.records[-1][1] / 10
    full_pipeline_jits = True
    print(f"  full log_likelihood = {full_result}")
except jax.errors.TracerArrayConversionError as e:
    full_pipeline_blocker = (
        "Grid2DIrregular.grid_2d_via_deflection_grid_from does not propagate xp; "
        "model_data ends up with _xp=np while holding JAX tracers, so "
        "squared_distances_to_coordinate_from calls np.square on a tracer."
    )
    print(
        "\n  >>> BLOCKER: full-pipeline source-plane likelihood does NOT JIT.\n"
        f"  >>> Cause:   {full_pipeline_blocker}\n"
        "  >>> See module docstring for the proposed library fix."
    )


# ===================================================================
# PART C — JIT-able prefix: tracer ray-trace of observed positions
# ===================================================================
#
# Even though the full pipeline is blocked, the dominant work in the
# source-plane likelihood — ray-tracing the observed image positions to
# the source plane via the tracer's deflection field — IS JIT-traceable
# when the input/output stay as raw arrays.  We profile that prefix here
# so the JIT-able portion of the source-plane path is still measured.

print("\n" + "=" * 70)
print("JIT-ABLE PREFIX: ray-trace observed positions to source plane")
print("=" * 70)

observed_positions_raw = jnp.array(dataset.positions.array)


def ray_trace_to_source_plane(params_tree, positions_raw):
    """Ray-trace observed image positions to the source plane (raw arrays)."""
    tracer = al.Tracer(galaxies=list(params_tree.galaxies))
    grid_in = aa.Grid2DIrregular(values=positions_raw, xp=jnp)
    deflections = tracer.deflections_yx_2d_from(grid=grid_in, xp=jnp)
    # Source-plane positions = observed - deflections.
    return positions_raw - deflections.array


_, source_plane_positions = jit_profile(
    ray_trace_to_source_plane,
    "raytrace_prefix",
    params_tree,
    observed_positions_raw,
)
prefix_per_call = timer.records[-1][1] / 10

print(f"  source-plane positions shape: {source_plane_positions.shape}")
print(f"  source-plane positions value: {np.array(source_plane_positions)}")


# ===================================================================
# PART D — vmap over the JIT-able prefix
# ===================================================================

print("\n--- vmap over ray-trace prefix ---")

batch_size = 3

batched_params = jax.tree_util.tree_map(
    lambda leaf: jnp.broadcast_to(leaf, (batch_size, *leaf.shape)),
    params_tree,
)
batched_positions = jnp.broadcast_to(
    observed_positions_raw, (batch_size, *observed_positions_raw.shape)
)

vmapped_prefix = jax.jit(jax.vmap(ray_trace_to_source_plane))

with timer.section("vmap_prefix_first_call"):
    result_vmap = vmapped_prefix(batched_params, batched_positions)
    block(result_vmap)

n_vmap_repeats = 10
with timer.section(f"vmap_prefix_steady_x{n_vmap_repeats}"):
    for _ in range(n_vmap_repeats):
        result_vmap = vmapped_prefix(batched_params, batched_positions)
        block(result_vmap)

vmap_batch_time = timer.records[-1][1] / n_vmap_repeats
vmap_per_call = vmap_batch_time / batch_size
vmap_speedup = prefix_per_call / vmap_per_call

print(f"  vmap batch={batch_size}: {vmap_batch_time:.6f} s")
print(f"  vmap per call:         {vmap_per_call:.6f} s")
print(f"  single JIT per call:   {prefix_per_call:.6f} s")
print(f"  vmap speedup:          {vmap_speedup:.1f}x faster per ray-trace")

# Eager ray-trace truth — compare against vmap output to lock the prefix.
eager_grid = aa.Grid2DIrregular(values=np.array(observed_positions_raw))
eager_deflections = al.Tracer(galaxies=list(instance.galaxies)).deflections_yx_2d_from(
    grid=eager_grid, xp=np
)
eager_source_positions = np.array(observed_positions_raw) - eager_deflections.array

np.testing.assert_allclose(
    np.array(source_plane_positions),
    eager_source_positions,
    rtol=1e-4,
    err_msg="point_source/source_plane: JIT ray-trace prefix mismatch with eager NumPy",
)
np.testing.assert_allclose(
    np.array(result_vmap),
    eager_source_positions[None, :, :].repeat(batch_size, axis=0),
    rtol=1e-4,
    err_msg="point_source/source_plane: vmap ray-trace prefix mismatch with eager NumPy",
)
print("  Eager vs JIT vs vmap (prefix) assertion PASSED")


# ===================================================================
# PART E — Static memory analysis (JIT-able prefix)
# ===================================================================

print("\n--- Static memory analysis (JIT-able prefix) ---")

lowered_batched = vmapped_prefix.lower(batched_params, batched_positions)
compiled_batched = lowered_batched.compile()
mem = compiled_batched.memory_analysis()
print(f"  Output size:  {mem.output_size_in_bytes / 1024**2:.3f} MB")
print(f"  Temp size:    {mem.temp_size_in_bytes / 1024**2:.3f} MB")
print(
    f"  Total:        "
    f"{(mem.output_size_in_bytes + mem.temp_size_in_bytes) / 1024**2:.3f} MB"
)


# ===================================================================
# Summary + outputs
# ===================================================================

al_version = al.__version__

print("\n" + "=" * 70)
print(f"JAX LIKELIHOOD SUMMARY — POINT SOURCE SOURCE-PLANE — v{al_version}")
print("=" * 70)
print(f"  Dataset:                    {dataset_name}")
print(f"  Observed image positions:   {n_observed_positions}")
print(f"  Position noise sigma:       {positions_noise_sigma}")
print(f"  Free parameters:            {model.total_free_parameters}")
print(f"  fit_positions_cls:          FitPositionsSource (source-plane chi-squared)")
print("-" * 70)
print(f"  Eager full likelihood:      {eager_per_call:.6f} s/call  ({log_likelihood_ref:.6f})")
if full_pipeline_jits:
    print(f"  Full pipeline (JIT):        {full_pipeline_per_call:.6f} s/call")
else:
    print("  Full pipeline (JIT):        BLOCKED (see module docstring)")
print(f"  JIT-able prefix (raytrace): {prefix_per_call:.6f} s/call")
print(f"  vmap prefix per-call (b={batch_size}): {vmap_per_call:.6f} s")
print(f"  vmap speedup vs single JIT prefix: {vmap_speedup:.1f}x")
print("=" * 70)

likelihood_summary = {
    "autolens_version": al_version,
    "dataset": dataset_name,
    "fit_positions_cls": "FitPositionsSource",
    "configuration": {
        "observed_image_positions": int(n_observed_positions),
        "positions_noise_sigma": positions_noise_sigma,
        "free_parameters": int(model.total_free_parameters),
    },
    "eager_per_call": eager_per_call,
    "eager_log_likelihood": log_likelihood_ref,
    "full_pipeline_jits": full_pipeline_jits,
    "full_pipeline_blocker": full_pipeline_blocker,
    "full_pipeline_single_jit": full_pipeline_per_call,
    "jit_able_prefix": {
        "name": "ray-trace observed positions to source plane",
        "per_call": prefix_per_call,
    },
    "vmap_prefix": {
        "batch_size": batch_size,
        "batch_time": vmap_batch_time,
        "per_call": vmap_per_call,
        "speedup_vs_single_jit_prefix": round(vmap_speedup, 1),
    },
}

results_dir = _script_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

dict_path = results_dir / f"source_plane_summary_v{al_version}.json"
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Bar chart ---

labels = [
    "Eager full likelihood",
    "JIT-able prefix (raytrace)",
    f"vmap prefix per-call (batch={batch_size})",
]
times = [eager_per_call, prefix_per_call, vmap_per_call]
colors = ["#8172B3", "#4C72B0", "#55A868"]
if full_pipeline_jits:
    labels.insert(1, "Full pipeline (JIT)")
    times.insert(1, full_pipeline_per_call)
    colors.insert(1, "#C44E52")

fig, ax = plt.subplots(figsize=(10, 4.5))
y_pos = range(len(labels))
bars = ax.barh(y_pos, times, color=colors, edgecolor="white", height=0.6)

for bar, t in zip(bars, times):
    ax.text(
        bar.get_width() + max(times) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.6f} s",
        va="center",
        fontsize=9,
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Time per call (s)", fontsize=11)
fig.suptitle(
    "Point-Source Likelihood — Source-Plane Chi-Squared",
    fontsize=12,
    fontweight="bold",
)
title_extra = (
    " | full pipeline JIT BLOCKED" if not full_pipeline_jits else ""
)
ax.set_title(
    f"AutoLens v{al_version}  |  {n_observed_positions} positions  |  "
    f"{model.total_free_parameters} free params{title_extra}",
    fontsize=9,
)
ax.margins(x=0.20)
fig.tight_layout()

chart_path = results_dir / f"source_plane_summary_v{al_version}.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")


# ===================================================================
# Regression assertions (eager and full-pipeline JIT)
# ===================================================================
#
# Seeded simulator (noise_seed=1 in simulators/point_source.py) + prior-median
# parameter vector make the source-plane log-likelihood deterministic. Both
# the eager numpy and the full-pipeline JIT paths now agree to float64
# precision, following the Richardson-extrapolation fix to
# LensCalc.hessian_from in PyAutoGalaxy (PR #358).
EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE = -4491.83220547254

np.testing.assert_allclose(
    log_likelihood_ref,
    EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE,
    rtol=1e-4,
    err_msg=(
        f"point_source/source_plane: regression — eager log_likelihood drifted "
        f"(got {log_likelihood_ref}, expected {EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE})"
    ),
)
print(
    f"  Eager regression assertion PASSED: log_likelihood matches "
    f"{EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE:.6f}"
)

if full_pipeline_jits:
    np.testing.assert_allclose(
        float(full_result),
        EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE,
        rtol=1e-4,
        err_msg=(
            f"point_source/source_plane: regression — JIT log_likelihood drifted "
            f"(got {float(full_result)}, expected {EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE})"
        ),
    )
    print(
        f"  JIT regression assertion PASSED: log_likelihood matches "
        f"{EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE:.6f}"
    )

timer.summary()
