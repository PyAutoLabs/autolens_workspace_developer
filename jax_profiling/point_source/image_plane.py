"""
JAX Profiling: Point-Source Likelihood — Image-Plane Chi-Squared
=================================================================

Profiles ``AnalysisPoint.log_likelihood_function`` for a lensed point-source
``PointDataset`` using the **image-plane** chi-squared
(``al.FitPositionsImagePairAll``).

Image-plane fitting solves for the model multiple-image positions in the
image plane via the ``PointSolver`` (which JIT-traces a triangle-refinement
loop), pairs each model image with the closest observed image, and computes
a chi-squared in image-plane coordinates.

Unlike the source-plane variant (see ``source_plane.py``), the full
image-plane pipeline IS JIT-traceable end-to-end because ``PointSolver``
threads ``xp=jnp`` through every step and ``FitPositionsImagePairAll``
constructs its model-data via JAX-friendly operations.

Pytree-native parameter inputs
------------------------------

This script uses ``af.ModelInstance`` as the JIT input via PyAutoFit's
opt-in pytree registration (``autofit.jax.register_model``, PRs #1220 /
#1221 / #1222).  The JIT'd closure consumes the registered instance
directly, mirroring the pattern in ``../imaging/mge.py``.

Three-tier numerical assertions
-------------------------------

1. **eager ≡ JIT**: numpy-path log-likelihood matches single-JIT result.
2. **JIT ≡ vmap**: every entry of the batched vmap output matches the
   single-JIT result.
3. **regression constant**: hardcoded
   ``EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE`` guards against silent drift in
   the underlying solver / chi-squared stack.  This depends on the seeded
   simulator (``noise_seed=1`` in ``simulators/point_source.py``) staying
   bit-stable.
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
import autolens as al
from autofit.jax import register_model as _register_model_pytrees


# ---------------------------------------------------------------------------
# Profiling helpers (mirrors imaging/mge.py and source_plane.py)
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
# Eager baseline — full FitPointDataset (image-plane chi-squared)
# ---------------------------------------------------------------------------

print("\n--- Eager FitPointDataset (image-plane) ---")

analysis_eager = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAll,
    use_jax=False,
)

with timer.section("fit_eager"):
    fit_eager = analysis_eager.fit_from(instance=instance)
    log_likelihood_ref = float(fit_eager.log_likelihood)
    figure_of_merit_ref = float(fit_eager.figure_of_merit)

n_eager_repeats = 10
with timer.section(f"eager_log_likelihood_x{n_eager_repeats}"):
    for _ in range(n_eager_repeats):
        analysis_eager.log_likelihood_function(instance=instance)
eager_per_call = timer.records[-1][1] / n_eager_repeats

print(f"  log_likelihood   = {log_likelihood_ref}")
print(f"  figure_of_merit  = {figure_of_merit_ref}")
print(f"  eager per-call   = {eager_per_call:.6f} s")


# ===================================================================
# PART B — Full-pipeline JIT
# ===================================================================

print("\n" + "=" * 70)
print("FULL-PIPELINE JIT (image-plane)")
print("=" * 70)

analysis_jax = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAll,
    use_jax=True,
)


def full_pipeline_from_params(params_tree):
    return analysis_jax.log_likelihood_function(instance=params_tree)


_, full_result = jit_profile(full_pipeline_from_params, "full_pipeline", params_tree)
full_pipeline_per_call = timer.records[-1][1] / 10
print(f"  full log_likelihood = {full_result}")


# ===================================================================
# PART C — vmap over the full pipeline
# ===================================================================

print("\n--- vmap batched evaluation ---")

batch_size = 3

batched_params = jax.tree_util.tree_map(
    lambda leaf: jnp.broadcast_to(leaf, (batch_size, *leaf.shape)),
    params_tree,
)

vmapped_full = jax.jit(jax.vmap(full_pipeline_from_params))

with timer.section("vmap_first_call"):
    result_vmap = vmapped_full(batched_params)
    block(result_vmap)

n_vmap_repeats = 10
with timer.section(f"vmap_steady_x{n_vmap_repeats}"):
    for _ in range(n_vmap_repeats):
        result_vmap = vmapped_full(batched_params)
        block(result_vmap)

vmap_batch_time = timer.records[-1][1] / n_vmap_repeats
vmap_per_call = vmap_batch_time / batch_size
vmap_speedup = full_pipeline_per_call / vmap_per_call

print(f"  batch results = {result_vmap}")
print(f"  vmap batch of {batch_size}:   {vmap_batch_time:.6f} s")
print(f"  vmap per call:         {vmap_per_call:.6f} s")
print(f"  single JIT per call:   {full_pipeline_per_call:.6f} s")
print(f"  vmap speedup:          {vmap_speedup:.1f}x faster per likelihood")


# ===================================================================
# PART D — Three-tier numerical assertions
# ===================================================================
#
# Tier 1: eager (NumPy path) ≡ single JIT
# Tier 2: single JIT ≡ every entry of vmap output
# Tier 3: hardcoded regression constant (deterministic via seeded simulator)

np.testing.assert_allclose(
    log_likelihood_ref,
    float(full_result),
    rtol=1e-4,
    err_msg=(
        f"point_source/image_plane: eager vs JIT mismatch — "
        f"eager={log_likelihood_ref} vs JIT={float(full_result)}"
    ),
)
print("  Tier 1: eager ≡ JIT assertion PASSED")

np.testing.assert_allclose(
    np.array(result_vmap),
    float(full_result),
    rtol=1e-4,
    err_msg="point_source/image_plane: JIT vs vmap mismatch",
)
print("  Tier 2: JIT ≡ vmap assertion PASSED")


# ===================================================================
# PART E — Static memory analysis
# ===================================================================

print("\n--- Static memory analysis ---")

lowered_batched = vmapped_full.lower(batched_params)
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
print(f"JAX LIKELIHOOD SUMMARY — POINT SOURCE IMAGE-PLANE — v{al_version}")
print("=" * 70)
print(f"  Dataset:                    {dataset_name}")
print(f"  Observed image positions:   {n_observed_positions}")
print(f"  Position noise sigma:       {positions_noise_sigma}")
print(f"  Free parameters:            {model.total_free_parameters}")
print(f"  fit_positions_cls:          FitPositionsImagePairAll (image-plane chi-squared)")
print("-" * 70)
print(f"  Eager full likelihood:      {eager_per_call:.6f} s/call  ({log_likelihood_ref:.6f})")
print(f"  Full pipeline (JIT):        {full_pipeline_per_call:.6f} s/call")
print(f"  vmap per-call (batch={batch_size}):    {vmap_per_call:.6f} s")
print(f"  vmap speedup vs single JIT:           {vmap_speedup:.1f}x")
print("=" * 70)

likelihood_summary = {
    "autolens_version": al_version,
    "dataset": dataset_name,
    "fit_positions_cls": "FitPositionsImagePairAll",
    "configuration": {
        "observed_image_positions": int(n_observed_positions),
        "positions_noise_sigma": positions_noise_sigma,
        "free_parameters": int(model.total_free_parameters),
    },
    "eager_per_call": eager_per_call,
    "eager_log_likelihood": log_likelihood_ref,
    "full_pipeline_single_jit": full_pipeline_per_call,
    "full_pipeline_log_likelihood": float(full_result),
    "vmap": {
        "batch_size": batch_size,
        "batch_time": vmap_batch_time,
        "per_call": vmap_per_call,
        "speedup_vs_single_jit": round(vmap_speedup, 1),
    },
}

results_dir = _script_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

dict_path = results_dir / f"image_plane_summary_v{al_version}.json"
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Bar chart ---

labels = [
    "Eager full likelihood",
    "Full pipeline (JIT)",
    f"vmap per-call (batch={batch_size})",
]
times = [eager_per_call, full_pipeline_per_call, vmap_per_call]
colors = ["#8172B3", "#C44E52", "#55A868"]

fig, ax = plt.subplots(figsize=(10, 4.0))
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
    "Point-Source Likelihood — Image-Plane Chi-Squared",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f"AutoLens v{al_version}  |  {n_observed_positions} positions  |  "
    f"{model.total_free_parameters} free params  |  "
    f"vmap speedup: {vmap_speedup:.1f}x",
    fontsize=9,
)
ax.margins(x=0.20)
fig.tight_layout()

chart_path = results_dir / f"image_plane_summary_v{al_version}.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")


# ===================================================================
# Tier 3: regression assertion — deterministic via seeded simulator
# ===================================================================
#
# Seeded simulator (noise_seed=1 in simulators/point_source.py) + prior-median
# parameter vector make the image-plane log-likelihood deterministic. Hardcoded
# value guards against silent regressions in the PointSolver / chi-squared stack.
EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE = 0.3936326580483207

np.testing.assert_allclose(
    log_likelihood_ref,
    EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE,
    rtol=1e-4,
    err_msg=(
        f"point_source/image_plane: regression — eager log_likelihood drifted "
        f"(got {log_likelihood_ref}, expected {EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE})"
    ),
)
print(
    f"  Eager regression assertion PASSED: log_likelihood matches "
    f"{EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE:.6f}"
)
np.testing.assert_allclose(
    float(full_result),
    EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE,
    rtol=1e-4,
    err_msg=(
        f"point_source/image_plane: regression — JIT log_likelihood drifted "
        f"(got {float(full_result)}, expected {EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE})"
    ),
)
np.testing.assert_allclose(
    np.array(result_vmap),
    EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE,
    rtol=1e-4,
    err_msg=(
        f"point_source/image_plane: regression — vmap log_likelihood drifted "
        f"(expected {EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE})"
    ),
)
print(
    f"  Tier 3: regression assertion PASSED: log_likelihood matches "
    f"{EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE:.6f}"
)

timer.summary()
