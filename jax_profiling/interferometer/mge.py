"""
JAX Profiling: MGE Interferometer Likelihood
=============================================

Profiles the JAX likelihood function for an interferometer dataset where the
source galaxy's light is modelled with a multi-Gaussian expansion (MGE) and
the lens galaxy is an Isothermal + ExternalShear.

Unlike the imaging MGE profiling script, this script deliberately does not
per-step JIT the inversion pipeline. The interferometer path exercises a
Fourier-transformed mapping matrix, a visibilities-space data vector /
curvature matrix, and an NNLS solve whose ``xp=jnp`` threading has not been
fully characterised. Per-step decomposition risks missing cross-step XLA
fusion that matters in practice, and risks hitting library-level JAX
blockers that we would want to raise as separate issues rather than work
around here. Once the full-pipeline JIT is stable on interferometer, the
per-step breakdown can land as a follow-up.

Instead, this script measures:

1. Eager baseline: ``FitInterferometer`` with ``xp=np``, print
   ``figure_of_merit`` / ``log_likelihood``.
2. Full-pipeline JIT: ``jax.jit(analysis.log_likelihood_function)`` on a
   pytree-registered ``ModelInstance``. Measure lower / compile / first-call /
   steady-state per-call.
3. Batched evaluation: ``jax.jit(jax.vmap(full_pipeline))``. Measure
   per-likelihood cost and speedup vs the single-JIT path.
4. Correctness: eager vs JIT log-likelihood agreement at ``rtol=1e-4``.
5. Static memory analysis of the batched program.
6. Results JSON + PNG written to ``results/`` using the same schema as the
   imaging profiling scripts so they can be compared side-by-side.

Pytree-native parameter inputs
------------------------------

Uses ``af.ModelInstance`` as the JIT input via PyAutoFit's opt-in pytree
registration (``autofit.jax.register_model``). This matches the pattern in
``jax_profiling/imaging/mge.py`` and exercises the ``TuplePrior`` pytree
support landed in PyAutoFit#1222.
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
import subprocess
import sys
from pathlib import Path
from contextlib import contextmanager

import autofit as af
import autolens as al
from autofit.jax import register_model as _register_model_pytrees

# ---------------------------------------------------------------------------
# Instrument configuration
# ---------------------------------------------------------------------------

INSTRUMENTS = {
    "sma": {"pixel_scale": 0.1, "real_space_shape": (256, 256)},
    "alma": {"pixel_scale": 0.05, "real_space_shape": (256, 256)},
}

instrument = "sma"  # <-- change this to profile a different instrument


# ---------------------------------------------------------------------------
# Profiling helpers (copied verbatim from imaging/mge.py)
# ---------------------------------------------------------------------------

class Timer:
    """Accumulates named timing measurements and prints a summary."""

    def __init__(self):
        self.records: list[tuple[str, float]] = []

    @contextmanager
    def section(self, label: str):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.records.append((label, elapsed))
        print(f"  [{label}] {elapsed:.4f} s")


def block(x):
    """Call block_until_ready if available (JAX arrays)."""
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def jit_profile(func, label, *args, n_repeats=10):
    """JIT-compile *func*, time lower / compile / first call / steady state."""
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

# ===================================================================
# PART A — Setup (not JIT-compiled)
# ===================================================================

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

print(f"\n--- Dataset loading [{instrument}] ---")

_script_dir = Path(__file__).resolve().parent
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
real_space_shape = INSTRUMENTS[instrument]["real_space_shape"]
dataset_path = Path("jax_profiling") / "interferometer" / "dataset" / "interferometer" / instrument

if al.util.dataset.should_simulate(str(dataset_path)):
    print(f"  Simulating {instrument} dataset...")
    subprocess.run(
        [
            sys.executable,
            str(_script_dir / "simulators" / "interferometer.py"),
            "--instrument", instrument,
        ],
        cwd=str(_script_dir),
        check=True,
    )

mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=real_space_shape,
    pixel_scales=pixel_scale,
    radius=mask_radius,
)

with timer.section("dataset_load"):
    dataset = al.Interferometer.from_fits(
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerDFT,
    )

n_visibilities = dataset.uv_wavelengths.shape[0]
print(f"  Total visibilities: {n_visibilities}")

# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

with timer.section("model_build"):
    mass = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")

# ---------------------------------------------------------------------------
# 3. Instantiate concrete objects from prior medians
# ---------------------------------------------------------------------------

print("\n--- Instantiate concrete model ---")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

with timer.section("register_pytrees"):
    _register_model_pytrees(model)

# JIT input: the instance itself, with all parameter leaves promoted to JAX
# arrays. The eager NumPy instance is retained for the eager FitInterferometer
# baseline below.
params_tree = jax.tree_util.tree_map(jnp.asarray, instance)

tracer = al.Tracer(galaxies=list(instance.galaxies))

print(f"  Tracer planes: {tracer.total_planes}")

# ---------------------------------------------------------------------------
# 4. Configuration summary
# ---------------------------------------------------------------------------

from autogalaxy.profiles.basis import Basis as _Basis
_basis_list = [b for g in instance.galaxies for b in g.cls_list_from(cls=_Basis)]
n_linear_gaussians = sum(len(b.profile_list) for b in _basis_list)

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {instrument}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Real-space mask radius:  {mask_radius} arcsec")
print(f"  Real-space grid shape:   {real_space_shape[0]} x {real_space_shape[1]}")
print(f"  Visibilities:            {n_visibilities}")
print(f"  Linear Gaussians:        {n_linear_gaussians}")

# ---------------------------------------------------------------------------
# 5. Full-pipeline reference (FitInterferometer) — eager baseline
# ---------------------------------------------------------------------------

print("\n--- Full FitInterferometer (eager baseline) ---")

with timer.section("fit_interferometer_eager"):
    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        xp=np,
    )
    figure_of_merit_ref = fit.figure_of_merit
    log_likelihood_ref = fit.log_likelihood

print(f"  figure_of_merit = {figure_of_merit_ref}")
print(f"  log_likelihood  = {log_likelihood_ref}")


# ===================================================================
# PART B — Full-pipeline JIT
# ===================================================================

print("\n" + "=" * 70)
print("FULL-PIPELINE JIT")
print("=" * 70)

analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=True)

def full_pipeline_from_params(params_tree):
    """Full interferometer likelihood from a pytree-shaped ``ModelInstance``.

    No flat-vector unpacking inside the trace — the instance crosses the JIT
    boundary directly, with constants (redshifts, etc.) kept static via the
    ``aux_data`` partition set up by ``autofit.jax.register_model``.
    """
    return analysis.log_likelihood_function(instance=params_tree)

_, full_result = jit_profile(full_pipeline_from_params, "full_pipeline", params_tree)
full_pipeline_per_call = timer.records[-1][1] / 10

print(f"  full log_likelihood = {full_result}")

# ===================================================================
# PART C — vmap + correctness
# ===================================================================

print("\n--- vmap batched evaluation ---")

batch_size = 3

parameters = jax.tree_util.tree_map(
    lambda leaf: jnp.broadcast_to(leaf, (batch_size, *leaf.shape)),
    params_tree,
)

vmapped_full = jax.jit(jax.vmap(full_pipeline_from_params))

with timer.section("vmap_first_call"):
    result_vmap = vmapped_full(parameters)
    block(result_vmap)

n_vmap_repeats = 10
with timer.section(f"vmap_steady_x{n_vmap_repeats}"):
    for _ in range(n_vmap_repeats):
        result_vmap = vmapped_full(parameters)
        block(result_vmap)

vmap_batch_time = timer.records[-1][1] / n_vmap_repeats
vmap_per_call = vmap_batch_time / batch_size
vmap_speedup = full_pipeline_per_call / vmap_per_call

print(f"  batch results = {result_vmap}")
print(f"  vmap batch of {batch_size}:   {vmap_batch_time:.6f} s")
print(f"  vmap per call:         {vmap_per_call:.6f} s")
print(f"  single JIT per call:   {full_pipeline_per_call:.6f} s")
print(f"  vmap speedup:          {vmap_speedup:.1f}x faster per likelihood")

# Correctness: full-pipeline JIT must match eager FitInterferometer.log_likelihood
# (NOT figure_of_merit — the analysis.log_likelihood_function returns the
# log-likelihood scalar directly, whereas FitInterferometer.figure_of_merit may
# be a log-evidence when an inversion is present).
np.testing.assert_allclose(
    float(full_result),
    float(log_likelihood_ref),
    rtol=1e-4,
    err_msg="interferometer/mge: JIT log_likelihood does not match eager FitInterferometer",
)
print("  Eager-vs-JIT correctness PASSED")

np.testing.assert_allclose(
    np.array(result_vmap),
    float(full_result),
    rtol=1e-4,
    err_msg="interferometer/mge: JAX vmap likelihood mismatch",
)
print("  vmap-vs-single-JIT correctness PASSED")

# ===================================================================
# PART D — Static memory analysis
# ===================================================================

print("\n--- Static memory analysis ---")

lowered_batched = vmapped_full.lower(parameters)
compiled_batched = lowered_batched.compile()

memory_analysis = compiled_batched.memory_analysis()
print(f"  Output size:  {memory_analysis.output_size_in_bytes / 1024**2:.3f} MB")
print(f"  Temp size:    {memory_analysis.temp_size_in_bytes / 1024**2:.3f} MB")
print(
    f"  Total:        "
    f"{(memory_analysis.output_size_in_bytes + memory_analysis.temp_size_in_bytes) / 1024**2:.3f} MB"
)

# ===================================================================
# JAX Likelihood Function Summary + artefacts
# ===================================================================

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

print("\n" + "=" * 70)
print(f"JAX LIKELIHOOD FUNCTION SUMMARY — {instrument.upper()} — v{al_version}")
print("=" * 70)
print(f"  Instrument:              {instrument}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Real-space mask radius:  {mask_radius} arcsec")
print(f"  Real-space grid shape:   {real_space_shape[0]} x {real_space_shape[1]}")
print(f"  Visibilities:            {n_visibilities}")
print(f"  Linear Gaussians:        {n_linear_gaussians}")
print("-" * 70)
print(f"  Eager log_likelihood:    {log_likelihood_ref}")
print(f"  JIT  log_likelihood:     {float(full_result)}")
print("-" * 70)
print(f"  Full pipeline per call:  {full_pipeline_per_call:.6f} s")
print(f"  vmap batch={batch_size} per call:   {vmap_per_call:.6f} s")
print(f"  vmap speedup:            {vmap_speedup:.1f}x")
print("=" * 70)

# --- Save results dictionary ---

likelihood_summary = {
    "autolens_version": al_version,
    "instrument": instrument,
    "model": "mge",
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "real_space_shape": list(real_space_shape),
        "visibilities": int(n_visibilities),
        "linear_gaussians": int(n_linear_gaussians),
    },
    "log_likelihood_eager": float(log_likelihood_ref),
    "log_likelihood_jit": float(full_result),
    "full_pipeline_single_jit": full_pipeline_per_call,
    "vmap": {
        "batch_size": batch_size,
        "batch_time": vmap_batch_time,
        "per_call": vmap_per_call,
        "speedup_vs_single_jit": round(vmap_speedup, 1),
    },
    "memory_mb": {
        "output": memory_analysis.output_size_in_bytes / 1024**2,
        "temp": memory_analysis.temp_size_in_bytes / 1024**2,
    },
}

results_dir = _script_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

dict_path = results_dir / f"mge_likelihood_summary_{instrument}_v{al_version}.json"
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart ---

labels = [
    f"Full pipeline (single JIT)",
    f"vmap batch={batch_size} (per call)",
]
times = [full_pipeline_per_call, vmap_per_call]

fig, ax = plt.subplots(figsize=(10, 3.5))
y_pos = range(len(labels))
bars = ax.barh(y_pos, times, color=["#4C72B0", "#55A868"], edgecolor="white", height=0.55)

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
    f"MGE Interferometer Likelihood — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f"AutoLens v{al_version}  |  {pixel_scale}\"/px  |  "
    f"{real_space_shape[0]}x{real_space_shape[1]} real-space  |  "
    f"{n_visibilities} visibilities  |  {n_linear_gaussians} Gaussians  |  "
    f"vmap speedup: {vmap_speedup:.1f}x",
    fontsize=9,
)
ax.margins(x=0.2)
fig.tight_layout()

chart_path = results_dir / f"mge_likelihood_summary_{instrument}_v{al_version}.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")


# ===================================================================
# Regression assertion — realistic-scale deterministic log-likelihood
# ===================================================================
#
# Seeded simulator (noise_seed=1 in simulators/interferometer.py) + fixed
# SMA uv-coverage + fixed model parameters make the full-pipeline
# log-likelihood deterministic. Guards against regressions in the
# visibility transform / MGE inversion / chi-squared stack.
EXPECTED_LOG_LIKELIHOOD_SMA = -3154.8053574023816

np.testing.assert_allclose(
    log_likelihood_ref,
    EXPECTED_LOG_LIKELIHOOD_SMA,
    rtol=1e-4,
    err_msg=(
        f"interferometer/mge[{instrument}]: regression — eager log_likelihood drifted "
        f"(got {log_likelihood_ref}, expected {EXPECTED_LOG_LIKELIHOOD_SMA})"
    ),
)
print(
    f"  Eager regression assertion PASSED: log_likelihood matches "
    f"{EXPECTED_LOG_LIKELIHOOD_SMA:.6f}"
)
np.testing.assert_allclose(
    float(full_result),
    EXPECTED_LOG_LIKELIHOOD_SMA,
    rtol=1e-4,
    err_msg=f"interferometer/mge[{instrument}]: regression — full log_likelihood drifted",
)
np.testing.assert_allclose(
    np.array(result_vmap),
    EXPECTED_LOG_LIKELIHOOD_SMA,
    rtol=1e-4,
    err_msg=f"interferometer/mge[{instrument}]: regression — vmap log_likelihood drifted",
)
print(f"  Regression assertion PASSED: log_likelihood matches {EXPECTED_LOG_LIKELIHOOD_SMA:.6f}")
