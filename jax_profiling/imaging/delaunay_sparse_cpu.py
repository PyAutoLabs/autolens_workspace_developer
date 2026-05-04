"""
CPU/Numba Profiling: Delaunay Imaging Likelihood (Step-by-Step)
==================================================================

Companion to ``delaunay.py`` (JAX). Mirrors the structure of
``pixelization_sparse_cpu.py`` but for the Delaunay + ConstantSplit pixelisation
path used by the Euclid science pipeline (with ``apply_sparse_operator_cpu()``).

Two reference fits are built at the same evaluation point:

* ``fit_no_sparse`` — plain ``FitImaging(xp=np)`` on the un-augmented dataset.
  This is the gold reference; it MUST match ``EXPECTED_LOG_EVIDENCE_HST`` from
  ``delaunay.py`` to within ``rtol=1e-4`` (asserted at the bottom).
* ``fit_sparse``    — ``FitImaging(xp=np)`` on a dataset with an attached
  CPU sparse operator (``apply_sparse_operator_cpu()``). Its ``log_evidence``
  is also asserted to match ``fit_no_sparse`` (within ``rtol=1e-4``) so this
  script doubles as a CPU-sparse Delaunay regression test.

  The rectangular sister ``pixelization_sparse_cpu.py`` was added after
  PyAutoArray PR #296 fixed an out-of-bounds read in ``psf_precision_value_from``
  on the Rectangular path. The Delaunay numba kernels in
  ``autoarray/inversion/inversion/imaging_numba/sparse.py`` and
  ``autoarray/inversion/mappers/mapper_numba_util.py`` are *not* exercised by
  the rectangular profile; this script is the parity check for them.

Per-step timings are eager-only: numba functions are compiled on first call
and reused on subsequent calls, so each section reports a first-call cost
(includes numba compile) plus a steady-state average across 10 repeats.
"""

import numpy as np
import time
import subprocess
import sys
from pathlib import Path
from contextlib import contextmanager

import autofit as af
import autolens as al

# ---------------------------------------------------------------------------
# Instrument configuration
# ---------------------------------------------------------------------------

INSTRUMENTS = {
    "euclid": {"pixel_scale": 0.1},
    "hst": {"pixel_scale": 0.05},
    "jwst": {"pixel_scale": 0.03},
    "ao": {"pixel_scale": 0.01},
}

instrument = "hst"  # <-- change this to profile a different instrument


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------

class Timer:
    """Accumulates named timing measurements and prints a summary."""

    def __init__(self):
        self.records: list[tuple[str, float]] = []

    @contextmanager
    def section(self, label: str):
        """Context manager that records wall-clock time for *label*."""
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
    """No-op for numpy arrays (matches the JAX script's helper)."""
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def eager_profile(label, build_fn, n_repeats=10):
    """Time *build_fn* once (first call) then *n_repeats* times (steady state).

    *build_fn* is a zero-arg closure that performs all setup needed so the
    timing measures the full chain end-to-end. This mirrors the role
    ``jit_profile`` plays in the JAX script: each step gets a first-call cost
    (which on the numba path includes the one-time numba compile) plus a
    steady-state per-call average.
    """
    with timer.section(f"{label}_first_call"):
        result = build_fn()
        block(result)

    with timer.section(f"{label}_steady_x{n_repeats}"):
        for _ in range(n_repeats):
            result = build_fn()
            block(result)

    per_call = timer.records[-1][1] / n_repeats
    print(f"    -> per-call avg: {per_call:.6f} s")
    return result


timer = Timer()
likelihood_steps = []  # (label, per_call_seconds) for the final summary

# ===================================================================
# PART A — Setup (matches delaunay.py exactly)
# ===================================================================

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

print(f"\n--- Dataset loading & masking [{instrument}] ---")

_script_dir = Path(__file__).resolve().parent
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
dataset_path = Path("jax_profiling") / "imaging" / "dataset" / "imaging" / instrument

if al.util.dataset.should_simulate(str(dataset_path)):
    print(f"  Simulating {instrument} dataset...")
    subprocess.run(
        [
            sys.executable,
            str(_script_dir / "simulators" / "imaging.py"),
            "--instrument", instrument,
        ],
        cwd=str(_script_dir),
        check=True,
    )

with timer.section("dataset_load"):
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=pixel_scale,
    )

with timer.section("mask_and_oversample"):
    mask_radius = 3.5

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=4,
        over_sample_size_pixelization=1,
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
# 2. Image mesh + edge points (Delaunay-specific)
# ---------------------------------------------------------------------------

print("\n--- Image mesh construction (Delaunay) ---")

overlay_shape = (26, 26)
edge_n_points = 30

with timer.section("image_mesh_overlay"):
    image_mesh = al.image_mesh.Overlay(shape=overlay_shape)
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=dataset.mask)

with timer.section("edge_points"):
    edge_pixels_total = image_plane_mesh_grid.shape[0]
    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=(0.0, 0.0),
        radius=mask_radius,
        n_points=edge_n_points,
    )
    edge_pixels_total = image_plane_mesh_grid.shape[0] - edge_pixels_total

n_mesh_vertices = image_plane_mesh_grid.shape[0]
print(f"  Overlay shape: {overlay_shape}")
print(f"  Mesh vertices (incl. edge): {n_mesh_vertices}")
print(f"  Edge points added: {edge_pixels_total}")

# ---------------------------------------------------------------------------
# 3. Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

with timer.section("model_build"):
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
        al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear
    )

    mesh = al.mesh.Delaunay(
        pixels=n_mesh_vertices,
        zeroed_pixels=edge_pixels_total,
    )
    regularization = al.reg.ConstantSplit(coefficient=1.0)
    pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Delaunay pixels: {n_mesh_vertices}")
print(f"  Zeroed edge pixels: {edge_pixels_total}")

# ---------------------------------------------------------------------------
# 4. Instantiate concrete objects from prior medians
# ---------------------------------------------------------------------------

print("\n--- Instantiate concrete model ---")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

tracer = al.Tracer(galaxies=list(instance.galaxies))

# `FitImaging` receives the pre-built image_plane_mesh_grid through `adapt_images`.
# The Delaunay mesh model itself carries no image_mesh; `delaunay.py` constructs
# the mesh grid via `al.image_mesh.Overlay(...)` then bypasses FitImaging entirely.
# Here we re-enter the FitImaging path by attaching the same mesh grid to the
# source galaxy's adapt_images dict, matching the tutorial pattern in
# `autolens_workspace/scripts/imaging/features/pixelization/delaunay.py`.
adapt_images = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={instance.galaxies.source: image_plane_mesh_grid},
)

print(f"  Tracer planes: {tracer.total_planes}")

# ---------------------------------------------------------------------------
# 5. Configuration that dictates run time
# ---------------------------------------------------------------------------

n_image_pixels = dataset.data.shape[0]
n_over_sampled_pixels = dataset.grids.lp.over_sampled.shape[0]

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {instrument}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Mask radius:             {mask_radius} arcsec")
print(f"  Image pixels (masked):   {n_image_pixels}")
print(f"  Over-sampled pixels:     {n_over_sampled_pixels}")
print(f"  Overlay shape:           {overlay_shape}")
print(f"  Mesh vertices:           {n_mesh_vertices}")
print(f"  Edge pixels:             {edge_pixels_total}")


# ===================================================================
# PART B — Three-way log-evidence reference
# ===================================================================

print("\n" + "=" * 70)
print("THREE-WAY LOG-EVIDENCE REFERENCE")
print("=" * 70)

# --- Reference 1: non-sparse CPU path (gold reference) -----------------------

print("\n--- fit_no_sparse: FitImaging on un-augmented dataset (numpy) ---")

settings = al.Settings(use_border_relocator=True)

with timer.section("fit_no_sparse_build_and_eval"):
    fit_no_sparse = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=settings,
        xp=np,
    )
    log_evidence_no_sparse = fit_no_sparse.figure_of_merit
    log_likelihood_no_sparse = fit_no_sparse.log_likelihood

# Steady-state per-call timing for fit_no_sparse — the apples-to-apples
# counterpart to the JAX script's "Full pipeline (single JIT)" number.

def _build_fit_no_sparse_full():
    fit = al.FitImaging(
        dataset=dataset, tracer=tracer, adapt_images=adapt_images,
        settings=settings, xp=np,
    )
    return fit.figure_of_merit

eager_profile("fit_no_sparse_full_likelihood", _build_fit_no_sparse_full)
fit_no_sparse_per_call = timer.records[-1][1] / 10
print(
    f"  -> Numba CPU non-sparse full-likelihood per call: "
    f"{fit_no_sparse_per_call:.6f} s"
)

print(f"  log_evidence   = {log_evidence_no_sparse}")
print(f"  log_likelihood = {log_likelihood_no_sparse}")

# --- Reference 2: CPU sparse-operator path (the suspected-buggy one) ---------

print("\n--- Apply CPU sparse operator (precompute) ---")

with timer.section("apply_sparse_operator_cpu"):
    dataset_sparse = dataset.apply_sparse_operator_cpu()

print("\n--- fit_sparse: FitImaging on sparse-operator dataset (numpy) ---")

with timer.section("fit_sparse_build_and_eval"):
    fit_sparse = al.FitImaging(
        dataset=dataset_sparse,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=settings,
        xp=np,
    )
    log_evidence_sparse = fit_sparse.figure_of_merit
    log_likelihood_sparse = fit_sparse.log_likelihood

print(f"  log_evidence   = {log_evidence_sparse}")
print(f"  log_likelihood = {log_likelihood_sparse}")

# --- Reference 3: JAX gold value imported from delaunay.py -------------------
# Must stay in sync with delaunay.py's EXPECTED_LOG_EVIDENCE_HST.

EXPECTED_LOG_EVIDENCE_HST = 29179.9490711974

# --- Three-way comparison ----------------------------------------------------

print("\n--- Three-way comparison ---")
print(f"  log_evidence (CPU non-sparse)  = {log_evidence_no_sparse}")
print(f"  log_evidence (CPU sparse op)   = {log_evidence_sparse}")
print(f"  log_evidence (JAX expected)    = {EXPECTED_LOG_EVIDENCE_HST}")
print(f"  delta sparse - non_sparse      = "
      f"{log_evidence_sparse - log_evidence_no_sparse:+.6e}")
print(f"  delta sparse - JAX expected    = "
      f"{log_evidence_sparse - EXPECTED_LOG_EVIDENCE_HST:+.6e}")
print(f"  delta non_sparse - JAX expected= "
      f"{log_evidence_no_sparse - EXPECTED_LOG_EVIDENCE_HST:+.6e}")

# Whether the sparse and non-sparse paths agree: a quick boolean for the JSON.
sparse_vs_non_sparse_rtol = abs(
    (log_evidence_sparse - log_evidence_no_sparse) / log_evidence_no_sparse
)
sparse_matches_non_sparse = sparse_vs_non_sparse_rtol < 1e-4
print(
    f"  sparse matches non_sparse @ rtol=1e-4? "
    f"{'YES' if sparse_matches_non_sparse else 'NO'}  "
    f"(observed rtol = {sparse_vs_non_sparse_rtol:.3e})"
)


# ===================================================================
# PART C — Per-step eager profiling of the CPU sparse path
# ===================================================================

print("\n" + "=" * 70)
print("PER-STEP EAGER PROFILING (CPU sparse path)")
print("=" * 70)

# ---------------------------------------------------------------------------
# Step 1: Ray-trace grids
# ---------------------------------------------------------------------------

print("\n--- Step 1: Ray-trace grids ---")

grid_pix = dataset.grids.pixelization
grid_lp = dataset.grids.lp
grid_blurring = dataset.grids.blurring

def _ray_trace():
    return tracer.traced_grid_2d_list_from(grid=grid_pix, xp=np)

eager_profile("ray_trace", _ray_trace)
likelihood_steps.append(("Ray-trace grids", timer.records[-1][1] / 10))

# ---------------------------------------------------------------------------
# Step 2: Lens light images (pre-PSF)
# ---------------------------------------------------------------------------

print("\n--- Step 2: Lens light images (pre-PSF) ---")

def _lens_light():
    img = tracer.image_2d_from(grid=grid_lp, xp=np)
    blur = tracer.image_2d_from(grid=grid_blurring, xp=np)
    return img, blur

eager_profile("lens_light_images", _lens_light)
likelihood_steps.append(("Lens light images (pre-PSF)", timer.records[-1][1] / 10))

# ---------------------------------------------------------------------------
# Step 3: Blurred image (PSF convolution)
# ---------------------------------------------------------------------------

print("\n--- Step 3: Blurred image (PSF convolution) ---")

def _blurred_image():
    return tracer.blurred_image_2d_from(
        grid=grid_lp, psf=dataset.psf, blurring_grid=grid_blurring, xp=np,
    )

blurred_image = eager_profile("blurred_image", _blurred_image)
likelihood_steps.append(("Blurred image (PSF convolution)", timer.records[-1][1] / 10))

# ---------------------------------------------------------------------------
# Step 4: Profile-subtracted image
# ---------------------------------------------------------------------------

print("\n--- Step 4: Profile-subtracted image ---")

data_array = np.asarray(dataset.data.array)
blurred_array = np.asarray(blurred_image.array)

def _profile_subtract():
    return data_array - blurred_array

eager_profile("profile_subtract", _profile_subtract)
likelihood_steps.append(("Profile-subtracted image", timer.records[-1][1] / 10))

# ---------------------------------------------------------------------------
# Step 5: Inversion setup (border relocate -> mesh -> mapper -> sparse triplets)
# ---------------------------------------------------------------------------

print("\n--- Step 5: Inversion setup (border + mesh + mapper + preloads) ---")

def _build_fit_sparse():
    return al.FitImaging(
        dataset=dataset_sparse, tracer=tracer, adapt_images=adapt_images,
        settings=settings, xp=np,
    )

eager_profile("inversion_setup", _build_fit_sparse)
likelihood_steps.append(
    ("Inversion setup (steps 4-7 combined)", timer.records[-1][1] / 10)
)

# ---------------------------------------------------------------------------
# Step 6: Data vector D (sparse path)
# ---------------------------------------------------------------------------

print("\n--- Step 6: Data vector D (sparse path) ---")

def _data_vector():
    fit = al.FitImaging(
        dataset=dataset_sparse, tracer=tracer, adapt_images=adapt_images,
        settings=settings, xp=np,
    )
    return fit.inversion.data_vector

data_vector = eager_profile("data_vector", _data_vector)
likelihood_steps.append(("Data vector (D)", timer.records[-1][1] / 10))

print(f"  data_vector shape: {data_vector.shape}")

# ---------------------------------------------------------------------------
# Step 7: Curvature matrix F (sparse path)
# ---------------------------------------------------------------------------

print("\n--- Step 7: Curvature matrix F (sparse path) ---")

def _curvature_matrix():
    fit = al.FitImaging(
        dataset=dataset_sparse, tracer=tracer, adapt_images=adapt_images,
        settings=settings, xp=np,
    )
    return fit.inversion.curvature_matrix

curvature_matrix = eager_profile("curvature_matrix", _curvature_matrix)
likelihood_steps.append(("Curvature matrix (F)", timer.records[-1][1] / 10))

print(f"  curvature_matrix shape: {curvature_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 8: Regularization matrix H
# ---------------------------------------------------------------------------

print("\n--- Step 8: Regularization matrix H ---")

def _regularization_matrix():
    fit = al.FitImaging(
        dataset=dataset_sparse, tracer=tracer, adapt_images=adapt_images,
        settings=settings, xp=np,
    )
    return fit.inversion.regularization_matrix

regularization_matrix = eager_profile("regularization_matrix", _regularization_matrix)
likelihood_steps.append(("Regularization matrix (H)", timer.records[-1][1] / 10))

print(f"  regularization_matrix shape: {regularization_matrix.shape}")

# ---------------------------------------------------------------------------
# Step 9: Reconstruction s = (F + H)^{-1} D
# ---------------------------------------------------------------------------

print("\n--- Step 9: Reconstruction ---")

def _reconstruction():
    fit = al.FitImaging(
        dataset=dataset_sparse, tracer=tracer, adapt_images=adapt_images,
        settings=settings, xp=np,
    )
    return fit.inversion.reconstruction

reconstruction = eager_profile("reconstruction", _reconstruction)
likelihood_steps.append(("Regularized reconstruction", timer.records[-1][1] / 10))

print(f"  reconstruction shape: {reconstruction.shape}")

# ---------------------------------------------------------------------------
# Step 10: Mapped reconstruction + log evidence (full FitImaging.figure_of_merit)
# ---------------------------------------------------------------------------

print("\n--- Step 10: Mapped recon + log evidence (figure_of_merit) ---")

def _log_evidence():
    fit = al.FitImaging(
        dataset=dataset_sparse, tracer=tracer, adapt_images=adapt_images,
        settings=settings, xp=np,
    )
    return fit.figure_of_merit

log_evidence_step = eager_profile("log_evidence", _log_evidence)
likelihood_steps.append(("Mapped recon + log evidence", timer.records[-1][1] / 10))

print(f"  log_evidence (sparse) = {log_evidence_step}")


# ===================================================================
# PART D — Summary tables + JSON + bar chart
# ===================================================================

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

al_version = al.__version__

print("\n" + "=" * 70)
print(f"CPU SPARSE DELAUNAY LIKELIHOOD SUMMARY — {instrument.upper()} — v{al_version}")
print("=" * 70)
print(f"  Instrument:            {instrument}")
print(f"  Pixel scale:           {pixel_scale} arcsec/pixel")
print(f"  Mask radius:           {mask_radius} arcsec")
print(f"  Image pixels (masked): {n_image_pixels}")
print(f"  Over-sampled pixels:   {n_over_sampled_pixels}")
print(f"  Overlay shape:         {overlay_shape}")
print(f"  Mesh vertices:         {n_mesh_vertices}")
print(f"  Edge pixels:           {edge_pixels_total}")
print("-" * 70)

max_label = max(len(label) for label, _ in likelihood_steps)
step_total = 0.0
for i, (label, per_call) in enumerate(likelihood_steps, 1):
    print(f"  {i:>2}. {label:<{max_label}}  {per_call:>12.6f} s")
    step_total += per_call

print("-" * 70)
print(f"      {'TOTAL (step-by-step, sparse)':<{max_label}}  {step_total:>12.6f} s")
print(
    f"      {'Full likelihood, non-sparse (steady)':<{max_label}}  "
    f"{fit_no_sparse_per_call:>12.6f} s"
)
print("=" * 70)

print("\n--- Three-way log_evidence ---")
print(f"      {'CPU non-sparse':<28}  {log_evidence_no_sparse:>20.10f}")
print(f"      {'CPU sparse op':<28}  {log_evidence_sparse:>20.10f}")
print(f"      {'JAX expected':<28}  {EXPECTED_LOG_EVIDENCE_HST:>20.10f}")
print(f"      {'delta sparse - non_sparse':<28}  "
      f"{log_evidence_sparse - log_evidence_no_sparse:>+20.6e}")
print("=" * 70)

# --- Save results dictionary -------------------------------------------------

likelihood_summary = {
    "autolens_version": al_version,
    "instrument": instrument,
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "image_pixels_masked": int(n_image_pixels),
        "over_sampled_pixels": int(n_over_sampled_pixels),
        "overlay_shape": list(overlay_shape),
        "mesh_vertices": int(n_mesh_vertices),
        "edge_pixels": int(edge_pixels_total),
    },
    "steps": {label: per_call for label, per_call in likelihood_steps},
    "total_step_by_step": step_total,
    "fit_no_sparse_full_likelihood_per_call": fit_no_sparse_per_call,
    "log_evidence_no_sparse": float(log_evidence_no_sparse),
    "log_evidence_sparse": float(log_evidence_sparse),
    "log_evidence_jax_expected": EXPECTED_LOG_EVIDENCE_HST,
    "delta_sparse_minus_no_sparse": float(
        log_evidence_sparse - log_evidence_no_sparse
    ),
    "delta_sparse_minus_jax_expected": float(
        log_evidence_sparse - EXPECTED_LOG_EVIDENCE_HST
    ),
    "sparse_matches_non_sparse_rtol_1em4": bool(sparse_matches_non_sparse),
}

results_dir = _script_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

dict_path = (
    results_dir
    / f"delaunay_sparse_cpu_likelihood_summary_{instrument}_v{al_version}.json"
)
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart ----------------------------------------------------------

labels = [label for label, _ in likelihood_steps]
times = [per_call for _, per_call in likelihood_steps]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = range(len(labels))
bars = ax.barh(y_pos, times, color="#4C72B0", edgecolor="white", height=0.6)

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
    f"Delaunay CPU/Numba Sparse Likelihood — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f"AutoLens v{al_version}  |  {pixel_scale}\"/px  |  {n_image_pixels} pixels  |  "
    f"{n_over_sampled_pixels} over-sampled  |  {n_mesh_vertices} vertices  |  "
    f"total: {step_total:.6f} s",
    fontsize=9,
)
ax.margins(x=0.15)
fig.tight_layout()

chart_path = (
    results_dir
    / f"delaunay_sparse_cpu_likelihood_summary_{instrument}_v{al_version}.png"
)
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")


# ===================================================================
# PART E — Regression assertions
# ===================================================================
#
# Same dual-assertion structure as pixelization_sparse_cpu.py:
#  1. Non-sparse CPU must match JAX gold (catches numerical drift in either backend).
#  2. Sparse CPU must match non-sparse CPU (catches Delaunay sparse-CPU regressions
#     in autoarray/inversion/inversion/imaging_numba/sparse.py and
#     autoarray/inversion/mappers/mapper_numba_util.py).

np.testing.assert_allclose(
    log_evidence_no_sparse,
    EXPECTED_LOG_EVIDENCE_HST,
    rtol=1e-4,
    err_msg=(
        f"imaging/delaunay_sparse_cpu[{instrument}]: regression — CPU "
        f"non-sparse log_evidence drifted from JAX reference "
        f"(got {log_evidence_no_sparse}, expected {EXPECTED_LOG_EVIDENCE_HST})"
    ),
)
print(
    f"\n  Non-sparse regression assertion PASSED: "
    f"log_evidence matches JAX reference {EXPECTED_LOG_EVIDENCE_HST:.6f}"
)

np.testing.assert_allclose(
    log_evidence_sparse,
    log_evidence_no_sparse,
    rtol=1e-4,
    err_msg=(
        f"imaging/delaunay_sparse_cpu[{instrument}]: regression — CPU "
        f"sparse-operator Delaunay log_evidence diverges from non-sparse "
        f"(sparse={log_evidence_sparse}, non_sparse={log_evidence_no_sparse}, "
        f"observed rtol={sparse_vs_non_sparse_rtol:.3e}). Likely regression "
        f"in PyAutoArray Delaunay sparse-CPU numba kernels."
    ),
)
print(
    f"  Sparse regression assertion PASSED: "
    f"sparse log_evidence matches non-sparse (rtol = "
    f"{sparse_vs_non_sparse_rtol:.3e})."
)
