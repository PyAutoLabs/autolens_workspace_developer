"""
Can multi-start gradient MAP optimizers recover the mass basin with a
PIXELIZED source? (autolens_workspace_developer#100)
=====================================================================

The gradient is settled: ``probe_grad_pix.py`` on the A100 shows the SLaM-pix-1
objective is FD-matched to ~1e-6 on every mass/shear parameter with a kernel-CDF
mesh. The open question is the *search* one, and it is the same question the MGE
benchmark answered: from **broad** starts, can a multi-start gradient optimizer
find the truth basin — where a single cold start lands in the wrong one?

So this script deliberately uses the **default (broad) priors**, exactly as the
MGE benchmark did (``_setup.build_model``), NOT the truth-centred priors the FD
probe needs. Truth: einstein_radius = 1.6.

Model (SLaM SOURCE_PIX[1] style):
  - lens light : MGE **linear**, non-linear geometry **fixed** at truth
  - lens mass  : Isothermal + ExternalShear, **free**, broad default priors
  - source     : one of the three gradient-certified pixelized meshes,
                 selected by ``PIX_MESH`` (autolens_workspace_developer#117):

    - ``rectangular`` (default): ``RectangularAdaptDensity(bandwidth=0.1)`` +
      free ``reg.Constant`` — the kernel-CDF mesh (C^inf continuous-density
      transform, differentiable at os_pix=1). This is the #100/#101 objective;
      the class took the plain name in the rectangular-mesh consolidation
      (formerly ``RectangularKernelAdaptDensity``).
    - ``knn``: ``KNearestNeighbor(pixels=300, zeroed_pixels=30)`` + free
      ``reg.AdaptSplit`` — Wendland-C4 weights, pure JAX. The neighbour-based
      ``reg.Constant``/``reg.Adapt`` schemes call scipy Delaunay on the traced
      mesh grid and raise ``TracerArrayConversionError`` under grad, so the
      split-family regularization is mandatory (certified in
      autolens_workspace_test ``jax_grad/knn.py``).
    - ``delaunay``: ``Delaunay(pixels=300, zeroed_pixels=30)`` + free
      ``reg.AdaptSplit`` — a.e.-exact frozen-tables gradient (certified in
      ``jax_grad/delaunay.py``; the qhull tables callback is
      ``vmap_method="sequential"``, one host call per vmap lane).

    ``knn``/``delaunay`` are Delaunay-family meshes: their vertices come from a
    Hilbert image-mesh computed once from static adapt data (the observed
    image), so the analysis carries ``al.AdaptImages`` mirroring the certified
    harness.

Usage::

    python searches_minimal/pix_multi_start.py adam|adabelief|lion|nautilus [n_starts]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import jax

# float64 throughout: dropping to fp32 to dodge the OOM would be a science
# compromise. The memory is instead bounded by the search's `batch_size`
# (PyAutoFit#1373/#1374), which tiles the vmapped value_and_grad via
# jax.lax.map. Unbatched at n_starts=16 the jvp fusion is
# f64[16, 15361, 2, 31, 512] = 58 GiB and OOMs an 80 GB A100.
jax.config.update("jax_enable_x64", True)

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402

from searches_minimal._setup import build_dataset, MASK_RADIUS  # noqa: E402

TRUTH_ELL = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
TRUTH_EINSTEIN_RADIUS = 1.6
# Simulator truth for the mass model (jax_profiling/simulators/imaging.py):
# Isothermal(centre=(0,0), einstein_radius=1.6, ell_comps=from(0.9, 45)),
# ExternalShear(0.05, 0.05). Used by pix_prodigy.py's truth-bar mode.
TRUTH_MASS_ELL = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
TRUTH_SHEAR = (0.05, 0.05)
BASIN_TOL = 0.3  # same basin criterion as the MGE benchmark
OS_PIX = 1
MESH_SHAPE = (30, 30)
# Delaunay-family (knn/delaunay) mesh scale, mirroring the certified
# jax_grad harness: 300 Hilbert vertices + 30 zeroed circle-edge points.
MESH_PIXELS = 300
EDGE_PIXELS = 30
# Which gradient-certified pixelized mesh to fit: rectangular | knn | delaunay.
PIX_MESH = os.environ.get("PIX_MESH") or "rectangular"
# Regularization-scheme override (#117 Stage-3): None = the mesh's default
# scheme (Constant for rectangular, AdaptSplit for the Delaunay family);
# "matern" = al.reg.MaternKernel with free (coefficient, scale), nu pinned at
# 0.5 — the OTHER JAX-safe pairing for the Delaunay family (kernel regs build
# from pairwise distances, no scipy neighbors). Discriminates "free
# regularization breaks the search" from "AdaptSplit's lambda^4-fragile
# parametrization breaks the search".
PIX_REG = os.environ.get("PIX_REG") or None
# Kernel-CDF bandwidth for the rectangular mesh (#117 Stage-3 lever). The
# scripts' historical 0.1 is sharp — C-infinity on paper but with narrow
# gradient support; the library default 1.0 is the smooth landscape. The
# rectangular stall (both broad AND fixed-reg arms flat where other meshes'
# fixed-reg converged in ~150 steps) implicates exactly this knob.
PIX_BANDWIDTH = float(os.environ.get("PIX_BANDWIDTH") or 0.1)
# lr override for the #101 phase-2a lr sweep (e.g. PIX_LR=3e-3). None = the
# rule's benchmark default (adam/adabelief 1e-2, lion 1e-3).
PIX_LR = float(os.environ.get("PIX_LR") or 0) or None
# Fix the regularization coefficient (e.g. PIX_FIX_REG=1.0) instead of leaving
# it free — the #101 NaN-mortality probe: descent drives reg toward the
# non-positive-definite curvature regime where the JAX Cholesky NaN-resamples.
PIX_FIX_REG = float(os.environ.get("PIX_FIX_REG") or 0) or None
# Inversion evidence log-det method (PyAutoArray#392, autolens_workspace_developer#112).
# None/"cholesky" = default, byte-identical to prior behaviour (the baseline that
# NaN-walled in #100/#101). "slogdet" = the shipped gradient-safe opt-in: finite +
# differentiable where the Cholesky log-det NaN-resamples. The A/B is one env var
# apart, so both arms run from the same seeds and script.
PIX_LOGDET = os.environ.get("PIX_LOGDET") or None


def mesh_and_regularization():
    """The ``PIX_MESH``-selected mesh + regularization (model components).

    Regularization is free unless ``PIX_FIX_REG`` pins it. For the split-family
    schemes the fix scales the certified (inner=0.1, outer=10) pair by
    ``PIX_FIX_REG`` (so ``PIX_FIX_REG=1`` reproduces the certified config);
    ``signal_scale`` stays pinned at 1.0 either way, so the free reg is
    2-parameter (inner + outer), keeping the certified variants' surface.
    """
    def _matern():
        if PIX_FIX_REG:
            return al.reg.MaternKernel(
                coefficient=PIX_FIX_REG, scale=1.0, nu=0.5
            )
        regularization = af.Model(al.reg.MaternKernel)
        regularization.nu = 0.5
        return regularization

    if PIX_MESH == "rectangular":
        mesh = al.mesh.RectangularAdaptDensity(shape=MESH_SHAPE, bandwidth=PIX_BANDWIDTH)
        if PIX_REG == "matern":
            regularization = _matern()
        elif PIX_FIX_REG:
            regularization = al.reg.Constant(coefficient=PIX_FIX_REG)
        else:
            regularization = al.reg.Constant
    elif PIX_MESH in ("knn", "delaunay"):
        mesh_cls = al.mesh.KNearestNeighbor if PIX_MESH == "knn" else al.mesh.Delaunay
        mesh = mesh_cls(pixels=MESH_PIXELS, zeroed_pixels=EDGE_PIXELS)
        if PIX_REG == "matern":
            regularization = _matern()
        elif PIX_FIX_REG:
            regularization = al.reg.AdaptSplit(
                inner_coefficient=0.1 * PIX_FIX_REG,
                outer_coefficient=10.0 * PIX_FIX_REG,
                signal_scale=1.0,
            )
        else:
            regularization = af.Model(al.reg.AdaptSplit)
            regularization.signal_scale = 1.0
    else:
        raise ValueError(
            f"PIX_MESH={PIX_MESH!r} — expected rectangular | knn | delaunay"
        )
    return mesh, regularization


def build_model() -> af.Collection:
    """Broad-prior SLaM-pix-1 model (the search problem, not the FD probe)."""
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=MASK_RADIUS, total_gaussians=20, centre_prior_is_uniform=True
    )
    # Lens light: MGE amplitudes stay linear; non-linear geometry FIXED at truth.
    for profile in lens_bulge.profile_list:
        profile.centre = (0.0, 0.0)
        profile.ell_comps = TRUTH_ELL

    # Mass + shear: DEFAULT (broad) priors — the whole point of multi-start.
    mass = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)
    lens = af.Model(
        al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear
    )

    mesh, regularization = mesh_and_regularization()
    pixelization = af.Model(
        al.Pixelization,
        mesh=mesh,
        regularization=regularization,
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def build_adapt_images(dataset) -> al.AdaptImages | None:
    """AdaptImages for the Delaunay-family meshes (``knn``/``delaunay``).

    Mirrors the certified jax_grad harness: a Hilbert image-mesh computed once
    from the observed image as static adapt data, appended with a circle of
    edge points whose reconstruction values the mesh zeroes
    (``zeroed_pixels=EDGE_PIXELS``). Keys MUST be the stringified path tuple
    (``str(("galaxies", "source"))``) — a raw tuple key silently yields
    ``adapt_data=None`` downstream. The rectangular kernel-CDF mesh derives its
    density in-graph and needs no adapt images.
    """
    if PIX_MESH == "rectangular":
        return None

    galaxy_image_name_dict = {
        "('galaxies', 'lens')": dataset.data,
        "('galaxies', 'source')": dataset.data,
    }
    image_mesh = al.image_mesh.Hilbert(
        pixels=MESH_PIXELS, weight_power=3.5, weight_floor=0.01
    )
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask,
        adapt_data=galaxy_image_name_dict["('galaxies', 'source')"],
    )
    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=dataset.mask.mask_centre,
        radius=MASK_RADIUS + dataset.mask.pixel_scale / 2.0,
        n_points=EDGE_PIXELS,
    )
    return al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid
        },
    )


def build_analysis(dataset):
    dataset = dataset.apply_over_sampling(over_sample_size_pixelization=OS_PIX)
    # None -> AnalysisImaging's own default settings (cholesky); pass a Settings
    # object only to opt into slogdet, so the default arm stays byte-identical.
    settings = al.Settings(log_det_method=PIX_LOGDET) if PIX_LOGDET else None
    return al.AnalysisImaging(
        dataset=dataset,
        adapt_images=build_adapt_images(dataset),
        settings=settings,
        raise_inversion_positions_likelihood_exception=False,
        use_jax=True,
    )


SEARCHES = {
    "adam": lambda n, b: af.MultiStartAdam(
        n_starts=n, n_steps=300, learning_rate=PIX_LR or 1e-2, batch_size=b
    ),
    "adabelief": lambda n, b: af.MultiStartADABelief(
        n_starts=n, n_steps=300, learning_rate=PIX_LR or 1e-2, batch_size=b
    ),
    "lion": lambda n, b: af.MultiStartLion(
        n_starts=n, n_steps=300, learning_rate=PIX_LR or 1e-3, batch_size=b
    ),
    # Baseline. n_batch is Nautilus's OWN algorithmic knob (exposed as
    # search.batch_size) — how many points it proposes per iteration, i.e. the
    # likelihood batch, hence the VRAM driver. Without the sparse operator the
    # dense pix path costs ~1 GB/replica, so the default n_batch=100 would need
    # ~100 GB and OOM the 80 GB card; 16 is the value autolens_profiling's vram
    # table uses for pix at HST scale.
    #
    # force_x1_cpu / use_jax_vmap are MANDATORY on a JAX row: nautilus.Sampler
    # forks a multiprocessing pool otherwise, which corrupts JAX state
    # (autolens_profiling/searches/_samplers.py).
    "nautilus": lambda n, b: af.Nautilus(
        n_live=100,
        n_batch=b or 16,
        number_of_cores=1,
        force_x1_cpu=True,
        use_jax_vmap=True,
    ),
}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "adam"
    n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    # batch_size: simultaneous model evaluations = the VRAM driver.
    # 0/absent -> None (unbatched, the pre-#1374 behaviour).
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else None
    batch_size = batch_size or None

    print(f"JAX backend: {jax.default_backend()}  x64={jax.config.jax_enable_x64}")
    print(f"Sampler: {which}  n_starts={n_starts}  batch_size={batch_size}  lr={PIX_LR or 'default'}")
    mesh_desc = (
        f"RectangularAdaptDensity{MESH_SHAPE} bandwidth=0.1"
        if PIX_MESH == "rectangular"
        else f"{PIX_MESH} pixels={MESH_PIXELS} zeroed={EDGE_PIXELS} (Hilbert+AdaptImages)"
    )
    print(f"mesh={mesh_desc} os_pix={OS_PIX}  (no sparse operator)")
    print(f"log_det_method={PIX_LOGDET or 'cholesky (default)'}")

    dataset = build_dataset()
    analysis = build_analysis(dataset)
    model = build_model()
    print(f"Free (non-linear) parameters: {model.prior_count}")

    search = SEARCHES[which](n_starts, batch_size)

    # VRAM estimate for this batch (search.batch_size is the generic
    # "simultaneous model evaluations" contract). OFF by default: it triggers a
    # full vmapped compile of its own, which on a pixelized cell costs as much
    # as the fit (its docstring's "20-30 seconds" is an MGE-scale claim).
    if os.environ.get("PIX_VRAM") == "1":
        try:
            analysis.print_vram_use(model=model, batch_size=batch_size or n_starts)
        except Exception as exc:
            print(f"(vram estimate unavailable: {exc!r})")

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    wall = time.time() - t0

    instance = result.samples.max_log_likelihood()
    r_e = float(instance.galaxies.lens.mass.einstein_radius)
    logl = float(result.samples.max_log_likelihood_sample.log_likelihood)

    in_basin = abs(r_e - TRUTH_EINSTEIN_RADIUS) < BASIN_TOL
    print("\n================ RESULT ================")
    print(f"sampler            : {which}")
    print(f"wall_s             : {wall:.1f}")
    print(f"max log likelihood : {logl:.3f}")
    print(f"einstein_radius    : {r_e:.4f}   (truth {TRUTH_EINSTEIN_RADIUS})")
    print(f"IN TRUTH BASIN     : {in_basin}")

    # Per-start basin diagnostics (multi-start searches record every start's
    # final point as zero-weight diagnostic samples after the best-basin sample).
    try:
        params = np.asarray(result.samples.parameter_lists)
        if which != "nautilus" and len(params) > 1:
            idx = model.prior_count and [
                i for i, (p, _) in enumerate(model.path_priors_tuples)
                if "einstein_radius" in str(p)
            ][0]
            starts_re = params[1:, idx]
            hits = int(np.sum(np.abs(starts_re - TRUTH_EINSTEIN_RADIUS) < BASIN_TOL))
            print(f"per-start basin    : {hits}/{len(starts_re)}  (p_hit={hits/len(starts_re):.2f})")
    except Exception as exc:  # diagnostics must never fail the run
        print(f"per-start diagnostics unavailable: {exc!r}")

    print("=======================================")


if __name__ == "__main__":
    main()
