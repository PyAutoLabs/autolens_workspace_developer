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
  - source     : ``RectangularKernelAdaptDensity(bandwidth=0.1)`` + Constant reg
                 (C^inf continuous-density transform — differentiable at os_pix=1)

Usage::

    python searches_minimal/pix_multi_start.py adam|adabelief|lion|nautilus [n_starts]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402

from searches_minimal._setup import build_dataset, MASK_RADIUS  # noqa: E402

TRUTH_ELL = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
TRUTH_EINSTEIN_RADIUS = 1.6
BASIN_TOL = 0.3  # same basin criterion as the MGE benchmark
OS_PIX = 1
MESH_SHAPE = (30, 30)


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

    pixelization = af.Model(
        al.Pixelization,
        mesh=al.mesh.RectangularKernelAdaptDensity(shape=MESH_SHAPE, bandwidth=0.1),
        regularization=al.reg.Constant,
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def build_analysis(dataset):
    dataset = dataset.apply_over_sampling(over_sample_size_pixelization=OS_PIX)
    return al.AnalysisImaging(
        dataset=dataset,
        raise_inversion_positions_likelihood_exception=False,
        use_jax=True,
    )


SEARCHES = {
    "adam": lambda n: af.MultiStartAdam(n_starts=n, n_steps=300, learning_rate=1e-2),
    "adabelief": lambda n: af.MultiStartADABelief(
        n_starts=n, n_steps=300, learning_rate=1e-2
    ),
    "lion": lambda n: af.MultiStartLion(n_starts=n, n_steps=300, learning_rate=1e-3),
    "nautilus": lambda n: af.Nautilus(n_live=100, number_of_cores=1),
}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "adam"
    n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    print(f"JAX backend: {jax.default_backend()}  x64={jax.config.jax_enable_x64}")
    print(f"Sampler: {which}  n_starts={n_starts}  mesh=KernelAdaptDensity os_pix={OS_PIX}")

    dataset = build_dataset()
    analysis = build_analysis(dataset)
    model = build_model()
    print(f"Free (non-linear) parameters: {model.prior_count}")

    search = SEARCHES[which](n_starts)

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
