"""
FD probe: is the SLaM-pix-1 objective JAX-differentiable in the lens mass?
=========================================================================

Mirrors the certified harness
``autolens_workspace_test/scripts/jax_grad/imaging_pixelization.py``, which
already proves pixelized likelihoods are gradient-differentiable. This probe
confirms it for *the objective this experiment optimises* (autolens_workspace_developer#100):

  - lens light: MGE **linear**, non-linear geometry **fixed** at truth (SLaM pix-1),
  - lens mass: Isothermal + ExternalShear **free** (what the samplers optimise),
  - source: **kernel-CDF mesh** ``RectangularKernelAdaptDensity(bandwidth=0.1)`` —
    a C^inf continuous-density transform, strict-FD certified on ALL params
    (incl. mass/shear) even at pixelization over-sampling 1,
  - regularization coefficient free.

Methodology (the part my first probe got wrong):
  - **truth-centred Gaussian priors** so the evaluation point sits where the
    source arcs land on the mesh and every parameter has real sensitivity
    (broad uniform priors give garbage logL ~ -5e5 where FD is meaningless);
  - a **small relative FD step sweep** (1e-8..1e-6), not a single eps=1e-4 —
    the likelihood is steep, and isolated steps can land on measure-thin
    positive-only-solver branch flips (hence "best over sweep");
  - explicit ``over_sample_size_pixelization`` (the *adaptive* meshes are an
    exact staircase in mass at os_pix=1; kernel-CDF is fine at 1).

Truth of the HST benchmark dataset (jax_profiling/simulators/imaging.py):
Isothermal einstein_radius=1.6, centre=(0,0), q=0.9/45deg; shear=(0.05, 0.05).
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402

from searches_minimal._setup import build_dataset, MASK_RADIUS  # noqa: E402

TRUTH_ELL = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
OS_PIX = 1  # kernel-CDF is differentiable at 1; adaptive meshes would need 4


def build_pix_model(mesh_shape: tuple[int, int] = (30, 30)) -> af.Collection:
    """SLaM-pix-1 style: fixed-geometry MGE linear lens light, free mass, kernel-CDF source."""
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=MASK_RADIUS, total_gaussians=20, centre_prior_is_uniform=True
    )
    # Fix the MGE non-linear geometry at truth (amplitudes stay linear/inversion-solved).
    for profile in lens_bulge.profile_list:
        profile.centre = (0.0, 0.0)
        profile.ell_comps = TRUTH_ELL

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=TRUTH_ELL[0], sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=TRUTH_ELL[1], sigma=0.01)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
    shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)

    lens = af.Model(
        al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear
    )

    pixelization = af.Model(
        al.Pixelization,
        mesh=al.mesh.RectangularKernelAdaptDensity(shape=mesh_shape, bandwidth=0.1),
        regularization=al.reg.Constant,
    )
    pixelization.regularization.coefficient = af.GaussianPrior(mean=1.0, sigma=0.1)
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def build_probe_analysis(dataset):
    dataset = dataset.apply_over_sampling(over_sample_size_pixelization=OS_PIX)
    return al.AnalysisImaging(
        dataset=dataset,
        raise_inversion_positions_likelihood_exception=False,
        use_jax=True,
    )


def main() -> None:
    print(f"JAX backend: {jax.default_backend()}  x64={jax.config.jax_enable_x64}")
    print(f"Mesh: RectangularKernelAdaptDensity(bandwidth=0.1)  os_pix={OS_PIX}")

    dataset = build_dataset()
    analysis = build_probe_analysis(dataset)
    model = build_pix_model()
    names = [str(p) for p, _ in model.path_priors_tuples]
    ndim = model.prior_count
    print(f"Free (non-linear) parameters: {ndim}")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")

    def log_likelihood(params):
        instance = model.instance_from_vector(vector=params, xp=jnp)
        return analysis.log_likelihood_function(instance=instance)

    # JIT both: the FD sweep does ~40 evaluations, and an un-jitted call
    # re-traces the whole pixelized likelihood each time (the certified harness
    # likewise passes a jitted f_fd).
    value_and_grad = jax.jit(jax.value_and_grad(log_likelihood))
    f_jit = jax.jit(log_likelihood)

    # Truth-centred prior medians + a tiny perturbation off exact symmetry points.
    x = np.array(model.physical_values_from_prior_medians, dtype=float)
    rng = np.random.default_rng(42)
    x = x + rng.uniform(0.001, 0.005, size=x.shape)
    x = jnp.asarray(x)

    loss, grad = value_and_grad(x)
    loss = float(loss)
    grad = np.asarray(grad)
    print(f"\nlogL at (truth-centred median + perturbation) = {loss:.6f}")
    print(f"grad finite: {bool(np.all(np.isfinite(grad)))}")
    if not (np.isfinite(loss) and np.all(np.isfinite(grad))):
        print("\nVERDICT: FAIL_NAN_OR_INF")
        return

    # FD step sweep (best-of, as the certified harness does — isolated steps can
    # land on measure-thin solver branch flips).
    verdict = "OK_PIX_GRAD_VIABLE"
    mass_idx = [i for i, n in enumerate(names) if "mass" in n or "shear" in n]
    print("\nFD cross-check (best over relative steps 1e-8/1e-7/1e-6):")
    for i in mass_idx:
        ad = float(grad[i])
        best_rel, best_fd = np.inf, np.nan
        for rel in (1e-8, 1e-7, 1e-6):
            h = rel * max(abs(float(x[i])), 1.0)
            fd = (float(f_jit(x.at[i].add(h))) - float(f_jit(x.at[i].add(-h)))) / (2 * h)
            r = abs(fd - ad) / (abs(fd) + abs(ad) + 1e-12)
            if r < best_rel:
                best_rel, best_fd = r, fd
        live = abs(ad) > 1e-2
        ok = best_rel < 0.05 and live
        print(
            f"  {names[i]:<45} ad={ad:+.4e} fd={best_fd:+.4e} rel={best_rel:.2e} "
            f"{'ok' if ok else ('DEAD(staircase)' if not live else 'MISMATCH')}"
        )
        if not ok:
            verdict = "FAIL_FD_MISMATCH" if live else "FAIL_STAIRCASE_DEAD"

    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
