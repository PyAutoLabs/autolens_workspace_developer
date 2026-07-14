"""
Decisive feasibility probe: is a FIXED-mesh (adapt-image) pixelized source
JAX-differentiable w.r.t. the lens mass?
=========================================================================

Stage A (``probe_grad_pix.py``) showed the *mass-adaptive* meshes
(RectangularAdaptDensity, RectangularSplineAdaptDensity) give FD-mismatched
gradients: the source-plane mesh adapts to the mass, and that adaptation is
non-smooth. This probe tests the ``RectangularSplineAdaptImage`` mesh, whose
adaptive weighting comes from a **fixed adapt image** (source brightness from a
prior fit) rather than the mass — so the non-smooth part is constant and only
the (smooth) ray-tracing depends on the mass. If its ``jax.grad`` is
FD-faithful, pixelized gradient sampling is viable with a fixed adapt image.

Free (non-linear) params: lens mass (Isothermal + ExternalShear) + adaptive
regularization coefficients. The adapt image is bootstrapped once (numpy) from a
RectangularAdaptDensity + Constant inversion at a near-truth mass.
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402

from searches_minimal._setup import build_dataset  # noqa: E402

# The adapt-image binding (AdaptImages.updated_via_instance_from) looks up
# ``str(galaxy_name)`` against the dict keys, so the key must be the *stringified*
# path tuple, not the tuple itself.
SOURCE_PATH = str(("galaxies", "source"))


def build_adapt_images(dataset) -> al.AdaptImages:
    """Bootstrap a fixed source adapt image from a numpy AdaptDensity inversion
    at a near-truth mass (stands in for a SLaM SOURCE_LP/SOURCE_PIX parent fit)."""
    mass = al.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.05, 0.05)
    )
    lens = al.Galaxy(redshift=0.5, mass=mass)
    pix = al.Pixelization(
        mesh=al.mesh.RectangularAdaptDensity(shape=(30, 30)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    source = al.Galaxy(redshift=1.0, pixelization=pix)
    instance = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    fit = analysis.fit_from(instance=instance)
    source_image = fit.galaxy_model_image_dict[source]
    print(f"  bootstrap source adapt image: sum={float(np.sum(source_image)):.4e}")
    return al.AdaptImages(galaxy_name_image_dict={SOURCE_PATH: source_image})


def build_pix_model(mesh_shape: tuple[int, int] = (30, 30)) -> af.Collection:
    mass = af.Model(al.mp.Isothermal)
    mass.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    mass.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
    mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.2, upper_limit=3.0)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    shear.gamma_2 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

    regularization = af.Model(al.reg.Adapt)
    regularization.inner_coefficient = af.UniformPrior(lower_limit=0.1, upper_limit=10.0)
    regularization.outer_coefficient = af.UniformPrior(lower_limit=0.1, upper_limit=10.0)
    regularization.signal_scale = 0.5  # fixed

    pixelization = af.Model(
        al.Pixelization,
        mesh=al.mesh.RectangularSplineAdaptImage(shape=mesh_shape),
        regularization=regularization,
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def main() -> None:
    print(f"JAX backend: {jax.default_backend()}  x64={jax.config.jax_enable_x64}")
    dataset = build_dataset()
    print("Bootstrapping fixed adapt image (numpy AdaptDensity inversion)...")
    adapt_images = build_adapt_images(dataset)

    model = build_pix_model()
    analysis = al.AnalysisImaging(
        dataset=dataset, adapt_images=adapt_images, use_jax=True
    )
    ndim = model.prior_count
    print(f"Mesh: RectangularSplineAdaptImage (fixed adapt image)")
    print(f"Free (non-linear) parameters: {ndim}")

    def log_likelihood(params):
        instance = model.instance_from_vector(vector=params, xp=jnp)
        return analysis.log_likelihood_function(instance=instance)

    value_and_grad = jax.value_and_grad(log_likelihood)

    rng = np.random.default_rng(0)
    verdict = "OK_PIX_GRAD_VIABLE"
    for k in range(3):
        u = np.clip(0.5 + rng.normal(0, 0.05, size=ndim), 0.02, 0.98)
        x = jnp.asarray(model.vector_from_unit_vector(list(u)))
        loss, grad = value_and_grad(x)
        loss = float(loss)
        grad = np.asarray(grad)
        finite = np.isfinite(loss) and np.all(np.isfinite(grad))
        print(f"\n[point {k}] logL={loss:.4f}  grad finite={finite}")
        if not finite:
            verdict = "FAIL_NAN_OR_INF"
            continue
        for idx in (4, 5, 6):  # einstein_radius, shear gamma_1, gamma_2
            eps = 1e-4
            fd = (
                float(log_likelihood(x.at[idx].add(eps)))
                - float(log_likelihood(x.at[idx].add(-eps)))
            ) / (2 * eps)
            ad = float(grad[idx])
            rel = abs(fd - ad) / (abs(fd) + abs(ad) + 1e-12)
            flag = "ok" if rel < 1e-3 else "MISMATCH"
            print(f"    param[{idx}]: analytic={ad:+.4e} fd={fd:+.4e} rel={rel:.2e} {flag}")
            if rel >= 1e-3 and verdict == "OK_PIX_GRAD_VIABLE":
                verdict = "FAIL_FD_MISMATCH"

    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
