"""
Feasibility probe: is a PIXELIZED-source likelihood JAX-differentiable?
======================================================================

The gradient MAP-optimizer benchmark (``gpu_multi_start_adam.py`` etc.) used an
**MGE** source, whose linear light amplitudes are solved by a positive
inversion. The open question (autolens_workspace_developer#100) is whether the
same ``jax.grad``-driven optimizers work when the source is a **pixelization**
(a regularized inversion with a log-det evidence term), not an MGE.

This probe answers it mechanically, cheapest-first:

  Stage A: RectangularAdaptDensity + Constant regularization (no adapt image) —
           the simplest differentiable pixelization. If ``jax.grad`` of the
           pixelized log-likelihood is finite and finite-difference-faithful
           here, pixelized gradient sampling is viable in principle.

Free (non-linear) parameters: lens mass (Isothermal + ExternalShear) + the
regularization coefficient. Lens light is omitted in this minimal probe to
isolate the mass-gradient question (the full experiment adds a fixed-geometry
MGE lens light). Source pixels are the linear parameters, solved internally.

Run (x64, CPU)::

    python searches_minimal/probe_grad_pix.py
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402

from searches_minimal._setup import build_dataset  # noqa: E402


MESH_CLS = al.mesh.RectangularSplineAdaptDensity  # spline = smooth/differentiable


def build_pix_model(mesh_shape: tuple[int, int] = (30, 30)) -> af.Collection:
    """Isothermal + shear lens (free) with a spline-mesh + Constant pixelized
    source (reg coefficient free). Mass priors centred near the benchmark truth
    (einstein_radius ~ 1.6). The mesh is a *spline* adapt mesh, whose smooth
    interpolation is designed to be differentiable (unlike the plain
    RectangularAdaptDensity, whose hard adaptation is not)."""
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

    pixelization = af.Model(
        al.Pixelization,
        mesh=MESH_CLS(shape=mesh_shape),
        regularization=al.reg.Constant,
    )
    pixelization.regularization.coefficient = af.UniformPrior(
        lower_limit=0.1, upper_limit=10.0
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def main() -> None:
    print(f"JAX backend: {jax.default_backend()}  x64={jax.config.jax_enable_x64}")
    print(f"Mesh: {MESH_CLS.__name__}")
    dataset = build_dataset()
    model = build_pix_model()
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)
    ndim = model.prior_count
    print(f"Free (non-linear) parameters: {ndim}")

    def log_likelihood(params):
        instance = model.instance_from_vector(vector=params, xp=jnp)
        return analysis.log_likelihood_function(instance=instance)

    value_and_grad = jax.value_and_grad(log_likelihood)

    # Evaluate at a few points in a narrow band around the prior median (the
    # exact median has ell=shear=0, a mildly singular point — perturb off it).
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
        # Finite-difference cross-check on 3 entries (incl. einstein_radius = idx 4).
        for idx in (4, 5, 6):
            eps = 1e-4
            xp_ = x.at[idx].add(eps)
            xm_ = x.at[idx].add(-eps)
            fd = (float(log_likelihood(xp_)) - float(log_likelihood(xm_))) / (2 * eps)
            ad = float(grad[idx])
            rel = abs(fd - ad) / (abs(fd) + abs(ad) + 1e-12)
            flag = "ok" if rel < 1e-3 else "MISMATCH"
            print(f"    param[{idx}]: analytic={ad:+.4e} fd={fd:+.4e} rel={rel:.2e} {flag}")
            if rel >= 1e-3 and verdict == "OK_PIX_GRAD_VIABLE":
                verdict = "FAIL_FD_MISMATCH"

    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
