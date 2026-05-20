"""
Basis Regularization: MGE Lens Light
====================================

Adds a `Constant` regularization to an MGE (multi-Gaussian expansion) `Basis`
used for the lens galaxy's light. The full user-facing MGE example lives at
``autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/modeling.py``;
this developer-workspace script preserves the regularization branch that was
removed from the user-facing example because it is not used in any production
scientific analysis. See the prose below for the rationale.

__Why this lives in autolens_workspace_developer__

Regularization was originally added to the MGE to avoid a "positive / negative"
ringing effect in the lens light model reconstruction, whereby the Gaussians
went to a systematic solution which alternated between large positive and
negative values.

Regularization was intended to smooth over the `intensity` values of the
Gaussians, such that the solution would prefer a positive-only solution.
However, this did not work — even with high levels of regularization, the
Gaussians still went to negative values. The solution also became far from
optimal, often leaving significant residuals in the lens light model
reconstruction.

This problem was solved by switching to a positive-only linear algebra solver,
which is the default used in **PyAutoLens** and the one used for every fit in
the user-facing MGE example. The regularization branch shown here is preserved
for completeness, and may still be useful if you have a specific reason to
explore overfit-mitigation via regularization on a `Basis` of linear light
profiles.

__Description__

There is one downside to `Basis` functions: we may compose a model with too
much freedom. The Basis (e.g. our 60 Gaussians) may overfit noise in the data,
or possibly the lensed source galaxy emission — neither of which we want to
happen.

To circumvent this issue we add a regularization to the `Basis`. Regularization
penalizes solutions which are not smooth — it is essentially a prior that says
we expect the component the Basis represents (e.g. a bulge or disk) to be
smooth, in that its light changes smoothly as a function of radius.

This adds one extra parameter to the fit, the `coefficient`, which controls
the degree of smoothing applied.
"""
from autoconf import jax_wrapper  # Sets JAX environment before other imports

import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Loads the strong lens dataset `lens_light_asymmetric` from the sibling
autolens_workspace. If the dataset has not been simulated yet, run the
corresponding simulator there first:

    python autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/simulator.py
"""
dataset_name = "lens_light_asymmetric"
dataset_path = (
    Path("..") / "autolens_workspace" / "dataset" / "imaging" / dataset_name
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)
dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

"""
__Basis Setup__

Two groups of 30 Gaussians whose centres and elliptical components are linked
within each group. Sigma values are fixed log-spaced from 0.01 to mask_radius.
"""
total_gaussians = 30
gaussian_per_basis = 2

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

bulge_gaussian_list = []
for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )
    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]
    bulge_gaussian_list += gaussian_list

"""
__Regularized Model__

Wrap the Gaussians in a `Basis` with a `Constant` regularization. The source
uses a non-regularized linear `SersicCore`.
"""
regularization = af.Model(al.reg.Constant)
bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
    regularization=regularization,
)
mass = af.Model(al.mp.Isothermal)
lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

search = af.Nautilus(
    path_prefix=Path("basis_regularization"),
    name="mge_lens",
    unique_tag=dataset_name,
    n_live=150,
    force_x1_cpu=True,
)

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

result = search.fit(model=model, analysis=analysis)
