"""
Basis Regularization: MGE Galaxy Light (PyAutoGalaxy)
=====================================================

Adds a `Constant` regularization to an MGE `Basis` for a single galaxy (no
lensing). Counterpart to ``mge_lens.py``; uses PyAutoGalaxy rather than
PyAutoLens. The full user-facing MGE example lives at
``autogalaxy_workspace/scripts/imaging/features/multi_gaussian_expansion/modeling.py``;
the regularization branch was removed from there because it is not used in any
production scientific analysis.

__Why this lives in autolens_workspace_developer__

Regularization was originally added to the MGE to avoid a "positive / negative"
ringing effect, whereby the Gaussians went to a systematic solution which
alternated between positive and negative values. It was intended to smooth
over the `intensity` values such that the solution would prefer a positive-only
form, but did not in fact converge to one — even at high regularization the
Gaussians still went negative and the residuals were significant.

The fix was to switch to a positive-only linear algebra solver, which is the
default in PyAutoGalaxy. The regularization branch here is preserved for
completeness and for the occasional overfit-mitigation experiment.
"""
import numpy as np
from pathlib import Path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Loads the asymmetric galaxy dataset from the sibling autogalaxy_workspace. If
the dataset has not been simulated yet, run:

    python autogalaxy_workspace/scripts/imaging/features/multi_gaussian_expansion/simulator.py
"""
dataset_name = "asymmetric"
dataset_path = (
    Path("..") / "autogalaxy_workspace" / "dataset" / "imaging" / dataset_name
)

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)
dataset = dataset.apply_mask(mask=mask)

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Basis Setup__

Two groups of 30 Gaussians, sigma values log-spaced from 0.01 to 3.0".
"""
total_gaussians = 30
gaussian_per_basis = 2

mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

bulge_gaussian_list = []
for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(ag.lp_linear.Gaussian) for _ in range(total_gaussians)
    )
    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]
    bulge_gaussian_list += gaussian_list

"""
__Regularized Model__

Wrap the Gaussians in a `Basis` with `Constant` regularization on the single
galaxy's bulge.
"""
bulge = af.Model(
    ag.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
    regularization=ag.reg.Constant,
)
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

search = af.Nautilus(
    path_prefix=Path("basis_regularization"),
    name="mge_galaxy",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
)

analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)
