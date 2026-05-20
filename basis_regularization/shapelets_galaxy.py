"""
Basis Regularization: Shapelets (PyAutoGalaxy)
==============================================

Adds a `Constant` regularization to a shapelet `Basis` for a single galaxy.
Counterpart to ``shapelets_lens.py``; uses PyAutoGalaxy. The full user-facing
shapelets example lives at
``autogalaxy_workspace/scripts/imaging/features/shapelets/modeling.py``;
this developer-workspace script preserves the regularization branch that was
removed from there because the feature is not used by any production
scientific analysis.

__Why this lives in autolens_workspace_developer__

There is one downside to `Basis` functions: we may compose a model with too
much freedom. The shapelets may overfit noise in the data, which we do not
want. Regularization penalizes solutions which are not smooth — adding it
costs one extra parameter, the `coefficient` controlling the degree of
smoothing.

For shapelets specifically, regularization is not a substitute for the
positive-negative solver problem (shapelets *require* negative intensities to
work). It is preserved here for completeness.
"""
import numpy as np
from pathlib import Path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Loads `simple__sersic` from the sibling autogalaxy_workspace. If the dataset
has not been simulated yet, run the corresponding simulator there first.
"""
dataset_name = "simple__sersic"
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
__Shapelet Basis__

Polar shapelets with linked centres and `beta`.
"""
total_n = 5
total_m = sum(range(2, total_n + 1)) + 1

shapelets_bulge_list = af.Collection(
    af.Model(ag.lp_linear.ShapeletPolar) for _ in range(total_n + total_m)
)

n_count = 1
m_count = -1
for i, shapelet in enumerate(shapelets_bulge_list):
    shapelet.n = n_count
    shapelet.m = m_count
    m_count += 2
    if m_count > n_count:
        n_count += 1
        m_count = -n_count
    shapelet.centre = shapelets_bulge_list[0].centre
    shapelet.beta = shapelets_bulge_list[0].beta

"""
__Regularized Model__

Single-galaxy model whose bulge is a shapelet `Basis` with `Constant`
regularization.
"""
bulge = af.Model(
    ag.lp_basis.Basis,
    profile_list=shapelets_bulge_list,
    regularization=ag.reg.Constant,
)
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

search = af.Nautilus(
    path_prefix=Path("basis_regularization"),
    name="shapelets_galaxy",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
)

analysis = ag.AnalysisImaging(
    dataset=dataset,
    settings=ag.Settings(use_positive_only_solver=False),
)

result = search.fit(model=model, analysis=analysis)
