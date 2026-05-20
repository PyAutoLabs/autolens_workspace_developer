"""
Basis Regularization: Shapelets (PyAutoLens)
============================================

Adds a `Constant` regularization to a shapelet `Basis`. Counterpart to the
MGE variants in this folder. The full user-facing shapelets example lives at
``autolens_workspace/scripts/imaging/features/advanced/shapelets/modeling.py``;
this developer-workspace script preserves the regularization branch that was
removed from there because the feature is not used by any production
scientific analysis.

__Why this lives in autolens_workspace_developer__

There is one downside to `Basis` functions: we may compose a model with too
much freedom. The shapelets (~20 here) may overfit noise in the data, or
possibly the lensed source galaxy emission — neither of which we want.

Regularization penalizes solutions which are not smooth — it is essentially a
prior that says we expect the component the Basis represents (e.g. a bulge or
disk) to be smooth as a function of radius. Adding it costs one extra
parameter, the `coefficient` controlling the degree of smoothing.

For shapelets specifically, regularization is not a substitute for the
positive-negative solver problem (shapelets *require* negative intensities to
work). It is preserved here for completeness.
"""
from autoconf import jax_wrapper  # Sets JAX environment before other imports

import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Loads the `simple__no_lens_light` dataset from the sibling autolens_workspace.
If the dataset has not been simulated yet, run the corresponding simulator
there first.
"""
dataset_name = "simple__no_lens_light"
dataset_path = (
    Path("..") / "autolens_workspace" / "dataset" / "imaging" / dataset_name
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.4,
    outer_radius=3.0,
)
dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Shapelet Basis__

Polar shapelets with linked centres, elliptical components, and `beta`.
"""
total_n = 10
total_m = sum(range(2, total_n + 1)) + 1

shapelets_bulge_list = af.Collection(
    af.Model(al.lp_linear.ShapeletPolar) for _ in range(total_n + total_m + 1)
)

n_count = 1
m_count = -1
for i, shapelet in enumerate(shapelets_bulge_list):
    if i == 0:
        shapelet.n = 0
        shapelet.m = 0
    else:
        shapelet.n = n_count
        shapelet.m = m_count
        m_count += 2
        if m_count > n_count:
            n_count += 1
            m_count = -n_count
    shapelet.centre = shapelets_bulge_list[0].centre
    shapelet.ell_comps = shapelets_bulge_list[0].ell_comps
    shapelet.beta = shapelets_bulge_list[0].beta

"""
__Regularized Model__

A single-galaxy model whose bulge is a shapelet `Basis` with `Constant`
regularization. (Matches the structure of the regularization section that
was removed from the user-facing shapelets/modeling.py — note it does not
include a full lens+source model; it is a minimal demo of the regularization
API on a Basis.)
"""
bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=shapelets_bulge_list,
    regularization=al.reg.Constant,
)
galaxy = af.Model(al.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

search = af.Nautilus(
    path_prefix=Path("basis_regularization"),
    name="shapelets_lens",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    settings=al.Settings(use_positive_only_solver=False),
)

result = search.fit(model=model, analysis=analysis)
