"""
Misc: Scaling Relations
=======================

"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

output = aplt.Output(path=".", format="png")

"""
__Dataset__

First, lets load a strong lens dataset, which is a simulated group scale lens with 2 galaxies surrounding the
lensed source.

These three galaxies will be modeled using a scaling relation.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

aplt.plot_array(array=dataset.data)

"""
__Centres__

Before composing our scaling relation model, we need to define the centres of the galaxies. 

In this example, we know these centres perfectly from the simulated dataset. In a real analysis, we would have to
determine these centres beforehand (see discussion above).
"""
extra_galaxies_centre_list = [(3.5, 2.5), (-4.4, -5.0)]

"""
We can plot the centres over the strong lens dataset to check that they look like reasonable values.
"""
aplt.plot_array(array=dataset.data, output=output)

"""
__Luminosities__

We also need the luminosity of each galaxy, which in this example is the measured property we relate to mass via
the scaling relation.

We again uses the true values of the luminosities from the simulated dataset, but in a real analysis we would have
to determine these luminosities beforehand (see discussion above).

This could be other measured properties, like stellar mass or velocity dispersion.
"""
extra_galaxies_luminosity_list = [0.9, 0.9]

"""
__Scaling Relation__

We now compose our scaling relation models, using **PyAutoFits** relational model API, which works as follows:

- Define the free parameters of the scaling relation using priors (note how the priors below are outside the for loop,
  meaning that every extra galaxy is associated with the same scailng relation prior and therefore parameters).

- For every extra galaxy centre and lumnosity, create a model mass profile (using `af.Model(dPIEPotentialSph)`), where 
  the centre of the mass profile is the extra galaxy centres and its other parameters are set via the scaling relation 
  priors.

- Make each extra galaxy a model galaxy (via `af.Model(Galaxy)`) and associate it with the model mass profile, where the
  redshifts of the extra galaxies are set to the same values as the lens galaxy.
"""
ra_star = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)
rs_star = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)
b0_star = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)
luminosity_star = 1e9

extra_galaxies_list = []

for extra_galaxy_centre, extra_galaxy_luminosity in zip(
    extra_galaxies_centre_list, extra_galaxies_luminosity_list
):
    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = extra_galaxy_centre
    mass.ra = ra_star * (extra_galaxy_luminosity / luminosity_star) ** 0.5
    mass.rs = rs_star * (extra_galaxy_luminosity / luminosity_star) ** 0.5
    mass.b0 = b0_star * (extra_galaxy_luminosity / luminosity_star) ** 0.25

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

"""
__Model__

We compose the overall lens model using the normal API.
"""
mask_radius = 3.0

# Lens:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)

mass = af.Model(al.mp.IsothermalSph)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

# Source:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

"""
When creating the overall model, we include the extra galaxies as a separate collection of galaxies.

This is not strictly necessary (e.g. if we input them into the `galaxies` attribute of the model the code would still
function correctly).

However, to ensure results are easier to interpret we keep them separate.
"""
model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source)
    + af.Collection(extra_galaxies_list),
)

"""
The `model.info` shows the model we have composed.

The priors and values of parameters that are set via scaling relations can be seen in the printed info.

The number of free parameters is N=16, which breaks down as follows:

 - 4 for the lens galaxy's `SersicSph` bulge.
 - 3 for the lens galaxy's `IsothermalSph` mass.
 - 6 for the source galaxy's `Sersic` bulge.
 - 3 for the scaling relation parameters.

Had we modeled both extra galaxies independently as dPIE profiles, we would of had 6 parameters per extra galaxy, 
giving N=19. Furthermore, by using scaling relations we can add more extra galaxies to the model without increasing the 
number of free parameters. 
"""
print(model.info)

"""
__VRAM__

The `modeling` example explains how VRAM is used during GPU-based fitting and how to
print the estimated VRAM required by a model.

For extra light and mass profiles in the model, even when on scaling relations, extra VRAM is used. For 3-10 linear 
Sersic light profiles this is a tiny  amount of VRAM (e.g. < 10MB  per batched likelihood). Even for large batch 
sizes (e.g. over 100) you probably will not use enough VRAM to require monitoring when using scaling relations.

__Run Time__

Light and mass calculations of galaxies on scaling relations run the same speed as normal light and mass profiles, 
so using scaling relations does not slow down the likelihood evaluation time compared to modeling each galaxy
individually.

For models with many extra galaxies, the scaling relation can lead to fewer free parameters, because for each mass
profile we do not fits its mass individually but rather via the scaling relation parameters. This can speed up the 
overall run time of the model-fit, because sampling will converge in fewer iterations due to the simpler parameter space.

__Model Fit__

We now perform the usual steps to perform a model-fit, to see our scaling relation based fit in action!
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

search = af.Nautilus(
    path_prefix=Path("features"),
    name="scaling_relation",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
    iterations_per_quick_update=1000000,
    n_like_max=2500,
)

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)


from pathlib import Path
import numpy as np
from scipy.interpolate import griddata

import autofit as af
import autolens as al
import autolens.plot as aplt

from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=Path("output") / "features",
)

fit_agg = al.agg.FitImagingAgg(aggregator=agg)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_gen:
    # Only one `Analysis` so take first and only dataset.
    fit = fit_list[0]

    inversion = fit.inversion

    reconstruction = inversion.reconstruction

    mapper = inversion.cls_list_from(cls=al.Mapper)[
        0
    ]  # Extract the mapper from the inversion

    source_plane_mesh_grid = mapper.mapper_grids.source_plane_mesh_grid

    mapped_reconstructed_image = inversion.mapped_reconstructed_image

    interpolation_grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05)

    interpolated_reconstruction = griddata(
        points=source_plane_mesh_grid, values=reconstruction, xi=interpolation_grid
    )

    # As a pure 2D numpy array in case its useful for calculations
    interpolated_reconstruction_ndarray = interpolated_reconstruction.reshape(
        interpolation_grid.shape_native
    )

    interpolated_reconstruction = al.Array2D.no_mask(
        values=interpolated_reconstruction_ndarray,
        pixel_scales=interpolation_grid.pixel_scales,
    )

    magnification = np.sum(
        mapped_reconstructed_image * mapped_reconstructed_image.pixel_area
    ) / np.sum(interpolated_reconstruction * interpolated_reconstruction.pixel_area)
