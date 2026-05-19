"""
Simulator: No Lens Light
========================

Same simulated lens as experiment 1 (`results/1_with_lens_light/simulator.py`)
but with the Sersic lens bulge removed. Only the SIE+shear lens mass and the
SersicCore source remain. Outputs to `dataset/imaging/no_lens_light/`.

This isolates the source-only signal from the data so the lens-light fit can
no longer interfere with the source recovery. Experiment 2 in the source-
science series.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt


dataset_type = "imaging"
dataset_name = "no_lens_light"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
)

"""
__Over Sampling__

Adaptive over-sampling on the lens light is irrelevant here (no lens light),
but the source still has high-magnification regions that benefit. Matches
experiment 1 for apples-to-apples comparison.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
__PSF__
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
__Simulator__
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Tracer__

Identical to experiment 1 except the lens bulge is gone. Mass + shear and the
source bulge are unchanged so the magnification truth and source flux are
directly comparable.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
__Output__
"""
aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Source Flux & Magnification (Truth)__

Computed on a 400×400 @ 0.03 "/px grid identical to experiment 1, so the
truth source flux and magnification are directly comparable to the
with-lens-light values.
"""
science_grid = al.Grid2D.uniform(shape_native=(400, 400), pixel_scales=0.03)

source_plane_image = source_galaxy.bulge.image_2d_from(grid=science_grid)
total_source_plane_flux = float(np.sum(source_plane_image))

traced_grid_list = tracer.traced_grid_2d_list_from(grid=science_grid)
lensed_source_image = source_galaxy.bulge.image_2d_from(grid=traced_grid_list[1])
total_image_plane_flux = float(np.sum(lensed_source_image))

source_magnification = total_image_plane_flux / total_source_plane_flux

print(f"Source Plane Flux (truth):  {total_source_plane_flux}")
print(f"Image Plane Flux (truth):   {total_image_plane_flux}")
print(f"Source Magnification:       {source_magnification}")

source_science_dict = {
    "image_plane_flux": total_image_plane_flux,
    "source_plane_flux": total_source_plane_flux,
    "source_magnification": source_magnification,
}

al.output_to_json(
    file_path=dataset_path / "source_science.json",
    obj=source_science_dict,
)

print(f"\nDataset written to {dataset_path}")
