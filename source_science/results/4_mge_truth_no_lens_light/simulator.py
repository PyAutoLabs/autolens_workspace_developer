"""
Simulator: MGE source truth, NO lens light.

Same lens setup as test 2 (`results/2_no_lens_light/simulator.py`) — SIE +
shear, no Sersic bulge — but the source is the MGE galaxy extracted from
the test-2 mge_source fit (`source_science/results/mge_truth_source.json`).
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt


dataset_type = "imaging"
dataset_name = "mge_truth_no_lens_light"
dataset_path = Path("dataset", dataset_type, dataset_name)
mge_truth_path = Path("source_science") / "results" / "mge_truth_source.json"

if not mge_truth_path.exists():
    raise FileNotFoundError(
        f"{mge_truth_path} missing — run `python source_science/extract_mge_truth.py`"
        " first."
    )

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.from_json(file_path=mge_truth_path)
print(f"Loaded MGE truth source: {type(source_galaxy).__name__}")

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)
al.output_to_json(obj=tracer, file_path=Path(dataset_path, "tracer.json"))

science_grid = al.Grid2D.uniform(shape_native=(400, 400), pixel_scales=0.03)
source_plane_image = source_galaxy.image_2d_from(grid=science_grid)
total_source_plane_flux = float(np.sum(source_plane_image))

traced_grid_list = tracer.traced_grid_2d_list_from(grid=science_grid)
lensed_source_image = source_galaxy.image_2d_from(grid=traced_grid_list[1])
total_image_plane_flux = float(np.sum(lensed_source_image))
source_magnification = total_image_plane_flux / total_source_plane_flux

print(f"Source Plane Flux (truth):  {total_source_plane_flux}")
print(f"Image Plane Flux (truth):   {total_image_plane_flux}")
print(f"Source Magnification:       {source_magnification}")

al.output_to_json(
    file_path=dataset_path / "source_science.json",
    obj={
        "image_plane_flux": total_image_plane_flux,
        "source_plane_flux": total_source_plane_flux,
        "source_magnification": source_magnification,
    },
)
print(f"\nDataset written to {dataset_path}")
