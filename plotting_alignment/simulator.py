"""
Simulator: HST
==============

The majority of simulator scripts are based on Euclid resolution imaging.

This is chosen because it is high enough resolution to resolve the lens and lensed source galaxy's detailed structure
when demonstrating **PyAutoLens** lens modeling features, but low enough resolution for the computation run-time to be
relatively fast.

The instrument simulator scripts in this folder simulate datasets assuming different instruments, which can be used
to gauge the impact of different instrument properties on lens modeling and run-time.

This script simulates imaging representative of the Hubble Space Telescope, which is higher resolution than the
majority of Euclid-like simulator scripts.

__Model__

This script simulates `Imaging` of a 'galaxy-scale' strong lens where:

 - The resolution, PSF and S/N are representative of Hubble Space Telescope imaging.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. 
"""
dataset_type = "plotting_alignment"

# dataset_name = "mass_centre_source_right"
dataset_name = "mass_centre_source_up_more"
# dataset_name = "mass_centre_source_up_right"
# dataset_name = "mass_centre_source_down"
# dataset_name = "mass_centre_source_x2"

# dataset_name = "mass_right_source_right"
# dataset_name = "mass_right_source_up"
# dataset_name = "mass_right_source_up_right"

dataset_path = Path("dataset", "imaging", dataset_type, dataset_name)

"""
__Simulate__

Simulate the image using a (y,x) grid with the adaptive over sampling scheme.
"""
grid = al.Grid2D.uniform(
    shape_native=(240, 240),
    pixel_scales=0.05,
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(21, 21), sigma=0.05, pixel_scales=grid.pixel_scales, normalize=True
)

"""
To simulate the `Imaging` dataset we first create a simulator, which defines the exposure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = al.SimulatorImaging(
    exposure_time=2000.0,
    psf=psf,
    background_sky_level=1.0,
    add_poisson_noise_to_data=True,
)

"""
__Ray Tracing__

Setup the lens galaxy's mass (SIE+Shear) and source galaxy light (elliptical Sersic) for this simulated lens.
"""
centre = (0.0, 0.0)

if (
    dataset_name == "mass_right_source_right"
    or dataset_name == "mass_right_source_up"
    or dataset_name == "mass_right_source_up_right"
):
    centre = (0.0, 0.3)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=centre,
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
)

if dataset_name == "mass_centre_source_right":
    centre = (0.0, 0.3)
elif dataset_name == "mass_centre_source_up":
    centre = (0.3, 0.0)
elif dataset_name == "mass_centre_source_up_right":
    centre = (0.3, 0.3)
elif dataset_name == "mass_centre_source_down":
    centre = (-0.3, 0.0)
elif dataset_name == "mass_centre_source_up_more":
    centre = (1.0, 0.0)
elif dataset_name == "mass_right_source_right":
    centre = (0.0, 0.6)
elif dataset_name == "mass_right_source_up":
    centre = (0.3, 0.3)
elif dataset_name == "mass_right_source_up_right":
    centre = (0.3, 0.6)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=centre,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid))

"""
Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
dataset.output_to_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
aplt.plot_array(array=dataset.data)

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png files.
"""
aplt.plot_array(array=dataset.data, output=aplt.Output(path=dataset_path, format="png"))
aplt.subplot_tracer(tracer=tracer, grid=grid, output=aplt.Output(path=dataset_path, format="png"))

"""
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `tracer = al.from_json()`.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
The dataset can be viewed in the folder `autolens_workspace/imaging/instruments/hst`.
"""
