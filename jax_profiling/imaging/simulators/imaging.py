"""
Simulator: Instrument-Based Imaging Datasets
=============================================

Simulates imaging datasets for different astronomical instruments, each
with a characteristic pixel scale:

    euclid  — 0.1  arcsec/pixel
    hst     — 0.05 arcsec/pixel
    jwst    — 0.03 arcsec/pixel
    ao      — 0.01 arcsec/pixel

All instruments use a 21x21 PSF kernel by default.

Usage
-----
Run directly to simulate a specific instrument::

    python simulators/imaging.py --instrument hst

Or import and call from another script::

    from simulators.imaging import simulate
    simulate("hst")

The auto-simulation pattern in ``mge.py`` calls this via subprocess when
the dataset does not already exist on disk.
"""

import argparse
import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

# ---------------------------------------------------------------------------
# Instrument definitions
# ---------------------------------------------------------------------------

INSTRUMENTS = {
    "euclid": {"pixel_scale": 0.1, "psf_shape": (21, 21), "psf_sigma": 0.1},
    "hst": {"pixel_scale": 0.05, "psf_shape": (21, 21), "psf_sigma": 0.05},
    "jwst": {"pixel_scale": 0.03, "psf_shape": (21, 21), "psf_sigma": 0.03},
    "ao": {"pixel_scale": 0.01, "psf_shape": (21, 21), "psf_sigma": 0.01},
}


def simulate(instrument: str, mask_radius: float = 3.5):
    """Simulate an imaging dataset for the given instrument.

    Parameters
    ----------
    instrument
        One of "euclid", "hst", "jwst", "ao".
    mask_radius
        The mask radius in arcseconds, used to set the grid extent.
    """
    if instrument not in INSTRUMENTS:
        raise ValueError(
            f"Unknown instrument '{instrument}'. "
            f"Choose from: {list(INSTRUMENTS.keys())}"
        )

    config = INSTRUMENTS[instrument]
    pixel_scale = config["pixel_scale"]
    psf_shape = config["psf_shape"]
    psf_sigma = config["psf_sigma"]

    dataset_path = Path("dataset") / "imaging" / instrument

    # Grid — sized so the mask_radius fits within the image
    shape_pixels = int(np.ceil(2 * mask_radius / pixel_scale))
    if shape_pixels % 2 == 0:
        shape_pixels += 1  # keep odd for symmetric centering

    grid = al.Grid2D.uniform(
        shape_native=(shape_pixels, shape_pixels),
        pixel_scales=pixel_scale,
    )

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid,
        sub_size_list=[32, 8, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

    # PSF
    psf = al.Convolver.from_gaussian(
        shape_native=psf_shape,
        sigma=psf_sigma,
        pixel_scales=grid.pixel_scales,
    )

    # Simulator
    simulator = al.SimulatorImaging(
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_poisson_noise_to_data=True,
    )

    # Galaxies — lens with Sersic light + Isothermal mass, source with cored Sersic
    lens_galaxy = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
            intensity=2.0,
            effective_radius=0.6,
            sersic_index=3.0,
        ),
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

    # Simulate and output
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    aplt.fits_imaging(
        dataset=dataset,
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        overwrite=True,
    )

    al.output_to_json(
        obj=tracer,
        file_path=dataset_path / "tracer.json",
    )

    print(f"  Dataset simulated: {dataset_path}")
    print(f"    Instrument:   {instrument}")
    print(f"    Pixel scale:  {pixel_scale} arcsec/pixel")
    print(f"    Grid shape:   {shape_pixels} x {shape_pixels}")
    print(f"    PSF shape:    {psf_shape[0]} x {psf_shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate imaging dataset for a given instrument."
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="hst",
        choices=list(INSTRUMENTS.keys()),
        help="Instrument name (default: hst)",
    )
    parser.add_argument(
        "--mask-radius",
        type=float,
        default=3.5,
        help="Mask radius in arcseconds (default: 3.5)",
    )
    args = parser.parse_args()
    simulate(args.instrument, args.mask_radius)
