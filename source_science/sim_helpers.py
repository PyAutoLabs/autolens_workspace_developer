"""
Shared simulator helpers for the source-science robustness study.

Per-test driver scripts under `results/<config>/<test>/` call these
functions to build the lens galaxy + source and write the simulated
dataset + truth tracer + truth source-science quantities.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import autolens as al
import autolens.plot as aplt

from source_science.lens_configs import SERSIC_SOURCE_TRUTH


GRID_SHAPE = (100, 100)
PIXEL_SCALE = 0.1
PSF_SHAPE = (11, 11)
PSF_SIGMA = 0.1
EXPOSURE_TIME = 300.0
BACKGROUND_SKY = 0.1
SCIENCE_GRID_SHAPE = (400, 400)
SCIENCE_PIXEL_SCALE = 0.03


def _grid():
    grid = al.Grid2D.uniform(shape_native=GRID_SHAPE, pixel_scales=PIXEL_SCALE)
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid,
        sub_size_list=[32, 8, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    return grid.apply_over_sampling(over_sample_size=over_sample_size)


def _simulator():
    return al.SimulatorImaging(
        exposure_time=EXPOSURE_TIME,
        psf=al.Convolver.from_gaussian(
            shape_native=PSF_SHAPE, sigma=PSF_SIGMA, pixel_scales=PIXEL_SCALE
        ),
        background_sky_level=BACKGROUND_SKY,
        add_poisson_noise_to_data=True,
    )


def sersic_lens_bulge(config: Dict) -> al.lp.Sersic:
    p = config["bulge_sersic"]
    return al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(
            axis_ratio=p["axis_ratio"], angle=p["angle"]
        ),
        intensity=p["intensity"],
        effective_radius=p["effective_radius"],
        sersic_index=p["sersic_index"],
    )


def lens_galaxy(
    config: Dict, lens_light_truth: str, mge_lens_truth_path: Optional[Path] = None
) -> al.Galaxy:
    """Build a truth lens galaxy.

    Parameters
    ----------
    lens_light_truth
        One of "none", "sersic", "mge". For "mge" the MGE lens light truth
        is loaded from `mge_lens_truth_path` (must be supplied).
    """
    mass = al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=config["mass"]["einstein_radius"],
        ell_comps=al.convert.ell_comps_from(
            axis_ratio=config["mass"]["axis_ratio"],
            angle=config["mass"]["angle"],
        ),
    )
    shear = al.mp.ExternalShear(
        gamma_1=config["shear"]["gamma_1"], gamma_2=config["shear"]["gamma_2"]
    )

    if lens_light_truth == "none":
        return al.Galaxy(redshift=0.5, mass=mass, shear=shear)
    if lens_light_truth == "sersic":
        return al.Galaxy(
            redshift=0.5,
            bulge=sersic_lens_bulge(config),
            mass=mass,
            shear=shear,
        )
    if lens_light_truth == "mge":
        if mge_lens_truth_path is None:
            raise ValueError("mge_lens_truth_path required for lens_light_truth='mge'")
        truth_galaxy = al.from_json(file_path=mge_lens_truth_path)
        return al.Galaxy(redshift=0.5, bulge=truth_galaxy.bulge, mass=mass, shear=shear)
    raise ValueError(f"Unknown lens_light_truth: {lens_light_truth}")


def source_galaxy(
    source_truth: str, mge_source_truth_path: Optional[Path] = None
) -> al.Galaxy:
    """Build a truth source galaxy.

    Parameters
    ----------
    source_truth
        One of "sersic", "mge". For "mge" the source truth is loaded from
        `mge_source_truth_path` (must be supplied).
    """
    if source_truth == "sersic":
        p = SERSIC_SOURCE_TRUTH
        return al.Galaxy(
            redshift=1.0,
            bulge=al.lp.SersicCore(
                centre=p["centre"],
                ell_comps=al.convert.ell_comps_from(
                    axis_ratio=p["axis_ratio"], angle=p["angle"]
                ),
                intensity=p["intensity"],
                effective_radius=p["effective_radius"],
                sersic_index=p["sersic_index"],
            ),
        )
    if source_truth == "mge":
        if mge_source_truth_path is None:
            raise ValueError("mge_source_truth_path required for source_truth='mge'")
        return al.from_json(file_path=mge_source_truth_path)
    raise ValueError(f"Unknown source_truth: {source_truth}")


def simulate_and_save(
    *,
    config: Dict,
    source_truth: str,
    lens_light_truth: str,
    dataset_path: Path,
    mge_source_truth_path: Optional[Path] = None,
    mge_lens_truth_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Build tracer → simulate → write .fits + tracer.json + source_science.json.

    Returns the truth source-science quantities (also written to disk).
    """
    dataset_path.mkdir(parents=True, exist_ok=True)
    grid = _grid()
    sim = _simulator()

    lens = lens_galaxy(
        config=config,
        lens_light_truth=lens_light_truth,
        mge_lens_truth_path=mge_lens_truth_path,
    )
    source = source_galaxy(
        source_truth=source_truth,
        mge_source_truth_path=mge_source_truth_path,
    )
    tracer = al.Tracer(galaxies=[lens, source])

    dataset = sim.via_tracer_from(tracer=tracer, grid=grid)
    aplt.fits_imaging(
        dataset=dataset,
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        overwrite=True,
    )
    al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")

    sci_grid = al.Grid2D.uniform(
        shape_native=SCIENCE_GRID_SHAPE, pixel_scales=SCIENCE_PIXEL_SCALE
    )
    source_plane_image = source.image_2d_from(grid=sci_grid)
    source_plane_flux = float(np.sum(source_plane_image))
    traced_grid_list = tracer.traced_grid_2d_list_from(grid=sci_grid)
    lensed_source_image = source.image_2d_from(grid=traced_grid_list[1])
    image_plane_flux = float(np.sum(lensed_source_image))
    source_magnification = image_plane_flux / source_plane_flux

    truth = {
        "image_plane_flux": image_plane_flux,
        "source_plane_flux": source_plane_flux,
        "source_magnification": float(source_magnification),
    }
    al.output_to_json(file_path=dataset_path / "source_science.json", obj=truth)
    return truth
