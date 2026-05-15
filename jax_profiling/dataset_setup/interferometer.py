"""
Simulator: Instrument-Based Interferometer Datasets
====================================================

Simulates interferometer datasets for different observatories, each with a
characteristic (u, v)-coverage and pixel scale:

    sma    — ~190 visibilities, 0.1  arcsec/pixel  (low resolution)
    alma   — ~1000 visibilities, 0.05 arcsec/pixel (high resolution)

Synthetic (u, v)-coverage is generated procedurally from a Gaussian blob in
the uv-plane with a seeded RNG, so runs are reproducible without requiring
an external uv_wavelengths .fits file. The scale of the blob is chosen so
that each preset probes a realistic range of angular frequencies for its
instrument.

Usage
-----
Run directly to simulate a specific instrument::

    python simulators/interferometer.py --instrument sma

Or import and call from another script::

    from simulators.interferometer import simulate
    simulate("sma")

The auto-simulation pattern in ``mge.py`` / ``pixelization.py`` calls this
via subprocess when the dataset does not already exist on disk.
"""

import argparse
import numpy as np
from pathlib import Path

import autolens as al


# ---------------------------------------------------------------------------
# Instrument definitions
# ---------------------------------------------------------------------------

INSTRUMENTS = {
    "sma": {
        "n_visibilities": 190,
        "uv_scale": 3.0e5,
        "pixel_scale": 0.1,
        "shape_native": (256, 256),
        "noise_sigma": 1000.0,
        "seed": 1,
        "transformer_class": "dft",
    },
    "alma": {
        "n_visibilities": 1000,
        "uv_scale": 2.0e6,
        "pixel_scale": 0.05,
        "shape_native": (256, 256),
        "noise_sigma": 100.0,
        "seed": 1,
        "transformer_class": "dft",
    },
    "hannah": {
        "n_visibilities": 16984,
        "uv_scale": 2.0e6,
        "pixel_scale": 0.125,
        "shape_native": (40, 40),
        "noise_sigma": 100.0,
        "seed": 1,
        "transformer_class": "nufft",
    },
}


def _synthetic_uv_wavelengths(n_visibilities: int, uv_scale: float, seed: int) -> np.ndarray:
    """Generate a reproducible synthetic (u, v) baseline distribution.

    The baselines are drawn from a 2D isotropic Gaussian in the uv-plane whose
    standard deviation is ``uv_scale / 3`` so that the 3-sigma envelope matches
    ``uv_scale``. This is a crude but sufficient stand-in for real instrument
    coverage when the goal is profiling, not imaging fidelity.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=uv_scale / 3.0, size=(n_visibilities, 2)).astype(np.float64)


def simulate(instrument: str):
    """Simulate an interferometer dataset for the given instrument preset.

    Writes ``data.fits`` (visibilities, stacked real/imag), ``noise_map.fits``
    (same shape), ``uv_wavelengths.fits`` and ``positions.json`` into
    ``dataset/interferometer/<instrument>/``.
    """
    if instrument not in INSTRUMENTS:
        raise ValueError(
            f"Unknown instrument '{instrument}'. "
            f"Choose from: {list(INSTRUMENTS.keys())}"
        )

    config = INSTRUMENTS[instrument]
    pixel_scale = config["pixel_scale"]
    shape_native = config["shape_native"]

    dataset_path = Path("jax_profiling") / "dataset" / "interferometer" / instrument
    dataset_path.mkdir(parents=True, exist_ok=True)

    grid = al.Grid2D.uniform(shape_native=shape_native, pixel_scales=pixel_scale)

    uv_wavelengths = _synthetic_uv_wavelengths(
        n_visibilities=config["n_visibilities"],
        uv_scale=config["uv_scale"],
        seed=config["seed"],
    )

    print(f"  Total visibilities: {uv_wavelengths.shape[0]}")

    transformer_choice = config.get("transformer_class", "dft").lower()
    transformer_class = {
        "dft": al.TransformerDFT,
        "nufft": al.TransformerNUFFT,
    }[transformer_choice]

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=uv_wavelengths,
        exposure_time=300.0,
        noise_sigma=config["noise_sigma"],
        transformer_class=transformer_class,
        noise_seed=1,
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

    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.SersicCore(
            centre=(0.1, 0.1),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=0.3,
            effective_radius=1.0,
            sersic_index=2.5,
        ),
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    al.output_to_fits(
        values=np.stack([dataset.data.real, dataset.data.imag], axis=-1),
        file_path=dataset_path / "data.fits",
        overwrite=True,
    )
    al.output_to_fits(
        values=np.stack([dataset.noise_map.real, dataset.noise_map.imag], axis=-1),
        file_path=dataset_path / "noise_map.fits",
        overwrite=True,
    )
    al.output_to_fits(
        values=dataset.uv_wavelengths,
        file_path=dataset_path / "uv_wavelengths.fits",
        overwrite=True,
    )

    solver = al.PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
    )

    positions = solver.solve(
        tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre
    )

    al.output_to_json(
        obj=positions,
        file_path=dataset_path / "positions.json",
    )

    al.output_to_json(
        obj=tracer,
        file_path=dataset_path / "tracer.json",
    )

    print(f"  Dataset simulated: {dataset_path}")
    print(f"    Instrument:           {instrument}")
    print(f"    Pixel scale:          {pixel_scale} arcsec/pixel")
    print(f"    Real-space grid:      {shape_native[0]} x {shape_native[1]}")
    print(f"    Visibilities:         {uv_wavelengths.shape[0]}")
    print(f"    Multiple-image pos:   {len(positions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate interferometer dataset for a given instrument preset."
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="sma",
        choices=list(INSTRUMENTS.keys()),
        help="Instrument preset (default: sma)",
    )
    args = parser.parse_args()
    simulate(args.instrument)
