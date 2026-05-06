"""
Simulator: Point-Source Dataset
================================

Simulates a lensed point-source `PointDataset` for the JAX JIT profiling
scripts in ``jax_profiling/point_source/``.

The dataset contains the image-plane (y, x) positions found by the
``PointSolver`` for an Isothermal lens + ``Point`` source, with seeded
Gaussian noise added to the positions so the resulting log-likelihood is
deterministic across runs.  This is the point-source equivalent of the
``noise_seed=1`` option used by ``simulators/imaging.py`` and the seeded
RNG used by ``simulators/interferometer.py``.

Usage
-----
Run directly to simulate the canonical ``simple`` dataset::

    python simulators/point_source.py --name simple

Or import and call from another script::

    from simulators.point_source import simulate
    simulate("simple")

The auto-simulation pattern in ``source_plane.py`` / ``image_plane.py``
calls this via subprocess when the dataset does not already exist on disk.
"""

import argparse
import numpy as np
from pathlib import Path

import autolens as al


# Single dataset config — kept dict-shaped so adding more (e.g. quad image
# system, low-S/N variant) later mirrors the imaging/interferometer pattern.
DATASETS = {
    "simple": {
        "lens_centre": (0.01, 0.01),
        "lens_einstein_radius": 1.6,
        "lens_ell_comps": (0.01, 0.01),
        "source_centre": (0.07, 0.07),
        "grid_shape_native": (100, 100),
        "grid_pixel_scales": 0.2,
        "pixel_scale_precision": 0.001,
        "magnification_threshold": 0.1,
        "noise_seed": 1,
    },
}


def simulate(name: str = "simple"):
    """Simulate a point-source dataset with seeded positions noise.

    Parameters
    ----------
    name
        Key into ``DATASETS``.  Currently only ``"simple"`` is provided.
    """
    if name not in DATASETS:
        raise ValueError(
            f"Unknown point-source dataset '{name}'. "
            f"Choose from: {list(DATASETS.keys())}"
        )

    config = DATASETS[name]

    dataset_path = Path("jax_profiling") / "dataset" / "point_source" / name
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Build the truth tracer.
    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=config["lens_centre"],
            einstein_radius=config["lens_einstein_radius"],
            ell_comps=config["lens_ell_comps"],
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        point_0=al.ps.Point(centre=config["source_centre"]),
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    # Solve for image-plane multiple images.
    grid = al.Grid2D.uniform(
        shape_native=config["grid_shape_native"],
        pixel_scales=config["grid_pixel_scales"],
    )

    solver = al.PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=config["pixel_scale_precision"],
        magnification_threshold=config["magnification_threshold"],
    )

    positions = solver.solve(
        tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
    )

    # Seeded Gaussian noise injection — sigma matches the canonical workspace simulator
    # at autolens_workspace/scripts/point_source/simulator.py (5 mas, HST PSF-centroiding
    # precision, *not* the imaging pixel scale). Uses np.random.default_rng so reruns
    # are bit-identical.
    position_noise = 0.005
    rng = np.random.default_rng(seed=config["noise_seed"])
    positions_with_noise = positions + rng.normal(
        loc=0.0, scale=position_noise, size=positions.shape
    )
    positions_with_noise = al.Grid2DIrregular(values=positions_with_noise)

    dataset = al.PointDataset(
        name="point_0",
        positions=positions_with_noise,
        positions_noise_map=position_noise,
    )

    al.output_to_json(
        obj=dataset,
        file_path=dataset_path / "point_dataset_positions_only.json",
    )
    al.output_to_json(
        obj=tracer,
        file_path=dataset_path / "tracer.json",
    )

    print(f"  Dataset simulated: {dataset_path}")
    print(f"    Name:                {name}")
    print(f"    Lens centre:         {config['lens_centre']}")
    print(f"    Einstein radius:     {config['lens_einstein_radius']}")
    print(f"    Source centre:       {config['source_centre']}")
    print(f"    Image positions:     {positions.shape[0]}")
    print(f"    Position noise sigma:{grid.pixel_scale}")
    print(f"    Noise seed:          {config['noise_seed']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate a point-source PointDataset (seeded RNG)."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="simple",
        choices=list(DATASETS.keys()),
        help="Dataset name (default: simple)",
    )
    args = parser.parse_args()
    simulate(args.name)
