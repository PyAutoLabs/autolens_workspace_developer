"""Simulator for the rect_adapt_duo demo.

Builds a strong-lensing imaging dataset with two compact Sérsic sources at
source-plane positions ``(+0.5, +0.5)`` and ``(−0.5, −0.5)`` behind an
isothermal lens. The diagonal placement is what makes the demo work:
adaptive meshes that use a separable per-axis CDF (the existing
``RectangularSplineAdaptImage``) will outer-product their marginals and
place dense pixels at all four ``(±0.5, ±0.5)`` cross-products — two real
peaks plus two ghosts. The PCA-rotated variant aligns to the +45° diagonal
and avoids the ghosts.

Outputs (``./dataset/``):
- ``data.fits`` — image (data + Poisson noise + sky background)
- ``noise_map.fits`` — RMS noise per pixel
- ``psf.fits`` — Gaussian PSF
- ``tracer.json`` — truth tracer (lens + two sources) for re-use by
  ``adapt_image.py`` and ``compare_meshes.py``
"""
from pathlib import Path

import autolens as al

PIXEL_SCALES = 0.05
SHAPE_NATIVE = (150, 150)

LENS_REDSHIFT = 0.5
SOURCE_REDSHIFT = 1.0
EINSTEIN_RADIUS = 1.2

SOURCE_INTENSITY = 0.6
SOURCE_EFFECTIVE_RADIUS = 0.1
SOURCE_SERSIC_INDEX = 2.5
SOURCE_POSITIONS = [(+0.5, +0.5), (-0.5, -0.5)]


def build_tracer() -> al.Tracer:
    lens = al.Galaxy(
        redshift=LENS_REDSHIFT,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            ell_comps=(0.0, 0.0),
            einstein_radius=EINSTEIN_RADIUS,
        ),
    )
    sources = []
    for (cy, cx) in SOURCE_POSITIONS:
        sources.append(
            al.Galaxy(
                redshift=SOURCE_REDSHIFT,
                bulge=al.lp.Sersic(
                    centre=(cy, cx),
                    ell_comps=(0.0, 0.0),
                    intensity=SOURCE_INTENSITY,
                    effective_radius=SOURCE_EFFECTIVE_RADIUS,
                    sersic_index=SOURCE_SERSIC_INDEX,
                ),
            )
        )
    return al.Tracer(galaxies=[lens, *sources])


def main():
    dataset_path = Path(__file__).parent / "dataset"
    dataset_path.mkdir(exist_ok=True)

    grid = al.Grid2D.uniform(
        shape_native=SHAPE_NATIVE, pixel_scales=PIXEL_SCALES, over_sample_size=4
    )

    psf = al.Convolver.from_gaussian(
        shape_native=(11, 11), sigma=0.1, pixel_scales=PIXEL_SCALES
    )

    simulator = al.SimulatorImaging(
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_poisson_noise_to_data=True,
        noise_seed=1,
    )

    tracer = build_tracer()
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    import autolens.plot as aplt
    aplt.fits_imaging(
        dataset=dataset,
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        psf_path=dataset_path / "psf.fits",
        overwrite=True,
    )

    print(f"simulator: wrote dataset to {dataset_path}")
    print("  (use build_tracer() from this module to recover the truth tracer)")


if __name__ == "__main__":
    main()
