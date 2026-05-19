"""
Shared visualisation helpers for the source-science experiment series.

Pure functions — no globals, no side effects beyond returning numpy arrays.
The per-test `make_diagnostics.py` scripts in `results/N_*/` import these.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import autofit as af
import autolens as al


# Default integration grid for the 2D source-plane brightness panel.
DEFAULT_SOURCE_2D_SHAPE = (400, 400)
DEFAULT_SOURCE_2D_PIXEL_SCALE = 0.005


def source_radial_profile(
    tracer: al.Tracer,
    r_max: float = 0.5,
    n: int = 1000,
    along: str = "x",
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the source-plane surface brightness along a 1D radial cut.

    Parameters
    ----------
    tracer
        Source galaxy is taken from `tracer.planes[1]`. The brightness is
        evaluated in the source plane (not lensed).
    r_max
        Outer radius in arcseconds.
    n
        Number of sample points between 0 and `r_max`.
    along
        "x" → cut along the x-axis through (0,0). Quick and reads cleanly
        for circular-ish sources.

    Returns
    -------
    (radii, brightness)
        Both shape (n,). `radii` in arcseconds, `brightness` in the same
        units as the source intensity (e- s⁻¹ arcsec⁻²).
    """
    if along != "x":
        raise NotImplementedError("Only along='x' is supported")

    radii = np.linspace(0.0, r_max, n)
    grid_radial = al.Grid2DIrregular(values=[(0.0, r) for r in radii])
    image_1d = tracer.planes[1].image_2d_from(grid=grid_radial)
    return radii, np.asarray(image_1d)


def source_cumulative_flux(
    tracer: al.Tracer,
    r_max: float = 3.0,
    n: int = 80,
    pixel_scale: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative integrated source flux inside circular apertures.

    Computed by evaluating the source-plane brightness on a fine 2D grid
    once, then summing pixels inside each aperture radius. This shows
    *where in radius* the integrated flux differs between models — the
    bias-source diagnostic.

    Parameters
    ----------
    tracer
        Source galaxy is taken from `tracer.planes[1]`.
    r_max
        Outer aperture radius in arcseconds.
    n
        Number of aperture radii to evaluate between 0 and `r_max`.
    pixel_scale
        Pixel scale of the underlying integration grid in arcseconds.

    Returns
    -------
    (radii, cumulative_flux)
        Both shape (n,). `radii` in arcseconds; `cumulative_flux` summed
        over pixels with the same area weighting per pixel (so total at
        large radius matches `np.sum(image_2d)`).
    """
    extent = int(np.ceil(2.0 * r_max / pixel_scale))
    grid = al.Grid2D.uniform(shape_native=(extent, extent), pixel_scales=pixel_scale)
    brightness = np.asarray(tracer.planes[1].image_2d_from(grid=grid))

    yx = np.asarray(grid.array)
    r_pixel = np.sqrt(yx[:, 0] ** 2 + yx[:, 1] ** 2)

    radii = np.linspace(0.0, r_max, n)
    cumulative = np.array(
        [float(np.sum(brightness[r_pixel <= r])) for r in radii]
    )
    return radii, cumulative


def source_2d_image(
    tracer: al.Tracer,
    shape: Tuple[int, int] = DEFAULT_SOURCE_2D_SHAPE,
    pixel_scale: float = DEFAULT_SOURCE_2D_PIXEL_SCALE,
) -> al.Array2D:
    """2D source-plane surface-brightness image on a regular grid."""
    grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=pixel_scale)
    image = tracer.planes[1].image_2d_from(grid=grid)
    return image  # autoarray Array2D


def solved_tracer_from_instance(
    instance, dataset: al.Imaging, has_linear_lp: bool
) -> al.Tracer:
    """Build a tracer from a `ModelInstance`; for MGE/linear-LP models,
    run the FitImaging inversion to populate per-Gaussian intensities."""
    tracer = al.Tracer(galaxies=instance.galaxies)
    if not has_linear_lp:
        return tracer
    fit = al.FitImaging(dataset=dataset, tracer=tracer)
    return fit.tracer_linear_light_profiles_to_light_profiles


def _draw_indices_from_pdf(samples, n_draws: int, rng=None) -> np.ndarray:
    """Robust replacement for `samples.draw_randomly_via_pdf` that
    re-normalises the weight list locally so floating-point drift in the
    cached samples can't break `np.random.choice`."""
    if rng is None:
        rng = np.random.default_rng()
    weights = np.asarray(samples.weight_list, dtype=np.float64)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError(
            f"sample weights do not sum to a positive finite number (got {total})"
        )
    weights = weights / total
    return rng.choice(len(weights), size=n_draws, p=weights)


def posterior_radial_profile_band(
    samples,
    dataset: al.Imaging,
    has_linear_lp: bool,
    r_max: float = 0.5,
    n: int = 1000,
    n_draws: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median + 1σ envelope of the source radial profile across N draws.

    Returns (radii, median, lower_1sigma, upper_1sigma). The 1σ band uses
    the 15.865 / 84.135 percentiles at each radius.
    """
    radii = np.linspace(0.0, r_max, n)
    profiles = []

    indices = _draw_indices_from_pdf(samples=samples, n_draws=n_draws)
    for idx in indices:
        vector = samples.parameter_lists[idx]
        try:
            instance = samples.model.instance_from_vector(
                vector=vector, ignore_assertions=True
            )
            tracer = solved_tracer_from_instance(
                instance=instance, dataset=dataset, has_linear_lp=has_linear_lp
            )
            _, brightness = source_radial_profile(tracer=tracer, r_max=r_max, n=n)
        except Exception as e:  # noqa: BLE001 - skip bad draws
            print(f"  posterior radial draw failed: {e}")
            continue
        profiles.append(brightness)

    if not profiles:
        nan = np.full_like(radii, np.nan)
        return radii, nan, nan, nan

    arr = np.stack(profiles, axis=0)
    median = np.median(arr, axis=0)
    lower = np.percentile(arr, 15.865, axis=0)
    upper = np.percentile(arr, 84.135, axis=0)
    return radii, median, lower, upper
