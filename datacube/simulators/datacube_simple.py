"""
Simulator: Datacube — list of Interferometer channels with a Gaussian emission line
====================================================================================

Phase 1 prototype dataset for ALMA datacube modeling. A "datacube" here is
a Python list of ``Interferometer`` objects, one per spectral channel. Every
channel shares the same lens galaxy and observes the same source morphology;
only the source intensity varies channel-to-channel, following a Gaussian
emission-line profile centred on a peak channel.

This is the deliberately simple end of Aris's two-option design from
``PyAutoPrompt/issued/alma_datacube.md``. The fast path that re-uses
``L^T W~ L`` across channels is a follow-up. Here every channel runs its own
NUFFT and its own inversion; the shared lens model is identified across
channels by the ``FactorGraphModel`` on the modeling side.

Layout written to disk
----------------------
``dataset/datacube/sim_simple/``
    ``channel_000/``
        ``data.fits``            — visibilities, shape (n_vis, 2) (real, imag)
        ``noise_map.fits``       — visibility noise, shape (n_vis, 2)
        ``uv_wavelengths.fits``  — (u, v) baselines, shape (n_vis, 2)
        ``tracer.json``          — true tracer for this channel
    ``channel_001/`` ...
    ``cube_summary.json``        — emission-line parameters used

Every channel writes its own copy of ``uv_wavelengths.fits``. Phase 1 keeps
all channels symmetric: the modeling side will load each channel as an
independent ``Interferometer`` object via ``Interferometer.from_fits``.

Usage
-----
Run from the ``autolens_workspace_developer`` repo root::

    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \\
        python datacube/simulators/datacube_simple.py
"""

import json
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

import autolens as al


# ---------------------------------------------------------------------------
# Cube configuration
# ---------------------------------------------------------------------------

N_CHANNELS = 4
PEAK_CHANNEL = 1.5            # emission-line peak (between channel 1 and 2)
SIGMA_CHANNEL = 1.2           # emission-line width in channels
PEAK_INTENSITY = 0.6          # source intensity at the line peak

INSTRUMENT = {
    "n_visibilities": 190,
    "uv_scale": 3.0e5,
    "pixel_scale": 0.1,
    "shape_native": (256, 256),
    "noise_sigma": 1000.0,
    "uv_seed": 1,
    "noise_seed": 1,
}


def _synthetic_uv_wavelengths(n_visibilities: int, uv_scale: float, seed: int) -> np.ndarray:
    """Reproducible synthetic (u, v) baseline distribution.

    Same recipe as ``jax_profiling/interferometer/simulators/interferometer.py``
    so the cube is comparable to the single-channel SMA preset.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=uv_scale / 3.0, size=(n_visibilities, 2)).astype(np.float64)


def _channel_intensity(channel: int) -> float:
    """Gaussian emission-line profile in channel index."""
    return PEAK_INTENSITY * float(
        np.exp(-0.5 * ((channel - PEAK_CHANNEL) / SIGMA_CHANNEL) ** 2)
    )


def _lens_galaxy() -> al.Galaxy:
    return al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.6,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        ),
        shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
    )


def _source_galaxy(intensity: float) -> al.Galaxy:
    return al.Galaxy(
        redshift=1.0,
        bulge=al.lp.SersicCore(
            centre=(0.1, 0.1),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=intensity,
            effective_radius=1.0,
            sersic_index=2.5,
        ),
    )


def _plot_cube_overview(cube_path, channel_intensities, datasets, tracers, grid):
    """Row-per-channel sanity plot.

    Columns: lensed image (real space) | Re(visibilities) in the uv-plane |
    Im(visibilities) in the uv-plane | |visibilities| vs baseline length.
    """
    fig, axes = plt.subplots(
        N_CHANNELS, 4, figsize=(16, 3.4 * N_CHANNELS), squeeze=False
    )

    for c in range(N_CHANNELS):
        dataset = datasets[c]
        tracer = tracers[c]
        uv = np.asarray(dataset.uv_wavelengths)
        re = np.asarray(dataset.data.real)
        im = np.asarray(dataset.data.imag)
        amp = np.hypot(re, im)
        baseline = np.hypot(uv[:, 0], uv[:, 1])

        image = tracer.image_2d_from(grid=grid)
        axes[c, 0].imshow(np.asarray(image.native), origin="lower", cmap="hot")
        axes[c, 0].set_title(
            f"channel {c}: lensed image (I={channel_intensities[c]:.3f})"
        )
        axes[c, 0].set_axis_off()

        sc = axes[c, 1].scatter(uv[:, 0], uv[:, 1], c=re, s=8, cmap="RdBu_r")
        axes[c, 1].set_title("Re(visibilities)")
        axes[c, 1].set_aspect("equal")
        axes[c, 1].set_xlabel("u")
        axes[c, 1].set_ylabel("v")
        plt.colorbar(sc, ax=axes[c, 1], fraction=0.046)

        sc = axes[c, 2].scatter(uv[:, 0], uv[:, 1], c=im, s=8, cmap="RdBu_r")
        axes[c, 2].set_title("Im(visibilities)")
        axes[c, 2].set_aspect("equal")
        axes[c, 2].set_xlabel("u")
        axes[c, 2].set_ylabel("v")
        plt.colorbar(sc, ax=axes[c, 2], fraction=0.046)

        axes[c, 3].scatter(baseline, amp, s=8, alpha=0.6)
        axes[c, 3].set_title("|vis| vs baseline length")
        axes[c, 3].set_xlabel(r"$\sqrt{u^2 + v^2}$")
        axes[c, 3].set_ylabel("|visibilities|")

    fig.tight_layout()
    fig.savefig(cube_path / "cube_overview.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(N_CHANNELS), channel_intensities, "o-")
    ax.set_xlabel("channel index")
    ax.set_ylabel("source intensity")
    ax.set_title(
        f"emission-line spectrum (peak={PEAK_CHANNEL}, sigma={SIGMA_CHANNEL})"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(cube_path / "spectrum.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def simulate():
    pixel_scale = INSTRUMENT["pixel_scale"]
    shape_native = INSTRUMENT["shape_native"]

    cube_path = Path("dataset") / "datacube" / "sim_simple"
    cube_path.mkdir(parents=True, exist_ok=True)

    grid = al.Grid2D.uniform(shape_native=shape_native, pixel_scales=pixel_scale)

    uv_wavelengths = _synthetic_uv_wavelengths(
        n_visibilities=INSTRUMENT["n_visibilities"],
        uv_scale=INSTRUMENT["uv_scale"],
        seed=INSTRUMENT["uv_seed"],
    )

    lens_galaxy = _lens_galaxy()

    channel_intensities = []
    datasets = []
    tracers = []

    for channel in range(N_CHANNELS):
        intensity = _channel_intensity(channel)
        channel_intensities.append(intensity)

        channel_path = cube_path / f"channel_{channel:03d}"
        channel_path.mkdir(parents=True, exist_ok=True)

        source_galaxy = _source_galaxy(intensity=intensity)
        tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

        simulator = al.SimulatorInterferometer(
            uv_wavelengths=uv_wavelengths,
            exposure_time=300.0,
            noise_sigma=INSTRUMENT["noise_sigma"],
            transformer_class=al.TransformerDFT,
            noise_seed=INSTRUMENT["noise_seed"] + channel,
        )

        dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
        datasets.append(dataset)
        tracers.append(tracer)

        al.output_to_fits(
            values=np.stack([dataset.data.real, dataset.data.imag], axis=-1),
            file_path=channel_path / "data.fits",
            overwrite=True,
        )
        al.output_to_fits(
            values=np.stack([dataset.noise_map.real, dataset.noise_map.imag], axis=-1),
            file_path=channel_path / "noise_map.fits",
            overwrite=True,
        )
        al.output_to_fits(
            values=dataset.uv_wavelengths,
            file_path=channel_path / "uv_wavelengths.fits",
            overwrite=True,
        )
        al.output_to_json(
            obj=tracer,
            file_path=channel_path / "tracer.json",
        )

        print(
            f"  channel {channel:03d}: intensity={intensity:.4f}, "
            f"|vis|_max={np.max(np.abs(dataset.data)):.3e}"
        )

    _plot_cube_overview(
        cube_path=cube_path,
        channel_intensities=channel_intensities,
        datasets=datasets,
        tracers=tracers,
        grid=grid,
    )

    summary = {
        "n_channels": N_CHANNELS,
        "peak_channel": PEAK_CHANNEL,
        "sigma_channel": SIGMA_CHANNEL,
        "peak_intensity": PEAK_INTENSITY,
        "channel_intensities": channel_intensities,
        "instrument": {
            "n_visibilities": INSTRUMENT["n_visibilities"],
            "uv_scale": INSTRUMENT["uv_scale"],
            "pixel_scale": INSTRUMENT["pixel_scale"],
            "shape_native": list(INSTRUMENT["shape_native"]),
            "noise_sigma": INSTRUMENT["noise_sigma"],
            "uv_seed": INSTRUMENT["uv_seed"],
            "noise_seed": INSTRUMENT["noise_seed"],
        },
    }
    with open(cube_path / "cube_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  cube simulated: {cube_path}")
    print(f"    channels:           {N_CHANNELS}")
    print(f"    visibilities/chan:  {uv_wavelengths.shape[0]}")
    print(f"    real-space grid:    {shape_native[0]} x {shape_native[1]}")
    print(f"    peak intensity:     {PEAK_INTENSITY}")


if __name__ == "__main__":
    simulate()
