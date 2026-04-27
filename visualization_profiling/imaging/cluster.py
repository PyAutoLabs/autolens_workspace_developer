"""
Visualization Profiling: Cluster Simulator
==========================================

Profiles the image-rendering and visualization phases of the cluster-scale
simulator at ``autolens_workspace/scripts/cluster/simulator.py``.

The cluster simulator was originally suspected to be slow because of its
``PointSolver`` call. JAX-jitting the solver (issue #89) brought solver runtime
down to ~22s including compile, which exposed two larger bottlenecks that
together still account for ~80% of the simulator's wall-clock time:

 - ``SimulatorImaging.via_tracer_from(grid=imaging_grid)`` — multi-plane
   ray-tracing of every pixel in the 1000x1000 high-resolution imaging grid,
   with sub-sampling up to 32x32 around each cluster member centre. ~92s on
   a warm CPU run.

 - ``aplt.subplot_tracer(tracer, grid=viz_grid)`` — multi-panel visualization
   on a coarse 200x200 grid. Each subplot internally re-ray-traces through
   the multi-plane tracer; the panels are rendered in numpy with no JAX
   acceleration. ~51s on a warm CPU run.

This script reproduces the simulator's geometry and instruments each rendering
and visualization step independently so the two costs can be attacked without
also re-running the (now-fast) solver pipeline. It mirrors the structure of
``jax_profiling/imaging/mge.py`` (per-step ``Timer.section`` blocks, summary
table at the end).

The script does not write any persistent dataset to disk — visualizations are
discarded into a temp directory so re-runs do not pollute the workspace data
folder.
"""

import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from autoconf import jax_wrapper  # noqa: F401  — sets JAX env

import autolens as al
import autolens.plot as aplt


# ---------------------------------------------------------------------------
# Profiling helpers (mirrors jax_profiling/imaging/mge.py)
# ---------------------------------------------------------------------------


class Timer:
    def __init__(self):
        self.records: list[tuple[str, float]] = []

    @contextmanager
    def section(self, label: str):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.records.append((label, elapsed))
        print(f"  [{label}] {elapsed:.4f} s")

    def summary(self):
        print("\n" + "=" * 70)
        print("VISUALIZATION PROFILING SUMMARY")
        print("=" * 70)
        max_label = max(len(r[0]) for r in self.records)
        total = 0.0
        for label, elapsed in self.records:
            print(f"  {label:<{max_label}}  {elapsed:>10.4f} s")
            total += elapsed
        print("-" * 70)
        print(f"  {'TOTAL':<{max_label}}  {total:>10.4f} s")
        print("=" * 70)


timer = Timer()


# ---------------------------------------------------------------------------
# Cluster geometry — kept in sync with autolens_workspace/scripts/cluster/simulator.py
# ---------------------------------------------------------------------------

print("\n--- Build cluster tracer ---")

with timer.section("build_tracer"):
    redshift_lens = 0.5
    source_redshifts = [1.0, 2.0]

    main_lens_centres = [
        (0.0, 0.0),      # BCG
        (10.0, 8.0),     # satellite
    ]
    host_halo_centre = (0.0, 0.0)
    source_centres = [
        (0.3, 0.5),
        (-0.8, 1.2),
    ]

    main_lens_dpie_params = [(8.0, 20.0, 3.0), (5.0, 12.0, 1.2)]
    main_lens_sersic_params = [(1.5, 3.0, 4.0), (0.8, 1.5, 3.5)]

    main_lens_galaxies = []
    for centre, (ra, rs, b0), (intensity, eff_r, sersic_n) in zip(
        main_lens_centres, main_lens_dpie_params, main_lens_sersic_params
    ):
        bulge = al.lp.SersicSph(
            centre=centre,
            intensity=intensity,
            effective_radius=eff_r,
            sersic_index=sersic_n,
        )
        mass = al.mp.dPIEMassSph(centre=centre, ra=ra, rs=rs, b0=b0)
        main_lens_galaxies.append(
            al.Galaxy(redshift=redshift_lens, bulge=bulge, mass=mass)
        )

    host_halo = al.mp.NFWMCRLudlowSph(
        centre=host_halo_centre,
        mass_at_200=10**15.3,
        redshift_object=redshift_lens,
        redshift_source=max(source_redshifts),
    )
    host_halo_galaxy = al.Galaxy(redshift=redshift_lens, dark=host_halo)

    source_galaxies = []
    for i, (centre, src_z) in enumerate(zip(source_centres, source_redshifts)):
        bulge = al.lp.SersicCore(
            centre=centre,
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=0.8, angle=60.0 + 30.0 * i
            ),
            intensity=2.0,
            effective_radius=0.3,
            sersic_index=1.0,
        )
        point = al.ps.Point(centre=centre)
        source_galaxies.append(
            al.Galaxy(redshift=src_z, bulge=bulge, **{f"point_{i}": point})
        )

    tracer = al.Tracer(
        galaxies=main_lens_galaxies + [host_halo_galaxy] + source_galaxies
    )


# ---------------------------------------------------------------------------
# Image-rendering profiling
# ---------------------------------------------------------------------------

print("\n--- Image rendering: SimulatorImaging.via_tracer_from ---")

# Variants compared:
#  1. baseline                            — 1000x1000 @ 0.1, sub_size=[32,8,2]
#  2. half-resolution                     — 500x500 @ 0.2, sub_size=[32,8,2]
#  3. half-res + lighter over-sampling    — 500x500 @ 0.2, sub_size=[8,4,2]
#  4. baseline shape, no over-sampling    — 1000x1000 @ 0.1, sub_size=1

variants = [
    ("baseline_1000x1000_oversample_32_8_2", (1000, 1000), 0.1, [32, 8, 2]),
    ("half_500x500_oversample_32_8_2",        (500, 500),   0.2, [32, 8, 2]),
    ("half_500x500_oversample_8_4_2",         (500, 500),   0.2, [8, 4, 2]),
    ("baseline_1000x1000_no_oversample",      (1000, 1000), 0.1, None),
]

psf = None  # built once below

for label, shape, pixel_scale, sub_size_list in variants:
    with timer.section(f"build_grid::{label}"):
        grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=pixel_scale)
        if sub_size_list is not None:
            over_sample = al.util.over_sample.over_sample_size_via_radial_bins_from(
                grid=grid,
                sub_size_list=sub_size_list,
                radial_list=[0.3, 0.6],
                centre_list=main_lens_centres,
            )
            grid = grid.apply_over_sampling(over_sample_size=over_sample)

    if psf is None:
        psf = al.Convolver.from_gaussian(
            shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
        )

    simulator = al.SimulatorImaging(
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_poisson_noise_to_data=True,
    )

    with timer.section(f"via_tracer_from::{label}"):
        _ = simulator.via_tracer_from(tracer=tracer, grid=grid)


# ---------------------------------------------------------------------------
# Visualization profiling
# ---------------------------------------------------------------------------

print("\n--- Visualization plotters ---")

# Use a throwaway output directory so successive runs do not pollute disk.
tmp_dir = Path(tempfile.mkdtemp(prefix="cluster_viz_profile_"))

# A fast reference dataset for the imaging-dataset plotter — built once on
# the half-resolution grid, since we are profiling plotting cost not rendering.
ref_grid = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.2)
ref_simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=al.Convolver.from_gaussian(
        shape_native=(11, 11), sigma=0.1, pixel_scales=ref_grid.pixel_scales
    ),
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)
ref_dataset = ref_simulator.via_tracer_from(tracer=tracer, grid=ref_grid)

with timer.section("subplot_imaging_dataset"):
    aplt.subplot_imaging_dataset(dataset=ref_dataset)

# subplot_tracer / subplot_galaxies_images on a series of grid resolutions.
viz_shapes = [(50, 50), (100, 100), (200, 200), (500, 500)]
for ny, nx in viz_shapes:
    viz_pixel_scale = 100.0 / ny  # always span the same 100" field
    viz_grid = al.Grid2D.uniform(shape_native=(ny, nx), pixel_scales=viz_pixel_scale)

    with timer.section(f"subplot_tracer::{ny}x{nx}"):
        aplt.subplot_tracer(
            tracer=tracer,
            grid=viz_grid,
            output_path=tmp_dir,
            output_format="png",
        )

    with timer.section(f"subplot_galaxies_images::{ny}x{nx}"):
        aplt.subplot_galaxies_images(
            tracer=tracer,
            grid=viz_grid,
            output_path=tmp_dir,
            output_format="png",
        )

# ---------------------------------------------------------------------------
# Cleanup + summary
# ---------------------------------------------------------------------------

shutil.rmtree(tmp_dir, ignore_errors=True)

timer.summary()

print(
    "\nNotes:\n"
    "  - `via_tracer_from` cost scales with both pixel count and over_sample.\n"
    "  - `subplot_tracer` / `subplot_galaxies_images` cost scales with viz grid\n"
    "    resolution AND with the number of internal panels rendered. Compare\n"
    "    against the same plotters on a galaxy-scale tracer to isolate the\n"
    "    multi-plane / multi-galaxy contribution.\n"
)
