"""
Simulator Profiling: Cluster Scale (Headliner)
================================================

Profiles `autolens_workspace/scripts/cluster/simulator.py` — the script the
user reported as slow despite JIT-wrapped `solver.solve`. Pinpoints where time
goes by timing every major phase separately:

- Pytree registration (register_model + register_instance_pytree)
- PointSolver construction (800×800 @ 0.1"/px)
- JIT-compiled solver.solve PER SOURCE (compile cost shown separately for each)
  so the per-source compile overhead is visible vs. steady-state
- `simulator.via_tracer_from` on 1000×1000 over-sampled imaging grid
  (single call — too slow to repeat; this is the suspected numpy bottleneck)
- FITS + JSON + CSV output

The summary JSON and bar chart make the gap between `solver_compile_total` and
`via_tracer_from` clearly visible — that's the diagnostic the user needs.

Run from any path:
    python jax_profiling/simulators/cluster.py
"""

from autoconf import jax_wrapper  # noqa: F401 — must be first

import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import autofit as af
import autolens as al
import autolens.plot as aplt

from autofit.jax import register_model as _register_model_pytrees
from autoarray.abstract_ndarray import register_instance_pytree
from autolens.lens.tracer import Tracer


# ---------------------------------------------------------------------------
# Helpers
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
        print("PROFILING SUMMARY")
        print("=" * 70)
        max_label = max(len(r[0]) for r in self.records)
        total = 0.0
        for label, elapsed in self.records:
            print(f"  {label:<{max_label}}  {elapsed:>10.4f} s")
            total += elapsed
        print("-" * 70)
        print(f"  {'TOTAL':<{max_label}}  {total:>10.4f} s")
        print("=" * 70)


def block(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def jit_profile(func, label, *args, n_repeats=10):
    jitted = jax.jit(func)

    with timer.section(f"{label}_lower"):
        lowered = jitted.lower(*args)

    with timer.section(f"{label}_compile"):
        compiled = lowered.compile()

    with timer.section(f"{label}_first_call"):
        result = compiled(*args)
        block(result)

    with timer.section(f"{label}_steady_x{n_repeats}"):
        for _ in range(n_repeats):
            result = compiled(*args)
            block(result)

    per_call = timer.records[-1][1] / n_repeats
    print(f"    -> per-call avg: {per_call:.6f} s")
    return compiled, result


timer = Timer()

_script_dir = Path(__file__).resolve().parent
_workspace_root = _script_dir.parents[1]

dataset_name = "simple"
dataset_path = _workspace_root / "jax_profiling" / "dataset" / "cluster" / dataset_name
dataset_path.mkdir(parents=True, exist_ok=True)


# === PART 1 — Setup ===

print("\n--- PART 1: Setup ---")

redshift_lens = 0.5
source_redshifts = [1.0, 2.0]

main_lens_centres = [
    (0.0, 0.0),
    (10.0, 8.0),
]
host_halo_centre = (0.0, 0.0)
source_centres = [
    (0.3, 0.5),
    (-0.8, 1.2),
]

with timer.section("setup_imaging_grid"):
    imaging_grid = al.Grid2D.uniform(shape_native=(1000, 1000), pixel_scales=0.1)
    imaging_over_sample = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=imaging_grid,
        sub_size_list=[32, 8, 2],
        radial_list=[0.3, 0.6],
        centre_list=main_lens_centres,
    )
    imaging_grid = imaging_grid.apply_over_sampling(
        over_sample_size=imaging_over_sample
    )

with timer.section("setup_galaxies"):
    main_lens_dpie_params = [
        (8.0, 20.0, 3.0),
        (5.0, 12.0, 1.2),
    ]
    main_lens_sersic_params = [
        (1.5, 3.0, 4.0),
        (0.8, 1.5, 3.5),
    ]
    main_lens_galaxies = []
    for centre, (ra, rs, b0), (intensity, effective_radius, sersic_index) in zip(
        main_lens_centres, main_lens_dpie_params, main_lens_sersic_params
    ):
        bulge = al.lp.SersicSph(
            centre=centre,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
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
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0 + 30.0 * i),
            intensity=2.0,
            effective_radius=0.3,
            sersic_index=1.0,
        )
        point = al.ps.Point(centre=centre)
        source_galaxies.append(
            al.Galaxy(redshift=src_z, bulge=bulge, **{f"point_{i}": point})
        )

with timer.section("setup_tracer"):
    tracer = al.Tracer(
        galaxies=main_lens_galaxies + [host_halo_galaxy] + source_galaxies
    )


# === PART 2 — Pytree registration ===

print("\n--- PART 2: Pytree registration ---")

with timer.section("register_pytrees"):
    _lens_models = [
        af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=af.Model(
                al.lp.SersicSph,
                centre=g.bulge.centre,
                intensity=g.bulge.intensity,
                effective_radius=g.bulge.effective_radius,
                sersic_index=g.bulge.sersic_index,
            ),
            mass=af.Model(
                al.mp.dPIEMassSph,
                centre=g.mass.centre,
                ra=g.mass.ra,
                rs=g.mass.rs,
                b0=g.mass.b0,
            ),
        )
        for g in main_lens_galaxies
    ]
    _halo_model = af.Model(
        al.Galaxy,
        redshift=redshift_lens,
        dark=af.Model(
            al.mp.NFWMCRLudlowSph,
            centre=host_halo_centre,
            mass_at_200=host_halo.mass_at_200,
            redshift_object=redshift_lens,
            redshift_source=max(source_redshifts),
        ),
    )
    _source_models = [
        af.Model(
            al.Galaxy,
            redshift=src_z,
            bulge=af.Model(
                al.lp.SersicCore,
                centre=src_centre,
                ell_comps=al.convert.ell_comps_from(
                    axis_ratio=0.8, angle=60.0 + 30.0 * i
                ),
                intensity=2.0,
                effective_radius=0.3,
                sersic_index=1.0,
            ),
            **{
                f"point_{i}": af.Model(al.ps.Point, centre=src_centre),
            },
        )
        for i, (src_centre, src_z) in enumerate(zip(source_centres, source_redshifts))
    ]
    _registration_model = af.Collection(
        galaxies=af.Collection(*(_lens_models + [_halo_model] + _source_models))
    )
    _register_model_pytrees(_registration_model)
    register_instance_pytree(Tracer, no_flatten=("cosmology",))


# === PART 3 — PointSolver build ===

print("\n--- PART 3: PointSolver build ---")

with timer.section("solver_build"):
    solver = al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(800, 800), pixel_scales=0.1),
        pixel_scale_precision=0.001,
        magnification_threshold=0.1,
    )


# === PART 4 — JIT solve per source (compile cost visible per-source) ===

print("\n--- PART 4: jitted_solve per source ---")

@jax.jit
def jitted_solve(tracer, source_plane_coordinate):
    return solver.solve(
        tracer=tracer,
        source_plane_coordinate=source_plane_coordinate,
        xp=jnp,
        remove_infinities=False,
    ).array


positions_list = []

for i, src_centre in enumerate(source_centres):
    coord = jnp.asarray(src_centre)

    # Use jit_profile so lower / compile / first / steady are all recorded
    # independently for each source — this is what makes per-source compile
    # cost visible.
    _, raw = jit_profile(
        lambda c, _tracer=tracer: jitted_solve(_tracer, c),
        f"jitted_solve_src{i}",
        coord,
        n_repeats=3,
    )

    raw_np = np.asarray(raw)
    finite_mask = ~(np.isinf(raw_np).any(axis=1) | np.isnan(raw_np).any(axis=1))
    positions_list.append(al.Grid2DIrregular(raw_np[finite_mask]))
    print(f"  src{i}: {len(positions_list[-1])} image positions found")


# === PART 5 — simulator.via_tracer_from (expected numpy bottleneck) ===

print("\n--- PART 5: simulator.via_tracer_from (1000×1000 over-sampled) ---")

with timer.section("setup_psf_simulator"):
    psf = al.Convolver.from_gaussian(
        shape_native=(11, 11), sigma=0.1, pixel_scales=imaging_grid.pixel_scales
    )
    simulator = al.SimulatorImaging(
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_poisson_noise_to_data=True,
    )

# Single call only — too slow to repeat on 1000×1000 over-sampled grid
np.random.seed(1)
with timer.section("via_tracer_from"):
    imaging_dataset = simulator.via_tracer_from(tracer=tracer, grid=imaging_grid)


# === PART 6 — Outputs (after timing) ===

print("\n--- PART 6: outputs ---")

with timer.section("output_fits"):
    aplt.fits_imaging(
        dataset=imaging_dataset,
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        overwrite=True,
    )

with timer.section("output_point_datasets"):
    # Position noise = 5 mas (HST PSF-centroiding precision), not the imaging pixel scale.
    position_noise = 0.005
    dataset_list = []
    for i, positions in enumerate(positions_list):
        ds = al.PointDataset(
            name=f"point_{i}",
            positions=positions,
            positions_noise_map=position_noise,
            redshift=source_redshifts[i],
        )
        dataset_list.append(ds)
        al.output_to_json(
            obj=ds, file_path=dataset_path / f"point_dataset_{i}.json"
        )
    al.output_to_csv(
        datasets=dataset_list,
        file_path=dataset_path / "point_datasets.csv",
    )

with timer.section("output_json"):
    al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")
    al.output_to_json(
        obj=al.Grid2DIrregular(main_lens_centres),
        file_path=dataset_path / "main_lens_centres.json",
    )
    al.output_to_json(
        obj=al.Grid2DIrregular([host_halo_centre]),
        file_path=dataset_path / "host_halo_centre.json",
    )
    al.output_to_json(
        obj=al.Grid2DIrregular(source_centres),
        file_path=dataset_path / "source_centres.json",
    )


# === Summary — make compile vs via_tracer_from gap obvious ===

al_version = al.__version__
results_dir = _workspace_root / "jax_profiling" / "results" / "simulators"
results_dir.mkdir(parents=True, exist_ok=True)

phases = dict(timer.records)

# Aggregate compile totals across both sources
solver_compile_total = sum(
    v for k, v in phases.items() if "compile" in k and "solve" in k
)
solver_lower_total = sum(
    v for k, v in phases.items() if "lower" in k and "solve" in k
)
solver_steady_total = sum(
    v for k, v in phases.items() if "steady" in k and "solve" in k
)

via_tracer_time = phases.get("via_tracer_from", 0.0)

print("\n" + "=" * 70)
print("CLUSTER DIAGNOSTIC: compile vs via_tracer_from")
print("=" * 70)
print(f"  solver_lower_total  (both sources):  {solver_lower_total:.4f} s")
print(f"  solver_compile_total (both sources): {solver_compile_total:.4f} s")
print(f"  solver_steady_total  (both sources): {solver_steady_total:.4f} s")
print(f"  via_tracer_from:                     {via_tracer_time:.4f} s")
if via_tracer_time > 0:
    ratio = solver_compile_total / via_tracer_time if via_tracer_time > 0 else float("inf")
    print(f"  compile / via_tracer ratio:          {ratio:.2f}x")
print("=" * 70)

results_summary = {
    "autolens_version": al_version,
    "type": "cluster",
    "configuration": {
        "imaging_grid_shape": [1000, 1000],
        "solver_grid_shape": [800, 800],
        "pixel_scales": 0.1,
        "n_sources": len(source_centres),
        "source_redshifts": source_redshifts,
        "n_lens_galaxies": len(main_lens_centres),
        "host_halo_mass_at_200": 10**15.3,
        "over_sample_sub_sizes": [32, 8, 2],
    },
    "phases": phases,
    "diagnostic": {
        "solver_compile_total_s": solver_compile_total,
        "solver_lower_total_s": solver_lower_total,
        "solver_steady_total_s": solver_steady_total,
        "via_tracer_from_s": via_tracer_time,
        "compile_vs_via_tracer_ratio": (
            round(solver_compile_total / via_tracer_time, 2)
            if via_tracer_time > 0
            else None
        ),
    },
    "positions_found": {
        f"src{i}": len(p) for i, p in enumerate(positions_list)
    },
}

json_path = results_dir / f"cluster_summary_v{al_version}.json"
json_path.write_text(json.dumps(results_summary, indent=2))
print(f"\n  Results saved to: {json_path}")

# Bar chart — all phases; compile phases highlighted
labels = [r[0] for r in timer.records]
times = [r[1] for r in timer.records]

# Colour compile phases red, via_tracer_from orange, rest blue
colors = []
for lbl in labels:
    if "compile" in lbl:
        colors.append("#C44E52")
    elif "via_tracer" in lbl:
        colors.append("#DD8452")
    elif "lower" in lbl:
        colors.append("#937860")
    else:
        colors.append("#4C72B0")

fig, ax = plt.subplots(figsize=(13, max(5.0, len(labels) * 0.45)))
y_pos = range(len(labels))
bars = ax.barh(y_pos, times, color=colors, edgecolor="white", height=0.6)
for bar, t in zip(bars, times):
    ax.text(
        bar.get_width() + max(times) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.4f} s",
        va="center",
        fontsize=8,
    )
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Time (s)", fontsize=11)
fig.suptitle(
    "Simulator Profiling: Cluster Scale (Headliner)",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f"AutoLens v{al_version}  |  1000×1000 imaging / 800×800 solver  |  "
    f"red=compile, orange=via_tracer",
    fontsize=8,
)
ax.margins(x=0.22)
fig.tight_layout()
chart_path = results_dir / f"cluster_summary_v{al_version}.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to: {chart_path}")

timer.summary()
