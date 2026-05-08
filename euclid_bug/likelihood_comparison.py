"""
Three-way log_evidence + per-step value comparison for the Euclid vis_pix
model on a single real lens (Tile102005065...).

Mirrors the pattern in
``autolens_workspace_developer/jax_profiling/misc/pixelization_sparse_cpu.py``,
but driven by:
  - the actual Euclid dataset (loaded via util.load_vis_dataset, with
    artefact_binary noise scaling and radial-bin over-sampling), and
  - the actual vis_pix model (Hilbert image-mesh + Delaunay mesh with
    zeroed_pixels + AdaptSplit regularization),

i.e. the exact configuration that misbehaves on HPC. The synthetic
HST-simulator comparison in pixelization_sparse_cpu.py shows agreement to
rtol=1e-4; this script asks whether the same agreement holds for the
production setup, and if not, *which step* diverges.

Workflow:
  1. Load the Euclid dataset.
  2. Run a fast vis_lp inline (small n_live) to get a source result for the
     adapt-image machinery.
  3. Build the vis_pix model, instantiate a tracer at the prior-medians
     evaluation point.
  4. Build three FitImaging objects:
       - CPU non-sparse  (xp=np, no apply_sparse_operator_cpu)
       - CPU sparse op   (xp=np, with apply_sparse_operator_cpu)
       - JAX             (xp=jnp, no sparse op)        [if jax installed]
  5. Compare log_evidence three-way.
  6. Compare D / F / H / s / mapped-image arrays element-wise across paths.
  7. Emit a JSON summary.

Run from any cwd:

    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \\
        python autolens_workspace_developer/euclid_bug/likelihood_comparison.py
"""

import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PYAUTO_LABS = _SCRIPT_DIR.parents[1]
_EUCLID_ROOT = _PYAUTO_LABS / "z_projects" / "euclid"
_EUCLID_SCRIPTS = _EUCLID_ROOT / "scripts"
sys.path.insert(0, str(_EUCLID_SCRIPTS))
import util  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_NAME = "Tile102005065RA0135279431487DECNEG0701599765928"
SAMPLE_NAME = None
COMPARISON_TAG = "likelihood_comparison"

# vis_lp is just to produce a source image for the Hilbert mesh / adapt path.
# Smaller n_live is fine; we don't need a converged mass model, just a
# reasonable source reconstruction.
VIS_LP_N_LIVE = 100
VIS_LP_N_LIKE_MAX = 10000

LOCAL_OUTPUT_PATH = _SCRIPT_DIR / "output"
RESULTS_DIR = _SCRIPT_DIR / "results"

# Pair-wise log_evidence agreement tolerance (also the per-step matrix
# tolerance flag in the report).
RTOL = 1e-4

# Whether to also try the JAX path. Auto-disabled if JAX is unavailable
# or if the JAX FitImaging build raises (caught and logged).
TRY_JAX = True


def _arr(x):
    if hasattr(x, "array"):
        return np.asarray(x.array)
    return np.asarray(x)


def _rel_diff(a, b):
    a = _arr(a)
    b = _arr(b)
    if a.shape != b.shape:
        return float("nan"), f"shape mismatch {a.shape} vs {b.shape}"
    denom = max(float(np.max(np.abs(a))), 1e-30)
    return float(np.max(np.abs(a - b)) / denom), "ok"


def main():
    from autoconf import conf
    conf.instance.push(
        new_path=_EUCLID_ROOT / "config",
        output_path=LOCAL_OUTPUT_PATH,
    )

    import autofit as af
    import autolens as al

    LOCAL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    have_jax = False
    if TRY_JAX:
        try:
            import jax.numpy as jnp  # noqa: F401
            have_jax = True
            print("[likelihood_comparison] JAX detected — will run JAX path.")
        except ImportError:
            print("[likelihood_comparison] JAX not importable — JAX path skipped.")

    # ===================================================================
    # PART A — Load dataset, run quick vis_lp
    # ===================================================================

    print("\n=== PART A: load dataset + run fast vis_lp ===")
    d = util.load_vis_dataset(DATASET_NAME, sample_name=SAMPLE_NAME)
    print(f"[A] mask_radius     = {d.mask_radius}")
    print(f"[A] dataset_centre  = {d.dataset_centre}")
    print(f"[A] image shape     = {d.dataset.shape_native}")

    settings_search = af.SettingsSearch(
        path_prefix=(
            (Path(SAMPLE_NAME) / DATASET_NAME)
            if SAMPLE_NAME is not None
            else Path(DATASET_NAME)
        ) / COMPARISON_TAG,
        unique_tag="vis_lp_prep",
        info={"magzero": d.magzero},
        session=None,
    )

    redshift_lens = 0.5
    redshift_source = 1.0

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=d.mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        centre=d.dataset_centre,
    )

    lp_mass = af.Model(al.mp.Isothermal)
    lp_mass.centre.centre_0 = d.dataset_centre[0]
    lp_mass.centre.centre_1 = d.dataset_centre[1]

    source_bulge = al.model_util.mge_model_from(
        mask_radius=d.mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=False,
    )

    lp_model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                mass=lp_mass,
                shear=af.Model(al.mp.ExternalShear),
            ),
            source=af.Model(
                al.Galaxy, redshift=redshift_source, bulge=source_bulge
            ),
        )
    )

    lp_analysis = al.AnalysisImaging(
        dataset=d.dataset,
        positions_likelihood_list=d.positions_likelihood_list,
        use_jax=False,
    )

    lp_search = af.Nautilus(
        name="vis_lp",
        **settings_search.search_dict,
        n_live=VIS_LP_N_LIVE,
        batch_size=50,
        n_like_max=VIS_LP_N_LIKE_MAX,
    )

    source_lp_result = lp_search.fit(
        model=lp_model, analysis=lp_analysis, **settings_search.fit_dict
    )
    print(
        f"[A] vis_lp prep done: "
        f"max_log_likelihood = {source_lp_result.max_log_likelihood_fit.log_likelihood}"
    )

    # ===================================================================
    # PART B — Build vis_pix model + adapt machinery (mirrors production)
    # ===================================================================

    print("\n=== PART B: build vis_pix model + adapt machinery ===")
    base_dataset = d.dataset
    mask = base_dataset.mask
    mask_radius = d.mask_radius

    image_mesh = al.image_mesh.Hilbert(
        pixels=500, weight_power=3.5, weight_floor=0.01
    )
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=base_dataset.mask,
        adapt_data=galaxy_image_name_dict["('galaxies', 'source')"],
    )

    edge_pixels_total = 30
    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=mask_radius + mask.pixel_scale / 2.0,
        n_points=edge_pixels_total,
    )
    print(
        f"[B] image_plane_mesh_grid.shape = {image_plane_mesh_grid.shape}, "
        f"edge_pixels_total = {edge_pixels_total}"
    )

    adapt_images = al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid
        },
    )

    signal_to_noise_threshold = 3.0
    over_sample_size_pixelization = np.where(
        galaxy_image_name_dict["('galaxies', 'source')"]
        > signal_to_noise_threshold,
        4,
        2,
    )
    over_sample_size_pixelization = al.Array2D(
        values=over_sample_size_pixelization, mask=mask
    )

    # CPU non-sparse and JAX paths share this dataset.
    dataset_no_sparse = base_dataset.apply_over_sampling(
        over_sample_size_lp=base_dataset.grids.lp.over_sample_size,
        over_sample_size_pixelization=over_sample_size_pixelization,
    )

    # CPU sparse-operator path mirrors the production sequence: apply sparse
    # operator first, then re-apply the pixelization over-sampling.
    dataset_sparse = base_dataset.apply_sparse_operator_cpu().apply_over_sampling(
        over_sample_size_lp=base_dataset.grids.lp.over_sample_size,
        over_sample_size_pixelization=over_sample_size_pixelization,
    )

    pix_mass = af.Model(al.mp.Isothermal)
    pix_mass.centre.centre_0 = af.UniformPrior(
        lower_limit=d.dataset_centre[0] - 0.1,
        upper_limit=d.dataset_centre[0] + 0.1,
    )
    pix_mass.centre.centre_1 = af.UniformPrior(
        lower_limit=d.dataset_centre[1] - 0.1,
        upper_limit=d.dataset_centre[1] + 0.1,
    )
    shear = source_lp_result.model.galaxies.lens.shear

    pix_model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                mass=pix_mass,
                shear=shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=al.mesh.Delaunay(
                        pixels=image_plane_mesh_grid.shape[0],
                        zeroed_pixels=edge_pixels_total,
                    ),
                    regularization=al.reg.AdaptSplit,
                ),
            ),
        ),
    )

    # Concrete tracer at prior medians. Same instance is used by all three
    # paths so any divergence is purely backend / sparse-op related.
    param_vector = pix_model.physical_values_from_prior_medians
    instance = pix_model.instance_from_vector(vector=param_vector)
    tracer = al.Tracer(galaxies=list(instance.galaxies))
    print("[B] vis_pix tracer built from prior medians.")

    # ===================================================================
    # PART C — Three-way log_evidence
    # ===================================================================

    print("\n=== PART C: three-way log_evidence ===")
    settings = al.Settings(use_border_relocator=True)

    fit_no_sparse = al.FitImaging(
        dataset=dataset_no_sparse,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=settings,
        xp=np,
    )
    log_ev_no_sparse = float(fit_no_sparse.figure_of_merit)
    log_lk_no_sparse = float(fit_no_sparse.log_likelihood)
    print(
        f"[C] CPU non-sparse: log_evidence = {log_ev_no_sparse}, "
        f"log_likelihood = {log_lk_no_sparse}"
    )

    fit_sparse = al.FitImaging(
        dataset=dataset_sparse,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=settings,
        xp=np,
    )
    log_ev_sparse = float(fit_sparse.figure_of_merit)
    log_lk_sparse = float(fit_sparse.log_likelihood)
    print(
        f"[C] CPU sparse op : log_evidence = {log_ev_sparse}, "
        f"log_likelihood = {log_lk_sparse}"
    )

    log_ev_jax = None
    fit_jax = None
    if have_jax:
        import jax.numpy as jnp
        try:
            fit_jax = al.FitImaging(
                dataset=dataset_no_sparse,
                tracer=tracer,
                adapt_images=adapt_images,
                settings=settings,
                xp=jnp,
            )
            log_ev_jax = float(fit_jax.figure_of_merit)
            print(f"[C] JAX path      : log_evidence = {log_ev_jax}")
        except Exception as exc:
            print(f"[C] JAX path raised, skipping: {type(exc).__name__}: {exc}")
            fit_jax = None

    print(
        f"[C] delta sparse - non_sparse = "
        f"{log_ev_sparse - log_ev_no_sparse:+.6e}"
    )
    if log_ev_jax is not None:
        print(
            f"[C] delta jax    - non_sparse = "
            f"{log_ev_jax - log_ev_no_sparse:+.6e}"
        )

    # ===================================================================
    # PART D — Per-step value comparison across paths
    # ===================================================================

    print("\n=== PART D: per-step inversion value comparison ===")

    inv_no = fit_no_sparse.inversion
    inv_sp = fit_sparse.inversion
    inv_jx = fit_jax.inversion if fit_jax is not None else None

    step_attrs = [
        "data_vector",
        "curvature_matrix",
        "regularization_matrix",
        "reconstruction",
        "mapped_reconstructed_image",
    ]

    step_diffs = {}
    for attr in step_attrs:
        try:
            a_no = getattr(inv_no, attr)
        except Exception as exc:
            print(f"[D] {attr:<28} no-sparse access failed: {exc}")
            step_diffs[attr] = {"error": f"no-sparse: {exc}"}
            continue
        try:
            a_sp = getattr(inv_sp, attr)
        except Exception as exc:
            print(f"[D] {attr:<28} sparse access failed: {exc}")
            step_diffs[attr] = {"error": f"sparse: {exc}"}
            continue

        rel_sp, msg_sp = _rel_diff(a_no, a_sp)

        rel_jx_str = "(no jax)"
        rel_jx_val = None
        if inv_jx is not None:
            try:
                a_jx = getattr(inv_jx, attr)
                rel_jx_val, msg_jx = _rel_diff(a_no, a_jx)
                rel_jx_str = f"{rel_jx_val:.3e} ({msg_jx})"
            except Exception as exc:
                rel_jx_str = f"jax err: {exc}"

        shape = _arr(a_no).shape
        flag_sp = "" if rel_sp < RTOL else "  ⚠ DISAGREE"
        print(
            f"[D] {attr:<28} shape={shape}  "
            f"sparse-vs-non={rel_sp:.3e} ({msg_sp})  "
            f"jax-vs-non={rel_jx_str}{flag_sp}"
        )
        step_diffs[attr] = {
            "shape": list(shape),
            "sparse_vs_no_sparse": rel_sp,
            "jax_vs_no_sparse": rel_jx_val,
        }

    # ===================================================================
    # PART E — Summary + JSON
    # ===================================================================

    sparse_rtol = (
        abs((log_ev_sparse - log_ev_no_sparse) / log_ev_no_sparse)
        if log_ev_no_sparse
        else float("inf")
    )
    jax_rtol = None
    if log_ev_jax is not None and log_ev_no_sparse:
        jax_rtol = abs((log_ev_jax - log_ev_no_sparse) / log_ev_no_sparse)

    print("\n=== Summary ===")
    print(
        f"  sparse vs non_sparse: rtol = {sparse_rtol:.3e}  "
        f"({'AGREE' if sparse_rtol < RTOL else 'DISAGREE'})"
    )
    if jax_rtol is not None:
        print(
            f"  jax    vs non_sparse: rtol = {jax_rtol:.3e}  "
            f"({'AGREE' if jax_rtol < RTOL else 'DISAGREE'})"
        )
    else:
        print("  jax    vs non_sparse: (no jax)")

    summary = {
        "dataset_name": DATASET_NAME,
        "mask_radius": float(d.mask_radius),
        "image_shape": list(d.dataset.shape_native),
        "n_source_pixels": int(image_plane_mesh_grid.shape[0]),
        "edge_pixels_total": int(edge_pixels_total),
        "log_evidence_no_sparse": log_ev_no_sparse,
        "log_evidence_sparse": log_ev_sparse,
        "log_evidence_jax": log_ev_jax,
        "delta_sparse_minus_no_sparse": log_ev_sparse - log_ev_no_sparse,
        "delta_jax_minus_no_sparse": (
            log_ev_jax - log_ev_no_sparse if log_ev_jax is not None else None
        ),
        "sparse_vs_no_sparse_rtol": sparse_rtol,
        "jax_vs_no_sparse_rtol": jax_rtol,
        "sparse_matches_no_sparse": bool(sparse_rtol < RTOL),
        "jax_matches_no_sparse": (
            bool(jax_rtol < RTOL) if jax_rtol is not None else None
        ),
        "rtol_threshold": RTOL,
        "per_step_diffs": step_diffs,
    }

    out_json = RESULTS_DIR / f"likelihood_comparison_{DATASET_NAME}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to: {out_json}")


if __name__ == "__main__":
    main()
