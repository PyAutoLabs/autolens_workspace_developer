"""
Local reproduction of the vis_pix CPU sparse-operator bug.

Mirrors ``z_projects/euclid/scripts/initial_lens_model_pix_cpu.py`` body
verbatim, but redirects autoconf's output_path to
``autolens_workspace_developer/euclid_bug/output/`` and hard-codes the test
lens. The goal is to run the *exact* HPC code path locally, on a small mask
(``info.json`` is set to ``mask_radius=2.3``), so the suspected source-
reconstruction bug can be observed and bisected without HPC.

Usage:

    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \\
        python autolens_workspace_developer/euclid_bug/run_initial_lens_model_pix_cpu.py

Outputs land in:
    autolens_workspace_developer/euclid_bug/output/<dataset>/initial_lens_model/
"""

import sys
from pathlib import Path

import numpy as np

# The Euclid util module that the production HPC runs use lives outside this
# repo. We import it directly so ``load_vis_dataset``, ``AnalysisImaging``,
# segmentation/artefact noise scaling, etc. stay byte-for-byte identical to
# what the HPC sees. Cross-repo relative path; intentional.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PYAUTO_LABS = _SCRIPT_DIR.parents[1]
_EUCLID_ROOT = _PYAUTO_LABS / "z_projects" / "euclid"
_EUCLID_SCRIPTS = _EUCLID_ROOT / "scripts"
sys.path.insert(0, str(_EUCLID_SCRIPTS))
import util  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — edit constants here to debug a different lens / scenario.
# ---------------------------------------------------------------------------

DATASET_NAME = "Tile102005065RA0135279431487DECNEG0701599765928"
SAMPLE_NAME = None
ITERATIONS_PER_QUICK_UPDATE = 5000
NUMBER_OF_CORES = 1
USE_CPU = True   # True triggers dataset.apply_sparse_operator_cpu() in vis_pix
SKIP_PIX = False

LOCAL_OUTPUT_PATH = _SCRIPT_DIR / "output"


def fit():
    from autoconf import conf

    conf.instance.push(
        new_path=_EUCLID_ROOT / "config",
        output_path=LOCAL_OUTPUT_PATH,
    )

    import autofit as af
    import autolens as al

    print(f"[run_initial_lens_model_pix_cpu] dataset = {DATASET_NAME}")
    print(f"[run_initial_lens_model_pix_cpu] use_cpu = {USE_CPU}")
    print(f"[run_initial_lens_model_pix_cpu] output  = {LOCAL_OUTPUT_PATH}")

    d = util.load_vis_dataset(DATASET_NAME, sample_name=SAMPLE_NAME)
    print(f"[run_initial_lens_model_pix_cpu] mask_radius = {d.mask_radius}")
    print(f"[run_initial_lens_model_pix_cpu] dataset_centre = {d.dataset_centre}")

    settings_search = af.SettingsSearch(
        path_prefix=Path(SAMPLE_NAME) / DATASET_NAME if SAMPLE_NAME is not None else Path(DATASET_NAME),
        unique_tag="initial_lens_model",
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

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = d.dataset_centre[0]
    mass.centre.centre_1 = d.dataset_centre[1]

    source_bulge = al.model_util.mge_model_from(
        mask_radius=d.mask_radius, total_gaussians=20, centre_prior_is_uniform=False,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                mass=mass,
                shear=af.Model(al.mp.ExternalShear),
            ),
            source=af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge),
        )
    )

    analysis = util.AnalysisImaging(
        dataset=d.dataset,
        positions_likelihood_list=d.positions_likelihood_list,
        use_jax=False,
        dataset_main_path=d.dataset_main_path,
        title_prefix="VIS",
        plot_rgb=True,
        psf_lowest_resolution=d.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=d.psf_lowest_resolution_fwhm,
        pixel_wcs=d.pixel_wcs,
        **settings_search.info,
    )

    search = af.Nautilus(
        name="vis_lp",
        **settings_search.search_dict,
        n_live=500,
        batch_size=50,
        iterations_per_quick_update=ITERATIONS_PER_QUICK_UPDATE,
        n_like_max=100000,
    )

    source_lp_result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    if SKIP_PIX:
        return source_lp_result

    # ------------------------------------------------------------------
    # vis_pix — the suspected-buggy stage on HPC
    # ------------------------------------------------------------------

    dataset = d.dataset
    mask = dataset.mask
    mask_radius = d.mask_radius

    if USE_CPU:
        dataset = dataset.apply_sparse_operator_cpu()

    hilbert_pixels = 500

    image_mesh = al.image_mesh.Hilbert(pixels=hilbert_pixels, weight_power=3.5, weight_floor=0.01)

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )

    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask, adapt_data=galaxy_image_name_dict["('galaxies', 'source')"]
    )

    edge_pixels_total = 30

    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=mask_radius + mask.pixel_scale / 2.0,
        n_points=edge_pixels_total,
    )

    adapt_images = al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid
        },
    )

    signal_to_noise_threshold = 3.0
    over_sample_size_pixelization = np.where(
        galaxy_image_name_dict["('galaxies', 'source')"] > signal_to_noise_threshold,
        4,
        2,
    )
    over_sample_size_pixelization = al.Array2D(
        values=over_sample_size_pixelization, mask=mask
    )

    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=dataset.grids.lp.over_sample_size,
        over_sample_size_pixelization=over_sample_size_pixelization,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
        ],
        use_jax=not USE_CPU,
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.UniformPrior(lower_limit=d.dataset_centre[0]-0.1, upper_limit=d.dataset_centre[0]+0.1)
    mass.centre.centre_1 = af.UniformPrior(lower_limit=d.dataset_centre[1]-0.1, upper_limit=d.dataset_centre[1]+0.1)

    shear = source_lp_result.model.galaxies.lens.shear

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                mass=mass,
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

    vis_pix_search_dict = {**settings_search.search_dict, "number_of_cores": NUMBER_OF_CORES}

    search = af.Nautilus(
        name="vis_pix",
        **vis_pix_search_dict,
        n_live=150,
        n_batch=15,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


if __name__ == "__main__":
    fit()
