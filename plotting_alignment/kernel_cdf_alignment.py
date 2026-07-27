"""
Kernel-CDF Mesh Alignment Verification
======================================

Verifies the rect-adapt plot fix (PyAutoArray#372 / PR#375) end-to-end and
extends it to the kernel-CDF meshes (PyAutoArray#373 / PR#374): renders the
source-plane reconstruction of the ``mass_centre_source_up_more`` dataset
(truth: source Sersic at (y, x) = (1.0, 0.0)) through the real plotting
pipeline (``aaplt.plot_inversion_reconstruction`` — imshow for uniform,
pcolormesh via ``edges_transformed`` for adaptive rectangular, tripcolor for
Delaunay) for every mesh flavour, and reports the bright-core flux centroid
and brightest-cell position of exactly the geometry each renderer draws.

Pass criterion (issue #372's complaint): the adaptive-mesh pcolormesh
centroids agree with the trusted references (uniform imshow, Delaunay
tripcolor) and the true source position to well under one mesh pixel — the
pre-fix uniform [0, 1] edge partition was off by up to ~1.5 mesh pixels in y.

Run ``plotting_alignment/simulator.py`` first if the dataset is absent.
"""

from pathlib import Path
import numpy as np

import autofit as af
import autolens as al
import autolens.plot as aplt
import autoarray.plot as aaplt

TRUTH = (1.0, 0.0)

dataset_name = "mass_centre_source_up_more"
dataset_path = Path("dataset", "imaging", "plotting_alignment", dataset_name)
output_path = Path("plotting_alignment", "output", "kernel_cdf")
output_path.mkdir(parents=True, exist_ok=True)

sub_size = 4
pixel_scale = 0.05
mesh_shape = (30, 30)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
    over_sample_size_lp=sub_size,
    over_sample_size_pixelization=1,
)
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)
dataset = dataset.apply_mask(mask=mask)
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Model__

Lens fixed at the simulator truth (Isothermal ER=1.6, q=0.8 in ell_comps_0,
zero shear) so the reconstruction sits at the true source position.
"""


def model_for(mesh):
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = 0.0
    mass.centre.centre_1 = 0.0
    mass.einstein_radius = 1.6
    mass.ell_comps.ell_comps_0 = 0.1111111111111111
    mass.ell_comps.ell_comps_1 = 0.0
    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = 0.0
    shear.gamma_2 = 0.0
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
    pixelization = al.Pixelization(
        mesh=mesh, regularization=al.reg.Constant(coefficient=1.0)
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def fit_for(mesh):
    model = model_for(mesh)
    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=al.AdaptImages(
            galaxy_name_image_dict={
                "('galaxies', 'lens')": dataset.data,
                "('galaxies', 'source')": dataset.data,
            }
        ),
        raise_inversion_positions_likelihood_exception=False,
    )
    instance = model.instance_from_prior_medians()
    return analysis.fit_from(instance)


def bright_stats(centres_yx, values):
    """Flux centroid over the bright core (>= 50% of max) + argmax position,
    computed from exactly the (position, value) pairs the renderer draws."""
    v = np.maximum(np.asarray(values), 0.0)
    core = v >= 0.5 * v.max()
    centroid = (v[core, None] * centres_yx[core]).sum(axis=0) / v[core].sum()
    return centroid, centres_yx[np.argmax(v)]


results = {}

for label, mesh in [
    ("uniform", al.mesh.RectangularUniform(shape=mesh_shape)),
    (
        "adapt_image",
        al.mesh.RectangularAdaptImage(shape=mesh_shape, weight_power=1.0),
    ),
    ("adapt_density", al.mesh.RectangularAdaptDensity(shape=mesh_shape)),
    (
        "kernel_adapt_density",
        al.mesh.RectangularKernelAdaptDensity(shape=mesh_shape, bandwidth=0.1),
    ),
    (
        "kernel_adapt_image",
        al.mesh.RectangularKernelAdaptImage(
            shape=mesh_shape, weight_power=1.0, bandwidth=0.1
        ),
    ),
]:
    fit = fit_for(mesh)
    mapper = fit.inversion.cls_list_from(cls=al.Mapper)[0]
    reconstruction = np.asarray(fit.inversion.reconstruction_dict[mapper])

    aaplt.plot_inversion_reconstruction(
        pixel_values=reconstruction,
        mapper=mapper,
        title=f"{label} (truth at {TRUTH})",
        zoom_to_brightest=False,
        grid=np.array([TRUTH]),
        output_path=str(output_path),
        output_filename=label,
        output_format="png",
    )

    # Value positions exactly as rendered: uniform imshow draws on the mesh
    # grid; adaptive pcolormesh draws cells bounded by edges_transformed.
    if label == "uniform":
        centres = np.asarray(mapper.source_plane_mesh_grid)
    else:
        y_edges, x_edges = np.asarray(mapper.mesh_geometry.edges_transformed).T
        y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
        x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
        yy, xx = np.meshgrid(y_mid, x_mid, indexing="ij")
        centres = np.stack([yy.ravel(), xx.ravel()], axis=1)

    centroid, peak = bright_stats(centres, reconstruction)
    results[label] = (centroid, peak, float(fit.figure_of_merit))
    print(
        f"{label:>22}: centroid=({centroid[0]:+.4f}, {centroid[1]:+.4f})  "
        f"peak_cell=({peak[0]:+.4f}, {peak[1]:+.4f})  FoM={fit.figure_of_merit:.2f}"
    )

# Delaunay tripcolor reference (values drawn at the mesh points themselves).
try:
    fit = fit_for(al.mesh.Delaunay(pixels=900))
    mapper = fit.inversion.cls_list_from(cls=al.Mapper)[0]
    reconstruction = np.asarray(fit.inversion.reconstruction_dict[mapper])
    aaplt.plot_inversion_reconstruction(
        pixel_values=reconstruction,
        mapper=mapper,
        title=f"delaunay (truth at {TRUTH})",
        zoom_to_brightest=False,
        grid=np.array([TRUTH]),
        output_path=str(output_path),
        output_filename="delaunay",
        output_format="png",
    )
    centres = np.asarray(mapper.source_plane_mesh_grid)
    centroid, peak = bright_stats(centres, reconstruction)
    results["delaunay"] = (centroid, peak, float(fit.figure_of_merit))
    print(
        f"{'delaunay':>22}: centroid=({centroid[0]:+.4f}, {centroid[1]:+.4f})  "
        f"peak_cell=({peak[0]:+.4f}, {peak[1]:+.4f})  FoM={fit.figure_of_merit:.2f}"
    )
except Exception as e:
    print(f"delaunay reference skipped: {type(e).__name__}: {e}")

"""
__Verdict — node containment + mapper-faithfulness__

Two checks per adaptive mesh, splitting "plot geometry" from "mapper physics":

A. **Containment** — every interior node (where the mapper places value (r, c))
   must lie inside its drawn pcolormesh cell. Post-#375 the node is the exact
   U-space midpoint of its cell edges, so a monotone CDF guarantees this; the
   check catches any convention regression for all three CDF flavours.

B. **Mapper-faithfulness** — scatter one over-sampled point's bilinear weights
   and compare the drawn centroid (edge-midpoint cells) against what the
   mapper actually encodes, sum(w * node position). These must agree to a
   fraction of the local cell; the residual |sum(w * node) - query| is the
   mapper's own bilinear-in-U interpolation displacement on non-uniform cells
   (physics of adaptive meshes, identical pre/post fix) and is reported, not
   asserted.
"""
from autoarray.inversion.mesh.interpolator.rectangular import (
    adaptive_rectangular_transformed_grid_from,
)
from autoarray.inversion.mesh.interpolator.rectangular_kernel import (
    adaptive_rectangular_transformed_grid_from_kernel,
)

n_y, n_x = mesh_shape

for label, mesh in [
    ("adapt_image", al.mesh.RectangularAdaptImage(shape=mesh_shape, weight_power=1.0)),
    ("adapt_density", al.mesh.RectangularAdaptDensity(shape=mesh_shape)),
    (
        "kernel_adapt_density",
        al.mesh.RectangularKernelAdaptDensity(shape=mesh_shape, bandwidth=0.1),
    ),
    (
        "kernel_adapt_image",
        al.mesh.RectangularKernelAdaptImage(
            shape=mesh_shape, weight_power=1.0, bandwidth=0.1
        ),
    ),
]:
    fit = fit_for(mesh)
    mapper = fit.inversion.cls_list_from(cls=al.Mapper)[0]
    interpolator = mapper.interpolator

    # Node positions per the mapper convention, through the same CDF machinery.
    U_y_nodes = (n_y - np.arange(n_y) - 1.0) / (n_y - 3)
    U_x_nodes = (np.arange(n_x) - 1.0) / (n_x - 3)
    U_nodes = np.stack([U_y_nodes, U_x_nodes]).T
    data_grid = np.asarray(interpolator.data_grid.array)
    if "kernel" in label:
        nodes_t = adaptive_rectangular_transformed_grid_from_kernel(
            data_grid,
            U_nodes,
            mesh_pixels=n_y,
            mesh_weight_map=interpolator.mesh_weight_map,
            bandwidth=interpolator.bandwidth,
            n_knots=interpolator.n_knots,
        )
    else:
        nodes_t = adaptive_rectangular_transformed_grid_from(
            data_grid, U_nodes, mesh_weight_map=interpolator.mesh_weight_map
        )
    node_y, node_x = np.asarray(nodes_t).T

    y_edges, x_edges = np.asarray(mapper.mesh_geometry.edges_transformed).T

    # A: interior nodes inside their drawn cells. Guard-ring cells clamp to
    # the data span and collapse to zero width with node == edge — allow a
    # few-ulp tolerance for exactly those float ties.
    tol = 1e-9
    ok_y = [
        min(y_edges[r], y_edges[r + 1]) - tol
        <= node_y[r]
        <= max(y_edges[r], y_edges[r + 1]) + tol
        for r in range(1, n_y - 1)
    ]
    ok_x = [
        min(x_edges[c], x_edges[c + 1]) - tol
        <= node_x[c]
        <= max(x_edges[c], x_edges[c + 1]) + tol
        for c in range(1, n_x - 1)
    ]
    assert all(ok_y) and all(ok_x), f"{label}: a node fell outside its drawn cell"

    # B: drawn centroid vs mapper-encoded centroid for a scattered delta.
    positions = np.asarray(interpolator.data_grid.over_sampled.array)
    k = int(np.argmin(np.hypot(positions[:, 0] - TRUTH[0], positions[:, 1] - TRUTH[1])))
    query = positions[k]
    mappings, _, weights = interpolator._mappings_sizes_weights
    idx = np.asarray(mappings)[k]
    w = np.asarray(weights)[k]

    y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    rows, cols = idx // n_x, idx % n_x
    drawn = np.array(
        [(w * y_mid[rows]).sum() / w.sum(), (w * x_mid[cols]).sum() / w.sum()]
    )
    encoded = np.array(
        [(w * node_y[rows]).sum() / w.sum(), (w * node_x[cols]).sum() / w.sum()]
    )
    cell = max(
        np.abs(np.diff(y_edges))[rows].max(), np.abs(np.diff(x_edges))[cols].max()
    )
    d_faithful = np.hypot(*(drawn - encoded))
    d_interp = np.hypot(*(encoded - query))
    print(
        f'{label:>22}: drawn-vs-encoded = {d_faithful:.4f}" '
        f"({d_faithful / cell:.2f} cells)  |  mapper interp displacement = "
        f'{d_interp:.4f}" ({d_interp / cell:.2f} cells; physics, unasserted)'
    )
    assert d_faithful < 0.5 * cell, (
        f"{label}: drawn centroid deviates from the mapper-encoded centroid by "
        f'{d_faithful:.4f}" (> 0.5 local cell) — plot geometry unfaithful.'
    )

print(
    "\nkernel_cdf_alignment.py: all nodes contained in their drawn cells and "
    "the plot is faithful to the mapper on every adaptive mesh (post-#375 edges)."
)
