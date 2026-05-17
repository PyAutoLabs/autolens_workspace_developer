"""Side-by-side comparison: baseline `RectangularSplineAdaptImage` vs new
`RectangularRotatedAdaptImage` on a two-source diagonal scenario.

The lens mass model is pinned to truth (no non-linear search) so the
comparison isolates mesh behaviour from mass-model variance. The adapt
image is the noiseless lensed-source image from the true tracer — the
"perfect prior" — fair to both mesh classes.

Output: ``output/comparison.png`` — 2×3 grid:

  row 0: baseline (RectangularSplineAdaptImage)
  row 1: rotated  (RectangularRotatedAdaptImage)
  cols: source-plane mesh + reconstruction, image-plane model, residuals
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import autoarray as aa
import autolens as al

from simulator import build_tracer, PIXEL_SCALES, SHAPE_NATIVE, SOURCE_POSITIONS

MESH_SHAPE = (40, 40)
REG_COEFF = 1.0
MASK_RADIUS = 3.0


def load_dataset(dataset_path: Path) -> al.Imaging:
    return al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        psf_path=dataset_path / "psf.fits",
        pixel_scales=PIXEL_SCALES,
    )


def make_adapt_image(grid: aa.Grid2D) -> aa.Array2D:
    """Truth-derived adapt image: noiseless lensed source-light image."""
    tracer = build_tracer()
    return tracer.image_2d_from(grid=grid)


def run_fit(dataset: al.Imaging, mesh_cls, adapt_image):
    """Build a `FitImaging` for the given mesh class at the true mass model.

    Uses `Constant` regularization. Returns the fit and the inversion's
    mapper so we can introspect the source-plane mesh geometry.
    """
    mesh = mesh_cls(shape=MESH_SHAPE)
    pixelization = al.Pixelization(
        mesh=mesh,
        regularization=al.reg.Constant(coefficient=REG_COEFF),
    )
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.0, 0.0), einstein_radius=1.2
        ),
    )
    source = al.Galaxy(redshift=1.0, pixelization=pixelization)
    tracer = al.Tracer(galaxies=[lens, source])

    adapt_images = al.AdaptImages(galaxy_image_dict={source: adapt_image})
    fit = al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt_images)
    return fit


# ---------------------------------------------------------------------------
# Source-plane mesh geometry helpers — mirror the experiment scripts'
# logic. For the rotated mesh, edges live in PCA frame and need to be
# un-rotated to source frame; the geometry exposes rotation_matrix and
# rotation_centroid for this.
# ---------------------------------------------------------------------------


def warped_mesh_centres_source_frame(mapper):
    """Return CDF-warped pixel centres + corner grid for plotting.

    Works for both the baseline and rotated mesh classes:
    - Baseline: edges already in source frame, no un-rotation needed.
    - Rotated: edges in PCA frame, un-rotate via geometry's rotation_matrix
      and rotation_centroid.
    """
    geom = mapper.interpolator.mesh_geometry
    edges = np.asarray(geom.edges_transformed)
    y_edges, x_edges = edges[:, 0], edges[:, 1]
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])

    yy, xx = np.meshgrid(y_centres, x_centres, indexing="ij")
    centres = np.stack([yy.ravel(), xx.ravel()], axis=1)

    yyE, xxE = np.meshgrid(y_edges, x_edges, indexing="ij")
    corners = np.stack([yyE.ravel(), xxE.ravel()], axis=1)

    R = getattr(geom, "rotation_matrix", None)
    centroid = getattr(geom, "rotation_centroid", None)
    if R is not None and centroid is not None:
        centres = centres @ R + centroid
        corners = corners @ R + centroid

    return centres, corners, y_edges.size  # corners are still rectangular in their frame


def draw_image_panel(ax, arr, title, what, vmax):
    """`what` ∈ {'model', 'residual'} — image-plane visualisation.

    ``vmax`` is supplied by the caller so both rows can use the SAME scale
    on the same column — otherwise per-panel auto-scaling makes visual
    comparison meaningless.
    """
    cmap = "RdBu_r" if what == "residual" else "inferno"
    vmin = -vmax if what == "residual" else 0.0
    extent = (
        -SHAPE_NATIVE[1] * PIXEL_SCALES / 2,
        +SHAPE_NATIVE[1] * PIXEL_SCALES / 2,
        -SHAPE_NATIVE[0] * PIXEL_SCALES / 2,
        +SHAPE_NATIVE[0] * PIXEL_SCALES / 2,
    )
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
                   extent=extent)
    ax.set_title(title)
    ax.set_xlabel("image x [arcsec]")
    ax.set_ylabel("image y [arcsec]")
    return im


def main():
    here = Path(__file__).parent
    dataset_path = here / "dataset"
    output_path = here / "output"
    output_path.mkdir(exist_ok=True)

    dataset = load_dataset(dataset_path)
    grid = al.Grid2D.uniform(
        shape_native=SHAPE_NATIVE, pixel_scales=PIXEL_SCALES, over_sample_size=4
    )
    adapt_image = make_adapt_image(grid)

    cases = [
        ("baseline", al.mesh.RectangularSplineAdaptImage),
        ("rotated",  al.mesh.RectangularRotatedAdaptImage),
    ]

    # First pass — run both fits, collect data, compute shared scales.
    results = []
    for label, mesh_cls in cases:
        print(f"running {label}: {mesh_cls.__name__}")
        fit = run_fit(dataset, mesh_cls, adapt_image)
        mapper = fit.inversion.linear_obj_list[0]
        centres, corners, _ = warped_mesh_centres_source_frame(mapper)
        model = np.asarray(fit.model_data.native.array)
        residual = np.asarray(fit.residual_map.native.array)
        chi2 = float(np.sum((residual / np.asarray(dataset.noise_map.native.array)) ** 2))
        print(
            f"  chi^2 = {chi2:.1f}, "
            f"|residual|_max = {np.nanmax(np.abs(residual)):.3f}, "
            f"recon range = [{np.nanmin(fit.inversion.reconstruction):.2f}, "
            f"{np.nanmax(fit.inversion.reconstruction):.2f}]"
        )
        results.append({
            "label": label, "fit": fit, "centres": centres, "corners": corners,
            "model": model, "residual": residual, "chi2": chi2,
        })

    # Shared scales per column
    model_vmax = max(np.nanmax(r["model"]) for r in results)
    residual_vmax = max(np.nanmax(np.abs(r["residual"])) for r in results)
    recon_vmax = max(np.nanmax(r["fit"].inversion.reconstruction) for r in results)

    # Second pass — plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    src_sc = None
    img_im = None
    res_im = None
    for row, r in zip(axes, results):
        # Source-plane mesh + reconstruction with shared colour range
        recon = np.asarray(r["fit"].inversion.reconstruction)
        sc = row[0].scatter(
            r["centres"][:, 1], r["centres"][:, 0],
            c=recon, cmap="hot", s=12, edgecolors="none",
            vmin=0.0, vmax=recon_vmax,
        )
        for (py, px) in SOURCE_POSITIONS:
            row[0].plot(px, py, "o", markersize=14, markerfacecolor="none",
                        markeredgecolor="lime", markeredgewidth=2)
        ghosts = [(py, px) for py, _ in SOURCE_POSITIONS for _, px in SOURCE_POSITIONS
                  if (py, px) not in SOURCE_POSITIONS]
        for (py, px) in ghosts:
            row[0].plot(px, py, "x", color="red", markersize=14, markeredgewidth=2)
        row[0].set_xlim(-1.4, 1.4)
        row[0].set_ylim(-1.4, 1.4)
        row[0].set_aspect("equal")
        row[0].set_title(f"{r['label']}: source mesh + recon  (χ²={r['chi2']:.0f})")
        row[0].set_xlabel("source x [arcsec]")
        row[0].set_ylabel("source y [arcsec]")
        src_sc = sc

        img_im = draw_image_panel(row[1], r["model"],
                                  f"{r['label']}: image-plane model",
                                  "model", vmax=model_vmax)
        res_im = draw_image_panel(row[2], r["residual"],
                                  f"{r['label']}: residuals (data − model)",
                                  "residual", vmax=residual_vmax)

    fig.colorbar(src_sc, ax=axes[:, 0].tolist(), label="reconstruction",
                 fraction=0.04, pad=0.04)
    fig.colorbar(img_im, ax=axes[:, 1].tolist(), label="model flux",
                 fraction=0.04, pad=0.04)
    fig.colorbar(res_im, ax=axes[:, 2].tolist(), label="residual",
                 fraction=0.04, pad=0.04)
    out_path = output_path / "comparison.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nfigure written: {out_path}")


if __name__ == "__main__":
    main()
