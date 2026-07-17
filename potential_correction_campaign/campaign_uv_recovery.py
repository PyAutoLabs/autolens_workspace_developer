"""
Potential-Correction Campaign: Iterative dkappa Recovery on Realistic uv Coverage
=================================================================================

Certifies iterative dkappa RECOVERY for the visibility-space potential-correction engine
(`al.pc.IterFitDpsiSrcInterferometer`, PyAutoLens#623/#627) on realistic interferometer coverage: an ALMA-like
earth-rotation-synthesis uv distribution (dense, ring-like sampling with honest sidelobe structure — unlike the
uniform-random coverage of the smoke tests), at a configuration where the smooth-model source fit reaches
chi2/dof ~ 1, so the corrections respond to the subhalo rather than absorbing source-model error.

Usage (tiers):
    python campaign_uv_recovery.py --tier local    # ~1.1k vis, 72x72 — laptop validation
    python campaign_uv_recovery.py --tier mid      # ~10k vis, 100x100 — workstation / RAL CPU
    python campaign_uv_recovery.py --tier full     # ~70k vis, 128x128 — RAL A100 (xp=jnp)

Outputs metrics JSON + figures under ``output/potential_correction_campaign/<tier>/``.

Cites Cao et al. 2025 (https://github.com/caoxiaoyue/lensing_potential_correction; cite via
https://github.com/caoxiaoyue/potential_correction_paper); targets the sensitivity regime of the JVAS B1938+666
detections (Powell et al. 2025, Nat. Astron. 9, 1714; Vegetti et al. 2026).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import autoarray as aa
import autolens as al

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


TIERS = {
    # n_ant, n_times, shape, pixel_scale, noise_sigma, n_iter
    "local": dict(n_ant=18, n_times=8, shape=72, pixel_scale=0.08, noise=2.0, n_iter=6),
    "mid": dict(n_ant=32, n_times=20, shape=100, pixel_scale=0.06, noise=4.0, n_iter=10),
    "full": dict(n_ant=40, n_times=90, shape=128, pixel_scale=0.05, noise=0.5, n_iter=15),
}


def synthesis_uv_from(n_ant: int, n_times: int, max_baseline_wavelengths: float, seed: int = 0):
    """
    An ALMA-like earth-rotation-synthesis uv distribution: Gaussian-distributed
    antenna positions observed over an hour-angle track, giving the dense
    elliptical-ring sampling of a real synthesis observation.
    """
    rng = np.random.default_rng(seed)
    antennas = rng.normal(scale=max_baseline_wavelengths / 4.0, size=(n_ant, 2))
    i_idx, j_idx = np.triu_indices(n_ant, k=1)
    baselines = antennas[i_idx] - antennas[j_idx]  # [n_base, 2]

    hour_angles = np.linspace(-np.pi / 3.0, np.pi / 3.0, n_times)
    declination = np.deg2rad(-23.0)

    uv_list = []
    for ha in hour_angles:
        u = baselines[:, 0] * np.cos(ha) - baselines[:, 1] * np.sin(ha)
        v = (
            baselines[:, 0] * np.sin(ha) * np.sin(declination)
            + baselines[:, 1] * np.cos(ha) * np.sin(declination)
        )
        uv_list.append(np.stack([u, v], axis=1))
    return np.concatenate(uv_list, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="local", choices=sorted(TIERS))
    parser.add_argument("--subhalo-mass", type=float, default=1.0e10)
    parser.add_argument("--dpsi-coeff", type=float, default=2000.0)
    parser.add_argument("--src-coeff", type=float, default=1.0)
    parser.add_argument("--reg-optimize", type=int, default=None, help="re-optimize reg strengths by evidence every N accepted LM iterations")
    parser.add_argument("--warm-start", action="store_true", help="initialize the iterative LM from the one-shot solution")
    parser.add_argument("--dpsi-scale", type=float, default=4.0)
    parser.add_argument("--use-jax", action="store_true", help="xp=jnp for the LM kernels")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = TIERS[args.tier]
    tag = f"{args.tier}_c{args.dpsi_coeff:.0e}_s{args.dpsi_scale:g}_sc{args.src_coeff:g}" + (f"_ro{args.reg_optimize}" if args.reg_optimize else "") + ("_ws" if args.warm_start else "")
    out = Path(args.out or Path("output") / "potential_correction_campaign" / tag)
    out.mkdir(parents=True, exist_ok=True)

    shape = (cfg["shape"], cfg["shape"])
    pixel_scale = cfg["pixel_scale"]
    # keep max baseline at ~60% of the real-space Nyquist limit
    nyquist = 1.0 / (2.0 * pixel_scale * np.pi / 180.0 / 3600.0)
    uv_wavelengths = synthesis_uv_from(
        cfg["n_ant"], cfg["n_times"], max_baseline_wavelengths=0.6 * nyquist
    )
    log(f"tier={args.tier}: {uv_wavelengths.shape[0]} visibilities, grid {shape} @ {pixel_scale}\"")

    # --- truth ---
    subhalo_centre = (1.41, 0.0)
    true_subhalo = al.mp.NFWMCRLudlowSph(
        centre=subhalo_centre,
        mass_at_200=args.subhalo_mass,
        redshift_object=0.2,
        redshift_source=0.6,
    )
    lens_true = al.Galaxy(
        redshift=0.2,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.4,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=0.0),
        ),
        subhalo=true_subhalo,
    )
    source_true = al.Galaxy(
        redshift=0.6,
        bulge0=al.lp.Gaussian(
            centre=(0.0, 0.0),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.6, angle=45.0),
            intensity=5.0,
            sigma=0.15,
        ),
        bulge1=al.lp.Gaussian(
            centre=(0.0, 0.4),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.4, angle=135.0),
            intensity=3.0,
            sigma=0.1,
        ),
    )

    real_space_mask = al.Mask2D.circular(
        shape_native=shape, pixel_scales=pixel_scale, radius=2.6
    )
    grid = al.Grid2D.from_mask(mask=real_space_mask)

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=uv_wavelengths,
        exposure_time=300.0,
        noise_sigma=cfg["noise"],
        noise_seed=1,
    )
    dataset = simulator.via_tracer_from(
        tracer=al.Tracer(galaxies=[lens_true, source_true]), grid=grid
    )
    dataset = al.Interferometer(
        data=dataset.data,
        noise_map=dataset.noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
    )

    # the precision operator is T^H C^-1 T — noise-dependent; key the cache on the noise config
    precision_path = out / f"nufft_precision_operator_n{cfg['noise']:g}.npy"
    if precision_path.exists():
        log("loading cached precision operator")
        dataset = dataset.apply_sparse_operator(
            nufft_precision_operator=np.load(precision_path)
        )
    else:
        log("computing precision operator (cached after)...")
        dataset = dataset.apply_sparse_operator()
        # recover the preload from the operator state for caching
        np.save(precision_path, np.real(np.fft.ifft2(np.asarray(dataset.sparse_operator.Khat))))
    log("sparse operator ready")

    # --- smooth model + pixelizations ---
    lens_smooth = al.Galaxy(redshift=0.2, mass=lens_true.mass)
    source_start = al.pc.AnalyticSrcFactory(source_galaxy=source_true)

    # arc-restricted dpsi mesh from the smooth-model image geometry
    tracer_smooth = al.Tracer(galaxies=[lens_smooth, source_true])
    arc_image = np.asarray(tracer_smooth.image_2d_from(grid=grid).native)
    arc_mask = al.pc.util.arc_mask_from(
        arc_image / (0.05 * arc_image.max()), threshold=3.0, ignore_size=10, ext_size=3
    )
    dpsi_mask = ~((~arc_mask) & (~np.asarray(real_space_mask)))
    log(f"dpsi mesh: {int(np.count_nonzero(~dpsi_mask))} arc pixels")

    grid_slim = dataset.grid.slim
    source_shape = (
        int(float(grid_slim[:, 0].max() - grid_slim[:, 0].min()) / pixel_scale / 2.0),
        int(float(grid_slim[:, 1].max() - grid_slim[:, 1].min()) / pixel_scale / 2.0),
    )
    src_pixelization = al.Pixelization(
        mesh=al.mesh.KNearestNeighbor(pixels=int(np.prod(source_shape))),
        regularization=al.reg.Constant(coefficient=args.src_coeff),
    )
    src_image_mesh = al.image_mesh.Overlay(shape=source_shape)

    dpsi_pixelization = al.pc.DpsiPixelization(
        mesh=al.pc.RegularDpsiMesh(factor=2),
        regularization=al.reg.MaternKernel(
            coefficient=args.dpsi_coeff, scale=args.dpsi_scale, nu=2.5
        ),
    )

    results = {"tier": tag, "n_vis": int(uv_wavelengths.shape[0]),
               "subhalo_mass": args.subhalo_mass}

    # --- regime gate: the smooth-model source fit must reach chi2/dof ~ 1 ---
    from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages

    source_galaxy_pix = al.Galaxy(redshift=0.6, pixelization=src_pixelization)
    image_plane_mesh_grid = src_image_mesh.image_plane_mesh_grid_from(
        mask=real_space_mask
    )
    smooth_fit = al.FitInterferometer(
        dataset=dataset,
        tracer=al.Tracer(galaxies=[lens_smooth, source_galaxy_pix]),
        adapt_images=AdaptImages(
            galaxy_image_plane_mesh_grid_dict={source_galaxy_pix: image_plane_mesh_grid}
        ),
        settings=aa.Settings(use_positive_only_solver=True, use_border_relocator=True),
    )
    n_dof = 2 * uv_wavelengths.shape[0]
    chi2_dof = float(smooth_fit.chi_squared) / n_dof
    results["smooth_chi2_dof"] = chi2_dof
    log(f"REGIME GATE: smooth-model chi2/dof = {chi2_dof:.2f} (target ~1-3)")

    def dkappa_metrics(pair_obj, dkappa_rec, tag):
        points = np.vstack([pair_obj.ygrid_dpsi_1d, pair_obj.xgrid_dpsi_1d]).T
        dkappa_true = np.asarray(
            true_subhalo.convergence_2d_from(grid=al.Grid2DIrregular(values=points))
        )
        r = np.hypot(points[:, 0] - subhalo_centre[0], points[:, 1] - subhalo_centre[1])
        local = r < 1.0
        corr = float(np.corrcoef(dkappa_rec, dkappa_true)[0, 1])
        corr_local = float(np.corrcoef(dkappa_rec[local], dkappa_true[local])[0, 1])
        peak = points[int(np.argmax(dkappa_rec))]
        dist = float(np.hypot(peak[0] - subhalo_centre[0], peak[1] - subhalo_centre[1]))
        far = r > 1.5
        significance = float(dkappa_rec.max() / dkappa_rec[far].std())
        log(
            f"{tag}: corr={corr:.3f} corr_local={corr_local:.3f} "
            f"peak_dist={dist:.2f}\" significance={significance:.1f}"
        )
        np.save(out / f"{tag}_dkappa_rec.npy", dkappa_rec)
        return dict(
            corr=corr, corr_local=corr_local, peak=[float(peak[0]), float(peak[1])],
            peak_dist=dist, significance=significance,
        )

    # --- one-shot joint inversion ---
    log("one-shot joint fit...")
    fit = al.pc.FitDpsiSrcInterferometer(
        dataset=dataset,
        lens_start=lens_smooth,
        source_start=source_start,
        dpsi_pixelization=dpsi_pixelization,
        src_pixelization=src_pixelization,
        src_image_mesh=src_image_mesh,
        dpsi_mask=dpsi_mask,
        use_sparse_operator=True,
    )
    ev1 = fit.log_evidence
    results["oneshot_log_evidence"] = float(ev1)
    log(f"one-shot log evidence = {ev1:.4e}")
    dkappa1 = np.asarray(fit.pair_dpsi_data_obj.hamiltonian_dpsi @ fit.best_fit_dpsi)
    results["oneshot"] = dkappa_metrics(fit.pair_dpsi_data_obj, dkappa1, "oneshot")

    # --- iterative LM engine ---
    xp = np
    if args.use_jax:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        xp = jnp
    log(f"iterative LM fit (n_iter={cfg['n_iter']}, gauge, xp={'jnp' if args.use_jax else 'np'})...")
    iter_fit = al.pc.IterFitDpsiSrcInterferometer(
        dataset=dataset,
        lens_start=lens_smooth,
        dpsi_pixelization=dpsi_pixelization,
        src_pixelization=src_pixelization,
        src_image_mesh=src_image_mesh,
        dpsi_mask=dpsi_mask,
        gauge_constraints=True,
        n_iter=cfg["n_iter"],
        reg_optimize_every=args.reg_optimize,
        verbose=True,
    )
    x0 = np.asarray(fit.src_dpsi_slim) if args.warm_start else None
    if x0 is not None:
        log(f"warm-starting iterative from the one-shot solution ({x0.shape[0]} params)")
    s_opt, dpsi_opt = iter_fit.solve_joint_optimization(x0=x0)
    ev2 = iter_fit.log_evidence(s=s_opt, dpsi=dpsi_opt)
    results["iterative_log_evidence"] = float(ev2)
    log(f"iterative Laplace log evidence = {ev2:.4e}")
    dkappa2 = np.asarray(iter_fit.pair_dpsi_data_obj.hamiltonian_dpsi @ dpsi_opt)
    results["iterative"] = dkappa_metrics(iter_fit.pair_dpsi_data_obj, dkappa2, "iterative")
    np.save(out / "iterative_s_opt.npy", s_opt)
    np.save(out / "iterative_dpsi_opt.npy", dpsi_opt)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    log(f"CAMPAIGN TIER DONE — results at {out}")
    print("RESULTS:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
