"""
Does MultiStartProdigy work for pixelized source meshes?
(autolens_workspace_developer#117)
========================================================

The campaign runner for the pixelized multi-start-gradient question, on the
library searches (PyAutoFit#1398/#1400) rather than the hand-rolled loop of
``pix_lr_free.py`` — per-start vmapped optimizer state (the blocker that forced
the hand-rolled loop) now lives in PyAutoFit's multi-start gradient base,
along with ``resurrect`` (redraw dead starts each step) and ``batch_size``
memory tiling.

Objective: ``pix_multi_start.build_model``/``build_analysis`` — the SLaM
source_pix[1]-style problem (fixed truth MGE lens-light geometry, free broad
Isothermal+shear mass, free regularization), with the mesh selected by
``PIX_MESH`` (rectangular kernel-CDF | knn | delaunay | delaunay_nn; see that
module).

Success bars ("the right solution"):
  - converged Nautilus, where it exists — rectangular: max logL **+17419** at
    r_E = 1.31 (#101, RAL job 330513); knn/delaunay baselines are launched by
    this campaign (``nautilus`` mode below).
  - the truth-point bar (``truth-bar`` mode): the objective evaluated at the
    simulator truth mass/shear with the regularization coefficient(s) scanned.
    A recovery reference — a converged fit may sit above it because mass and
    shear remain free (DelaunayNN: +30374 fit vs +30304 truth point).

Resume-chaining: fixed ``name``/``path_prefix`` per (rule, mesh) means the
search's persisted ``search_internal`` resumes across successive SLURM jobs —
CPU throughput is ~5-6 min/step at 16 starts (job 330953), so multi-thousand
step budgets span multiple 24 h submissions.

Usage (from the workspace root)::

    python -m searches_minimal.pix_prodigy [prodigy|adam|adabelief|lion]
    python -m searches_minimal.pix_prodigy nautilus     # per-mesh baseline
    python -m searches_minimal.pix_prodigy truth-bar    # reg-scan at truth

Env: ``PIX_MESH`` (rectangular), ``PIX_N_STARTS`` (16), ``PIX_N_STEPS`` (300),
``PIX_BATCH`` (4; 0 = unbatched), ``PIX_RESURRECT`` (1), plus
``PIX_LOGDET``/``PIX_FIX_REG``/``PIX_LR`` via ``pix_multi_start``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import autofit as af  # noqa: E402
import autoarray as aa  # noqa: E402
import autolens as al  # noqa: E402

from searches_minimal.pix_multi_start import (  # noqa: E402
    build_analysis,
    build_dataset,
    build_model,
    BASIN_TOL,
    PIX_FIX_REG,
    PIX_LOGDET,
    PIX_LR,
    PIX_MESH,
    PIX_REG,
    TRUTH_EINSTEIN_RADIUS,
    TRUTH_MASS_ELL,
    TRUTH_SHEAR,
)

N_STARTS = int(os.environ.get("PIX_N_STARTS") or 16)
N_STEPS = int(os.environ.get("PIX_N_STEPS") or 300)
BATCH = int(os.environ.get("PIX_BATCH") or 4) or None
RESURRECT = os.environ.get("PIX_RESURRECT", "1") == "1"
UPDATE_EVERY = int(os.environ.get("PIX_UPDATE_EVERY") or 50)
LOG_EVERY = int(os.environ.get("PIX_LOG_EVERY") or 10)
# Isolates throwaway runs (e.g. PIX_NAME_SUFFIX=_smoke) from the fixed-name
# resume chain: a completed short run under the chain's name would be loaded
# as the finished result by the real budget's resume.
NAME_SUFFIX = os.environ.get("PIX_NAME_SUFFIX", "")
# Stage-3 arm knob: narrow the multi-start draw band (library default
# 0.15-0.85 of the unit hypercube). A narrow-band arm that succeeds where the
# broad band stalls localises the failure to global discovery, not local
# landscape geometry.
START_LOW = float(os.environ.get("PIX_START_LOW") or 0.15)
START_HIGH = float(os.environ.get("PIX_START_HIGH") or 0.85)

# Per-mesh converged-sampler bars (max logL). Only rectangular exists so far —
# the knn/delaunay entries are filled in from this campaign's own `nautilus`
# baselines once they converge.
# knn/delaunay: this campaign's CPU baselines (jobs 331180/331181, 2026-07-27).
# CAVEAT: single n_live=100 runs; both modes sit FAR from truth (knn r_E=1.011,
# delaunay 0.962) and well below their truth-point bars — at these modest
# sampler settings they are floors, not converged references. Notably Prodigy
# BEAT the knn baseline by ~16.8k nats (+22515 vs +5704) in ~1/5 the wall.
NAUTILUS_BAR = {"rectangular": 17419.0, "knn": 5704.2, "delaunay": 19982.3}
NAUTILUS_MODE_R_E = {"rectangular": 1.31, "knn": 1.011, "delaunay": 0.962}
# Truth-point bars from this campaign's `truth-bar` reg scans (2026-07-27,
# local CPU): the objective at simulator-truth mass/shear, reg at its scan
# optimum. These are recovery references rather than upper bounds. The
# delaunay scan also showed NaN foms at high coefficients (inner>=10) — the
# #104 high-lambda fragility is visible in forward evals on this mesh.
TRUTH_BAR = {"rectangular": 27059.4, "knn": 28791.5, "delaunay": 30078.7}

OUT_DIR = os.path.join("searches_minimal", "output")
RESULTS_DIR = os.environ.get("PIX_RESULTS_DIR") or OUT_DIR
# Distinguishes reg-scheme override arms in output filenames/reports.
REG_TAG = f"_{PIX_REG}" if PIX_REG else ""
TRUTH_BAR_OVERRIDE = float(os.environ.get("PIX_TRUTH_BAR") or "nan")


def _source_revision(module) -> str | None:
    """Return the git revision backing an imported source package."""
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    for parent in Path(module_file).resolve().parents:
        if (parent / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or None
    return None


def _run_metadata() -> dict:
    """Hardware and source identity shared by every durable run artifact."""
    device = jax.devices()[0]
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "backend": jax.default_backend(),
        "device_kind": getattr(device, "device_kind", str(device)),
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "autofit_version": getattr(af, "__version__", None),
        "autoarray_version": getattr(aa, "__version__", None),
        "autolens_version": getattr(al, "__version__", None),
        "autofit_revision": _source_revision(af),
        "autoarray_revision": _source_revision(aa),
        "autolens_revision": _source_revision(al),
    }


def _run_config() -> dict:
    """The complete set of campaign knobs affecting an optimizer arm."""
    return {
        "mesh": PIX_MESH,
        "regularization": PIX_REG or "mesh-default",
        "fixed_regularization_scale": PIX_FIX_REG,
        "log_det_method": PIX_LOGDET or "cholesky",
        "n_starts": N_STARTS,
        "n_steps": N_STEPS,
        "batch_size": BATCH,
        "resurrect": RESURRECT,
        "iterations_per_full_update": UPDATE_EVERY,
        "iterations_per_log": LOG_EVERY,
        "start_lower_limit": START_LOW,
        "start_upper_limit": START_HIGH,
        "name_suffix": NAME_SUFFIX,
    }


def _artifact_path(kind: str) -> Path:
    """A stable, collision-resistant JSON path for one campaign arm."""
    raw_name = f"{kind}_{PIX_MESH}{REG_TAG}{NAME_SUFFIX}_{jax.default_backend()}"
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw_name).strip("_")
    path = Path(RESULTS_DIR) / f"{safe_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_artifact(kind: str, payload: dict) -> Path:
    """Write one structured result without changing the legacy text report."""
    path = _artifact_path(kind)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"artifact            : {path}")
    return path


def _failure_kind(exc: Exception) -> str:
    message = str(exc).lower()
    if "resource_exhausted" in message or "out of memory" in message:
        return "vram"
    if "overflow" in message:
        return "cap_overflow"
    if "nan" in message or "non-finite" in message or "nonfinite" in message:
        return "non_finite"
    return "exception"


def _delaunay_nn_diagnostics(analysis, instance) -> dict:
    """Read cap/degeneracy diagnostics from the best DelaunayNN mapper."""
    if PIX_MESH != "delaunay_nn":
        return {}
    try:
        fit = analysis.fit_from(instance=instance)
        for mapper in fit.inversion.linear_obj_list:
            interpolator = getattr(mapper, "interpolator", None)
            delaunay = getattr(interpolator, "delaunay", None)
            if delaunay is None or not hasattr(delaunay, "cavity_sizes"):
                continue
            return {
                "main_max_cavity": int(np.max(np.asarray(delaunay.cavity_sizes))),
                "main_max_neighbors": int(np.max(np.asarray(delaunay.sizes))),
                "split_max_cavity": int(
                    np.max(np.asarray(delaunay.split_cavity_sizes))
                ),
                "split_max_neighbors": int(np.max(np.asarray(delaunay.splitted_sizes))),
                "main_overflow_rows": int(np.sum(np.asarray(delaunay.overflow))),
                "main_degenerate_rows": int(np.sum(np.asarray(delaunay.degenerate))),
                "split_overflow_rows": int(np.sum(np.asarray(delaunay.split_overflow))),
                "split_degenerate_rows": int(
                    np.sum(np.asarray(delaunay.split_degenerate))
                ),
            }
        return {"diagnostic_error": "DelaunayNN mapper not found"}
    except Exception as exc:  # diagnostics must never invalidate a fit
        return {"diagnostic_error": repr(exc)}


def gradient_search(rule: str):
    """The library multi-start gradient search for `rule`.

    ``convergence`` is set explicitly: ``None`` silently ENABLES convergence
    checking (window 50 / rtol 1e-4), which would collapse fixed-step
    comparisons — the same trap autolens_profiling's ``_samplers.py``
    documents. (With ``resurrect=True`` the library skips the check anyway;
    explicit is still clearer for the resurrect=False A/B arm.)
    """
    cls = {
        "prodigy": af.MultiStartProdigy,
        "adam": af.MultiStartAdam,
        "adabelief": af.MultiStartADABelief,
        "lion": af.MultiStartLion,
    }[rule]
    kwargs = {}
    # Prodigy is learning-rate-free (learning_rate=None -> no lr arg to optax);
    # the adam family keeps its benchmark default unless PIX_LR overrides.
    if rule != "prodigy" and PIX_LR:
        kwargs["learning_rate"] = PIX_LR
    search = cls(
        name=f"pix_{rule}_{PIX_MESH}{REG_TAG}{NAME_SUFFIX}",
        path_prefix=os.path.join("searches_minimal", "pix_prodigy"),
        n_starts=N_STARTS,
        n_steps=N_STEPS,
        batch_size=BATCH,
        start_lower_limit=START_LOW,
        start_upper_limit=START_HIGH,
        resurrect=RESURRECT,
        convergence=af.MultiStartGradientConvergence(check_for_convergence=False),
        # Bound what a SLURM 24 h kill can lose: the persisted search_internal
        # (the resume point for the next chained job) refreshes every full
        # update. The chain targets ONE fixed n_steps across jobs — each job
        # resumes from the last persisted step and the search completes in
        # whichever job reaches n_steps.
        iterations_per_full_update=UPDATE_EVERY,
        iterations_per_log=LOG_EVERY,
        number_of_cores=1,
        **kwargs,
    )
    return search


def nautilus_search():
    """Per-mesh converged-sampler baseline (the definitive bar).

    force_x1_cpu / use_jax_vmap are MANDATORY on a JAX row (fork corrupts JAX
    state); n_batch=16 keeps the dense-path memory bounded (default 100 would
    need ~100 GB).
    """
    return af.Nautilus(
        name=f"pix_nautilus_{PIX_MESH}{REG_TAG}{NAME_SUFFIX}",
        path_prefix=os.path.join("searches_minimal", "pix_prodigy"),
        n_live=100,
        n_batch=16,
        number_of_cores=1,
        force_x1_cpu=True,
        use_jax_vmap=True,
    )


def truth_bar() -> None:
    """Scan the regularization coefficient(s) at the simulator-truth mass.

    Forward-only (no gradients — the 10.9 GiB cost was value_and_grad), eager
    JAX via the analysis, one fit per scan point. For the split-family regs the
    scan moves (inner, outer) together, keeping the certified 1:100 ratio.
    """
    dataset = build_dataset()
    analysis = build_analysis(dataset)
    model = build_model()

    # Pin mass + shear at the simulator truth; regularization stays free.
    model.galaxies.lens.mass.centre = (0.0, 0.0)
    model.galaxies.lens.mass.ell_comps = TRUTH_MASS_ELL
    model.galaxies.lens.mass.einstein_radius = TRUTH_EINSTEIN_RADIUS
    model.galaxies.lens.shear.gamma_1 = TRUTH_SHEAR[0]
    model.galaxies.lens.shear.gamma_2 = TRUTH_SHEAR[1]
    print(f"truth-bar: mesh={PIX_MESH}  free reg params={model.prior_count}")

    coefficients = np.logspace(-2, 4, 13)
    best = (-np.inf, None)
    rows = []
    lines = [f"# truth-point reg scan  mesh={PIX_MESH}  (figure_of_merit)"]
    for c in coefficients:
        if PIX_REG == "matern":
            # MaternKernel: (coefficient, scale) — scan coefficient at scale 1.
            physical = [c, 1.0]
        elif PIX_MESH == "rectangular":
            physical = [c]
        else:
            # AdaptSplit: (inner, outer) at the certified 1:100 ratio.
            physical = [0.1 * c, 10.0 * c]
        # instance_from_vector takes PHYSICAL values (the same convention as
        # Fitness.call); reg is the only free component, so the vector is just
        # its parameter(s) in model order.
        instance = model.instance_from_vector(physical)
        t0 = time.time()
        fit = analysis.fit_from(instance=instance)
        fom = float(fit.figure_of_merit)
        finite = bool(np.isfinite(fom))
        print(
            f"  coeff={c:9.3e}  fom={fom:14.3f}  ({time.time() - t0:.1f}s)",
            flush=True,
        )
        lines.append(f"{c:.6e}  {fom:.6f}")
        rows.append(
            {
                "coefficient_scale": float(c),
                "log_likelihood": fom if finite else None,
                "finite": finite,
            }
        )
        if finite and fom > best[0]:
            best = (fom, c)

    if best[1] is None:
        raise RuntimeError(f"truth-bar scan for {PIX_MESH} had no finite rows")
    print(f"\nTRUTH BAR ({PIX_MESH}): max fom = {best[0]:.3f} at coeff={best[1]:.3e}")
    lines.append(f"# max {best[0]:.6f} at {best[1]:.6e}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(
        os.path.join(RESULTS_DIR, f"pix_truth_bar_{PIX_MESH}{REG_TAG}.txt"), "w"
    ) as f:
        f.write("\n".join(lines) + "\n")
    _write_artifact(
        "truth_bar",
        {
            "kind": "truth_bar",
            "metadata": _run_metadata(),
            "config": _run_config(),
            "rows": rows,
            "best_log_likelihood": float(best[0]),
            "best_coefficient_scale": float(best[1]),
        },
    )


def report(rule: str, result, wall: float, analysis, model) -> None:
    samples = result.samples
    instance = samples.max_log_likelihood()
    r_e = float(instance.galaxies.lens.mass.einstein_radius)
    logl = float(samples.max_log_likelihood_sample.log_likelihood)
    bar = NAUTILUS_BAR.get(PIX_MESH)

    info = getattr(samples, "samples_info", None) or {}
    fom_history_raw = info.get("fom_history")
    fom_history = (
        np.asarray(fom_history_raw, dtype=float).tolist()
        if fom_history_raw is not None
        else []
    )
    total_steps = info.get("total_steps")
    n_resurrections = info.get("n_resurrections")
    stop_reason = info.get("stop_reason")

    # fom = -2 * log_posterior; -fom/2 approximates logL (prior offset aside),
    # good enough to read off when the best start crossed the bar.
    steps_to_bar = None
    if bar is not None:
        for i, fom in enumerate(fom_history, start=1):
            if -0.5 * float(fom) >= bar:
                steps_to_bar = i
                break

    truth_bar_value = (
        TRUTH_BAR_OVERRIDE
        if np.isfinite(TRUTH_BAR_OVERRIDE)
        else TRUTH_BAR.get(PIX_MESH)
    )
    steps_to_truth_bar = None
    if truth_bar_value is not None:
        for i, fom in enumerate(fom_history, start=1):
            if -0.5 * float(fom) >= truth_bar_value:
                steps_to_truth_bar = i
                break

    mass = instance.galaxies.lens.mass
    shear = instance.galaxies.lens.shear
    recovered = {
        "centre": [float(value) for value in mass.centre],
        "ell_comps": [float(value) for value in mass.ell_comps],
        "einstein_radius": r_e,
        "shear": [float(shear.gamma_1), float(shear.gamma_2)],
    }
    diagnostics = _delaunay_nn_diagnostics(analysis=analysis, instance=instance)
    wall_per_step = wall / total_steps if total_steps else None

    lines = [
        "================ RESULT ================",
        f"rule               : {rule}  (mesh={PIX_MESH})",
        f"n_starts/steps/batch: {N_STARTS}/{N_STEPS}/{BATCH}  resurrect={RESURRECT}",
        f"log_det={PIX_LOGDET or 'cholesky'}  fix_reg={PIX_FIX_REG or 'free'}"
        f"  reg={PIX_REG or 'mesh-default'}",
        f"wall_s             : {wall:.1f}",
        f"max log likelihood : {logl:.3f}",
        f"einstein_radius    : {r_e:.4f}   (truth {TRUTH_EINSTEIN_RADIUS}"
        + (
            f", Nautilus mode {NAUTILUS_MODE_R_E[PIX_MESH]}"
            if PIX_MESH in NAUTILUS_MODE_R_E
            else ""
        )
        + ")",
        f"in truth basin     : {abs(r_e - TRUTH_EINSTEIN_RADIUS) < BASIN_TOL}"
        "   (slack tol — converged-sampler logL is the real bar)",
        f"bar (Nautilus)     : {bar if bar is not None else 'none yet (baseline pending)'}",
        f"bar (truth-point)  : {truth_bar_value if truth_bar_value is not None else 'n/a'}"
        "   (recovery reference)",
        f"steps to bar proxy : {steps_to_bar if steps_to_bar is not None else 'not reached'}",
        "steps to truth proxy: "
        f"{steps_to_truth_bar if steps_to_truth_bar is not None else 'not reached'}",
        f"total_steps        : {total_steps}",
        f"wall per step      : {wall_per_step:.3f}s"
        if wall_per_step is not None
        else "wall per step      : n/a",
        f"n_resurrections    : {n_resurrections}",
        f"stop_reason        : {stop_reason}",
        "=======================================",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(
        os.path.join(RESULTS_DIR, f"pix_prodigy_{rule}_{PIX_MESH}{REG_TAG}.txt"),
        "a",
    ) as f:
        f.write(text + "\n\n")
        if fom_history:
            f.write(
                "fom_history (best -2logP per step): "
                + " ".join(f"{float(v):.1f}" for v in fom_history)
                + "\n\n"
            )
    _write_artifact(
        f"pix_prodigy_{rule}",
        {
            "kind": "gradient_search",
            "rule": rule,
            "metadata": _run_metadata(),
            "config": _run_config(),
            "metrics": {
                "wall_seconds": wall,
                "wall_seconds_per_step_including_compile": wall_per_step,
                "max_log_likelihood": logl,
                "best_log_posterior_proxy": (
                    -0.5 * min(fom_history) if fom_history else None
                ),
                "in_truth_basin": abs(r_e - TRUTH_EINSTEIN_RADIUS) < BASIN_TOL,
                "nautilus_bar": bar,
                "truth_point_bar": truth_bar_value,
                "steps_to_nautilus_bar_proxy": steps_to_bar,
                "steps_to_truth_bar_proxy": steps_to_truth_bar,
                "total_steps": total_steps,
                "n_resurrections": n_resurrections,
                "stop_reason": stop_reason,
            },
            "recovered": recovered,
            "delaunay_nn_diagnostics": diagnostics,
            "fom_history": fom_history,
        },
    )


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "prodigy"

    print(f"JAX backend: {jax.default_backend()}  x64={jax.config.jax_enable_x64}")
    print(
        f"mode={mode}  mesh={PIX_MESH}  n_starts={N_STARTS}  n_steps={N_STEPS}  "
        f"batch={BATCH}  resurrect={RESURRECT}  log_det={PIX_LOGDET or 'cholesky'}"
    )

    if mode == "truth-bar":
        truth_bar()
        return

    dataset = build_dataset()
    analysis = build_analysis(dataset)
    model = build_model()
    print(f"Free (non-linear) parameters: {model.prior_count}")

    search = nautilus_search() if mode == "nautilus" else gradient_search(mode)

    t0 = time.time()
    try:
        result = search.fit(model=model, analysis=analysis)
    except Exception as exc:
        wall = time.time() - t0
        _write_artifact(
            f"pix_{mode}_failure",
            {
                "kind": "failure",
                "rule": mode,
                "metadata": _run_metadata(),
                "config": _run_config(),
                "failure_kind": _failure_kind(exc),
                "exception": repr(exc),
                "wall_seconds": wall,
            },
        )
        raise
    wall = time.time() - t0

    if mode == "nautilus":
        instance = result.samples.max_log_likelihood()
        r_e = float(instance.galaxies.lens.mass.einstein_radius)
        logl = float(result.samples.max_log_likelihood_sample.log_likelihood)
        print("\n================ NAUTILUS BASELINE ================")
        print(f"mesh               : {PIX_MESH}")
        print(f"wall_s             : {wall:.1f}")
        print(f"max log likelihood : {logl:.3f}   <- the bar for this mesh")
        print(f"einstein_radius    : {r_e:.4f}   (truth {TRUTH_EINSTEIN_RADIUS})")
        print("===================================================")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, f"pix_nautilus_{PIX_MESH}.txt"), "a") as f:
            f.write(
                f"mesh={PIX_MESH} wall_s={wall:.1f} max_logL={logl:.3f} r_E={r_e:.4f}\n"
            )
    else:
        report(mode, result, wall, analysis=analysis, model=model)


if __name__ == "__main__":
    main()
