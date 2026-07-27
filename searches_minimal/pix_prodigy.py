"""
Does MultiStartProdigy work for pixelized source meshes?
(autolens_workspace_developer#117)
========================================================

The campaign runner for the pixelized multi-start-gradient question, on the
library searches (PyAutoFit#1398/#1400) rather than the hand-rolled loop of
``pix_lr_free.py`` — per-start vmapped optimizer state (the blocker that forced
the hand-rolled loop) now lives in ``af.AbstractMultiStartGradient``, along
with ``resurrect`` (redraw dead starts each step) and ``batch_size`` memory
tiling.

Objective: ``pix_multi_start.build_model``/``build_analysis`` — the SLaM
source_pix[1]-style problem (fixed truth MGE lens-light geometry, free broad
Isothermal+shear mass, free regularization), with the mesh selected by
``PIX_MESH`` (rectangular kernel-CDF | knn | delaunay; see that module).

Success bars ("the right solution"):
  - converged Nautilus, where it exists — rectangular: max logL **+17419** at
    r_E = 1.31 (#101, RAL job 330513); knn/delaunay baselines are launched by
    this campaign (``nautilus`` mode below).
  - the truth-point bar (``truth-bar`` mode): the objective evaluated at the
    simulator truth mass/shear with the regularization coefficient(s) scanned.
    An upper reference — a converged fit with free reg typically sits below it
    (rectangular: +25537 truth-point vs +17419 Nautilus).

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

import os
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import autofit as af  # noqa: E402
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
    TRUTH_EINSTEIN_RADIUS,
    TRUTH_MASS_ELL,
    TRUTH_SHEAR,
)

N_STARTS = int(os.environ.get("PIX_N_STARTS") or 16)
N_STEPS = int(os.environ.get("PIX_N_STEPS") or 300)
BATCH = int(os.environ.get("PIX_BATCH") or 4) or None
RESURRECT = os.environ.get("PIX_RESURRECT", "1") == "1"
# Isolates throwaway runs (e.g. PIX_NAME_SUFFIX=_smoke) from the fixed-name
# resume chain: a completed short run under the chain's name would be loaded
# as the finished result by the real budget's resume.
NAME_SUFFIX = os.environ.get("PIX_NAME_SUFFIX", "")

# Per-mesh converged-sampler bars (max logL). Only rectangular exists so far —
# the knn/delaunay entries are filled in from this campaign's own `nautilus`
# baselines once they converge.
NAUTILUS_BAR = {"rectangular": 17419.0}
NAUTILUS_MODE_R_E = {"rectangular": 1.31}
# Truth-point bars from this campaign's `truth-bar` reg scans (2026-07-27,
# local CPU): the objective at simulator-truth mass/shear, reg at its scan
# optimum. An UPPER reference — a converged free-reg fit sits below it
# (rectangular: +27059 truth-point vs +17419 Nautilus). The delaunay scan also
# showed NaN foms at high coefficients (inner>=10) — the #104 high-lambda
# fragility is visible in forward evals on this mesh.
TRUTH_BAR = {"rectangular": 27059.4, "knn": 28791.5, "delaunay": 30078.7}

OUT_DIR = os.path.join("searches_minimal", "output")


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
        name=f"pix_{rule}_{PIX_MESH}{NAME_SUFFIX}",
        path_prefix=os.path.join("searches_minimal", "pix_prodigy"),
        n_starts=N_STARTS,
        n_steps=N_STEPS,
        batch_size=BATCH,
        resurrect=RESURRECT,
        convergence=af.MultiStartGradientConvergence(check_for_convergence=False),
        # Bound what a SLURM 24 h kill can lose: the persisted search_internal
        # (the resume point for the next chained job) refreshes every full
        # update. The chain targets ONE fixed n_steps across jobs — each job
        # resumes from the last persisted step and the search completes in
        # whichever job reaches n_steps.
        iterations_per_full_update=50,
        number_of_cores=1,
        **kwargs,
    )
    # LIBRARY BUG HOTFIX (PyAutoFit, filed from #117): abstract_search coerces
    # iterations_per_full_update to float, and the multi-start _fit passes
    # min(float, steps_remaining) straight into range() — TypeError whenever
    # the cadence is below the remaining budget (config defaults are huge, so
    # min() returns the int and the crash never fired before). Overwrite with
    # the int post-construction until the library casts in _fit.
    search.iterations_per_full_update = 50
    return search


def nautilus_search():
    """Per-mesh converged-sampler baseline (the definitive bar).

    force_x1_cpu / use_jax_vmap are MANDATORY on a JAX row (fork corrupts JAX
    state); n_batch=16 keeps the dense-path memory bounded (default 100 would
    need ~100 GB).
    """
    return af.Nautilus(
        name=f"pix_nautilus_{PIX_MESH}{NAME_SUFFIX}",
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
    lines = [f"# truth-point reg scan  mesh={PIX_MESH}  (figure_of_merit)"]
    for c in coefficients:
        reg = model.galaxies.source.pixelization.regularization
        if PIX_MESH == "rectangular":
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
        print(
            f"  coeff={c:9.3e}  fom={fom:14.3f}  ({time.time() - t0:.1f}s)",
            flush=True,
        )
        lines.append(f"{c:.6e}  {fom:.6f}")
        if fom > best[0]:
            best = (fom, c)

    print(f"\nTRUTH BAR ({PIX_MESH}): max fom = {best[0]:.3f} at coeff={best[1]:.3e}")
    lines.append(f"# max {best[0]:.6f} at {best[1]:.6e}")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"pix_truth_bar_{PIX_MESH}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def report(rule: str, result, wall: float) -> None:
    samples = result.samples
    instance = samples.max_log_likelihood()
    r_e = float(instance.galaxies.lens.mass.einstein_radius)
    logl = float(samples.max_log_likelihood_sample.log_likelihood)
    bar = NAUTILUS_BAR.get(PIX_MESH)

    info = getattr(samples, "samples_info", None) or {}
    fom_history = info.get("fom_history") or []
    total_steps = info.get("total_steps")
    n_resurrections = info.get("n_resurrections")
    stop_reason = info.get("stop_reason")

    # fom = -2 * log_posterior; -fom/2 approximates logL (prior offset aside),
    # good enough to read off when the best start crossed the bar.
    steps_to_bar = None
    if bar is not None:
        for i, fom in enumerate(fom_history):
            if -0.5 * float(fom) >= bar:
                steps_to_bar = i
                break

    lines = [
        "================ RESULT ================",
        f"rule               : {rule}  (mesh={PIX_MESH})",
        f"n_starts/steps/batch: {N_STARTS}/{N_STEPS}/{BATCH}  resurrect={RESURRECT}",
        f"log_det={PIX_LOGDET or 'cholesky'}  fix_reg={PIX_FIX_REG or 'free'}",
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
        f"bar (truth-point)  : {TRUTH_BAR.get(PIX_MESH, 'n/a')}   (upper reference)",
        f"steps to bar       : {steps_to_bar if steps_to_bar is not None else 'not reached'}",
        f"total_steps        : {total_steps}",
        f"n_resurrections    : {n_resurrections}",
        f"stop_reason        : {stop_reason}",
        "=======================================",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(
        os.path.join(OUT_DIR, f"pix_prodigy_{rule}_{PIX_MESH}.txt"), "a"
    ) as f:
        f.write(text + "\n\n")
        if fom_history:
            f.write(
                "fom_history (best -2logP per step): "
                + " ".join(f"{float(v):.1f}" for v in fom_history)
                + "\n\n"
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
    result = search.fit(model=model, analysis=analysis)
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
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(
            os.path.join(OUT_DIR, f"pix_nautilus_{PIX_MESH}.txt"), "a"
        ) as f:
            f.write(
                f"mesh={PIX_MESH} wall_s={wall:.1f} "
                f"max_logL={logl:.3f} r_E={r_e:.4f}\n"
            )
    else:
        report(mode, result, wall)


if __name__ == "__main__":
    main()
