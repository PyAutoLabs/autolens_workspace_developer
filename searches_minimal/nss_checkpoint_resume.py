"""
af.NSS Checkpoint + Resume Smoke
--------------------------------

End-to-end resume integration smoke for Phases 2-3 of the
``nss_first_class_sampler`` roadmap (PyAutoLabs/PyAutoFit#1273).

Strategy:

1. **Capture pass** — run ``af.NSS`` once with a tight ``checkpoint_interval=2``
   on a tiny 2D Gaussian flat-likelihood problem. Monkey-patch ``_save_checkpoint``
   to copy the first written blob aside before the post-success cleanup deletes it.

2. **Resume pass** — wipe the output dir, run again with the same Paths, but
   restore the captured checkpoint to the expected resume location before
   ``fit()`` starts. Verify the run logs the resume message and produces a
   finite ``log_evidence`` + cleans up the checkpoint at the end.

Also validates the Phase 3 quick-update visualization fires by attaching a
counter to the analysis's ``visualize`` method.

Run from the workspace root:

    python searches_minimal/nss_checkpoint_resume.py
"""

import logging
import os
import shutil
from pathlib import Path

os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np

import autofit as af
from autofit.non_linear.search.nest.nss import search as nss_search_module


_VIZ_COUNT = {"n": 0}


class FlatLikelihoodModel:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y


class FlatLikelihoodAnalysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return 0.0

    def visualize(self, paths, instance, during_analysis):
        _VIZ_COUNT["n"] += 1


def _build_model_analysis():
    model = af.Model(FlatLikelihoodModel)
    model.x = af.GaussianPrior(mean=2.5, sigma=1.0)
    model.y = af.GaussianPrior(mean=-1.0, sigma=0.5)
    analysis = FlatLikelihoodAnalysis()
    return model, analysis


def _new_search(name: str = "nss_checkpoint_resume"):
    return af.NSS(
        name=name,
        path_prefix=str(Path("searches_minimal") / "output"),
        n_live=40,
        num_mcmc_steps=2,
        num_delete=5,
        termination=-1.0,
        seed=0,
        checkpoint_interval=2,
        iterations_per_quick_update=3,
    )


def _wipe(path: Path):
    if path.exists():
        shutil.rmtree(path)


def main():
    log_capture = []
    handler = logging.Handler()
    handler.emit = lambda record: log_capture.append(record.getMessage())
    logging.getLogger().addHandler(handler)

    # PyAutoFit composes the output path as `output/<path_prefix>/<name>/<unique_tag>/`.
    # We wipe the full sub-tree below the workspace root before each pass so
    # we don't trigger PyAutoFit's "Fit Already Completed" short-circuit.
    output_root = (
        Path("output") / "searches_minimal" / "output" / "nss_checkpoint_resume"
    )
    _wipe(output_root)

    model, analysis = _build_model_analysis()

    # --- Capture pass: snapshot a checkpoint before post-success cleanup ---
    print("Capture pass: running NSS with snapshot-on-save monkey patch...")
    _VIZ_COUNT["n"] = 0

    snapshot_path = output_root.parent / "snapshot_checkpoint.pkl"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        snapshot_path.unlink()

    original_save = nss_search_module._save_checkpoint
    snapshot_count = {"n": 0}

    def snapshot_save(path, state, dead, run_key, iteration):
        original_save(path, state, dead, run_key, iteration)
        if snapshot_count["n"] == 0:
            shutil.copy(path, snapshot_path)
            snapshot_count["n"] += 1

    nss_search_module._save_checkpoint = snapshot_save
    try:
        capture_search = _new_search()
        capture_result = capture_search.fit(model=model, analysis=analysis)
    finally:
        nss_search_module._save_checkpoint = original_save

    capture_checkpoint = (
        capture_search.paths.search_internal_path / "nss_checkpoint.pkl"
    )

    assert snapshot_count["n"] >= 1, (
        "No checkpoint write captured during capture pass — Phase 2 hook is broken."
    )
    assert snapshot_path.exists(), f"Snapshot not written at {snapshot_path}"
    assert not capture_checkpoint.exists(), (
        f"Post-success cleanup did not delete {capture_checkpoint}."
    )
    assert _VIZ_COUNT["n"] >= 1, (
        f"Quick-update visualize did not fire (count={_VIZ_COUNT['n']}). "
        f"Phase 3 hook is broken or the run was too short."
    )

    capture_log_evidence = float(
        capture_result.samples.samples_info["log_evidence"]
    )
    assert np.isfinite(capture_log_evidence), (
        f"Capture-pass log_evidence not finite: {capture_log_evidence!r}"
    )

    print(f"  Snapshot captured at: {snapshot_path}")
    print(f"  Quick-update viz fires: {_VIZ_COUNT['n']}")
    print(f"  Post-success cleanup: OK")
    print(f"  log_evidence (capture pass): {capture_log_evidence:.4f}")

    # --- Resume pass: pre-plant the snapshot and confirm resume ---
    print("\nResume pass: pre-planting snapshot and confirming resume...")
    _wipe(output_root)
    _VIZ_COUNT["n"] = 0

    resume_search = _new_search()
    # We have to trigger model-aware path resolution before pre-planting the
    # checkpoint, otherwise the unique-tag hash isn't computed yet and the
    # path we copy to differs from where ``_fit`` looks. Calling ``set_paths``
    # via a no-op fit step is awkward; instead we hash here using the same
    # path_prefix / name / model / search-identifier-fields chain.
    resume_search.paths.model = model
    resume_search.paths.search = resume_search
    target_dir = resume_search.paths.search_internal_path
    target_dir.mkdir(parents=True, exist_ok=True)
    target_checkpoint = target_dir / "nss_checkpoint.pkl"
    shutil.copy(snapshot_path, target_checkpoint)
    print(f"  Pre-planted checkpoint at: {target_checkpoint}")
    print(f"  Exists: {target_checkpoint.exists()}")
    print(f"  Resolved by NSS: {resume_search._nss_checkpoint_path}")

    log_capture.clear()
    resume_result = resume_search.fit(model=model, analysis=analysis)

    saw_resume_log = any("Resuming NSS from checkpoint" in m for m in log_capture)
    assert saw_resume_log, (
        f"Expected 'Resuming NSS from checkpoint' log message did not fire. "
        f"Recent log lines: {log_capture[-10:]}"
    )

    resume_log_evidence = float(
        resume_result.samples.samples_info["log_evidence"]
    )
    assert np.isfinite(resume_log_evidence), (
        f"Resume-pass log_evidence not finite: {resume_log_evidence!r}"
    )
    assert not target_checkpoint.exists(), (
        f"Post-success cleanup did not delete resumed-run checkpoint at "
        f"{target_checkpoint}."
    )

    print(f"  Resume log fired: OK")
    print(f"  log_evidence (resume pass): {resume_log_evidence:.4f}")
    print(f"  Post-success cleanup (resume pass): OK")

    print("\nnss_checkpoint_resume: all assertions passed")


if __name__ == "__main__":
    main()
