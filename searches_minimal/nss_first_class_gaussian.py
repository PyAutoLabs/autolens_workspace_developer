"""
af.NSS Wiring Smoke — Fast 2D Gaussian
--------------------------------------

End-to-end wiring smoke for the Phase 1 ``af.NSS`` ``NonLinearSearch``
wrapper (PyAutoLabs/PyAutoFit#1271) on a trivial 2-parameter
JAX-traceable Gaussian likelihood with ``GaussianPrior`` priors. Runs in
~30 seconds on CPU and validates the full path:

1. ``af.NSS().fit(model, analysis)`` builds JAX closures and calls
   ``nss.ns.run_nested_sampling``.
2. The returned ``final_state`` + ``results`` are repackaged into the
   ``_NSSInternal`` holder via ``_fit``.
3. ``samples_via_internal_from`` builds ``NSSamples``.
4. ``AbstractNest.perform_update`` writes ``samples.csv`` /
   ``samples_summary.json`` / etc. through ``paths``.
5. ``Result.max_log_likelihood_instance`` returns the expected type and
   recovers the prior mean (since the likelihood is constant and the
   prior is Gaussian).

This complements the heavier ``nss_first_class.py`` HST MGE numerical
smoke — that one validates the science (``einstein_radius=1.5996``),
this one validates the wrapper plumbing in 1/100 of the wall time.

Run from the workspace root:

    python searches_minimal/nss_first_class_gaussian.py
"""

import time
from pathlib import Path

import numpy as np

import autofit as af


class FlatLikelihoodModel:
    """Trivial 2-param model: ``x``, ``y``."""

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y


class FlatLikelihoodAnalysis(af.Analysis):
    """Flat log-likelihood — the posterior is the prior itself."""

    def log_likelihood_function(self, instance):
        return 0.0


def main():
    model = af.Model(FlatLikelihoodModel)
    model.x = af.GaussianPrior(mean=2.5, sigma=1.0)
    model.y = af.GaussianPrior(mean=-1.0, sigma=0.5)

    analysis = FlatLikelihoodAnalysis()

    search = af.NSS(
        name="nss_first_class_gaussian",
        path_prefix=str(Path("searches_minimal") / "output"),
        n_live=40,
        num_mcmc_steps=2,
        num_delete=5,
        termination=-1.0,
        seed=0,
    )

    print(
        "Running af.NSS on 2D Gaussian flat-likelihood smoke. "
        "JIT compile on first iteration may take 10-20 s.\n"
    )

    t_start = time.time()
    result = search.fit(model=model, analysis=analysis)
    t_elapsed = time.time() - t_start

    samples = result.samples
    info = samples.samples_info
    best_instance = result.max_log_likelihood_instance

    parameter_lists = samples.parameter_lists
    x_samples = np.array([p[0] for p in parameter_lists])
    y_samples = np.array([p[1] for p in parameter_lists])
    weights = np.array(samples.weight_list)

    weight_total = float(weights.sum())
    x_weighted_mean = float((x_samples * weights).sum() / weight_total)
    y_weighted_mean = float((y_samples * weights).sum() / weight_total)

    summary = f"""\
--- af.NSS Wiring Smoke (2D Gaussian, flat L) Results ---
Best instance:         x={best_instance.x:.4f}, y={best_instance.y:.4f}
Weighted posterior mean: x={x_weighted_mean:.4f} (target 2.5), y={y_weighted_mean:.4f} (target -1.0)
Log evidence:          {info["log_evidence"]:.4f} +/- {info["log_evidence_error"]:.4f}
Wall time:             {t_elapsed:.2f} s
Posterior samples:     {info["total_accepted_samples"]}
ESS:                   {info["ess"]}
Likelihood evals:      {info["total_samples"]}
"""
    print()
    print(summary)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{Path(__file__).stem}_summary.txt").write_text(summary)

    assert hasattr(best_instance, "x") and hasattr(best_instance, "y"), (
        "Result.max_log_likelihood_instance is missing model attributes; "
        "NSSamples conversion or Result construction is broken."
    )
    assert isinstance(best_instance.x, float), (
        f"best_instance.x has type {type(best_instance.x)}, expected float"
    )
    assert np.isfinite(info["log_evidence"]), (
        f"log_evidence = {info['log_evidence']!r} is not finite"
    )
    assert np.isfinite(info["log_evidence_error"]), (
        f"log_evidence_error = {info['log_evidence_error']!r} is not finite"
    )
    assert info["total_accepted_samples"] > 0, (
        "No accepted samples — NSS run produced an empty posterior."
    )
    assert weights.min() >= 0.0, "Negative posterior weight encountered"
    assert abs(weight_total - 1.0) < 1e-6, (
        f"Posterior weights do not sum to 1: total = {weight_total!r}"
    )
    # Loose recovery check — termination=-1 doesn't fully converge but the
    # posterior should at least be in the right ballpark of the prior mean.
    assert abs(x_weighted_mean - 2.5) < 1.5, (
        f"x weighted mean {x_weighted_mean:.4f} too far from prior mean 2.5"
    )
    assert abs(y_weighted_mean - (-1.0)) < 1.0, (
        f"y weighted mean {y_weighted_mean:.4f} too far from prior mean -1.0"
    )

    # Verify samples.csv was written through the Paths pipeline.
    samples_csv = Path(result.paths._files_path) / "samples.csv"
    assert samples_csv.exists(), (
        f"samples.csv not written to {samples_csv} — AbstractNest.perform_update "
        f"path appears broken for af.NSS."
    )

    print("nss_first_class_gaussian: all wiring assertions passed")


if __name__ == "__main__":
    main()
