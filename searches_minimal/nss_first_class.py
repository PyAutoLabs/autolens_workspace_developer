"""
NSS First-Class Search — `af.NSS` on the HST MGE Likelihood
-----------------------------------------------------------

End-to-end smoke for the Phase 1 `af.NSS` `NonLinearSearch` wrapper
(PyAutoLabs/PyAutoFit#1271). Drives the same HST MGE problem as the
bare-bones `nss_jit.py` reference, but through the production
`search = af.NSS(...).fit(model=model, analysis=analysis)` pipeline so
the result rides on `Paths`, `Result`, and the aggregator round-trip
that downstream users actually consume.

Expected outcome (from FINDINGS_v3 `c3_big_delete`):
- `result.max_log_likelihood_instance.galaxies.lens.mass.einstein_radius`
  ≈ 1.5996 ± 0.002.
- `result.samples.log_evidence` finite, similar magnitude to the bare
  reference (~ -31786).
- Wall time within 20% of bare `nss_jit.py` on the same machine.

Requirements:
    pip install git+https://github.com/yallup/nss.git

Run from the workspace root:
    cd autolens_workspace_developer
    python searches_minimal/nss_first_class.py
"""

import time
from pathlib import Path

import numpy as np

import autofit as af

from searches_minimal._setup import (
    build_analysis,
    build_dataset,
    build_model,
    format_best_fit,
)


def main():
    dataset = build_dataset()
    model = build_model()
    analysis = build_analysis(dataset, use_jax=True)

    print(f"Model free parameters: {model.total_free_parameters}")

    search = af.NSS(
        name="nss_first_class",
        path_prefix=str(Path("searches_minimal") / "output"),
        n_live=200,
        num_mcmc_steps=5,
        num_delete=10,
        termination=-3.0,
        seed=42,
    )

    print(
        "Running af.NSS via NonLinearSearch.fit (Phase 1 wrapper). JIT compile "
        "on the first iteration may take 25-30 s.\n"
    )

    t_start = time.time()
    result = search.fit(model=model, analysis=analysis)
    t_elapsed = time.time() - t_start

    best_instance = result.max_log_likelihood_instance
    samples = result.samples
    info = samples.samples_info

    max_logl = float(max(samples.log_likelihood_list))

    summary = f"""\
--- af.NSS (Phase 1 wrapper) Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_logl:.4f}
Log evidence:    {info.get("log_evidence"):.4f} +/- {info.get("log_evidence_error"):.4f}

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (includes JIT compile)
Sampling time:       {info.get("sampling_time"):.2f} s
Likelihood evals:    {info.get("total_samples")}
Time per eval:       {info.get("sampling_time") / max(info.get("total_samples"), 1) * 1e3:.3f} ms
ESS:                 {info.get("ess")}
Posterior samples:   {info.get("total_accepted_samples")}
Sampler config:      n_live={info.get("number_live_points")}, num_mcmc_steps={info.get("num_mcmc_steps")}, num_delete={info.get("num_delete")}, termination={info.get("termination")}

--- Correctness gates ---
"""

    einstein_radius = float(
        best_instance.galaxies.lens.mass.einstein_radius
    )
    log_evidence = float(info["log_evidence"])

    assertion_lines = []

    er_ok = abs(einstein_radius - 1.5996) < 0.002
    assertion_lines.append(
        f"einstein_radius = {einstein_radius:.4f} "
        f"(target 1.5996 +/- 0.002) -> {'PASS' if er_ok else 'FAIL'}"
    )

    lz_finite = np.isfinite(log_evidence)
    assertion_lines.append(
        f"log_evidence    = {log_evidence:.4f} "
        f"(must be finite)               -> {'PASS' if lz_finite else 'FAIL'}"
    )

    summary += "\n".join(assertion_lines) + "\n"

    print()
    print(summary)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{Path(__file__).stem}_summary.txt"
    summary_path.write_text(summary)
    print(f"Summary written to: {summary_path}")

    if not er_ok:
        raise AssertionError(
            f"einstein_radius {einstein_radius:.4f} outside the 1.5996 +/- 0.002 "
            f"correctness window — the af.NSS Phase 1 wrapper has regressed."
        )
    if not lz_finite:
        raise AssertionError(
            f"log_evidence {log_evidence!r} is not finite — the af.NSS Phase 1 "
            f"wrapper has not converged or the NSSamples conversion is broken."
        )

    print("nss_first_class: all correctness gates passed")


if __name__ == "__main__":
    main()
