"""
Sweep PocoMC (preconditioned MC + normalizing flows) settings on the HST
MGE imaging likelihood and record per-config evals/wall-time/log_Z so we
can pick the most efficient ``pocomc_simple.py`` configuration.

Seven configurations (cheapest first), each run to completion. Per-config
rows are appended to ``output/sweep_pocomc.csv`` immediately so a
Ctrl-C mid-sweep still leaves earlier rows on disk.

PocoMC is not JAX-native (PyTorch + NumPy backend); the JAX-jitted
likelihood crosses an ``np.asarray`` boundary on every call — same cost
as ``nautilus_jax``. PocoMC trains a Zuko normalising flow on PyTorch
which competes with JAX for VRAM, so we cap JAX at 50%.

Known limitation: configs run sequentially in the same process so neither
JAX nor PyTorch necessarily releases memory cleanly between configs. The
``c7_huge`` config (n_effective=2048) is HPC-targeted and may OOM on a
6 GB card; lower-resource configs work locally.
"""
import os
# Cap JAX VRAM at 50% so PyTorch (Zuko flow) has headroom on small cards.
# Must be set before any JAX import.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import uniform as scipy_uniform
import jax
import jax.numpy as jnp
import pocomc

from searches_minimal._setup import (
    build_analysis,
    build_dataset,
    build_model,
    format_best_fit,
)


@dataclass
class SweepConfig:
    name: str
    n_effective: int
    n_active: int
    n_total: int
    n_evidence: int


CONFIGS = [
    SweepConfig("c1_baseline",       n_effective=512,  n_active=256, n_total=4096,  n_evidence=4096),
    SweepConfig("c2_smaller",        n_effective=256,  n_active=128, n_total=2048,  n_evidence=2048),
    SweepConfig("c3_bigger",         n_effective=1024, n_active=512, n_total=8192,  n_evidence=8192),
    SweepConfig("c4_more_effective", n_effective=1024, n_active=256, n_total=4096,  n_evidence=4096),
    SweepConfig("c5_more_total",     n_effective=512,  n_active=256, n_total=8192,  n_evidence=4096),
    SweepConfig("c6_more_evidence",  n_effective=512,  n_active=256, n_total=4096,  n_evidence=8192),
    SweepConfig("c7_huge",           n_effective=2048, n_active=512, n_total=16384, n_evidence=8192),
]

CSV_FIELDS = [
    "name",
    "n_effective",
    "n_active",
    "n_total",
    "n_evidence",
    "evals",
    "wall_time",
    "log_Z",
    "log_Z_err",
    "max_logL",
    "best_fit",
]


def run_one(cfg: SweepConfig, likelihood_callable, eval_counter, model, ndim) -> dict:
    eval_counter[0] = 0  # reset per-config

    prior = pocomc.Prior([scipy_uniform(loc=0.0, scale=1.0)] * ndim)
    sampler = pocomc.Sampler(
        prior=prior,
        likelihood=likelihood_callable,
        n_dim=ndim,
        n_effective=cfg.n_effective,
        n_active=cfg.n_active,
        random_state=42,
    )

    print(
        f"\n=== {cfg.name}  (n_effective={cfg.n_effective} n_active={cfg.n_active}"
        f" n_total={cfg.n_total} n_evidence={cfg.n_evidence}) ===",
        flush=True,
    )

    t_start = time.time()
    sampler.run(n_total=cfg.n_total, n_evidence=cfg.n_evidence, progress=True)
    wall = time.time() - t_start

    samples, weights, logl, logp = sampler.posterior()
    logz, logz_err = sampler.evidence()

    best_idx = int(np.argmax(logl))
    best_cube = np.asarray(samples[best_idx])
    best_physical = np.asarray(
        model.vector_from_unit_vector(list(best_cube)), dtype=np.float64
    )
    best_instance = model.instance_from_vector(vector=list(best_physical))
    max_logl = float(np.max(logl))

    row = dict(
        name=cfg.name,
        n_effective=cfg.n_effective,
        n_active=cfg.n_active,
        n_total=cfg.n_total,
        n_evidence=cfg.n_evidence,
        evals=int(eval_counter[0]),
        wall_time=wall,
        log_Z=float(logz),
        log_Z_err=float(logz_err),
        max_logL=max_logl,
        best_fit=format_best_fit(best_instance),
    )

    print(
        f"  evals={row['evals']}  wall={wall:.1f}s"
        f"  log_Z={row['log_Z']:.3f}+/-{row['log_Z_err']:.3f}"
        f"  max_logL={row['max_logL']:.3f}",
        flush=True,
    )
    return row


def append_csv(path: Path, row: dict) -> None:
    new_file = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def main() -> int:
    print(f"JAX devices: {jax.devices()}")
    if jax.default_backend() != "gpu":
        print("ERROR: JAX is not on GPU. Aborting.", file=sys.stderr)
        return 2

    dataset = build_dataset()
    model = build_model()
    analysis = build_analysis(dataset, use_jax=True)
    print(f"Model free parameters: {model.prior_count}")

    ndim = model.prior_count

    def log_likelihood_jax(physical):
        instance = model.instance_from_vector(vector=physical, xp=jnp)
        return analysis.log_likelihood_function(instance=instance)

    jit_log_likelihood = jax.jit(log_likelihood_jax)

    print("JIT-compiling MGE likelihood (one-shot)...", flush=True)
    t0 = time.time()
    warmup_physical = jnp.asarray(
        model.vector_from_unit_vector([0.5] * ndim)
    )
    _ = float(jax.block_until_ready(jit_log_likelihood(warmup_physical)))
    print(f"  Compiled in {time.time() - t0:.2f} s", flush=True)

    # Mutable single-element list serves as a per-config eval counter that
    # the closure can mutate without ``nonlocal`` gymnastics across run_one.
    eval_counter = [0]

    def likelihood_for_pocomc(cube_np):
        eval_counter[0] += 1
        physical_np = np.asarray(
            model.vector_from_unit_vector(list(np.asarray(cube_np))),
            dtype=np.float64,
        )
        return float(jit_log_likelihood(jnp.asarray(physical_np)))

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sweep_pocomc.csv"
    summary_path = output_dir / "sweep_pocomc_summary.txt"
    if csv_path.exists():
        csv_path.unlink()

    rows: list[dict] = []
    for cfg in CONFIGS:
        try:
            row = run_one(cfg, likelihood_for_pocomc, eval_counter, model, ndim)
        except KeyboardInterrupt:
            print(f"\nAborted by user at {cfg.name}.", flush=True)
            break
        except Exception as exc:
            print(f"\n{cfg.name} failed: {exc}", flush=True)
            row = dict(
                name=cfg.name,
                n_effective=cfg.n_effective,
                n_active=cfg.n_active,
                n_total=cfg.n_total,
                n_evidence=cfg.n_evidence,
                evals="",
                wall_time="",
                log_Z="",
                log_Z_err="",
                max_logL="",
                best_fit=f"FAILED: {exc}",
            )
        append_csv(csv_path, row)
        rows.append(row)

    succeeded = [r for r in rows if isinstance(r.get("max_logL"), float)]
    if succeeded:
        best_logL = max(r["max_logL"] for r in succeeded)
        for r in succeeded:
            r["delta_logL"] = best_logL - r["max_logL"]
            r["converged"] = r["delta_logL"] < 1.0
        converged = [r for r in succeeded if r["converged"]]
        recommended = (
            min(converged, key=lambda r: r["evals"]) if converged else None
        )
    else:
        best_logL = float("nan")
        recommended = None

    lines = ["--- PocoMC settings sweep ---", ""]
    lines.append(
        f"{'name':18s} {'n_eff':>6s} {'n_act':>6s} {'n_tot':>6s} {'n_ev':>6s}"
        f" {'evals':>9s} {'wall_s':>9s} {'logZ':>11s} {'logZ_err':>9s}"
        f" {'maxL':>11s} {'dLogL':>8s} {'conv':>5s}"
    )
    for r in rows:
        if isinstance(r.get("max_logL"), float):
            d = r.get("delta_logL", float("nan"))
            conv = "yes" if r.get("converged") else "no"
            lines.append(
                f"{r['name']:18s} {r['n_effective']:>6d} {r['n_active']:>6d}"
                f" {r['n_total']:>6d} {r['n_evidence']:>6d}"
                f" {r['evals']:>9d} {r['wall_time']:>9.1f}"
                f" {r['log_Z']:>11.3f} {r['log_Z_err']:>9.3f}"
                f" {r['max_logL']:>11.3f} {d:>8.3f} {conv:>5s}"
            )
        else:
            lines.append(f"{r['name']:18s} FAILED  ({r['best_fit']})")
    lines.append("")
    lines.append(f"best max_logL across sweep: {best_logL:.3f}")
    if recommended is not None:
        lines.append(
            f"Recommended config (fewest evals among converged):"
            f" {recommended['name']}"
            f"  n_effective={recommended['n_effective']}"
            f"  n_active={recommended['n_active']}"
            f"  n_total={recommended['n_total']}"
            f"  n_evidence={recommended['n_evidence']}"
            f"  evals={recommended['evals']}"
            f"  wall={recommended['wall_time']:.1f}s"
        )
    else:
        lines.append("No config converged (within 1 nat of best max_logL).")
    lines.append("")
    lines.append("Best fit per config:")
    for r in rows:
        lines.append(f"  {r['name']:18s} -> {r['best_fit']}")

    summary_path.write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))
    print(f"\nWritten: {csv_path}")
    print(f"Written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
