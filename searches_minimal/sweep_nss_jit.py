"""
Sweep NSS (slice-based, pure JAX) sampler settings on the HST MGE
imaging likelihood and record per-config evals/wall-time/log_Z so we can
pick the most efficient ``nss_jit.py`` configuration.

Six configurations (cheapest first), each run to native ``termination``.
Per-config rows are appended to ``output/sweep_nss_jit.csv`` immediately
so a Ctrl-C mid-sweep still leaves earlier rows on disk.

Why no wall-clock cap: ``run_nested_sampling`` exposes only the
``termination`` (delta-logZ) knob — no ``max_steps``. Configs are picked
to fit in ~10–15 min on the RTX 2060; the largest (n_live=500) loosens
``termination`` to -1 to stay in budget.
"""
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from nss.ns import run_nested_sampling

from searches_minimal._setup import (
    build_analysis,
    build_dataset,
    build_model,
    format_best_fit,
)


@dataclass
class SweepConfig:
    name: str
    n_live: int
    num_mcmc_steps: int
    num_delete: int
    termination: int


CONFIGS = [
    SweepConfig("c1_baseline",   n_live=200, num_mcmc_steps=5,  num_delete=10, termination=-3),
    SweepConfig("c2_fewer_mcmc", n_live=200, num_mcmc_steps=3,  num_delete=10, termination=-3),
    SweepConfig("c3_more_mcmc",  n_live=200, num_mcmc_steps=10, num_delete=10, termination=-3),
    SweepConfig("c4_big_delete", n_live=200, num_mcmc_steps=5,  num_delete=50, termination=-3),
    SweepConfig("c5_half_live",  n_live=100, num_mcmc_steps=5,  num_delete=10, termination=-3),
    SweepConfig("c6_big_live",   n_live=500, num_mcmc_steps=5,  num_delete=10, termination=-1),
]

CSV_FIELDS = [
    "name",
    "n_live",
    "num_mcmc_steps",
    "num_delete",
    "termination",
    "evals",
    "sampling_time",
    "wall_time",
    "ess",
    "log_Z",
    "max_logL_live",
    "best_fit",
]


def make_initial_samples(model, n_live: int, key) -> jnp.ndarray:
    """Map unit-cube draws through the model priors so every starting
    point is in the prior support — same construction nss_jit.py uses."""
    ndim = model.prior_count
    unit_cube = np.asarray(jax.random.uniform(key, shape=(n_live, ndim)))
    physical = np.array([model.vector_from_unit_vector(list(u)) for u in unit_cube])
    return jnp.asarray(physical)


def run_one(cfg: SweepConfig, log_likelihood, log_prior, model) -> dict:
    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)
    initial_samples = make_initial_samples(model, cfg.n_live, init_key)

    print(
        f"\n=== {cfg.name}  (n_live={cfg.n_live} mcmc={cfg.num_mcmc_steps}"
        f" delete={cfg.num_delete} term={cfg.termination}) ===",
        flush=True,
    )
    t_start = time.time()
    final_state, results = run_nested_sampling(
        rng_key,
        loglikelihood_fn=log_likelihood,
        prior_logprob=log_prior,
        num_mcmc_steps=cfg.num_mcmc_steps,
        initial_samples=initial_samples,
        num_delete=cfg.num_delete,
        termination=cfg.termination,
    )
    wall = time.time() - t_start

    log_l_live = np.asarray(final_state.particles.loglikelihood)
    positions = np.asarray(final_state.particles.position)
    best_idx = int(np.argmax(log_l_live))
    best_instance = model.instance_from_vector(vector=positions[best_idx].tolist())

    row = dict(
        name=cfg.name,
        n_live=cfg.n_live,
        num_mcmc_steps=cfg.num_mcmc_steps,
        num_delete=cfg.num_delete,
        termination=cfg.termination,
        evals=int(results.evals),
        sampling_time=float(results.time),
        wall_time=wall,
        ess=float(results.ess),
        log_Z=float(jnp.asarray(results.logZs).mean()),
        max_logL_live=float(np.max(log_l_live)),
        best_fit=format_best_fit(best_instance),
    )

    print(
        f"  evals={row['evals']}  sampling_time={row['sampling_time']:.1f}s"
        f"  wall={row['wall_time']:.1f}s  ess={row['ess']:.1f}"
        f"  log_Z={row['log_Z']:.3f}  max_logL_live={row['max_logL_live']:.3f}",
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

    def log_likelihood(params):
        instance = model.instance_from_vector(vector=params, xp=jnp)
        return analysis.log_likelihood_function(instance=instance)

    def log_prior(params):
        return jnp.float64(0.0)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sweep_nss_jit.csv"
    summary_path = output_dir / "sweep_nss_jit_summary.txt"
    if csv_path.exists():
        csv_path.unlink()  # fresh sweep

    rows: list[dict] = []
    for cfg in CONFIGS:
        try:
            row = run_one(cfg, log_likelihood, log_prior, model)
        except KeyboardInterrupt:
            print(f"\nAborted by user at {cfg.name}.", flush=True)
            break
        except Exception as exc:
            print(f"\n{cfg.name} failed: {exc}", flush=True)
            row = dict(
                name=cfg.name,
                n_live=cfg.n_live,
                num_mcmc_steps=cfg.num_mcmc_steps,
                num_delete=cfg.num_delete,
                termination=cfg.termination,
                evals="",
                sampling_time="",
                wall_time="",
                ess="",
                log_Z="",
                max_logL_live="",
                best_fit=f"FAILED: {exc}",
            )
        append_csv(csv_path, row)
        rows.append(row)

    # Summary table sorted by evals among "converged" rows
    succeeded = [r for r in rows if isinstance(r.get("max_logL_live"), float)]
    if succeeded:
        best_logL = max(r["max_logL_live"] for r in succeeded)
        for r in succeeded:
            r["delta_logL"] = best_logL - r["max_logL_live"]
            r["converged"] = r["delta_logL"] < 1.0
        converged = [r for r in succeeded if r["converged"]]
        recommended = (
            min(converged, key=lambda r: r["evals"]) if converged else None
        )
    else:
        best_logL = float("nan")
        recommended = None

    lines = ["--- NSS slice-sampler settings sweep ---", ""]
    lines.append(
        f"{'name':14s} {'n_live':>6s} {'mcmc':>5s} {'del':>4s} {'term':>5s}"
        f" {'evals':>8s} {'wall_s':>9s} {'logZ':>11s} {'maxL_live':>12s}"
        f" {'dLogL':>8s} {'conv':>5s}"
    )
    for r in rows:
        if isinstance(r.get("max_logL_live"), float):
            d = r.get("delta_logL", float("nan"))
            conv = "yes" if r.get("converged") else "no"
            lines.append(
                f"{r['name']:14s} {r['n_live']:>6d} {r['num_mcmc_steps']:>5d}"
                f" {r['num_delete']:>4d} {r['termination']:>5d}"
                f" {r['evals']:>8d} {r['wall_time']:>9.1f}"
                f" {r['log_Z']:>11.3f} {r['max_logL_live']:>12.3f}"
                f" {d:>8.3f} {conv:>5s}"
            )
        else:
            lines.append(f"{r['name']:14s} FAILED  ({r['best_fit']})")
    lines.append("")
    lines.append(f"best max_logL_live across sweep: {best_logL:.3f}")
    if recommended is not None:
        lines.append(
            f"Recommended config (fewest evals among converged):"
            f" {recommended['name']}"
            f"  n_live={recommended['n_live']}"
            f"  num_mcmc_steps={recommended['num_mcmc_steps']}"
            f"  num_delete={recommended['num_delete']}"
            f"  evals={recommended['evals']}"
            f"  wall={recommended['wall_time']:.1f}s"
        )
    else:
        lines.append("No config converged (within 1 nat of best max_logL_live).")
    lines.append("")
    lines.append("Best fit per config:")
    for r in rows:
        lines.append(f"  {r['name']:14s} -> {r['best_fit']}")

    summary_path.write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))
    print(f"\nWritten: {csv_path}")
    print(f"Written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
