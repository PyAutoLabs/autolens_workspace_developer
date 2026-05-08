"""
Sweep NumPyro Ensemble Slice Sampling (ESS) settings on the HST MGE
imaging likelihood and record per-config evals/wall-time/diagnostics so
we can pick the most efficient ``numpyro_ess.py`` configuration.

Seven configurations (cheapest first), each run to completion. Per-config
rows are appended to ``output/sweep_numpyro_ess.csv`` immediately so a
Ctrl-C mid-sweep still leaves earlier rows on disk.

ESS does not estimate log evidence — the headline diagnostics here are
``r_hat_max`` (mixing) and ``n_eff_min`` (effective samples per
parameter). A converged config is one with ``r_hat < 1.1`` and reasonable
``n_eff_min``. NumPyro's ESS requires ``n_chains >= 2 * ndim`` and
``n_chains`` divisible by 2.

Known limitation: configs run sequentially in the same process so the
JAX BFC allocator does NOT release between configs. The ``c3_huge_chains``
config (n_chains=128) is HPC-targeted and may OOM on a 6 GB card; lower-
chain configs work locally.
"""
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.diagnostics as diagnostics
from numpyro.infer import MCMC, ESS

from searches_minimal._setup import (
    build_analysis,
    build_dataset,
    build_model,
    format_best_fit,
)


@dataclass
class SweepConfig:
    name: str
    n_chains: int
    num_warmup: int
    num_samples: int


CONFIGS = [
    SweepConfig("c1_baseline",     n_chains=32,  num_warmup=200,  num_samples=200),
    SweepConfig("c2_more_chains",  n_chains=64,  num_warmup=200,  num_samples=200),
    SweepConfig("c3_huge_chains",  n_chains=128, num_warmup=200,  num_samples=200),
    SweepConfig("c4_more_warmup",  n_chains=32,  num_warmup=500,  num_samples=200),
    SweepConfig("c5_less_warmup",  n_chains=32,  num_warmup=50,   num_samples=200),
    SweepConfig("c6_more_samples", n_chains=32,  num_warmup=200,  num_samples=500),
    SweepConfig("c7_long_run",     n_chains=32,  num_warmup=1000, num_samples=2000),
]

CSV_FIELDS = [
    "name",
    "n_chains",
    "num_warmup",
    "num_samples",
    "evals",
    "wall_time",
    "max_logL",
    "n_eff_min",
    "r_hat_max",
    "best_fit",
]


def run_one(cfg: SweepConfig, potential_fn, log_likelihood_jitvmap, model) -> dict:
    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key, run_key = jax.random.split(rng_key, 3)

    ndim = model.prior_count
    initial_walkers = jax.random.uniform(
        init_key, shape=(cfg.n_chains, ndim), minval=0.0, maxval=1.0
    )

    kernel = ESS(potential_fn=potential_fn)
    mcmc = MCMC(
        kernel,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        num_chains=cfg.n_chains,
        chain_method="vectorized",
        progress_bar=True,
    )

    print(
        f"\n=== {cfg.name}  (n_chains={cfg.n_chains} warmup={cfg.num_warmup}"
        f" samples={cfg.num_samples}) ===",
        flush=True,
    )

    t_start = time.time()
    mcmc.run(run_key, init_params=initial_walkers)
    jax.block_until_ready(mcmc.get_samples(group_by_chain=False))
    wall = time.time() - t_start

    samples = mcmc.get_samples(group_by_chain=False)
    samples_by_chain = mcmc.get_samples(group_by_chain=True)

    log_l_per_sample = log_likelihood_jitvmap(samples)
    max_logl = float(jnp.max(log_l_per_sample))
    best_idx = int(jnp.argmax(log_l_per_sample))
    best_cube = np.asarray(samples[best_idx])
    best_physical = np.asarray(
        model.vector_from_unit_vector(list(best_cube)), dtype=np.float64
    )
    best_instance = model.instance_from_vector(vector=list(best_physical))

    n_eff_per_param = diagnostics.effective_sample_size(np.asarray(samples_by_chain))
    r_hat_per_param = diagnostics.gelman_rubin(np.asarray(samples_by_chain))
    n_eff_min = float(np.min(n_eff_per_param))
    r_hat_max = float(np.max(r_hat_per_param))

    # Upper-bound eval count: ESS slices the potential along each move; the
    # accepted-move count is (warmup + samples) * n_chains. Per-slice eval
    # count is higher and unknown without instrumentation.
    evals = (cfg.num_warmup + cfg.num_samples) * cfg.n_chains

    row = dict(
        name=cfg.name,
        n_chains=cfg.n_chains,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        evals=evals,
        wall_time=wall,
        max_logL=max_logl,
        n_eff_min=n_eff_min,
        r_hat_max=r_hat_max,
        best_fit=format_best_fit(best_instance),
    )

    print(
        f"  evals={evals}  wall={wall:.1f}s  max_logL={max_logl:.3f}"
        f"  n_eff_min={n_eff_min:.1f}  r_hat_max={r_hat_max:.3f}",
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

    def _cube_to_physical_host(cube_np):
        return np.asarray(
            model.vector_from_unit_vector(list(np.asarray(cube_np))),
            dtype=np.float64,
        )

    _phys_shape = jax.ShapeDtypeStruct((ndim,), jnp.float64)

    def cube_to_physical(cube):
        return jax.pure_callback(
            _cube_to_physical_host, _phys_shape, cube, vmap_method="sequential"
        )

    def log_likelihood(cube):
        in_cube = jnp.all((cube >= 0.0) & (cube <= 1.0))
        safe_cube = jnp.clip(cube, 0.0, 1.0)
        physical = cube_to_physical(safe_cube)
        instance = model.instance_from_vector(vector=physical, xp=jnp)
        log_l = analysis.log_likelihood_function(instance=instance)
        return jnp.where(in_cube, log_l, -jnp.inf)

    def potential_fn(cube):
        return -log_likelihood(cube)

    print("JIT-compiling MGE likelihood (one-shot)...", flush=True)
    t0 = time.time()
    _ = float(jax.block_until_ready(jax.jit(log_likelihood)(jnp.full(ndim, 0.5))))
    print(f"  Compiled in {time.time() - t0:.2f} s", flush=True)

    log_likelihood_jitvmap = jax.jit(jax.vmap(log_likelihood))

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sweep_numpyro_ess.csv"
    summary_path = output_dir / "sweep_numpyro_ess_summary.txt"
    if csv_path.exists():
        csv_path.unlink()

    rows: list[dict] = []
    for cfg in CONFIGS:
        try:
            row = run_one(cfg, potential_fn, log_likelihood_jitvmap, model)
        except KeyboardInterrupt:
            print(f"\nAborted by user at {cfg.name}.", flush=True)
            break
        except Exception as exc:
            print(f"\n{cfg.name} failed: {exc}", flush=True)
            row = dict(
                name=cfg.name,
                n_chains=cfg.n_chains,
                num_warmup=cfg.num_warmup,
                num_samples=cfg.num_samples,
                evals="",
                wall_time="",
                max_logL="",
                n_eff_min="",
                r_hat_max="",
                best_fit=f"FAILED: {exc}",
            )
        append_csv(csv_path, row)
        rows.append(row)

    succeeded = [r for r in rows if isinstance(r.get("max_logL"), float)]
    if succeeded:
        best_logL = max(r["max_logL"] for r in succeeded)
        for r in succeeded:
            r["delta_logL"] = best_logL - r["max_logL"]
            # ESS "converged" means r_hat well below 1.1 AND within 1 nat of best.
            r["converged"] = r["delta_logL"] < 1.0 and r["r_hat_max"] < 1.1
        converged = [r for r in succeeded if r["converged"]]
        recommended = (
            min(converged, key=lambda r: r["evals"]) if converged else None
        )
    else:
        best_logL = float("nan")
        recommended = None

    lines = ["--- NumPyro ESS settings sweep ---", ""]
    lines.append(
        f"{'name':16s} {'chains':>6s} {'warm':>5s} {'samp':>5s}"
        f" {'evals':>9s} {'wall_s':>9s} {'maxL':>11s}"
        f" {'n_eff':>7s} {'r_hat':>6s} {'dLogL':>8s} {'conv':>5s}"
    )
    for r in rows:
        if isinstance(r.get("max_logL"), float):
            d = r.get("delta_logL", float("nan"))
            conv = "yes" if r.get("converged") else "no"
            lines.append(
                f"{r['name']:16s} {r['n_chains']:>6d} {r['num_warmup']:>5d}"
                f" {r['num_samples']:>5d} {r['evals']:>9d} {r['wall_time']:>9.1f}"
                f" {r['max_logL']:>11.3f} {r['n_eff_min']:>7.1f}"
                f" {r['r_hat_max']:>6.3f} {d:>8.3f} {conv:>5s}"
            )
        else:
            lines.append(f"{r['name']:16s} FAILED  ({r['best_fit']})")
    lines.append("")
    lines.append(f"best max_logL across sweep: {best_logL:.3f}")
    if recommended is not None:
        lines.append(
            f"Recommended config (fewest evals among converged):"
            f" {recommended['name']}"
            f"  n_chains={recommended['n_chains']}"
            f"  num_warmup={recommended['num_warmup']}"
            f"  num_samples={recommended['num_samples']}"
            f"  evals={recommended['evals']}"
            f"  wall={recommended['wall_time']:.1f}s"
        )
    else:
        lines.append(
            "No config converged (delta_logL < 1.0 nat AND r_hat < 1.1)."
        )
    lines.append("")
    lines.append("Best fit per config:")
    for r in rows:
        lines.append(f"  {r['name']:16s} -> {r['best_fit']}")

    summary_path.write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))
    print(f"\nWritten: {csv_path}")
    print(f"Written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
