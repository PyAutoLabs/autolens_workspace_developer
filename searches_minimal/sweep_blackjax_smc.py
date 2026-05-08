"""
Sweep BlackJAX SMC (adaptive tempered + RWM) sampler settings on the HST
MGE imaging likelihood and record per-config evals/wall-time/log_Z so we
can pick the most efficient ``blackjax_smc.py`` configuration.

Eight configurations (cheapest first), each run to native termination
(``tempering_param >= 1.0``). Per-config rows are appended to
``output/sweep_blackjax_smc.csv`` immediately so a Ctrl-C mid-sweep
still leaves earlier rows on disk.

Why no wall-clock cap: BlackJAX adaptive-tempered SMC terminates when
the temperature schedule reaches lambda=1.0; on this peaked likelihood
that may take 100-300 SMC steps depending on ``target_ess``. The
``c7_more_particles`` config (n_particles=512) is HPC-targeted and may
OOM on a 6 GB card; lower-particle configs work locally.

Known limitation: configs run sequentially in the same process so the
JAX BFC allocator does NOT release between configs (cf. the cascading
OOM seen in ``sweep_nss_jit.py``). On HPC this isn't an issue because
the 6 GB ceiling is irrelevant; locally, drop ``c7_more_particles`` if
you hit OOM and want to keep the rest of the sweep going.
"""
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import blackjax
import blackjax.smc.resampling as resampling
import blackjax.mcmc.random_walk as rw

from searches_minimal._setup import (
    build_analysis,
    build_dataset,
    build_model,
    format_best_fit,
)


@dataclass
class SweepConfig:
    name: str
    n_particles: int
    num_mcmc_steps: int
    target_ess: float
    rmh_sigma: float


CONFIGS = [
    SweepConfig("c1_baseline",        n_particles=256, num_mcmc_steps=5,  target_ess=0.5, rmh_sigma=0.05),
    SweepConfig("c2_fewer_mcmc",      n_particles=256, num_mcmc_steps=3,  target_ess=0.5, rmh_sigma=0.05),
    SweepConfig("c3_more_mcmc",       n_particles=256, num_mcmc_steps=10, target_ess=0.5, rmh_sigma=0.05),
    SweepConfig("c4_lower_ess",       n_particles=256, num_mcmc_steps=5,  target_ess=0.3, rmh_sigma=0.05),
    SweepConfig("c5_smaller_step",    n_particles=256, num_mcmc_steps=5,  target_ess=0.5, rmh_sigma=0.02),
    SweepConfig("c6_bigger_step",     n_particles=256, num_mcmc_steps=5,  target_ess=0.5, rmh_sigma=0.10),
    SweepConfig("c7_more_particles",  n_particles=512, num_mcmc_steps=5,  target_ess=0.5, rmh_sigma=0.05),
    SweepConfig("c8_fewer_particles", n_particles=128, num_mcmc_steps=5,  target_ess=0.5, rmh_sigma=0.05),
]

CSV_FIELDS = [
    "name",
    "n_particles",
    "num_mcmc_steps",
    "target_ess",
    "rmh_sigma",
    "n_smc_steps",
    "evals",
    "wall_time",
    "log_Z",
    "max_logL_final",
    "mean_acceptance",
    "best_fit",
]


def smc_step_fn(rng_key, state, logdensity_fn, sigma):
    proposal = rw.normal(sigma)
    kernel = rw.build_additive_step()
    return kernel(rng_key, state, logdensity_fn, proposal)


def run_one(cfg: SweepConfig, log_prior, log_likelihood, vmapped_log_l, model) -> dict:
    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)

    ndim = model.prior_count
    sigma_init = jnp.eye(ndim)[None, :, :] * cfg.rmh_sigma  # (1, N, N) shared

    smc = blackjax.adaptive_tempered_smc(
        logprior_fn=log_prior,
        loglikelihood_fn=log_likelihood,
        mcmc_step_fn=smc_step_fn,
        mcmc_init_fn=rw.init,
        mcmc_parameters={"sigma": sigma_init},
        resampling_fn=resampling.systematic,
        target_ess=cfg.target_ess,
        num_mcmc_steps=cfg.num_mcmc_steps,
    )

    initial_particles = jax.random.uniform(
        init_key, shape=(cfg.n_particles, ndim), minval=0.0, maxval=1.0
    )
    state = smc.init(initial_particles)

    print(
        f"\n=== {cfg.name}  (n_particles={cfg.n_particles} mcmc={cfg.num_mcmc_steps}"
        f" target_ess={cfg.target_ess} rmh_sigma={cfg.rmh_sigma}) ===",
        flush=True,
    )

    t_start = time.time()
    log_z = 0.0
    n_smc_steps = 0
    accept_sum = 0.0
    accept_count = 0
    while float(state.tempering_param) < 1.0:
        rng_key, sub_key = jax.random.split(rng_key)
        state, info = jax.block_until_ready(smc.step(sub_key, state))
        log_z += float(info.log_likelihood_increment)
        accept_sum += float(jnp.mean(info.update_info.acceptance_rate))
        accept_count += 1
        n_smc_steps += 1
    wall = time.time() - t_start

    final_log_l = vmapped_log_l(state.particles)
    best_idx = int(jnp.argmax(final_log_l))
    best_cube = np.asarray(state.particles[best_idx])
    best_physical = np.asarray(
        model.vector_from_unit_vector(list(best_cube)), dtype=np.float64
    )
    best_instance = model.instance_from_vector(vector=list(best_physical))

    # Upper-bound likelihood eval count: each SMC step runs ``num_mcmc_steps``
    # MCMC evals per particle plus the per-step ``vmapped_log_l`` we use for
    # max-tracking. JIT-traced proposals always evaluate even on rejected MH
    # moves, so this is an upper bound, not the literal count.
    evals = (
        cfg.n_particles
        + n_smc_steps * cfg.n_particles * (cfg.num_mcmc_steps + 1)
    )

    row = dict(
        name=cfg.name,
        n_particles=cfg.n_particles,
        num_mcmc_steps=cfg.num_mcmc_steps,
        target_ess=cfg.target_ess,
        rmh_sigma=cfg.rmh_sigma,
        n_smc_steps=n_smc_steps,
        evals=evals,
        wall_time=wall,
        log_Z=log_z,
        max_logL_final=float(jnp.max(final_log_l)),
        mean_acceptance=accept_sum / max(accept_count, 1),
        best_fit=format_best_fit(best_instance),
    )

    print(
        f"  n_smc_steps={n_smc_steps}  evals={evals}  wall={wall:.1f}s"
        f"  log_Z={row['log_Z']:.3f}  max_logL={row['max_logL_final']:.3f}"
        f"  acc={row['mean_acceptance']:.3f}",
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

    def log_prior(cube):
        in_cube = jnp.all((cube >= 0.0) & (cube <= 1.0))
        return jnp.where(in_cube, 0.0, -jnp.inf)

    def log_likelihood(cube):
        in_cube = jnp.all((cube >= 0.0) & (cube <= 1.0))
        safe_cube = jnp.clip(cube, 0.0, 1.0)
        physical = cube_to_physical(safe_cube)
        instance = model.instance_from_vector(vector=physical, xp=jnp)
        log_l = analysis.log_likelihood_function(instance=instance)
        return jnp.where(in_cube, log_l, 0.0)

    # JIT compile the likelihood once before the sweep so the first config
    # doesn't carry the compile cost. The cache is reused for all configs.
    print("JIT-compiling MGE likelihood (one-shot)...", flush=True)
    t0 = time.time()
    _ = float(jax.block_until_ready(jax.jit(log_likelihood)(jnp.full(ndim, 0.5))))
    print(f"  Compiled in {time.time() - t0:.2f} s", flush=True)

    vmapped_log_l = jax.jit(jax.vmap(log_likelihood))

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sweep_blackjax_smc.csv"
    summary_path = output_dir / "sweep_blackjax_smc_summary.txt"
    if csv_path.exists():
        csv_path.unlink()

    rows: list[dict] = []
    for cfg in CONFIGS:
        try:
            row = run_one(cfg, log_prior, log_likelihood, vmapped_log_l, model)
        except KeyboardInterrupt:
            print(f"\nAborted by user at {cfg.name}.", flush=True)
            break
        except Exception as exc:
            print(f"\n{cfg.name} failed: {exc}", flush=True)
            row = dict(
                name=cfg.name,
                n_particles=cfg.n_particles,
                num_mcmc_steps=cfg.num_mcmc_steps,
                target_ess=cfg.target_ess,
                rmh_sigma=cfg.rmh_sigma,
                n_smc_steps="",
                evals="",
                wall_time="",
                log_Z="",
                max_logL_final="",
                mean_acceptance="",
                best_fit=f"FAILED: {exc}",
            )
        append_csv(csv_path, row)
        rows.append(row)

    succeeded = [r for r in rows if isinstance(r.get("max_logL_final"), float)]
    if succeeded:
        best_logL = max(r["max_logL_final"] for r in succeeded)
        for r in succeeded:
            r["delta_logL"] = best_logL - r["max_logL_final"]
            r["converged"] = r["delta_logL"] < 1.0
        converged = [r for r in succeeded if r["converged"]]
        recommended = (
            min(converged, key=lambda r: r["evals"]) if converged else None
        )
    else:
        best_logL = float("nan")
        recommended = None

    lines = ["--- BlackJAX SMC settings sweep ---", ""]
    lines.append(
        f"{'name':18s} {'np':>4s} {'mcmc':>5s} {'tess':>5s} {'sig':>5s}"
        f" {'smc':>4s} {'evals':>9s} {'wall_s':>9s} {'logZ':>11s}"
        f" {'maxL':>11s} {'acc':>5s} {'dLogL':>8s} {'conv':>5s}"
    )
    for r in rows:
        if isinstance(r.get("max_logL_final"), float):
            d = r.get("delta_logL", float("nan"))
            conv = "yes" if r.get("converged") else "no"
            lines.append(
                f"{r['name']:18s} {r['n_particles']:>4d} {r['num_mcmc_steps']:>5d}"
                f" {r['target_ess']:>5.2f} {r['rmh_sigma']:>5.2f}"
                f" {r['n_smc_steps']:>4d} {r['evals']:>9d} {r['wall_time']:>9.1f}"
                f" {r['log_Z']:>11.3f} {r['max_logL_final']:>11.3f}"
                f" {r['mean_acceptance']:>5.2f} {d:>8.3f} {conv:>5s}"
            )
        else:
            lines.append(f"{r['name']:18s} FAILED  ({r['best_fit']})")
    lines.append("")
    lines.append(f"best max_logL_final across sweep: {best_logL:.3f}")
    if recommended is not None:
        lines.append(
            f"Recommended config (fewest evals among converged):"
            f" {recommended['name']}"
            f"  n_particles={recommended['n_particles']}"
            f"  num_mcmc_steps={recommended['num_mcmc_steps']}"
            f"  target_ess={recommended['target_ess']}"
            f"  rmh_sigma={recommended['rmh_sigma']}"
            f"  evals={recommended['evals']}"
            f"  wall={recommended['wall_time']:.1f}s"
        )
    else:
        lines.append("No config converged (within 1 nat of best max_logL_final).")
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
