"""
Learning-rate-free multi-start optimizers — HST MGE lens likelihood
-------------------------------------------------------------------

Phase-3 settled that wide multi-start Adam is the best fast, reliable MAP
optimizer on this likelihood, and that the *local update rule barely matters*
(Adam / ADABelief / Lion all hit ~15-18% per-start basin rates). But every rule
benchmarked so far carries a hand-set learning rate, and on the PIXELIZED cell
(autolens_workspace_developer#100) multi-start Adam went 0/16 with
``lr=1e-2`` mis-scaling a prime suspect.

This script benchmarks the optax.contrib **learning-rate-free** family —
Prodigy, D-Adaptation, DoG/DoWG, Mechanic, MoMo — plus ADOPT / AdEMAMix as
fixed-lr controls and schedule-free AdamW (schedule-free, *not* lr-free),
against the Adam reference, inside the same multi-start recipe as
``multi_start_adam.py`` (identical seeded broad starts, same basin accounting).

Two wiring points differ from the stacked-parameter trick the fixed-lr scripts
use, and both are load-bearing:

1. **Per-start optimizer state via ``jax.vmap`` over ``init``/``update``.**
   The lr-free rules estimate *global scalars* (Prodigy/D-Adapt's distance
   ``d``, DoG's ``max_dist``, Mechanic's scale, MoMo's Polyak step) with norms
   over the whole parameter tree. Under the usual stacked ``(N_STARTS, ndim)``
   trick those norms would couple every start into ONE shared estimate —
   silently breaking multi-start independence. vmapping the optimizer itself
   gives each start its own state. (Elementwise rules like Adam are unaffected
   either way; they are run through the same vmapped path for uniformity.)

2. **``optax.apply_if_finite`` as the NaN-step guard.** Broad starts can wander
   back through the ell_comps/shear = 0 gradient singularity mid-run (the
   ADABelief failure in phase 1). ``apply_if_finite`` rejects non-finite
   updates per-start instead of poisoning the optimizer state; the
   argmin-over-finite bookkeeping is kept as the outer safety net.

Run from the workspace root (all rules, or name a subset):

    python -m searches_minimal.lr_free_multistart
    python -m searches_minimal.lr_free_multistart prodigy dog mechanic

Env overrides: ``MULTISTART_N_STARTS`` (default 12), ``MULTISTART_N_STEPS``
(default 300).

Requirements: optax >= 0.2.5 (contrib rules).
"""

import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
import optax.contrib as contrib

from searches_minimal._grad_setup import (
    build_map_objective,
    time_compile,
    write_grad_summary,
)

# --- config -----------------------------------------------------------------
N_STARTS = int(os.environ.get("MULTISTART_N_STARTS", 12))
N_STEPS = int(os.environ.get("MULTISTART_N_STEPS", 300))
START_LOW, START_HIGH = 0.15, 0.85  # unit-cube spread of the random starts
EINSTEIN_TRUTH = 1.6
EINSTEIN_TOL = 0.3  # a start "found the basin" if |r_E - truth| < this
MAX_CONSECUTIVE_NAN = 8  # apply_if_finite: per-start rejected-step budget

# name -> (builder, needs_value, is_schedule_free, config note for the summary)
RULES = {
    "adam": (
        lambda: optax.adam(1e-2),
        False,
        False,
        "reference rule, lr=1e-2 (the phase-3 winner)",
    ),
    "adopt": (
        lambda: contrib.adopt(1e-2),
        False,
        False,
        "fixed-lr control, lr=1e-2 (Adam variant, any-beta2 convergence)",
    ),
    "ademamix": (
        lambda: contrib.ademamix(1e-2),
        False,
        False,
        "fixed-lr control, lr=1e-2 (dual-EMA momentum)",
    ),
    "prodigy": (
        lambda: contrib.prodigy(),
        False,
        False,
        "lr-free (D-Adaptation successor; default lr=1.0 multiplies adapted d)",
    ),
    "dadapt_adamw": (
        lambda: contrib.dadapt_adamw(),
        False,
        False,
        "lr-free (D-Adaptation AdamW)",
    ),
    "dog": (
        lambda: contrib.dog(),
        False,
        False,
        "lr-free (Distance over Gradients)",
    ),
    "dowg": (
        lambda: contrib.dowg(),
        False,
        False,
        "lr-free (Distance over weighted Gradients)",
    ),
    "mechanic": (
        lambda: contrib.mechanize(optax.adam(1.0)),
        False,
        False,
        "lr-free (Mechanic scale-learner wrapping Adam, base lr=1.0)",
    ),
    "momo_adam": (
        lambda: contrib.momo_adam(),
        True,  # Polyak-type step needs the loss value each update
        False,
        "adaptive-lr (MoMo-Adam Polyak step, default lr cap 1e-2)",
    ),
    "schedule_free_adamw": (
        lambda: contrib.schedule_free_adamw(),
        False,
        True,  # score at schedule_free_eval_params, not the train iterates
        "schedule-free AdamW (NOT lr-free; default peak lr 0.0025)",
    ),
}


def run_rule(name, obj, batched_vag, batched_log_l, params0, shared_compile_s):
    build, needs_value, is_schedule_free, note = RULES[name]

    # apply_if_finite rejects NaN/inf updates per start (each start has its own
    # notfinite_count under the vmapped state). It forwards extra update kwargs
    # (optax >= 0.2.5), so MoMo's `value=` passes through.
    opt = optax.apply_if_finite(build(), max_consecutive_errors=MAX_CONSECUTIVE_NAN)

    params = params0
    # Per-start independent optimizer state — see module docstring point 1.
    opt_states = jax.vmap(opt.init)(params)

    if needs_value:

        @jax.jit
        def step_update(grads, states, params, losses):
            return jax.vmap(
                lambda g, s, p, v: opt.update(g, s, p, value=v)
            )(grads, states, params, losses)

    else:

        @jax.jit
        def step_update(grads, states, params, losses):
            del losses
            return jax.vmap(opt.update)(grads, states, params)

    best_history: list[float] = []
    global_best_loss = np.inf
    global_best_params = params[0]

    print(f"\n[{name}] {N_STARTS}-start x {N_STEPS} steps  ({note})")
    t_start = time.time()
    for i in range(N_STEPS):
        losses, grads = batched_vag(params)  # (S,), (S, ndim)
        losses_np = np.asarray(losses)
        losses_np = np.where(np.isfinite(losses_np), losses_np, np.inf)
        j = int(np.argmin(losses_np))
        if losses_np[j] < global_best_loss:
            global_best_loss = float(losses_np[j])
            global_best_params = params[j]
        best_history.append(-global_best_loss)
        updates, opt_states = step_update(grads, opt_states, params, losses)
        params = optax.apply_updates(params, updates)
        if i % 50 == 0:
            print(f"  step {i:4d}: best log_posterior = {-global_best_loss:.2f}")
    loop_s = time.time() - t_start

    # Schedule-free's convergence guarantee is on the averaged (eval) iterates,
    # not the train sequence the loop steps through — score the final points
    # there. (The in-loop best tracking above is on train iterates; recorded.)
    if is_schedule_free:
        params = jax.vmap(
            lambda s, p: contrib.schedule_free_eval_params(s.inner_state, p)
        )(opt_states, params)
        final_losses = np.asarray(batched_vag(params)[0])
        finite = np.where(np.isfinite(final_losses), final_losses, np.inf)
        j = int(np.argmin(finite))
        if finite[j] < global_best_loss:
            global_best_loss = float(finite[j])
            global_best_params = params[j]

    # --- per-start outcome: how many reached the correct basin ----------------
    final_log_l = np.asarray(batched_log_l(params))
    final_r_e = np.array(
        [
            obj.model.instance_from_vector(
                vector=list(np.asarray(p))
            ).galaxies.lens.mass.einstein_radius
            for p in params
        ]
    )
    in_basin = np.abs(final_r_e - EINSTEIN_TRUTH) < EINSTEIN_TOL
    n_in_basin = int(np.sum(in_basin))

    for k in range(len(params)):
        tag = "  <-- correct basin" if in_basin[k] else ""
        print(
            f"  start {k:2d}: log L = {final_log_l[k]:12.2f}   "
            f"r_E = {final_r_e[k]:.3f}{tag}"
        )
    print(f"  {n_in_basin}/{N_STARTS} starts in the correct basin")

    write_grad_summary(
        name=f"lr_free_{name}",
        title=f"multi-start {name} (MAP)",
        obj=obj,
        best_params=global_best_params,
        log_posterior_history=best_history,
        wall_s=shared_compile_s + loop_s,
        compile_s=shared_compile_s,
        warm_ms_per_eval=loop_s / N_STEPS / N_STARTS * 1e3,
        n_evals=N_STARTS * N_STEPS,
        n_iters=N_STEPS,
        converged=(n_in_basin > 0),
        config_line=(
            f"n_starts={N_STARTS}, steps={N_STEPS}, {note}, "
            f"start_spread=U({START_LOW},{START_HIGH}), "
            f"{n_in_basin}/{N_STARTS} reached correct basin"
        ),
    )

    return {
        "name": name,
        "note": note,
        "loop_s": loop_s,
        "best_log_posterior": -global_best_loss,
        "best_r_e": float(
            obj.model.instance_from_vector(
                vector=list(np.asarray(global_best_params))
            ).galaxies.lens.mass.einstein_radius
        ),
        "n_in_basin": n_in_basin,
    }


def main() -> None:
    requested = sys.argv[1:] or list(RULES)
    unknown = [r for r in requested if r not in RULES]
    if unknown:
        raise SystemExit(f"Unknown rule(s) {unknown}; choose from {list(RULES)}")

    obj = build_map_objective()
    print(f"Model free parameters: {obj.ndim}")

    compile_s = time_compile(obj)
    print(f"JIT compile (value_and_grad): {compile_s:.1f} s")

    # One batched value_and_grad graph, shared by every rule (the heavy compile
    # is optimizer-independent); a batched log-L for scoring.
    batched_vag = jax.jit(jax.vmap(jax.value_and_grad(obj.neg_log_posterior_raw)))
    batched_log_l = jax.jit(jax.vmap(obj.log_likelihood))

    # --- identical seeded broad, finite-gradient starts for every rule --------
    rng = np.random.default_rng(0)
    starts = []
    tries = 0
    while len(starts) < N_STARTS and tries < N_STARTS * 20:
        tries += 1
        u = rng.uniform(START_LOW, START_HIGH, size=obj.ndim)
        x = jnp.asarray(obj.model.vector_from_unit_vector(list(u)))
        loss, grad = obj.value_and_grad(x)
        if np.isfinite(float(loss)) and np.all(np.isfinite(np.asarray(grad))):
            starts.append(x)
    params0 = jnp.stack(starts)  # (N_STARTS, ndim)
    print(f"Collected {len(starts)} finite-gradient starts (from {tries} draws)")

    t0 = time.time()
    l, g = batched_vag(params0)
    jax.block_until_ready(l)
    jax.block_until_ready(g)
    batched_compile_s = time.time() - t0
    print(f"Batched (vmap x{N_STARTS}) compile: {batched_compile_s:.1f} s")
    shared_compile_s = compile_s + batched_compile_s

    results = [
        run_rule(name, obj, batched_vag, batched_log_l, params0, shared_compile_s)
        for name in requested
    ]

    lines = [
        "Learning-rate-free multi-start comparison (MGE cell)",
        f"n_starts={N_STARTS} n_steps={N_STEPS} "
        f"start_spread=U({START_LOW},{START_HIGH}) seed=0 "
        f"(shared compile {shared_compile_s:.1f}s)",
        "",
        f"| {'rule':>19} | {'loop s':>7} | {'best log post':>13} "
        f"| {'best r_E':>8} | {'basin':>7} |",
        f"|{'-' * 21}|{'-' * 9}|{'-' * 15}|{'-' * 10}|{'-' * 9}|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']:>19} | {r['loop_s']:7.1f} | {r['best_log_posterior']:13.1f} "
            f"| {r['best_r_e']:8.4f} | {r['n_in_basin']:3d}/{N_STARTS:<3d} |"
        )
    table = "\n".join(lines)
    print(f"\n{table}")

    out_path = os.path.join("searches_minimal", "output", "lr_free_comparison.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(table + "\n")
    print(f"\nComparison table written to {out_path}")


if __name__ == "__main__":
    main()
