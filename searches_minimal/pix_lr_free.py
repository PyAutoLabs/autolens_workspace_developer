"""
Learning-rate-free multi-start rules on the PIXELIZED objective (#101 phase 2b)
-------------------------------------------------------------------------------

Runs the optax.contrib rule registry from ``lr_free_multistart.py`` against the
#100 pixelized-source objective (``pix_multi_start.build_model`` — broad-prior
Isothermal+shear mass, fixed MGE light geometry, kernel-CDF mesh, free reg).

The af.MultiStart* searches cannot carry these rules: the library inits its
optax state on the stacked ``(n_starts, ndim)`` params, which couples the
lr-free rules' *global scalar* estimates (Prodigy's d, DoG's max_dist,
Mechanic's scale) across starts. This script therefore mirrors the library's
``_fit`` loop — the same ``Fitness`` objective (``-2 * log_posterior``, NaN ->
-inf resample) and the same ``jax.lax.map(batch_size=...)`` memory tiling
(PyAutoFit#1374; unbatched the pixelized jvp fusion OOMs an 80 GB A100) — but
with **per-start optimizer state via ``jax.vmap`` over init/update** and
``optax.apply_if_finite`` as the per-start NaN-step guard.

The success bar (see #100/#101): converged Nautilus reaches max log L
**+17419** at r_E = 1.31; multi-start Adam@1e-2 managed only -39888. A rule
matches the sampler if it reaches ~+17419; r_E is reported against the truth
(1.6) and the Nautilus mode (1.31) separately — the slack 0.3 basin tolerance
alone is known to over-flag.

Usage (from the workspace root; defaults: all rules, 16 starts, batch 4)::

    python -m searches_minimal.pix_lr_free [rule ...]

Env: ``PIX_N_STARTS`` (16), ``PIX_N_STEPS`` (300), ``PIX_BATCH`` (4).
"""

import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
import optax.contrib as contrib

# x64 is also set on the pix_multi_start import below; set it first regardless.
jax.config.update("jax_enable_x64", True)

from autofit.non_linear.fitness import Fitness  # noqa: E402

from searches_minimal.pix_multi_start import (  # noqa: E402
    build_model,
    build_analysis,
    TRUTH_EINSTEIN_RADIUS,
    BASIN_TOL,
    MESH_SHAPE,
    OS_PIX,
)
from searches_minimal._setup import build_dataset  # noqa: E402
from searches_minimal.lr_free_multistart import (  # noqa: E402
    RULES,
    MAX_CONSECUTIVE_NAN,
    START_LOW,
    START_HIGH,
)

N_STARTS = int(os.environ.get("PIX_N_STARTS", 16))
N_STEPS = int(os.environ.get("PIX_N_STEPS", 300))
BATCH = int(os.environ.get("PIX_BATCH", 4))
# Start-band override: the broad default reproduces the #100 setup; the FD
# probe certified finite gradients only in the narrow [0.4, 0.6] band, so
# narrow-vs-broad separates "NaN cliffs everywhere" from "broad starts die".
PIX_START_LOW = float(os.environ.get("PIX_START_LOW", START_LOW))
PIX_START_HIGH = float(os.environ.get("PIX_START_HIGH", START_HIGH))
# Resurrection mode (#101): the pixelized likelihood has hard non-finite walls
# and every trajectory dies on one within ~50 steps (jobs 330588/89/92/93).
# With PIX_RESURRECT=1, a start whose objective goes non-finite is redrawn
# fresh (params + optimizer state) instead of latching dead — the probe for
# whether multi-start gradient MAP is viable on this landscape at all.
PIX_RESURRECT = os.environ.get("PIX_RESURRECT") == "1"
NAUTILUS_BAR_LOG_L = 17419.0  # converged Nautilus max log L (job 330513)
NAUTILUS_MODE_R_E = 1.31


def main() -> None:
    requested = sys.argv[1:] or list(RULES)
    unknown = [r for r in requested if r not in RULES]
    if unknown:
        raise SystemExit(f"Unknown rule(s) {unknown}; choose from {list(RULES)}")

    print(f"JAX backend: {jax.default_backend()}  x64={jax.config.jax_enable_x64}")
    print(
        f"Rules: {requested}  n_starts={N_STARTS} n_steps={N_STEPS} batch={BATCH}"
        f"  start_band=U({PIX_START_LOW},{PIX_START_HIGH})"
    )
    print(f"mesh=KernelAdaptDensity{MESH_SHAPE} os_pix={OS_PIX}  (no sparse operator)")

    dataset = build_dataset()
    analysis = build_analysis(dataset)
    model = build_model()
    print(f"Free (non-linear) parameters: {model.prior_count}")

    # The library's exact objective: Fitness.call = -2 * log_posterior with
    # invalid/NaN models resampled to -inf (never selected as best).
    fitness = Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=False,
        resample_figure_of_merit=-np.inf,
        convert_to_chi_squared=True,
    )
    _value_and_grad = jax.value_and_grad(fitness.call)
    _single = jax.jit(_value_and_grad)

    # The library's memory tiling (PyAutoFit#1374): lax.map vmaps within each
    # batch and scans across batches — numerically identical to one big vmap,
    # but the pixelized jvp fusion stays bounded (~58 GiB unbatched at 16
    # starts in f64 vs an 80 GB card).
    @jax.jit
    def batched_value_and_grad(params):
        return jax.lax.map(_value_and_grad, params, batch_size=BATCH)

    # --- identical seeded broad, finite-gradient starts for every rule --------
    rng = np.random.default_rng(0)
    starts = []
    tries = 0
    t0 = time.time()
    while len(starts) < N_STARTS and tries < N_STARTS * 30:
        tries += 1
        u = rng.uniform(PIX_START_LOW, PIX_START_HIGH, size=model.prior_count)
        x = jnp.asarray(model.vector_from_unit_vector(unit_vector=list(u), xp=jnp))
        loss, grad = _single(x)
        if np.isfinite(float(loss)) and np.all(np.isfinite(np.asarray(grad))):
            starts.append(x)
    params0 = jnp.stack(starts)
    print(
        f"Collected {len(starts)}/{N_STARTS} finite-gradient starts "
        f"(from {tries} draws; single-graph compile folded in: {time.time() - t0:.1f} s)"
    )

    t0 = time.time()
    l, g = batched_value_and_grad(params0)
    jax.block_until_ready(l)
    batched_compile_s = time.time() - t0
    print(f"Batched (lax.map batch={BATCH}) compile+first-eval: {batched_compile_s:.1f} s")

    rows = []
    for name in requested:
        build, needs_value, is_schedule_free, note = RULES[name]
        opt = optax.apply_if_finite(
            build(), max_consecutive_errors=MAX_CONSECUTIVE_NAN
        )
        params = params0
        opt_states = jax.vmap(opt.init)(params)

        if needs_value:

            @jax.jit
            def step_update(grads, states, params, losses, _opt=opt):
                return jax.vmap(
                    lambda g, s, p, v: _opt.update(g, s, p, value=v)
                )(grads, states, params, losses)

        else:

            @jax.jit
            def step_update(grads, states, params, losses, _opt=opt):
                del losses
                return jax.vmap(_opt.update)(grads, states, params)

        def reinit_starts(params, opt_states, dead_idx, rng):
            """Redraw dead starts in the unit band and reinit their optimizer
            state, leaving alive starts untouched (leaves are (S, ...))."""
            params_np = np.array(params)  # np.asarray of a jax array is read-only
            for k in dead_idx:
                u = rng.uniform(PIX_START_LOW, PIX_START_HIGH, size=model.prior_count)
                params_np[k] = np.asarray(
                    model.vector_from_unit_vector(unit_vector=list(u), xp=jnp)
                )
            params = jnp.asarray(params_np)
            fresh_states = jax.vmap(opt.init)(params)
            mask = np.zeros(N_STARTS, dtype=bool)
            mask[dead_idx] = True
            mask_j = jnp.asarray(mask)

            def merge(old, fresh):
                m = mask_j.reshape((N_STARTS,) + (1,) * (fresh.ndim - 1))
                return jnp.where(m, fresh, old)

            return params, jax.tree.map(merge, opt_states, fresh_states)

        best_loss = np.inf
        best_params = params[0]
        n_resurrections = 0
        # Per-start death diagnostics: the last step, params and loss at which
        # each start's objective was still finite (#101 NaN-mortality probe).
        death_step = np.full(N_STARTS, -1, dtype=int)
        last_finite_params = np.asarray(params0).copy()
        last_finite_loss = np.full(N_STARTS, np.inf)
        resurrect_rng = np.random.default_rng(1)
        print(
            f"\n[{name}] {N_STARTS}-start x {N_STEPS} steps  ({note})"
            f"{'  [RESURRECT]' if PIX_RESURRECT else ''}"
        )
        t_start = time.time()
        for i in range(N_STEPS):
            losses, grads = batched_value_and_grad(params)
            losses_np = np.asarray(losses)
            alive = np.isfinite(losses_np)
            death_step[alive] = i
            last_finite_params[alive] = np.asarray(params)[alive]
            last_finite_loss[alive] = losses_np[alive]
            if PIX_RESURRECT and not alive.all():
                dead_idx = np.flatnonzero(~alive)
                n_resurrections += len(dead_idx)
                params, opt_states = reinit_starts(
                    params, opt_states, dead_idx, resurrect_rng
                )
            losses_np = np.where(alive, losses_np, np.inf)
            j = int(np.argmin(losses_np))
            if losses_np[j] < best_loss:
                best_loss = float(losses_np[j])
                best_params = params[j]
            updates, opt_states = step_update(grads, opt_states, params, losses)
            params = optax.apply_updates(params, updates)
            if i % 25 == 0:
                n_finite_loss = int(np.sum(np.isfinite(np.asarray(losses))))
                n_finite_grad = int(
                    np.sum(np.all(np.isfinite(np.asarray(grads)), axis=1))
                )
                print(
                    f"  step {i:4d}: best log_posterior = {-0.5 * best_loss:.2f}"
                    f"   finite loss {n_finite_loss}/{N_STARTS}"
                    f" grad {n_finite_grad}/{N_STARTS}",
                    flush=True,
                )
        loop_s = time.time() - t_start

        if is_schedule_free:
            params = jax.vmap(
                lambda s, p: contrib.schedule_free_eval_params(s.inner_state, p)
            )(opt_states, params)
            final = np.asarray(batched_value_and_grad(params)[0])
            finite = np.where(np.isfinite(final), final, np.inf)
            j = int(np.argmin(finite))
            if finite[j] < best_loss:
                best_loss = float(finite[j])
                best_params = params[j]

        # Death report: where was each start when it last had a finite
        # objective, and which parameters had run to extremes? Reports r_E and
        # the reg coefficient (the non-PD-Cholesky suspect) if it is free.
        reg_idx = [
            k
            for k, (path, _) in enumerate(model.path_priors_tuples)
            if "coefficient" in str(path)
        ]
        print("  death report (last finite step; params at that point):")
        for k in range(N_STARTS):
            pl = list(last_finite_params[k])
            r_e_d = model.instance_from_vector(
                vector=pl
            ).galaxies.lens.mass.einstein_radius
            reg_s = f" reg={pl[reg_idx[0]]:.3e}" if reg_idx else ""
            print(
                f"    start {k:2d}: died after step {death_step[k]:3d}  "
                f"log_post={-0.5 * last_finite_loss[k]:12.2f}  "
                f"r_E={float(r_e_d):.4f}{reg_s}"
            )

        # Per-start scoring. log_posterior = -0.5 * fom; log L = log_post -
        # log_prior (prior evaluated numpy-side — no extra JAX compile).
        final_foms = np.asarray(batched_value_and_grad(params)[0])
        param_lists = [list(np.asarray(p)) for p in params]
        log_priors = np.asarray(model.log_prior_list_from(parameter_lists=param_lists))
        log_posts = -0.5 * np.where(np.isfinite(final_foms), final_foms, np.inf)
        log_ls = log_posts - log_priors
        r_es = np.array(
            [
                model.instance_from_vector(vector=pl).galaxies.lens.mass.einstein_radius
                for pl in param_lists
            ]
        )
        near_truth = np.abs(r_es - TRUTH_EINSTEIN_RADIUS) < BASIN_TOL
        near_naut = np.abs(r_es - NAUTILUS_MODE_R_E) < BASIN_TOL

        for k in range(len(params)):
            tags = []
            if near_truth[k]:
                tags.append("~truth r_E")
            if near_naut[k]:
                tags.append("~Nautilus r_E")
            print(
                f"  start {k:2d}: log L = {log_ls[k]:12.2f}   r_E = {r_es[k]:.4f}"
                f"   {' '.join(tags)}"
            )

        best_log_post = -0.5 * best_loss
        best_pl = list(np.asarray(best_params))
        best_r_e = float(
            model.instance_from_vector(vector=best_pl).galaxies.lens.mass.einstein_radius
        )
        best_log_l = best_log_post - float(
            np.sum(model.log_prior_list_from(parameter_lists=[best_pl]))
        )
        meets_bar = best_log_l >= NAUTILUS_BAR_LOG_L - 100.0
        print(
            f"  BEST: log L = {best_log_l:.2f} (bar {NAUTILUS_BAR_LOG_L:+.0f} -> "
            f"{'MATCHES SAMPLER' if meets_bar else 'below bar'})  r_E = {best_r_e:.4f}"
        )

        rows.append(
            f"| {name:>19} | {loop_s:8.1f} | {best_log_l:12.1f} | {best_r_e:8.4f} "
            f"| {int(np.sum(near_truth)):2d}/{N_STARTS:<3d} | {int(np.sum(near_naut)):2d}/{N_STARTS:<3d} "
            f"| {'yes' if meets_bar else 'no':>3} |"
        )

    table = "\n".join(
        [
            "Pixelized lr-free multi-start comparison (#101 phase 2b)",
            f"n_starts={N_STARTS} n_steps={N_STEPS} batch={BATCH} seed=0 "
            f"(bar = Nautilus max log L {NAUTILUS_BAR_LOG_L:+.0f} @ r_E {NAUTILUS_MODE_R_E})",
            "",
            f"| {'rule':>19} | {'loop s':>8} | {'best log L':>12} | {'best r_E':>8} "
            f"| ~truth | ~naut | bar |",
            f"|{'-' * 21}|{'-' * 10}|{'-' * 14}|{'-' * 10}|{'-' * 8}|{'-' * 7}|{'-' * 5}|",
        ]
        + rows
    )
    print(f"\n{table}")
    out_path = os.path.join("searches_minimal", "output", "pix_lr_free_comparison.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(table + "\n")
    print(f"\nComparison table written to {out_path}")


if __name__ == "__main__":
    main()
