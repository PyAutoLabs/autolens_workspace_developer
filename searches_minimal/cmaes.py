"""
CMA-ES (evosax) — interacting-population optimizer on the HST MGE lens
---------------------------------------------------------------------

Next-wave follow-up to the gradient-optimizer benchmark (#95). That run showed
*independent* multi-start Adam recovers the truth where single starts fail.
CMA-ES asks the next question: does an **interacting** population — one that
adapts a full covariance from the best members each generation — match (or beat)
independent multi-start, and **does the gradient even help** given a smart
gradient-free ES?

CMA-ES (Hansen) via `evosax` (JAX-native evolutionary strategies, installed).
The search runs in **unit-cube** space (respects every prior, same spread as the
multi-start scripts); each candidate is mapped to physical parameters and scored
by the **neg-log-posterior** (evosax minimises). Gradient-free, so the per-eval
cost is a forward likelihood only (cheaper than the fwd+grad optimizers).

Run from the workspace root:  python -m searches_minimal.cmaes

Requirements: evosax (JAX-native).
"""

import time

import numpy as np
import jax
import jax.numpy as jnp
from evosax.algorithms import CMA_ES

from searches_minimal._grad_setup import build_map_objective, write_grad_summary

# --- config -----------------------------------------------------------------
POPULATION_SIZE = 16
N_GENERATIONS = 200
STD_INIT = 0.30  # initial step size in unit-cube space (broad prior coverage)
EINSTEIN_TRUTH = 1.6
EINSTEIN_TOL = 0.3

obj = build_map_objective()
print(f"Model free parameters: {obj.ndim}")

# Batched neg-log-posterior over a physical-parameter population (fwd-only).
batched_neg = jax.jit(jax.vmap(obj.neg_log_posterior_raw))

# Compile timing on a dummy physical batch.
_dummy = jnp.stack([obj.x0] * POPULATION_SIZE)
t0 = time.time()
_ = batched_neg(_dummy)
jax.block_until_ready(_)
compile_s = time.time() - t0
print(f"JIT compile (batched neg-log-posterior): {compile_s:.1f} s")

t0 = time.time()
for _ in range(2):
    jax.block_until_ready(batched_neg(_dummy))
warm_ms = (time.time() - t0) / 2 / POPULATION_SIZE * 1e3
print(f"Warm fwd eval: {warm_ms:.1f} ms/single-eval")


def _unit_to_physical(u_batch):
    """Map a (P, ndim) unit-cube batch to physical parameters (numpy)."""
    u = np.clip(np.asarray(u_batch), 1e-4, 1.0 - 1e-4)
    return np.stack([obj.model.vector_from_unit_vector(list(ui)) for ui in u])


def fitness_fn(u_batch):
    """neg-log-posterior for a unit-cube population; NaN → large penalty."""
    phys = _unit_to_physical(u_batch)
    neg = np.asarray(batched_neg(jnp.asarray(phys)))
    return np.where(np.isfinite(neg), neg, 1e10)


solver = CMA_ES(population_size=POPULATION_SIZE, solution=jnp.zeros(obj.ndim))
import dataclasses

params = dataclasses.replace(solver.default_params, std_init=STD_INIT)
key = jax.random.key(0)
state = solver.init(key, mean=jnp.full(obj.ndim, 0.5), params=params)

best_history: list[float] = []
global_best_neg = np.inf
global_best_u = np.full(obj.ndim, 0.5)

print(
    f"\nRunning CMA-ES: pop={POPULATION_SIZE}, {N_GENERATIONS} gens, std_init={STD_INIT}..."
)
t_start = time.time()
for gen in range(N_GENERATIONS):
    key, k_ask, k_tell = jax.random.split(key, 3)
    population, state = solver.ask(k_ask, state, params)  # (P, ndim) unit-cube
    fitness = fitness_fn(population)
    state, _metrics = solver.tell(
        k_tell, population, jnp.asarray(fitness), state, params
    )
    j = int(np.argmin(fitness))
    if fitness[j] < global_best_neg:
        global_best_neg = float(fitness[j])
        global_best_u = np.asarray(population[j])
    best_history.append(-global_best_neg)  # best log posterior so far
    if gen % 25 == 0:
        print(f"  gen {gen:4d}: best log_posterior = {-global_best_neg:.2f}")
loop_s = time.time() - t_start

# Final population outcome: how many members reached the correct basin.
final_phys = _unit_to_physical(solver.ask(key, state, params)[0])
final_r_e = np.array(
    [
        obj.model.instance_from_vector(
            vector=list(p)
        ).galaxies.lens.mass.einstein_radius
        for p in final_phys
    ]
)
n_in_basin = int(np.sum(np.abs(final_r_e - EINSTEIN_TRUTH) < EINSTEIN_TOL))

best_phys = np.asarray(
    obj.model.vector_from_unit_vector(list(np.clip(global_best_u, 1e-4, 1 - 1e-4)))
)
best_r_e = float(
    obj.model.instance_from_vector(
        vector=list(best_phys)
    ).galaxies.lens.mass.einstein_radius
)
print(
    f"\nBest r_E = {best_r_e:.3f} (truth {EINSTEIN_TRUTH}); "
    f"{n_in_basin}/{POPULATION_SIZE} of final population in basin"
)

write_grad_summary(
    name="cmaes",
    title="CMA-ES (evosax)",
    obj=obj,
    best_params=jnp.asarray(best_phys),
    log_posterior_history=best_history,
    wall_s=compile_s + loop_s,
    compile_s=compile_s,
    warm_ms_per_eval=warm_ms,
    n_evals=POPULATION_SIZE * N_GENERATIONS,
    n_iters=N_GENERATIONS,
    converged=(abs(best_r_e - EINSTEIN_TRUTH) < EINSTEIN_TOL),
    config_line=(
        f"pop={POPULATION_SIZE}, gens={N_GENERATIONS}, std_init={STD_INIT}, "
        f"unit-cube search, gradient-free; best r_E={best_r_e:.3f}, "
        f"{n_in_basin}/{POPULATION_SIZE} final pop in basin"
    ),
)
