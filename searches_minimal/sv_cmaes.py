"""
SV-CMA-ES (evosax) — Stein-variational CMA-ES on the HST MGE lens
-----------------------------------------------------------------

Plain CMA-ES collapsed its single population onto one (wrong) mode (r_E → 8,
0/16 in the basin). SV-CMA-ES runs **several** CMA-ES sub-populations kept apart
by a **Stein repulsion** kernel — i.e. it adds back the diversity that plain
CMA-ES destroys, while staying **gradient-free** (so it keeps CMA-ES's cheap
~20 s compile, unlike SVGD whose 16 fused gradient graphs are compile-prohibitive
on CPU). This is the cheap, direct test of the wave's hypothesis: **does
repulsion preserve the multi-basin diversity that makes independent multi-start
robust?**

Used as an optimizer: take the best member across all sub-populations; report how
many sub-population means reach the true basin.

Run from the workspace root:  python -m searches_minimal.sv_cmaes

Requirements: evosax (JAX-native).
"""

import dataclasses
import time

import numpy as np
import jax
import jax.numpy as jnp
from evosax.algorithms import SV_CMA_ES

from searches_minimal._grad_setup import build_map_objective, write_grad_summary

# --- config -----------------------------------------------------------------
NUM_POPULATIONS = 8
POPULATION_SIZE = 8  # members per sub-population
N_GENERATIONS = 120
STD_INIT = 0.30
EINSTEIN_TRUTH = 1.6
EINSTEIN_TOL = 0.3

obj = build_map_objective()
print(f"Model free parameters: {obj.ndim}")

batched_neg = jax.jit(jax.vmap(obj.neg_log_posterior_raw))
_dummy = jnp.stack([obj.x0] * (NUM_POPULATIONS * POPULATION_SIZE))
t0 = time.time()
jax.block_until_ready(batched_neg(_dummy))
compile_s = time.time() - t0
print(f"JIT compile (batched neg-log-posterior): {compile_s:.1f} s")


def _unit_to_physical(u_flat):
    u = np.clip(np.asarray(u_flat), 1e-4, 1.0 - 1e-4)
    return np.stack([obj.model.vector_from_unit_vector(list(ui)) for ui in u])


def fitness_of(u_flat):
    phys = _unit_to_physical(u_flat)
    neg = np.asarray(batched_neg(jnp.asarray(phys)))
    return np.where(np.isfinite(neg), neg, 1e10)


solver = SV_CMA_ES(
    population_size=POPULATION_SIZE, num_populations=NUM_POPULATIONS, solution=jnp.zeros(obj.ndim)
)
params = dataclasses.replace(solver.default_params, std_init=STD_INIT)

# Seed each sub-population mean at a distinct broad unit-cube point (diverse
# starts); Stein repulsion then keeps them apart.
rng = np.random.default_rng(0)
means0 = jnp.asarray(rng.uniform(0.2, 0.8, size=(NUM_POPULATIONS, obj.ndim)))
key = jax.random.key(0)
state = solver.init(key, means0, params)

best_history: list[float] = []
global_best_neg = np.inf
global_best_u = np.full(obj.ndim, 0.5)

print(f"\nRunning SV-CMA-ES: {NUM_POPULATIONS} pops x {POPULATION_SIZE}, "
      f"{N_GENERATIONS} gens, std_init={STD_INIT}...")
t_start = time.time()
for gen in range(N_GENERATIONS):
    key, k_ask, k_tell = jax.random.split(key, 3)
    population, state = solver.ask(k_ask, state, params)  # (..., ndim)
    pop_shape = population.shape
    flat = np.asarray(population).reshape(-1, obj.ndim)
    fitness = fitness_of(flat)
    j = int(np.argmin(fitness))
    if fitness[j] < global_best_neg:
        global_best_neg = float(fitness[j])
        global_best_u = flat[j]
    best_history.append(-global_best_neg)
    fitness_shaped = jnp.asarray(fitness).reshape(pop_shape[:-1])
    state, _m = solver.tell(k_tell, population, fitness_shaped, state, params)
    if gen % 20 == 0:
        print(f"  gen {gen:4d}: best log_posterior = {-global_best_neg:.2f}")
loop_s = time.time() - t_start

# Per-sub-population outcome: how many population means reached the basin.
means_final = np.asarray(state.mean) if hasattr(state, "mean") else None
if means_final is not None and means_final.ndim == 2:
    means_phys = _unit_to_physical(means_final)
    means_r_e = np.array(
        [obj.model.instance_from_vector(vector=list(p)).galaxies.lens.mass.einstein_radius for p in means_phys]
    )
    n_in_basin = int(np.sum(np.abs(means_r_e - EINSTEIN_TRUTH) < EINSTEIN_TOL))
    denom = NUM_POPULATIONS
else:
    n_in_basin, denom = -1, NUM_POPULATIONS

best_phys = np.asarray(obj.model.vector_from_unit_vector(list(np.clip(global_best_u, 1e-4, 1 - 1e-4))))
best_r_e = float(obj.model.instance_from_vector(vector=list(best_phys)).galaxies.lens.mass.einstein_radius)
print(f"\nBest r_E = {best_r_e:.3f} (truth {EINSTEIN_TRUTH}); "
      f"{n_in_basin}/{denom} sub-population means in basin")

write_grad_summary(
    name="sv_cmaes",
    title="SV-CMA-ES (evosax)",
    obj=obj,
    best_params=jnp.asarray(best_phys),
    log_posterior_history=best_history,
    wall_s=compile_s + loop_s,
    compile_s=compile_s,
    warm_ms_per_eval=(loop_s / max(N_GENERATIONS, 1) / (NUM_POPULATIONS * POPULATION_SIZE) * 1e3),
    n_evals=NUM_POPULATIONS * POPULATION_SIZE * N_GENERATIONS,
    n_iters=N_GENERATIONS,
    converged=(abs(best_r_e - EINSTEIN_TRUTH) < EINSTEIN_TOL),
    config_line=(
        f"{NUM_POPULATIONS} pops x {POPULATION_SIZE}, gens={N_GENERATIONS}, "
        f"std_init={STD_INIT}, Stein repulsion, gradient-free; best r_E={best_r_e:.3f}, "
        f"{n_in_basin}/{denom} means in basin"
    ),
)
