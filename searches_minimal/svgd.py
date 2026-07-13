"""
SVGD as a mode-finder (blackjax) — HST MGE lens
-----------------------------------------------

Stein Variational Gradient Descent (Liu & Wang 2016) via `blackjax.svgd`
(installed, JAX-native). N particles do gradient ascent on the log-posterior
**plus a kernel repulsion term** that keeps them apart. Where CMA-ES has one
adapting population that *collapses* onto a single mode (and here picked the
wrong one, r_E → 8), SVGD's explicit repulsion is designed to **preserve
diversity** — the property that makes independent multi-start robust. So this
tests the sharper form of the question: does a *diversity-preserving* interacting
population beat independent multi-start?

Used here as an OPTIMIZER, not for the posterior: we take the **best particle**
(highest log-posterior) as the point estimate and report how many particles
reach the true basin.

Run from the workspace root:  python -m searches_minimal.svgd

Requirements: blackjax, optax (JAX).
"""

import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
import blackjax
from blackjax.vi.svgd import rbf_kernel, update_median_heuristic

from searches_minimal._grad_setup import build_map_objective, write_grad_summary

# --- config -----------------------------------------------------------------
N_PARTICLES = 16
N_STEPS = 300
LEARNING_RATE = 1e-2
MAX_GRAD_NORM = 1.0
START_LOW, START_HIGH = 0.15, 0.85
EINSTEIN_TRUTH = 1.6
EINSTEIN_TOL = 0.3

obj = build_map_objective()
print(f"Model free parameters: {obj.ndim}")

# log-density = log posterior (blackjax SVGD ascends this + repulsion).
def _grad_logdensity(particle):
    return jax.grad(lambda p: -obj.neg_log_posterior_raw(p))(particle)


# Batched log-posterior for scoring particles (fwd-only).
batched_logpost = jax.jit(jax.vmap(lambda p: -obj.neg_log_posterior_raw(p)))

# Broad, finite-gradient initial particles (same spread as multi-start).
rng = np.random.default_rng(0)
starts = []
tries = 0
while len(starts) < N_PARTICLES and tries < N_PARTICLES * 20:
    tries += 1
    u = rng.uniform(START_LOW, START_HIGH, size=obj.ndim)
    x = jnp.asarray(obj.model.vector_from_unit_vector(list(u)))
    loss, grad = obj.value_and_grad(x)
    if np.isfinite(float(loss)) and np.all(np.isfinite(np.asarray(grad))):
        starts.append(x)
initial_particles = jnp.stack(starts)
print(f"Collected {len(starts)} finite-gradient particles (from {tries} draws)")

optimizer = optax.chain(
    optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LEARNING_RATE)
)
svgd = blackjax.svgd(_grad_logdensity, optimizer, rbf_kernel, update_median_heuristic)
state = svgd.init(initial_particles, {"length_scale": 1.0})

# Compile timing on the first step.
print("\nRunning SVGD (compiling on step 0)...")
t0 = time.time()
state = svgd.step(state)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), state.particles)
compile_s = time.time() - t0
print(f"JIT compile (svgd.step): {compile_s:.1f} s")

best_history: list[float] = []
global_best_lp = -np.inf
global_best_p = np.asarray(initial_particles[0])

t_start = time.time()
for i in range(1, N_STEPS):
    state = svgd.step(state)
    lps = np.asarray(batched_logpost(state.particles))
    lps = np.where(np.isfinite(lps), lps, -np.inf)
    j = int(np.argmax(lps))
    if lps[j] > global_best_lp:
        global_best_lp = float(lps[j])
        global_best_p = np.asarray(state.particles[j])
    best_history.append(global_best_lp)
    if i % 25 == 0:
        print(f"  step {i:4d}: best log_posterior = {global_best_lp:.2f}")
loop_s = time.time() - t_start

# Final particle outcome.
final_r_e = np.array(
    [
        obj.model.instance_from_vector(vector=list(np.asarray(p))).galaxies.lens.mass.einstein_radius
        for p in state.particles
    ]
)
n_in_basin = int(np.sum(np.abs(final_r_e - EINSTEIN_TRUTH) < EINSTEIN_TOL))
best_r_e = float(obj.model.instance_from_vector(vector=list(global_best_p)).galaxies.lens.mass.einstein_radius)
print(f"\nBest r_E = {best_r_e:.3f} (truth {EINSTEIN_TRUTH}); "
      f"{n_in_basin}/{N_PARTICLES} particles in basin")

write_grad_summary(
    name="svgd",
    title="SVGD (blackjax)",
    obj=obj,
    best_params=jnp.asarray(global_best_p),
    log_posterior_history=best_history,
    wall_s=compile_s + loop_s,
    compile_s=compile_s,
    warm_ms_per_eval=(loop_s / max(N_STEPS - 1, 1) / N_PARTICLES * 1e3),
    n_evals=N_PARTICLES * N_STEPS,
    n_iters=N_STEPS,
    converged=(abs(best_r_e - EINSTEIN_TRUTH) < EINSTEIN_TOL),
    config_line=(
        f"n_particles={N_PARTICLES}, steps={N_STEPS}, lr={LEARNING_RATE}, "
        f"rbf+median-heuristic repulsion; best r_E={best_r_e:.3f}, "
        f"{n_in_basin}/{N_PARTICLES} particles in basin"
    ),
)
