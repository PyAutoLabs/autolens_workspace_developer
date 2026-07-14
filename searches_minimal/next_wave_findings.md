# Next-wave fast optimizers — does *interaction* beat independent multi-start?

Follow-up to `gradient_optimizer_findings.md` (#95). That run established the
baseline: **independent multi-start Adam** recovers the truth
(`einstein_radius = 1.600`, 2/12 starts in the basin) where every *single*
cold-start method fails. This wave asks the sharper question the win implies:

> The population principle is what buys robustness — but is it enough to have a
> *population*, or must the population **maintain diversity** across basins?

We test **interacting** population optimizers (which share information across the
population) against the **independent** multi-start baseline. All are fast
optimizers (MAP point estimates), JAX-native, open-source. Metric: fraction of
starts/particles that reach the true basin (r_E ≈ 1.6), plus end-to-end
wallclock and eval count.

## Results

| Method | Interaction | Diversity | Wall (s) | Compile (s) | Evals | Max log L | best r_E | in basin |
|--------|-------------|-----------|---------:|------------:|------:|----------:|---------:|:--------:|
| **multi-start Adam (12×)** *(baseline)* | none (independent) | by independence | ~1254 | ~561 | 3600 | **+31788** | **1.600** | **2/12 ✓** |
| CMA-ES (evosax) | full covariance adaptation | **collapses** | 232 | 20 | 3200 | −158018 | 7.999 | 0/16 ✗ |
| SVGD (blackjax) | kernel repulsion (gradient) | preserves | — | **CPU: prohibitive** / **A100: 44.5 s** | — | — | — | see A100 section |
| SV-CMA-ES (evosax) | Stein repulsion (gradient-free) | preserves | 232 | 28 | 7680 | −149670† | 2.605 | 0/8 (still improving) |

### A100 (RAL) results — GPU scaling

Run on an NVIDIA A100 80GB (RAL `gpu` partition, `euclid_jump`; overlay blackjax
1.3, base CUDA jax 0.4.38, **float32** — x64 off on that GPU, tolerated fine).

| Method | Device | Compile (s) | Evals | Max log L | best r_E | in basin |
|--------|--------|------------:|------:|----------:|---------:|:--------:|
| **multi-start Adam (128×)** | A100 | 84.7 | 38 400 | **+31787.8** | **1.600** | **23/128 (p_hit 0.18)** |
| **SVGD (16 particles)** | A100 | 90.1 | 4 800 | **+17999** | **1.595** | best particle = truth¶ |

¶ SVGD's *best* particle reaches the truth (r_E 1.595, log L +18k); 0/16 of the
*final* particles sit inside the tight ±0.3 basin because SVGD is a **posterior**
method — repulsion spreads the cloud rather than collapsing every particle to the
single MAP (hence its best log L +18k is shallower than the multi-start point
optimum +31.8k). As a *mode-finder* (best particle), it succeeds.

**Multi-start Adam scales exactly as GIGA-Lens predicts.** At 128 starts on the
A100 (compile 85 s), **23** land in the true basin — the per-start hit rate holds
at ~18% (2/12 on CPU → 23/128 on GPU), so P(≥1 hit) is effectively 1.0. The GPU
makes the wide parallel start-batch that guarantees robustness *cheap*.

**SVGD — the diversity-preserving gradient method — reaches the truth on the
A100** (best r_E 1.595), where it was prohibitive to even compile on CPU
(>26 min/10 GB; A100 compiles it in 90 s and runs 300 steps in 105 s). Two script
lessons: (1) the SVGD step **must be jitted** — an un-jitted Python loop retraces
the median-heuristic kernel update every step and never finishes; (2) the RAL GPU
runs **float32** (x64 off), tolerated here.

## Final verdict (CPU + A100)

Robust MAP-finding on this likelihood needs **diversity AND gradients**, and the
methods separate cleanly along both axes:

| method | diversity | gradient | reaches truth? |
|--------|:--:|:--:|:--:|
| single cold-start (Adam/L-BFGS/SVI) | ✗ | ✓ | ✗ |
| CMA-ES | ✗ (collapses) | ✗ | ✗ |
| SV-CMA-ES | ✓ (repulsion) | ✗ | ✗ (near, too slow) |
| **multi-start Adam** | ✓ (independence) | ✓ | ✓ **best (scales on GPU)** |
| **SVGD** | ✓ (repulsion) | ✓ | ✓ (mode-finder; posterior spread) |

**Practical recommendation:** **multi-start Adam** for a robust fast *point*
estimate — trivially parallel, scales to ~100% basin-hit on a GPU (GIGA-Lens).
**SVGD** when you also want a posterior — it reaches the basin and gives spread —
but jit the loop and expect a distribution, not the deepest MAP. Both need the
GPU to be practical at scale; SVGD needs it even to compile.

_r_E truth ≈ 1.6; a good fit has positive log L. Gradient-free methods (CMA-ES,
SV-CMA-ES) compile in ~20 s (forward likelihood only, ~68 ms/eval); the
gradient methods pay the ~280 s grad-graph compile._

**SVGD is compile-prohibitive on this CPU.** `blackjax.svgd` fuses one gradient
graph *per particle* into a single compile — ~16× the already-heavy MGE grad
graph — which was still compiling after 26 min at >10 GB RAM when stopped. SVGD
is a genuine GPU/HPC candidate (where the grad is faster and parallel), not a
laptop-CPU one. **SV-CMA-ES is the cheap stand-in for the same hypothesis:**
Stein repulsion to preserve diversity, but gradient-free, so it keeps CMA-ES's
20 s compile.

## The lesson so far: a population is not enough — it must stay diverse

**CMA-ES fails, and worse than a single start.** It drove its whole population
to `einstein_radius = 7.999` — the *opposite* prior wall from the single-start
optimizers (which hit ~5) — with **0/16** members in the true basin. CMA-ES
maintains **one** adapting Gaussian (mean + covariance); its covariance
adaptation **collapses the population onto a single mode** each generation, and
here it collapsed onto the wrong one. Its "population" exists to estimate a
descent direction, not to cover multiple basins — CMA-ES is fundamentally a
*unimodal* optimizer.

So the many-points robustness of multi-start Adam does **not** come from having a
population per se — it comes from **independence preserving basin diversity**
(each start descends its own basin; 2 of 12 happened to start in the right one).
An interacting population that *collapses* diversity (CMA-ES) is not just
unhelpful, it can be worse.

**The test of the hypothesis:** a method that *preserves* diversity while
interacting should retain multi-basin coverage where CMA-ES lost it. SVGD
(gradient + repulsion) is the natural test but is compile-prohibitive on this
CPU (above), so **SV-CMA-ES** (Stein repulsion, gradient-free) is the cheap
stand-in that isolates the *repulsion* variable against plain CMA-ES.

**SV-CMA-ES result — repulsion prevents the collapse.** With Stein repulsion,
the sub-populations did *not* collapse to the wall: best `einstein_radius`
reached **2.605** (vs plain CMA-ES's **7.999**), and the log-posterior was still
climbing steeply at gen 100 (−184k → −150k, not plateaued). So the repulsion
term demonstrably **restored the diversity** plain CMA-ES destroys — the search
stayed near the truth instead of collapsing to a wall. But it did **not** reach
the basin (0/8) in 120 generations: gradient-free, it explores diversely but
descends *slowly* into the narrow true basin.

† still-improving, not converged — the log-posterior was rising when the budget
ran out.

## Verdict — the many-points principle needs BOTH diversity and gradients

The wave answers the question cleanly. Robust MAP-finding on this likelihood
needs **two** ingredients, and the methods separate exactly along them:

| | maintains diversity? | uses gradient? | result |
|---|:--:|:--:|---|
| single cold-start (Adam/L-BFGS/SVI) | ✗ | ✓ | wrong basin |
| CMA-ES | ✗ (collapses) | ✗ | wrong basin (worst, r_E→8) |
| SV-CMA-ES | ✓ (repulsion) | ✗ | near truth (r_E 2.6), too slow to land |
| **multi-start Adam** | **✓ (independence)** | **✓** | **truth (r_E 1.600)** |
| SVGD | ✓ (repulsion) | ✓ | **predicted best — but GPU/HPC only on CPU compile** |

- **Diversity alone** (SV-CMA-ES) keeps you near the truth but can't descend the
  narrow basin fast — gradient-free exploration is slow.
- **Gradient alone** (single start) descends fast but into whatever basin the
  cold start fell in — usually the wrong one.
- **Both** (multi-start Adam) is the robust-and-fast winner; **SVGD** is the
  theoretically ideal unifier (repulsion-diversity + gradient) and the clear
  thing to run **on GPU**, where its 16-fused-gradient compile is affordable.

**Recommendation:** multi-start Adam for a robust CPU solve; **SVGD on GPU** as
the next step to try to beat it (diversity + gradient in one interacting
population). SV-CMA-ES with a longer budget / a final gradient-polish of its best
member is a cheap CPU compromise.

## GPU / HPC status (RESUME HERE — 2026-07-13 EOD)

The compute reality for the GPU candidates:

- **RAL A100** — the `autolens_profiling` submit scripts target it
  (`/mnt/ral/jnightin/…`, user `jnightin`), but it is **not reachable from the
  local machine** (no SSH host configured; the RAL flow runs *on* the RAL login
  node: `source activate.sh; sbatch hpc/batch_gpu/submit_…`). Submitting needs a
  human on RAL.
- **COSMA8** (Durham, `cosma8` in `~/.ssh/config`, A100 partition) — configured
  but **2FA-gated**; an agent can't authenticate.
- **Laptop GPU** (RTX 2060 6 GB, `~/venv/PyAutoGPU`, jax 0.6.2 cuda) — works for
  **optax** (multi-start Adam) but **NOT** SVGD/CMA-ES: PyAutoGPU has blackjax
  0.1.0b1 (too old for the 1.5 SVGD API) and no evosax/numpyro.

**In progress at EOD:** `gpu_multi_start_adam.py` running on the laptop GPU with
`N_STARTS=48` (GIGA-Lens scaling test — does the per-start ~17% hit rate push the
whole-run hit toward ~100%?). Result → `output/gpu_multi_start_adam_n48_summary.txt`.
**Read it first on resume.**

**Tomorrow:**
1. Read the 48-start GPU result; if it fit 6 GB, sweep N ∈ {24, 48, 96} for the
   p_hit scaling curve (or reduce N if it OOM'd).
2. **SVGD + SV-CMA-ES on RAL A100** — the diversity-preserving candidates that
   need the A100 (+ full deps: newer blackjax, evosax). Needs: confirm the RAL
   checkout layout (which repos under `/mnt/ral/jnightin/`), write RAL sbatch
   scripts for `searches_minimal/{svgd,cmaes,sv_cmaes}.py`, human runs them on
   the RAL login node.
3. Cheap multi-start ADABelief/Lion local-rule variants (CPU, optional).

## Candidates still to run

- **SV-CMA-ES** (running) — the cheap diversity-preservation test.
- **SVGD on GPU/HPC** — the gradient version, once off the laptop CPU.
- Cheap multi-start local-rule variants (ADABelief, Lion) — does the local rule
  matter within independent multi-start? (multi-start L-BFGS is expensive — the
  line search makes it ~hours — so it is capped/optional.)
- **jaxns** cameo — JAX-native nested sampling reference (a *sampler*, included
  for context: its many live points are the population principle done as
  inference).
