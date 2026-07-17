# Learning-rate-free multi-start optimizers (#101) — findings

Follow-up to `phase3_map_optimizer_findings.md`. Phase 3 settled that wide
multi-start Adam is the best fast, reliable MAP optimizer on the MGE cell and
that the local update rule barely matters — but every rule benchmarked carried
a hand-set learning rate, and on the pixelized cell (#100) multi-start Adam
went 0/16 with `lr=1e-2` mis-scaling a prime suspect. This experiment asks:
**do learning-rate-free optimizers (which estimate their own step scale) match
the tuned rules — and can they rescue the pixelized case?**

## Phase 1 — MGE cell (laptop RTX 2060, 12 starts × 300 steps, seed 0)

All ten rules share one batched `value_and_grad` compile and identical seeded
broad starts (`U(0.15, 0.85)` in the unit cube), via
`searches_minimal/lr_free_multistart.py`. Adam's phase-3 optimum on these
starts is +31787.84 (log posterior); `r_E` truth ≈ 1.6.

| rule | class | best log post | best r_E | basin hits |
|---|---|---:|---:|:--:|
| **adam** (lr 1e-2) | reference | **+31787.8** | 1.5997 | 3/12 |
| **prodigy** | **lr-free** | **+31787.8** | 1.5997 | 3/12 |
| ademamix (lr 1e-2) | fixed-lr control | +31786.0 | 1.5995 | 3/12 |
| adopt (lr 1e-2) | fixed-lr control | +31762.0 | 1.5993 | **5/12** |
| dadapt_adamw | lr-free | +31588.9 | 1.5995 | 2/12 |
| mechanic (wrapping Adam@1.0) | lr-free | +30798.7 | 1.6005 | 4/12 |
| dowg | lr-free | +27315.2 | 1.5985 | 2/12 |
| schedule_free_adamw (lr 0.0025) | schedule-free | +27281.4 | 1.5993 | 4/12 |
| dog | lr-free | +26637.8 | 1.5984 | 3/12 |
| momo_adam | adaptive-lr (Polyak) | +1646.5 | 1.6076 | 3/12 |

(Full per-run configs and timings: `output/lr_free_<rule>_summary.txt`.)

### Findings

1. **Every rule — lr-free included — recovers the truth basin.** The phase-3
   null ("the local rule barely matters; diversity is load-bearing") extends
   unchanged to the learning-rate-free family: basin hit rates sit at the
   familiar ~17–40% per start, with no rule failing.
2. **Prodigy is indistinguishable from hand-tuned Adam.** Its best optimum is
   bit-identical (+31787.84, same winning start), with *no* learning rate set.
   On this cell, deleting the lr hyperparameter costs nothing at all.
3. **Depth ordering within 300 steps:** adam = prodigy > ademamix > adopt >
   dadapt_adamw > mechanic > dowg ≈ schedule_free > dog ≫ momo_adam. The
   distance-estimating rules that warm up from tiny steps (DoG/DoWG,
   schedule-free's averaging, MoMo's conservative Polyak cap) are still
   descending at budget exhaustion — they find the basin but land shallower.
   More steps, not tuning, is their lever.
4. **Wiring is load-bearing (both apply to any future library promotion):**
   - Per-start optimizer state must be `jax.vmap`ed over `init`/`update`.
     The lr-free rules estimate global scalars (Prodigy/D-Adapt distance `d`,
     DoG `max_dist`, Mechanic's scale, MoMo's Polyak step) with whole-tree
     norms; the stacked-`(n_starts, ndim)` trick that is safe for elementwise
     rules (Adam family) would silently couple every start into one shared
     estimate. **`af.AbstractMultiStartGradient` currently inits its optimizer
     on the stacked params, so it cannot carry these rules as-is.**
   - `optax.apply_if_finite` works as the per-start NaN-step guard (the
     ell_comps/shear=0 singularity re-entry), and forwards MoMo's `value=`
     kwarg (optax ≥ 0.2.5).

## Phase 2 — pixelized cell (#100 model; A100)

Gate result (see #100): converged Nautilus (27,840 calls, N_eff 1233,
logZ +17345, 1h32 cold / ~22 min warm-cache) reaches **+17419 at
r_E = 1.31** — itself ~8k nats below the FD probe's +25537 at the truth point,
consistent with the pixelized-source degeneracy making r_E ≈ 1.31 the dominant
posterior mode. The optimizer bar is therefore **+17419** (matching the
converged sampler), not the slack r_E tolerance.

### The elimination chain (all A100, 16 starts, `batch_size=4`, warm cache)

| probe | job(s) | result |
|---|---|---|
| 2a: Adam lr sweep 1e-3 / 3e-3 / 3e-2 | 330529–31 | −39549 / **−28462** / −30659 — **lr ruled out** (1e-2 gave −39888 in #100) |
| 2b: prodigy / dadapt / mechanic (lr-free) | 330535 | −50683 / −49124 / −50375, all frozen from step ~25 — **update rule ruled out** |
| broad vs FD-certified narrow start band | 330588/89 | finite starts 16/16 → **0/16 by step 50 in BOTH bands** — start placement ruled out |
| reg pinned at 1.0 (7 params) | 330593 | mortality unchanged — **reg ruled out** |
| per-start death report | 330592 | deaths scattered across reg 1e-4…4e3, r_E 1.36…6.4; even step 0 has only 13–15/16 finite grads |

**Diagnosis: trajectory NaN mortality.** The pixelized likelihood has hard
non-finite walls (invalid inversions) throughout the broad parameter space.
Every gradient trajectory walks into one within ~25–50 steps; an unguarded
optimizer NaN-poisons outright, and `optax.apply_if_finite` zeroes updates
with the start latched *at* the non-finite point — dead either way. This —
not learning rate, rule, budget, or starts — is the #100 search failure. The
MGE cell (whose only wall is the measure-zero ell_comps/shear singularity)
never exposes it, which is why every MGE finding transferred except the one
that mattered.

### Resurrection: restart-on-death makes the landscape searchable

`PIX_RESURRECT=1` redraws a dead start (params + its vmapped optimizer state)
each step. Population stays alive (14–16/16 finite throughout) and descent
continues indefinitely (jobs 330595 / 330598):

| rule | 600 steps | 3000 steps (2.7 h) | best r_E |
|---|---:|---:|---:|
| adam@1e-2 | −21443 (improving) | **+1718 (still improving)** | 1.5704 (truth band) |
| prodigy | −26946 | −15844 (late jump, improving) | 1.2795 (Nautilus mode) |

The trajectories are long plateaus punctuated by breakthrough jumps —
resurrection churn (~6/16 dead at any step) plus slow kinked-valley descent.

### Verdict

1. **The pixelized landscape is searchable by gradient descent only with
   restart-on-death.** Any PyAutoFit promotion of the multi-start searches
   must implement resurrection (redraw + per-start state reinit), not just
   NaN-guards — `apply_if_finite` alone latches at the cliff edge, and the
   current unguarded `af.MultiStart*` dies silently (this retroactively
   explains #100's −39888 "0/16" result).
2. **Even so, Nautilus wins the pixelized cell decisively:** +17419 with full
   posterior/evidence in ~22 min (warm) vs +1718 after 2.7 h of resurrected
   Adam. The MGE conclusion (gradient MAP ~10× faster than Nautilus) **inverts**
   on inversion-heavy landscapes; multi-start gradient MAP stays the
   recommendation only for parametric (MGE-class) likelihoods, where prodigy
   makes it learning-rate-free at zero cost.
3. **The load-bearing library follow-up is the likelihood, not the
   optimizer:** localising and fixing the non-finite regions (the old
   sweep-findings "localise the NaN" item) would help every gradient consumer
   — MAP searches, HMC/NUTS samplers, SVGD — far more than optimizer work.
