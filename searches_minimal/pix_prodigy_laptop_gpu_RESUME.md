# Resume note — pixelized Prodigy laptop-GPU campaign (#125)

Written 2026-08-11 ~22:40. Everything below is committed and pushed to
`feature/pix-prodigy-gpu-compat` / PR #126.

## Where things stand

Phase-1 item 2 (the 16-start claim) is **done and confirmed**. Phase-2 settings
work is **substantially done** — see the "Follow-up" section of
`pix_prodigy_laptop_gpu_findings.md`, which carries the full tables, the
revised recommendations, and one correction to the phase-1 settings table.

11 of 13 planned cells have landed. Environment was validated against Codex's
earlier run bit-identically, and all three source revisions were confirmed
unchanged, so old and new numbers compose directly.

## The two cells that were still running at bedtime

They were launched detached and self-terminate; they may or may not have
survived the night (laptop sleep). **No corruption risk either way** — the
runner writes its artifact only on completion, so a killed run simply leaves no
artifact and can be re-run.

| tag | config | expected artifact |
|---|---|---|
| `dnn_s8_b4_300_freereg` | delaunay_nn, 8 starts, batch 4, 300 steps, **free** AdaptSplit | `pix_prodigy_prodigy_delaunay_nn_rtx2060_dnn_s8_b4_300_freereg_gpu.json` |
| `d_s16_b2_300` | delaunay, 16 starts, **batch 2**, 300 steps, fixed reg | `pix_prodigy_prodigy_delaunay_rtx2060_d_s16_b2_300_gpu.json` |

Check first: `ls searches_minimal/pix_prodigy_results/laptop_gpu/*freereg* *d_s16_b2*`

### If they completed
Fold both into the findings doc (§5 "Not covered" lists them as pending):
- The **free-reg** cell is the phase-2 regularization comparison for
  DelaunayNN — compare against the fixed-reg 8/4/300 row (+30374.8, bar
  crossed at step 175). Expect it to be worse or slower; the #117 lesson is
  that AdaptSplit's high-coefficient region is a NaN wall on Delaunay-family
  meshes.
- The **batch-2** cell tests the one genuinely open question in §1: whether
  Delaunay's 16-start GPU result (+24581.8, plateaued) underperforms the CPU
  reference (+30202) because of batch-4 divergence. If batch 2 lands near
  +30202, batching is the cause and the Delaunay recommendation should say
  batch 2 at 16 starts. If it also plateaus low, the cause is elsewhere and
  the gap stays open.

### To re-run either
Driver scripts are gone with the scratchpad; the invocation is:
```bash
cd ~/Code/PyAutoLabs-wt/pix-prodigy-gpu-compat/autolens_workspace_developer
JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu \
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
PIX_RESULTS_DIR=searches_minimal/pix_prodigy_results/laptop_gpu \
PIX_MESH=delaunay PIX_N_STARTS=16 PIX_BATCH=2 PIX_N_STEPS=300 \
PIX_UPDATE_EVERY=300 PIX_LOG_EVERY=10 PIX_RESURRECT=1 \
PIX_FIX_REG=0.316227766 PIX_NAME_SUFFIX=_rtx2060_d_s16_b2_300 \
~/venv/PyAutoGPU/bin/python -m searches_minimal.pix_prodigy prodigy
```
For the free-reg cell: `PIX_MESH=delaunay_nn PIX_N_STARTS=8 PIX_BATCH=2... ` —
drop `PIX_FIX_REG` entirely, set `PIX_BATCH=4`, `PIX_TRUTH_BAR=30304.030161`,
suffix `_rtx2060_dnn_s8_b4_300_freereg`.

**Traps worth remembering:** `PIX_NAME_SUFFIX` must be unique per cell or the
runner's resume-chaining loads a *completed* run as the finished result instead
of re-running. Wall time scales with `ceil(starts/batch)` passes, ~2.7 s/step
per pass for DelaunayNN. Batch 8 OOMs for every Delaunay-family mesh.

## Then

1. Update the findings doc, drop §5's pending bullet.
2. Commit + push to PR #126 (already open, mergeable).
3. Phase-2 prompt `pixelized_prodigy_laptop_gpu_phase_2_settings.md` is a
   `blocked` draft in PyAutoMind — it can move to formalised/complete once the
   two cells land, since its recommendation table is otherwise satisfied.
4. Separately still owed from PR #106: the **A100 DelaunayNN mapper profile**.
   Every mapper number there is RTX 2060.

## Open questions worth carrying forward

- Delaunay's GPU/CPU 16-start optimum gap (§1) — the batch-2 cell is the test.
- Whether 4-start DelaunayNN crosses its bar given >300 steps; it was still
  climbing at the cap, 156 nats short.
