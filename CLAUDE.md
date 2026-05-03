# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

**autolens_workspace_developer** is the developer workspace for profiling and optimising PyAutoLens JAX pipelines and prototyping minimal-search experiments. It is not a user-facing workspace — see `../autolens_workspace` for example scripts and tutorials, and `../autolens_workspace_test` for the integration test suite.

Dependencies: `autolens`, `autogalaxy`, `autofit`, `autoarray`, `jax`, `numba`. Python version: 3.11.

## Workspace Structure

```
jax_profiling/               JAX JIT profiling scripts for the imaging /
                             interferometer / point-source likelihood paths.
searches_minimal/            Minimal direct-sampler examples (NSS, Nautilus,
                             Dynesty, Emcee, LBFGS) that bypass the
                             NonLinearSearch wrapper, run on a real lens model.
slam_pipeline/               SLaM pipeline prototypes.
source_science/              Source-plane reconstruction experiments.
los/                         Line-of-sight modelling experiments.
plotting_alignment/          Plotting / visualisation alignment work.
scaling_relation_agg/        Scaling-relation aggregator prototypes.
visualization_profiling/     Visualisation-pipeline profiling.
dataset/                     Input data files.
output/                      Model-fit results written here at runtime.
```

## Running Scripts

Scripts run from the workspace root (`autolens_workspace_developer/`). All relative `dataset/` and output paths inside scripts must be written relative to the workspace root (e.g. `Path("jax_profiling") / "imaging" / "dataset" / ...`):

```bash
cd autolens_workspace_developer
python jax_profiling/imaging/mge.py
```

**Codex / sandboxed runs**: when running from Codex or any restricted environment, set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home or source-tree paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python jax_profiling/imaging/mge.py
```

This workspace is often imported from `/mnt/c/...` and Codex may not be able to write to module `__pycache__` directories or `/home/jammy/.cache`, which can cause import-time `numba` caching failures without this override.

## Raw Array Extraction via `.array`

All autoarray types (`Array2D`, `Grid2D`, `Grid2DIrregular`, `ArrayIrregular`, etc.)
inherit from `AbstractNDArray` and expose a `.array` property that returns the
underlying raw `np.ndarray` or `jax.Array`:

```python
grid_raw = grid.array          # shape (N, 2) raw array
data_raw = dataset.data.array  # shape (N,) raw array
noise_raw = dataset.noise_map.array
```

This is essential for JAX JIT profiling, because autoarray types are **not
registered as JAX pytrees** and cannot cross `jax.jit` boundaries as
inputs or outputs. Extract `.array` before a JIT boundary and pass raw arrays in:

```python
grid_raw = jnp.array(grid.array)

@jax.jit
def my_func(grid_raw):
    ...
    return result_raw  # must be a raw jax.Array, not an autoarray type

result = my_func(grid_raw)
```

Autoarray types *can* be constructed inside a JIT trace (they are consumed
internally), they just cannot be returned from one.

## The `xp` Parameter Pattern

Most PyAutoLens / PyAutoGalaxy / PyAutoArray functions accept an `xp` keyword
that selects the array backend:

- `xp=np` (default) -- pure NumPy path
- `xp=jnp` -- JAX path (`import jax.numpy as jnp`)

Pass `xp=jnp` when calling functions inside JIT-compiled code or when you want
JAX tracing to flow through the computation:

```python
image = tracer.image_2d_from(grid=grid, xp=jnp)
curvature = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=bmm, noise_map=noise, xp=jnp,
)
```

## Key Utility Functions (Inversion Pipeline)

These are the pure-array functions that make up the linear algebra core of the
MGE likelihood. All accept `xp=jnp` and work with raw arrays:

| Function | Module | Purpose |
|---|---|---|
| `data_vector_via_blurred_mapping_matrix_from` | `al.util.inversion_imaging` | Data vector D |
| `curvature_matrix_via_mapping_matrix_from` | `al.util.inversion` | Curvature matrix F |
| `reconstruction_positive_only_from` | `al.util.inversion` | NNLS solve |
| `mapped_reconstructed_data_via_mapping_matrix_from` | `al.util.inversion` | Map reconstruction to image |

## `LightProfileLinearObjFuncList`

The `mapping_matrix` and `operated_mapping_matrix_override` properties already
return **raw arrays** (not autoarray types), so they can be passed directly into
JIT-compiled functions after conversion to `jnp.array`.

## Line Endings -- Always Unix (LF)

All files must use Unix line endings (LF, `\n`). Never write `\r\n`.
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.
