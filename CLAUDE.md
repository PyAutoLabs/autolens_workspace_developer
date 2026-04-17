# autolens_workspace_developer

Developer workspace for profiling and optimising PyAutoLens JAX pipelines.

## Running Scripts

Scripts run from the workspace root (`autolens_workspace_developer/`). All relative `dataset/` and output paths inside scripts must be written relative to the workspace root (e.g. `Path("jax_profiling") / "imaging" / "dataset" / ...`):

```bash
cd autolens_workspace_developer
python jax_profiling/imaging/mge.py
```

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
