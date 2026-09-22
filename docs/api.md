# API Reference

## C ABI (`include/decimate.h`)

All functions are `extern "C"`. Inputs are row-major, contiguous, little-endian:
`vertices: float32[n_vert][3]`, `triangles: int32[n_tri][3]`.

### Types

```c
typedef enum {
    DECIMATE_MODE_NONE = 0,
    DECIMATE_MODE_PRESERVE_BOUNDARIES = 1,
    DECIMATE_MODE_PRESERVE_VOLUME = 2,
} decimate_mode;

typedef enum {
    DECIMATE_DEVICE_AUTO = 0,
    DECIMATE_DEVICE_CPU  = 1,
    DECIMATE_DEVICE_GPU  = 2,
} decimate_device;

typedef struct {
    decimate_mode   mode;
    decimate_device device;
    float           target_reduction;  // [0.0, 1.0)
    int32_t         num_threads;       // <=0 => hardware
    float           boundary_penalty;  // default 15.0
    float           curvature_gain;    // default 5.0
    uint64_t        seed;              // reserved (deterministic tiebreak)
} decimate_options;

typedef struct {
    float*      vertices;
    int32_t*    triangles;
    int32_t     n_vert_out;
    int32_t     n_tri_out;
    int64_t     elapsed_ns;
    const char* device_used;           // "cpu" | "gpu"
} decimate_result;

// Shape-regularisation (Lloyd / CVT) options
typedef struct {
    int32_t  iterations;     // Lloyd iterations, def 6, clamped [1,64]
    float    step;           // damping in (0,1], def 0.8
    int32_t  pin_boundary;   // 1 = keep boundary vertices fixed (def 1)
    int32_t  preserve_area;  // 1 = damped move keeps area ~constant (def 1)
    float    smooth;         // Laplacian blend in [0,1], def 0.2 (0 = pure Lloyd)
    int32_t  num_threads;    // OpenMP threads; <=0 => hardware
    uint64_t seed;           // reserved (deterministic)
} decimate_regularise_options;
```

### Status codes

| Code | Value |
|---|---|
| `DECIMATE_OK` | `0` |
| `DECIMATE_ERR_NULLPTR` | `-1` |
| `DECIMATE_ERR_BAD_ARGS` | `-2` |
| `DECIMATE_ERR_MODE` | `-3` |
| `DECIMATE_ERR_MEMORY` | `-4` |
| `DECIMATE_ERR_DEVICE` | `-5` |
| `DECIMATE_ERR_INPUT` | `-6` |
| `DECIMATE_ERR_TARGET` | `-7` |
| `DECIMATE_ERR_INTERNAL` | `-8` |

### Functions

```c
decimate_options decimate_default_options(void);

decimate_status decimate_mesh(
    const float*    vertices,
    const int32_t*  triangles,
    int32_t         n_vert,
    int32_t         n_tri,
    const decimate_options* opts,
    decimate_result*        out_result);

decimate_regularise_options decimate_default_regularise_options(void);

// Topology-preserving Lloyd / CVT shape-regularisation pass.
// Returns the same vertices/faces/boundary; only interior vertex
// coordinates are moved. Free with decimate_result_free.
decimate_status decimate_regularise(
    const float*    vertices,
    const int32_t*  triangles,
    int32_t         n_vert,
    int32_t         n_tri,
    const decimate_regularise_options* opts,   // NULL => defaults
    decimate_result*        out_result);

void         decimate_result_free(decimate_result* result); // NULL/twice-safe
const char*  decimate_status_str(decimate_status s);        // never NULL
decimate_status decimate_query(int32_t* gpu_available, int32_t* cpu_threads);
const char*  decimate_version(void);                         // e.g. "1.3.0"
```

### Error handling

- `decimate_mesh` and `decimate_regularise` return `DECIMATE_OK` on success
  and a `DECIMATE_ERR_*` code otherwise; on error `out_result` is left
  untouched and no buffers are allocated.
- Always pair a successful call with `decimate_result_free`.
- Use `decimate_status_str` for a human-readable message in diagnostics.

## Python (`python/decimate.py`)

### `decimate_mesh`

```python
decimate_mesh(
    vertices, triangles,
    target_reduction=0.5,
    mode=Mode.NONE,
    device=Device.AUTO,
    *, num_threads=0,
    boundary_penalty=15.0,
    curvature_gain=5.0,
    seed=0,
    regularise=True,
    reg_iterations=6, reg_step=0.8,
    reg_pin_boundary=True, reg_preserve_area=True,
    reg_smooth=0.2,
) -> Result
```

- `vertices`: `(n_vert, 3)` array, coerced to `float32`.
- `triangles`: `(n_tri, 3)` array, coerced to `int32`.
- Non-contiguous or wrong-dtype inputs are made C-contiguous copies as needed.
- Raises `DecimateError` if the native call returns non-OK.

### `regularise_mesh`

```python
regularise_mesh(
    vertices, triangles,
    iterations=6, step=0.8,
    pin_boundary=True, preserve_area=True,
    smooth=0.2, num_threads=0, seed=0,
) -> Result
```

- Topology-preserving Lloyd / centroidal-Voronoi shape-regularisation pass:
  the returned mesh has the **same vertex count and an identical face index
  array**; boundary vertices stay fixed; only interior vertex coordinates
  move (damped tangent move, see [algorithms.md](algorithms.md#shape-regularisation-lloyd--centroidal-voronoi)).
- `vertices` / `triangles`: same coercion rules as `decimate_mesh`.
- `iterations`: `int` in `[1, 64]` (out-of-range values are clamped).
- `step`: damping factor; values are clamped to `(0, 1]` (`<= 0` becomes `0.5`).
- `pin_boundary`: keep boundary vertices fixed (default `True`).
- `preserve_area`: keep moves in the tangent plane so local surface area is
  unchanged to first order (default `True`).
- `smooth`: mild Laplacian (neighbor-averaging) smoothing weight in `[0, 1]`
  (default `0.2`). Blends the mean of the vertex's unique neighbors into the
  Lloyd target to remove spurious vertices; `0` disables it (pure Lloyd).
- `num_threads`: `<= 0` uses the detected hardware concurrency.
- `seed`: reserved; the pass is deterministic.
- No `device` argument: on CUDA-enabled builds with a GPU present the pass
  runs on the GPU (reported via `Result.device_used == "gpu"`); otherwise it
  runs on the CPU (OpenMP).
- Raises `DecimateError` if the native call returns non-OK.
- The returned `Result` has the same `.free()` / `.copies()` / context-manager
  semantics as for `decimate_mesh`.

### `Result`

Owns the native buffers.

| Member | Type | Description |
|---|---|---|
| `.vertices` | `np.ndarray` | `(n_vert_out, 3) float32` view. |
| `.triangles` | `np.ndarray` | `(n_tri_out, 3) int32` view. |
| `.n_vert` | `int` | Output vertex count. |
| `.n_tri` | `int` | Output triangle count. |
| `.elapsed_ns` | `int` | Wall-clock ns. |
| `.device_used` | `str` | `"cpu"` or `"gpu"`. |
| `.copies()` | `(v, t)` | Independent owned copies. |
| `.free()` | — | Release native buffers (idempotent). |

Supports the context-manager protocol; `__del__` releases on GC.

### Enums

- `Mode.NONE / PRESERVE_BOUNDARIES / PRESERVE_VOLUME`
- `Device.AUTO / CPU / GPU`
- `Status.OK / ERR_*`, with `Status.name(code)`.

### Misc

- `query() -> (gpu_available: bool, cpu_threads: int)`
- `version() -> str`
- `DecimateError(RuntimeError)` — carries `.status`.
- `__version__` — wrapper version.
