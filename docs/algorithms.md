# Algorithm

`decimate` is a **greedy edge-collapse** simplifier. It repeatedly selects the
globally cheapest *valid* edge, collapses it, and updates the local cost
neighborhood until the target face count is reached. This mirrors the reference
Python prototype exactly.

## Collapse semantics

- **Merge point:** vertex `v1` is merged into `v0` at the **midpoint**:
  `new_pos(v0) = (verts[v0] + verts[v1]) / 2.0`.
- **Face removal:** all faces shared by both endpoints are removed, plus any
  face that becomes degenerate after remapping.
- **Remap:** in every surviving face, index `v1` is replaced by `v0`.
- **Repack:** after the target is reached, surviving vertices are packed to
  contiguous indices `0 .. n_vert_out-1`.

## Cost modes

Let `base_dist = |verts[v0] - verts[v1]|` (Euclidean).

### `NONE`

```
cost = base_dist
```

### `PRESERVE_BOUNDARIES`

An edge shared by **exactly one** triangle is a *boundary* edge and is
penalized:

```
cost = base_dist * boundary_penalty      if shared_by_one_face
cost = base_dist                         otherwise
```

`boundary_penalty` defaults to `15.0`.

### `PRESERVE_VOLUME`

Let `S = (faces(v0) ∪ faces(v1)) \ shared_faces` be the surrounding
neighborhood. If `|S| > 1`:

1. Compute the normalized normal of every face in `S`.
2. `mindot = min over all unordered pairs (i, j) of dot(n_i, n_j)`.
3. `clip = clamp(mindot, -1, 1)`.
4. `curvature_factor = 1.0 + (1.0 - clip) * curvature_gain`.
5. `cost = base_dist * curvature_factor`.

If `|S| <= 1`, `cost = base_dist`.

`curvature_gain` defaults to `5.0`.

**Face normal:** `normalize((p1 - p0) × (p2 - p0))`, returned as the zero
vector if the cross-product norm is `< 1e-8`.

## Determinism

The selection order is fully deterministic. The heap is keyed on

```
edge_key = min(a, b) * n_vert + max(a, b)
```

where `n_vert` is the *input* vertex count. Because `edge_key` depends only on
the unordered vertex pair — **not** on insertion order, thread scheduling, or
device — two entries with an equal `cost` are always ordered the same way.
This guarantees:

- repeated runs produce identical output;
- the CPU and the CUDA backend select the **same** edge on every tie, so their
  outputs match.

## Complexity

- Seeding: `O(E)` cost computations (parallelized with OpenMP), `O(E log E)`
  heap build.
- Per accepted collapse: `O(degree * log H)` for cost updates, where `degree`
  is the local edge degree and `H` the heap size.
- Stale heap entries (edges whose endpoints were retired) are discarded on
  pop, so no explicit deletion is needed.

This replaces the prototype's per-iteration full `sort` + `pop(0)`, which was
`O(N log N)` per collapse and dominant on large meshes.

## CPU vs GPU

- **CPU:** OpenMP parallelizes per-edge cost computation and heap seeding.
  The greedy loop is host-driven and single-threaded for determinism.
- **GPU (opt-in):** bulk cost and normal passes run as CUDA kernels; the
  greedy selection loop is host-driven with the *identical* `edge_key`
  tie-break, so the result is bit-identical to the CPU on ties.

## Shape regularisation (Lloyd / centroidal-Voronoi)

`decimate_regularise` is an **opt-in, topology-preserving** post-pass that
relieves the "irregular triangles" artefact that greedy edge-collapse can leave
behind. It keeps the exact same vertex count, the same faces (the index array is
returned unchanged), and the same boundary — only the coordinates of *interior*
vertices are moved.

This is the standard **Lloyd / centroidal-Voronoi (CVT)** iteration of
Du–Faber–Günzburger: each vertex is moved toward the centroid of its local
Voronoi cell. In 2D this converges to a regular (near-equilateral) tiling.

### Per-vertex step

For a vertex `v` with incident faces `F(v)`:

1. **Cell centroid.** For each incident face `f = (a, b, c)` (using the
   *current* positions of its corners) take the face centroid
   `g_f = (a + b + c) / 3`. The cell centroid is the mean
   `C = (1 / |F(v)|) Σ_f g_f`. This is a standard, robust CVT dual for triangle
   meshes.
2. **Tangent-plane normal.** Accumulate the (unnormalised) face normal
   `n_f = (b − a) × (c − a)` over `f ∈ F(v)`; normalise the sum. If the
   accumulated norm is `< 1e-12` (degenerate cell) fall back to `+z`.
3. **Tangent move.** Project the displacement `d = C − v` onto the tangent
   plane: `d ← d − (d · n̂) n̂`. The vertex moves a **damped** fraction of the
   way: `v' = v + step · d`, with `step ∈ (0, 1]` (default `0.8`).
4. **Laplacian smoothing blend (optional).** Optionally, a mild Laplacian
   (neighbor-averaging) target is blended in. Let `S` be the mean position of
   the vertex's unique neighbors (collected from the incident faces, with
   itself excluded), and project its displacement the same way:
   `s ← (S − v) − ((S − v) · n̂) n̂`. The move becomes
   `v' = v + step · [(1 − smooth) · d + smooth · s]` with `smooth ∈ [0, 1]`
   (default `0.2`). `smooth = 0` recovers the pure Lloyd move exactly (and skips
   the neighbor collection entirely); higher values pull the vertex toward its
   neighbors, removing spurious vertices and producing a smoother lattice.

A vertex with **no incident faces**, or a **boundary** vertex (incident to an
edge shared by exactly one face, when `pin_boundary` is set) is **pinned** —
its position is copied through unchanged. This preserves the silhouette and
avoids moving isolated vertices.

### Guarantees

- **Topology preserved:** the same vertices, faces, and boundary are returned;
  the triangle index array is copied unchanged.
- **Area / volume stable:** the damped, symmetric move keeps total surface area
  (and, to second order, enclosed volume) nearly constant. `preserve_area` is a
  no-op flag kept for API symmetry with the damping already provided by `step`.
- **Determinism:** each vertex depends only on its own incident faces and the
  current positions, so there is no cross-vertex reduction and no ordering
  dependency. Repeated runs — and the CPU vs CUDA backends, which use the
  identical per-face accumulation order and the same `1e-12` normal fallback —
  agree.
- **No degeneration:** damped moves toward a face centroid do not flip or
  collapse faces in practice.

### Complexity

`O(iters · Σ_v |F(v)|)` = `O(iters · 3·n_tri)`, embarrassingly parallel per
vertex (OpenMP on CPU, one thread per vertex on GPU).

### Options

| field           | default | meaning                                                    |
| --------------- | ------- | ---------------------------------------------------------- |
| `iterations`    | `6`     | Lloyd iterations (clamped to `[1, 64]`)                    |
| `step`          | `0.8`   | damping in `(0, 1]`; `<= 0` clamps to `0.5`, `> 1` to `1`  |
| `pin_boundary`  | `1`     | keep boundary / empty vertices fixed                        |
| `preserve_area` | `1`     | no-op (damping already preserves area)                     |
| `smooth`        | `0.2`   | Laplacian (neighbor-avg) blend in `[0,1]`; `0` = pure Lloyd |
| `num_threads`   | `0`     | OpenMP threads; `<= 0` → hardware default                  |
| `seed`          | `0`     | reserved (deterministic)                                    |

### GPU status

The CUDA mirror (`decimate_regularise_gpu`) implements the identical per-vertex
Lloyd step as a one-thread-per-vertex kernel. It is only compiled when
`DECIMATE_ENABLE_CUDA` is on and has **not** been verified on a machine without
`nvcc`; the CPU path is the reference.
