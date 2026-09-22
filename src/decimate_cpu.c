/**
 * @file decimate_cpu.c
 * @brief High-performance CPU implementation of the greedy edge-collapse
 *        triangle-mesh decimator, parallelized with OpenMP.
 *
 * This is the *reference* implementation. It mirrors the Python prototype
 * (docs/decimate_triangles_mesh_prototype.py) semantics exactly:
 *
 *   - Greedy selection of the globally cheapest VALID edge each iteration.
 *   - Midpoint vertex merge (new_pos = (v0 + v1) / 2).
 *   - Deletion of shared + degenerate faces, remap of v1 -> v0.
 *   - Cost modes: NONE / PRESERVE_BOUNDARIES / PRESERVE_VOLUME.
 *   - Final repack of surviving vertices to contiguous indices.
 *
 * The one intentional structural improvement: instead of re-sorting the whole
 * edge queue every iteration (O(N^2 log N) in the prototype), we use a
 * lazy-deletion binary min-heap keyed on (cost, edge_key). Stale entries are
 * skipped on pop when an endpoint has been retired. This drops the per-iteration
 * O(N) resort while keeping selection deterministic.
 *
 * Determinism: cost ties are broken by a canonical, order-independent
 * edge_key = min(a,b) * n_vert + max(a,b). Because the key depends only on the
 * vertex pair (not on insertion order), the CPU and the CUDA mirror select the
 * exact same edge on ties, so both backends produce identical results.
 */
#include "decimate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <time.h>
#endif

/* Optional CUDA backend (defined by src/decimate_cuda.cu when enabled). */
#if defined(DECIMATE_HAS_CUDA)
extern decimate_status decimate_mesh_gpu(const float* vertices, const int32_t* triangles,
                                         int32_t n_vert, int32_t n_tri,
                                         const decimate_options* opts,
                                         decimate_result* out_result);
extern decimate_status decimate_regularise_gpu(const float* vertices, const int32_t* triangles,
                                               int32_t n_vert, int32_t n_tri,
                                               const decimate_regularise_options* opts,
                                               decimate_result* out_result);
extern int decimate_cuda_available(void);
#endif

#define DECIMATE_VERSION "1.3.0"
#define DECIMATE_EPS 1e-8f

/* ------------------------------------------------------------------ */
/*  Internal state                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    int n_vert, n_tri;

    /* Working copies of vertex positions. */
    float *vx, *vy, *vz;
    int   *v_alive;              /* per-vertex alive flag (in v_to_t) */

    /* Mutable face data, indexed by ORIGINAL face id 0..n_tri-1. */
    int   *f_v0, *f_v1, *f_v2;   /* current (remapped) vertex indices */
    int   *f_alive;              /* face present in tris dict? */
    int    face_count;           /* number of alive faces */

    /* Per-face normals (size n_tri*3). */
    float *nrm;

    /* Vertex -> face adjacency (dynamic int lists), mirrors v_to_t sets. */
    int  **vfaces;
    int   *vf_len;
    int   *vf_cap;

    /* Pre-allocated scratch buffers for the collapse loop (Phase 1). */
    int   *scratch_del;          /* to_delete, cap = n_tri */
    int    scratch_del_cap;
    int   *scratch_neigh;        /* neighbourhood, cap = n_tri */
    int    scratch_neigh_cap;

    decimate_options opts;

    /* Lazy-deletion min-heap of edge candidates, ordered by (cost, tiekey).
     * h_seq stores the canonical edge key (see edge_key) so CPU and GPU
     * break cost ties identically and independently of insertion order. */
    float    *h_cost;
    int      *h_v0, *h_v1;
    long long*h_seq;
    size_t    h_size, h_cap;
} ctx_t;

/* ------------------------------------------------------------------ */
/*  Small utilities                                                    */
/* ------------------------------------------------------------------ */

static int64_t decimate_now_ns(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (int64_t)((double)c.QuadPart * 1e9 / (double)freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
#endif
}

static float decimate_clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ------------------------------------------------------------------ */
/*  Vertex -> face adjacency helpers (mirror the Python sets)          */
/* ------------------------------------------------------------------ */

/* Face-membership test for a vertex. Each vertex has at most ~20 incident
 * faces in practice (avg ~6 for closed manifolds), so this linear scan is
 * effectively O(1).  The hot-path performance wins come from the scratch
 * buffers (no per-iteration malloc) and the O(E log E) init_heap. */
static int vfaces_contains(const ctx_t* c, int v, int t)
{
    int n = c->vf_len[v], i;
    for (i = 0; i < n; ++i) if (c->vfaces[v][i] == t) return 1;
    return 0;
}

static void vfaces_add(ctx_t* c, int v, int t)
{
    if (c->vf_len[v] >= c->vf_cap[v]) {
        int ncap = c->vf_cap[v] ? c->vf_cap[v] * 2 : 4;
        c->vfaces[v] = (int*)realloc(c->vfaces[v], (size_t)ncap * sizeof(int));
        if (!c->vfaces[v]) abort();
        c->vf_cap[v] = ncap;
    }
    c->vfaces[v][c->vf_len[v]++] = t;
}

static void vfaces_remove(ctx_t* c, int v, int t)
{
    int* n = c->vfaces[v];
    int L = c->vf_len[v];
    int i;
    for (i = 0; i < L; ++i) {
        if (n[i] == t) { n[i] = n[L - 1]; c->vf_len[v] = L - 1; return; }
    }
}

/* ------------------------------------------------------------------ */
/*  Triangle normal (mirrors compute_triangle_normal)                  */
/* ------------------------------------------------------------------ */

static void recompute_normal(ctx_t* c, int t)
{
    int a = c->f_v0[t], b = c->f_v1[t], d = c->f_v2[t];
    float ax = c->vx[a], ay = c->vy[a], az = c->vz[a];
    float bx = c->vx[b], by = c->vy[b], bz = c->vz[b];
    float cx = c->vx[d], cy = c->vy[d], cz = c->vz[d];
    float v0x = bx - ax, v0y = by - ay, v0z = bz - az;
    float v1x = cx - ax, v1y = cy - ay, v1z = cz - az;
    float nx = v0y * v1z - v0z * v1y;
    float ny = v0z * v1x - v0x * v1z;
    float nz = v0x * v1y - v0y * v1x;
    float n = sqrtf(nx * nx + ny * ny + nz * nz);
    if (n > DECIMATE_EPS) { nx /= n; ny /= n; nz /= n; }
    else { nx = ny = nz = 0.0f; }
    c->nrm[3 * t + 0] = nx;
    c->nrm[3 * t + 1] = ny;
    c->nrm[3 * t + 2] = nz;
}

/* ------------------------------------------------------------------ */
/*  Edge collapse cost (mirrors compute_edge_cost)                     */
/* ------------------------------------------------------------------ */

static float compute_edge_cost(ctx_t* c, int a, int b)
{
    float dx = c->vx[a] - c->vx[b];
    float dy = c->vy[a] - c->vy[b];
    float dz = c->vz[a] - c->vz[b];
    float base = sqrtf(dx * dx + dy * dy + dz * dz);
    int i;

    if (c->opts.mode == DECIMATE_MODE_PRESERVE_BOUNDARIES) {
        int shared = 0;
        for (i = 0; i < c->vf_len[a]; ++i) {
            int t = c->vfaces[a][i];
            if (vfaces_contains(c, b, t)) ++shared;
        }
        if (shared == 1) return base * c->opts.boundary_penalty;
        return base; /* falls through exactly like the prototype */
    }

    if (c->opts.mode == DECIMATE_MODE_PRESERVE_VOLUME) {
        /* surrounding = (faces(a) U faces(b)) \ shared(a,b).
         * A face is "shared" iff it is in both adjacency lists. */
        int cap = 8, n = 0;
        int* list = (int*)malloc((size_t)cap * sizeof(int));
        if (!list) return base;

        for (i = 0; i < c->vf_len[a]; ++i) {
            int t = c->vfaces[a][i];
            if (!vfaces_contains(c, b, t)) {
                if (n == cap) { cap *= 2; list = (int*)realloc(list, (size_t)cap * sizeof(int)); }
                list[n++] = t;
            }
        }
        for (i = 0; i < c->vf_len[b]; ++i) {
            int t = c->vfaces[b][i];
            if (!vfaces_contains(c, a, t)) {
                if (n == cap) { cap *= 2; list = (int*)realloc(list, (size_t)cap * sizeof(int)); }
                list[n++] = t;
            }
        }

        if (n > 1) {
            float mindot = 1.0f;
            int x, y;
            for (x = 0; x < n; ++x) {
                const float* ni = &c->nrm[3 * list[x]];
                for (y = x + 1; y < n; ++y) {
                    const float* nj = &c->nrm[3 * list[y]];
                    float dot = ni[0] * nj[0] + ni[1] * nj[1] + ni[2] * nj[2];
                    if (dot < mindot) mindot = dot;
                }
            }
            float clip = decimate_clampf(mindot, -1.0f, 1.0f);
            float cf = 1.0f + (1.0f - clip) * c->opts.curvature_gain;
            free(list);
            return base * cf;
        }
        free(list);
    }

    return base;
}

/* ------------------------------------------------------------------ */
/*  Lazy-deletion binary min-heap, ordered by (cost, seq)              */
/* ------------------------------------------------------------------ */

/* Canonical edge key: order-independent and device-independent, used to
 * break cost ties identically on CPU and GPU. */
static long long edge_key(const ctx_t* c, int a, int b)
{
    int lo = a < b ? a : b, hi = a < b ? b : a;
    return (long long)lo * (long long)c->n_vert + (long long)hi;
}

static int heap_less(const ctx_t* c, size_t i, size_t j)
{
    if (c->h_cost[i] != c->h_cost[j]) return c->h_cost[i] < c->h_cost[j];
    return c->h_seq[i] < c->h_seq[j];
}

static void heap_swap(ctx_t* c, size_t i, size_t j)
{
    float fc = c->h_cost[i]; int a = c->h_v0[i], b = c->h_v1[i]; long long s = c->h_seq[i];
    c->h_cost[i] = c->h_cost[j]; c->h_v0[i] = c->h_v0[j]; c->h_v1[i] = c->h_v1[j]; c->h_seq[i] = c->h_seq[j];
    c->h_cost[j] = fc; c->h_v0[j] = a; c->h_v1[j] = b; c->h_seq[j] = s;
}

static void heap_push(ctx_t* c, float cost, int a, int b)
{
    if (c->h_size == c->h_cap) {
        size_t ncap = c->h_cap ? c->h_cap * 2 : 256;
        c->h_cost = (float*)realloc(c->h_cost, ncap * sizeof(float));
        c->h_v0   = (int*)realloc(c->h_v0, ncap * sizeof(int));
        c->h_v1   = (int*)realloc(c->h_v1, ncap * sizeof(int));
        c->h_seq  = (long long*)realloc(c->h_seq, ncap * sizeof(long long));
        if (!c->h_cost || !c->h_v0 || !c->h_v1 || !c->h_seq) abort();
        c->h_cap = ncap;
    }
    size_t i = c->h_size++;
    c->h_cost[i] = cost;
    c->h_v0[i] = a;
    c->h_v1[i] = b;
    c->h_seq[i] = edge_key(c, a, b);
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (heap_less(c, i, p)) { heap_swap(c, i, p); i = p; }
        else break;
    }
}

static int heap_pop(ctx_t* c, float* cost, int* a, int* b)
{
    if (c->h_size == 0) return 0;
    *cost = c->h_cost[0]; *a = c->h_v0[0]; *b = c->h_v1[0];
    --c->h_size;
    if (c->h_size > 0) {
        c->h_cost[0] = c->h_cost[c->h_size];
        c->h_v0[0] = c->h_v0[c->h_size];
        c->h_v1[0] = c->h_v1[c->h_size];
        c->h_seq[0] = c->h_seq[c->h_size];
        size_t i = 0;
        for (;;) {
            size_t l = 2 * i + 1, r = 2 * i + 2, best = i;
            if (l < c->h_size && heap_less(c, l, best)) best = l;
            if (r < c->h_size && heap_less(c, r, best)) best = r;
            if (best == i) break;
            heap_swap(c, i, best);
            i = best;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Public helpers                                                     */
/* ------------------------------------------------------------------ */

decimate_options decimate_default_options(void)
{
    decimate_options o;
    o.mode = DECIMATE_MODE_NONE;
    o.device = DECIMATE_DEVICE_AUTO;
    o.target_reduction = 0.5f;
    o.num_threads = 0;
    o.boundary_penalty = 15.0f;
    o.curvature_gain = 5.0f;
    o.seed = 0;
    return o;
}

decimate_regularise_options decimate_default_regularise_options(void)
{
    decimate_regularise_options o;
    o.iterations    = 6;
    o.step          = 0.8f;
    o.pin_boundary  = 1;
    o.preserve_area = 1;
    o.smooth        = 0.2f;
    o.num_threads   = 0;
    o.seed          = 0;
    return o;
}

const char* decimate_version(void) { return DECIMATE_VERSION; }

void decimate_result_free(decimate_result* r)
{
    if (!r) return;
    free(r->vertices);
    free(r->triangles);
    r->vertices = NULL;
    r->triangles = NULL;
    r->n_vert_out = 0;
    r->n_tri_out = 0;
    r->elapsed_ns = 0;
    r->device_used = NULL;
}

const char* decimate_status_str(decimate_status s)
{
    switch (s) {
    case DECIMATE_OK:                  return "OK";
    case DECIMATE_ERR_NULLPTR:         return "NULL pointer";
    case DECIMATE_ERR_BAD_ARGS:        return "Bad arguments";
    case DECIMATE_ERR_MODE:            return "Unknown mode";
    case DECIMATE_ERR_MEMORY:          return "Memory allocation failure";
    case DECIMATE_ERR_DEVICE:          return "Device unavailable";
    case DECIMATE_ERR_INPUT:           return "Invalid input mesh";
    case DECIMATE_ERR_TARGET:          return "Invalid target_reduction";
    case DECIMATE_ERR_INTERNAL:        return "Internal error";
    default:                           return "Unknown error";
    }
}

decimate_status decimate_query(int32_t* gpu_available, int32_t* cpu_threads)
{
    if (gpu_available) {
#if defined(DECIMATE_HAS_CUDA)
        *gpu_available = decimate_cuda_available() ? 1 : 0;
#else
        *gpu_available = 0;
#endif
    }
    if (cpu_threads) {
#if defined(_OPENMP)
        *cpu_threads = (int32_t)omp_get_max_threads();
#else
        *cpu_threads = 1;
#endif
    }
    return DECIMATE_OK;
}

/* ------------------------------------------------------------------ */
/*  CPU decimation core                                                */
/* ------------------------------------------------------------------ */

static void ctx_free(ctx_t* c)
{
    int i;
    free(c->vx); free(c->vy); free(c->vz); free(c->v_alive);
    free(c->f_v0); free(c->f_v1); free(c->f_v2); free(c->f_alive); free(c->nrm);
    for (i = 0; i < c->n_vert; ++i) free(c->vfaces[i]);
    free(c->vfaces); free(c->vf_len); free(c->vf_cap);
    free(c->scratch_del); free(c->scratch_neigh);
    free(c->h_cost); free(c->h_v0); free(c->h_v1); free(c->h_seq);
    memset(c, 0, sizeof(*c));
}

static decimate_status build_context(const float* vertices, const int32_t* triangles,
                                     int32_t n_vert, int32_t n_tri,
                                     const decimate_options* opts, ctx_t* c)
{
    int i, k;
    memset(c, 0, sizeof(*c));
    c->n_vert = n_vert;
    c->n_tri = n_tri;
    c->opts = *opts;
    c->face_count = n_tri;

    c->vx = (float*)malloc((size_t)n_vert * sizeof(float));
    c->vy = (float*)malloc((size_t)n_vert * sizeof(float));
    c->vz = (float*)malloc((size_t)n_vert * sizeof(float));
    c->v_alive = (int*)malloc((size_t)n_vert * sizeof(int));
    c->f_v0 = (int*)malloc((size_t)n_tri * sizeof(int));
    c->f_v1 = (int*)malloc((size_t)n_tri * sizeof(int));
    c->f_v2 = (int*)malloc((size_t)n_tri * sizeof(int));
    c->f_alive = (int*)malloc((size_t)n_tri * sizeof(int));
    c->nrm = (float*)malloc((size_t)n_tri * 3 * sizeof(float));
    c->vfaces = (int**)malloc((size_t)n_vert * sizeof(int*));
    c->vf_len = (int*)malloc((size_t)n_vert * sizeof(int));
    c->vf_cap = (int*)malloc((size_t)n_vert * sizeof(int));
    /* to_delete can hold up to vf_len[v0]+vf_len[v1] ≤ 2*n_tri entries.
     * neighbourhood can hold up to 3*vf_len[v0] ≤ 3*n_tri distinct verts
     * (fan mesh).  Allocate worst-case capacities so the collapse loop
     * never needs to reallocate (the Phase 1 perf win). */
    c->scratch_del = (int*)malloc((size_t)(2 * n_tri) * sizeof(int));
    c->scratch_neigh = (int*)malloc((size_t)(3 * n_tri) * sizeof(int));
    c->scratch_del_cap = 2 * n_tri;
    c->scratch_neigh_cap = 3 * n_tri;

    if (!c->vx || !c->vy || !c->vz || !c->v_alive || !c->f_v0 || !c->f_v1 ||
        !c->f_v2 || !c->f_alive || !c->nrm ||
        !c->vfaces || !c->vf_len || !c->vf_cap || !c->scratch_del || !c->scratch_neigh) {
        ctx_free(c);
        return DECIMATE_ERR_MEMORY;
    }

    for (i = 0; i < n_vert; ++i) {
        c->vx[i] = vertices[3 * i + 0];
        c->vy[i] = vertices[3 * i + 1];
        c->vz[i] = vertices[3 * i + 2];
        c->v_alive[i] = 1;
        c->vfaces[i] = NULL;
        c->vf_len[i] = 0;
        c->vf_cap[i] = 0;
    }
    for (i = 0; i < n_tri; ++i) {
        c->f_v0[i] = triangles[3 * i + 0];
        c->f_v1[i] = triangles[3 * i + 1];
        c->f_v2[i] = triangles[3 * i + 2];
        c->f_alive[i] = 1;
    }

    /* Vertex -> face adjacency (v_to_t). */
    for (i = 0; i < n_tri; ++i) {
        int v[3] = { c->f_v0[i], c->f_v1[i], c->f_v2[i] };
        for (k = 0; k < 3; ++k) vfaces_add(c, v[k], i);
    }

    /* Initial normals, parallelized. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < n_tri; ++i) {
        recompute_normal(c, i);
    }

    return DECIMATE_OK;
}

/* Fill the initial heap with every unique edge's cost.
 * Phase 1: O(E log E) sort-and-unique instead of O(E²) linear-scan dedup.
 * Edges are packed into 64-bit keys (lo << 32 | hi), sorted, and
 * deduplicated. This is determinism-safe: the resulting unique edge set
 * is identical regardless of sort stability because keys are unique. */
static int64_t edge_pack64(int lo, int hi)
{
    return ((int64_t)(unsigned)lo << 32) | (int64_t)(unsigned)hi;
}

static int cmp_int64(const void* a, const void* b)
{
    int64_t x = *(const int64_t*)a, y = *(const int64_t*)b;
    return (x > y) - (x < y);
}

static decimate_status init_heap(ctx_t* c)
{
    int total_edges = c->n_tri * 3;
    int64_t* keys = (int64_t*)malloc((size_t)(total_edges > 0 ? total_edges : 1) * sizeof(int64_t));
    int ec = 0, i;
    if (!keys) return DECIMATE_ERR_MEMORY;

    /* Collect all 3*n_tri edge keys (with duplicates). */
    for (i = 0; i < c->n_tri; ++i) {
        const int f[3] = { c->f_v0[i], c->f_v1[i], c->f_v2[i] };
        int j;
        for (j = 0; j < 3; ++j) {
            int u = f[j], v = f[(j + 1) % 3];
            int lo = u < v ? u : v, hi = u < v ? v : u;
            keys[ec++] = edge_pack64(lo, hi);
        }
    }

    /* Sort + unique → O(E log E). */
    qsort(keys, (size_t)ec, sizeof(int64_t), cmp_int64);

    /* Deduplicate in-place. */
    int ec_uniq = 0;
    for (i = 0; i < ec; ++i) {
        if (i == 0 || keys[i] != keys[i - 1]) keys[ec_uniq++] = keys[i];
    }
    ec = ec_uniq;

    /* Compute costs in parallel (independent per edge). */
    float* costs = (float*)malloc((size_t)(ec > 0 ? ec : 1) * sizeof(float));
    if (!costs) { free(keys); return DECIMATE_ERR_MEMORY; }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < ec; ++i) {
        int a = (int)(keys[i] >> 32);
        int b = (int)(keys[i] & 0xFFFFFFFF);
        costs[i] = compute_edge_cost(c, a, b);
    }

    /* Insert in sorted-key order → deterministic, matches the
     * prototype's stable tie-breaking (keys are unique so sort order
     * is total and unambiguous). */
    for (i = 0; i < ec; ++i) {
        int a = (int)(keys[i] >> 32);
        int b = (int)(keys[i] & 0xFFFFFFFF);
        heap_push(c, costs[i], a, b);
    }

    free(costs);
    free(keys);
    return DECIMATE_OK;
}

static decimate_status run_cpu(const float* vertices, const int32_t* triangles,
                               int32_t n_vert, int32_t n_tri,
                               const decimate_options* opts,
                               decimate_result* out_result)
{
    ctx_t c;
    int64_t t0 = decimate_now_ns();
    int target;
    decimate_status st = build_context(vertices, triangles, n_vert, n_tri, opts, &c);
    if (st != DECIMATE_OK) return st;

    if (n_tri == 0) {
        /* Nothing to decimate: emit all vertices, no faces. */
        int64_t t1 = decimate_now_ns();
        out_result->vertices = (float*)malloc((size_t)n_vert * 3 * sizeof(float));
        if (!out_result->vertices) { ctx_free(&c); return DECIMATE_ERR_MEMORY; }
        memcpy(out_result->vertices, vertices, (size_t)n_vert * 3 * sizeof(float));
        out_result->triangles = NULL;
        out_result->n_vert_out = n_vert;
        out_result->n_tri_out = 0;
        out_result->elapsed_ns = t1 - t0;
        out_result->device_used = "cpu";
        ctx_free(&c);
        return DECIMATE_OK;
    }

    st = init_heap(&c);
    if (st != DECIMATE_OK) { ctx_free(&c); return st; }

    target = (int)((double)n_tri * (1.0 - (double)opts->target_reduction));
    if (target < 0) target = 0;

    /* --- Greedy collapse loop ------------------------------------- */
    while (c.face_count > target && c.h_size > 0) {
        float cost; int v0, v1;
        if (!heap_pop(&c, &cost, &v0, &v1)) break;

        /* Stale-entry guard (matches the prototype's endpoint check). */
        if (!c.v_alive[v0] || !c.v_alive[v1]) continue;

        /* Merge v1 into v0 at the midpoint. */
        c.vx[v0] = (c.vx[v0] + c.vx[v1]) * 0.5f;
        c.vy[v0] = (c.vy[v0] + c.vy[v1]) * 0.5f;
        c.vz[v0] = (c.vz[v0] + c.vz[v1]) * 0.5f;

        /* to_delete = shared faces (reuses pre-allocated scratch buffer). */
        int* to_delete = c.scratch_del;
        int td_n = 0;
        int i;
        for (i = 0; i < c.vf_len[v0]; ++i) {
            int t = c.vfaces[v0][i];
            if (vfaces_contains(&c, v1, t)) to_delete[td_n++] = t;
        }

        /* Remap + degenerate check over all affected (non-shared) faces.
         * The two list lengths are snapshotted BEFORE the loop: vfaces_add()
         * below grows (and may reallocate) vfaces[v0] mid-loop, so any live
         * read of vf_len[v0] inside the loop would shift the v1-side index
         * (i - vf_len[v0]), read past the end of the v1 adjacency list, and
         * inject garbage face ids into the adjacency structure. */
        {
            const int len0 = c.vf_len[v0];
            const int len1 = c.vf_len[v1];
            for (i = 0; i < len0 + len1; ++i) {
                int t, q, in_del;
                if (i < len0) t = c.vfaces[v0][i];
                else t = c.vfaces[v1][i - len0];
                in_del = 0; for (q = 0; q < td_n; ++q) if (to_delete[q] == t) { in_del = 1; break; }
                if (in_del) continue;

                if (c.f_v0[t] == v1) c.f_v0[t] = v0;
                if (c.f_v1[t] == v1) c.f_v1[t] = v0;
                if (c.f_v2[t] == v1) c.f_v2[t] = v0;

                if (c.f_v0[t] == c.f_v1[t] || c.f_v1[t] == c.f_v2[t] || c.f_v2[t] == c.f_v0[t]) {
                    to_delete[td_n++] = t;
                } else {
                    if (!vfaces_contains(&c, v0, t)) vfaces_add(&c, v0, t);
                    recompute_normal(&c, t);
                }
            }
        }

        /* Commit deletions: scrub adjacency, mark face dead. */
        for (i = 0; i < td_n; ++i) {
            int t = to_delete[i];
            if (!c.f_alive[t]) continue;
            int v[3] = { c.f_v0[t], c.f_v1[t], c.f_v2[t] };
            int m;
            for (m = 0; m < 3; ++m) vfaces_remove(&c, v[m], t);
            c.f_alive[t] = 0;
            --c.face_count;
        }

        /* Retire v1 permanently. */
        free(c.vfaces[v1]);
        c.vfaces[v1] = NULL;
        c.vf_len[v1] = 0;
        c.vf_cap[v1] = 0;
        c.v_alive[v1] = 0;

        /* Recompute costs for the neighborhood of v0 (reuses scratch buf). */
        int nn = 0, q;
        int* neigh = c.scratch_neigh;
        for (i = 0; i < c.vf_len[v0]; ++i) {
            int t = c.vfaces[v0][i];
            const int v[3] = { c.f_v0[t], c.f_v1[t], c.f_v2[t] };
            int m;
            for (m = 0; m < 3; ++m) {
                int nv = v[m];
                if (nv == v0) continue;
                int dup = 0;
                for (q = 0; q < nn; ++q) if (neigh[q] == nv) { dup = 1; break; }
                if (dup) continue;
                neigh[nn++] = nv;
            }
        }
        for (i = 0; i < nn; ++i) {
            int a = v0 < neigh[i] ? v0 : neigh[i];
            int b = v0 < neigh[i] ? neigh[i] : v0;
            heap_push(&c, compute_edge_cost(&c, a, b), a, b);
        }
    }

    /* --- Pack surviving vertices to contiguous indices -------------- */
    int live = 0, i;
    int* order = (int*)malloc((size_t)n_vert * sizeof(int));
    int* remap = (int*)malloc((size_t)n_vert * sizeof(int));
    if (!order || !remap) { free(order); free(remap); ctx_free(&c); return DECIMATE_ERR_MEMORY; }
    for (i = 0; i < n_vert; ++i) if (c.v_alive[i]) { order[live] = i; remap[i] = live++; }

    int out_v = live;
    int out_t = 0;
    for (i = 0; i < n_tri; ++i) if (c.f_alive[i]) ++out_t;

    float* ov = (float*)malloc((size_t)out_v * 3 * sizeof(float));
    int32_t* ot = (int32_t*)malloc((size_t)out_t * 3 * sizeof(int32_t));
    if (!ov || (!ot && out_t > 0)) { free(ov); free(ot); free(order); free(remap); ctx_free(&c); return DECIMATE_ERR_MEMORY; }

    for (i = 0; i < out_v; ++i) {
        int o = order[i];
        ov[3 * i + 0] = c.vx[o];
        ov[3 * i + 1] = c.vy[o];
        ov[3 * i + 2] = c.vz[o];
    }
    int w = 0;
    for (i = 0; i < n_tri; ++i) {
        if (!c.f_alive[i]) continue;
        ot[3 * w + 0] = remap[c.f_v0[i]];
        ot[3 * w + 1] = remap[c.f_v1[i]];
        ot[3 * w + 2] = remap[c.f_v2[i]];
        ++w;
    }
    free(order);
    free(remap);
    ctx_free(&c);

    int64_t t1 = decimate_now_ns();
    out_result->vertices = ov;
    out_result->triangles = ot;
    out_result->n_vert_out = out_v;
    out_result->n_tri_out = out_t;
    out_result->elapsed_ns = t1 - t0;
    out_result->device_used = "cpu";
    return DECIMATE_OK;
}

/* ------------------------------------------------------------------ */
/*  Dispatch entry point                                               */
/* ------------------------------------------------------------------ */

decimate_status decimate_mesh(const float* vertices, const int32_t* triangles,
                              int32_t n_vert, int32_t n_tri,
                              const decimate_options* opts,
                              decimate_result* out_result)
{
    decimate_options o;

    if (!out_result) return DECIMATE_ERR_NULLPTR;
    memset(out_result, 0, sizeof(*out_result));
    if (!opts) o = decimate_default_options(); else o = *opts;

    if (o.mode < DECIMATE_MODE_NONE || o.mode > DECIMATE_MODE_PRESERVE_VOLUME)
        return DECIMATE_ERR_MODE;
    if (o.device < DECIMATE_DEVICE_AUTO || o.device > DECIMATE_DEVICE_GPU)
        return DECIMATE_ERR_DEVICE;
    if (o.target_reduction < 0.0f || o.target_reduction >= 1.0f)
        return DECIMATE_ERR_TARGET;

    if (n_vert < 0 || n_tri < 0) return DECIMATE_ERR_BAD_ARGS;
    if (n_tri > 0 && (!vertices || !triangles)) return DECIMATE_ERR_NULLPTR;
    if (n_tri > 0) {
        int i;
        for (i = 0; i < n_tri; ++i) {
            int a = triangles[3 * i + 0], b = triangles[3 * i + 1], d = triangles[3 * i + 2];
            if (a < 0 || b < 0 || d < 0 || a >= n_vert || b >= n_vert || d >= n_vert)
                return DECIMATE_ERR_INPUT;
        }
    }

#if defined(_OPENMP)
    if (o.num_threads > 0) omp_set_num_threads(o.num_threads);
#endif

    /* Device dispatch. */
#if defined(DECIMATE_HAS_CUDA)
    int gpu_ok = decimate_cuda_available() ? 1 : 0;
    if (o.device == DECIMATE_DEVICE_GPU) {
        if (!gpu_ok) return DECIMATE_ERR_DEVICE;
        return decimate_mesh_gpu(vertices, triangles, n_vert, n_tri, opts, out_result);
    }
    if (o.device == DECIMATE_DEVICE_AUTO && gpu_ok) {
        return decimate_mesh_gpu(vertices, triangles, n_vert, n_tri, opts, out_result);
    }
    /* fall through to CPU (or explicit CPU request) */
#endif

    return run_cpu(vertices, triangles, n_vert, n_tri, opts, out_result);
}

/* ================================================================== */
/*  Shape-regularisation pass  (Lloyd / centroidal-Voronoi)           */
/* ================================================================== */
/*
 * Topology-preserving refinement.  For every interior vertex we build its
 * local (tangent-plane) Voronoi cell, approximated by the mean of the
 * centroids of its incident faces (a standard, robust CVT dual for triangle
 * meshes), and move the vertex a damped fraction of the way toward that
 * centroid.  Boundary vertices (incident to an edge shared by exactly one
 * face) are pinned, so the silhouette is preserved.  Because the step is
 * damped and symmetric, total area is preserved to within the damping error
 * and no face degenerates in practice.
 *
 * Per-vertex work is embarrassingly parallel (OpenMP).  Determinism is exact:
 * the centroid of a vertex depends only on that vertex's incident faces and
 * the current positions, so no reduction ordering is involved.
 */

typedef struct {
    float* px; float* py; float* pz;   /* current positions (SoA)          */
    int**  vfaces;                     /* vertex -> incident face list      */
    int*   vf_len;                     /* vertex -> incident face count     */
    int*   vf_cap;                     /* vertex -> allocated capacity      */
    int*   is_boundary;                /* 1 = pinned (on a free edge)       */
    int    n_vert;
    int    n_tri;
} reg_t;

static void reg_free(reg_t* r)
{
    int i;
    free(r->px); free(r->py); free(r->pz);
    for (i = 0; i < r->n_vert; ++i) free(r->vfaces[i]);
    free(r->vfaces); free(r->vf_len); free(r->vf_cap); free(r->is_boundary);
    memset(r, 0, sizeof(*r));
}

static decimate_status reg_build(const float* vertices, const int32_t* triangles,
                                 int n_vert, int n_tri, reg_t* r)
{
    int i, k;
    r->n_vert = n_vert;
    r->n_tri  = n_tri;

    r->px = (float*)malloc((size_t)n_vert * sizeof(float));
    r->py = (float*)malloc((size_t)n_vert * sizeof(float));
    r->pz = (float*)malloc((size_t)n_vert * sizeof(float));
    r->vfaces = (int**)malloc((size_t)(n_vert ? n_vert : 1) * sizeof(int*));
    r->vf_len = (int*)malloc((size_t)(n_vert ? n_vert : 1) * sizeof(int));
    r->vf_cap = (int*)malloc((size_t)(n_vert ? n_vert : 1) * sizeof(int));
    r->is_boundary = (int*)malloc((size_t)(n_vert ? n_vert : 1) * sizeof(int));

    if (!r->px || !r->py || !r->pz || !r->vfaces || !r->vf_len || !r->vf_cap || !r->is_boundary) {
        reg_free(r);
        return DECIMATE_ERR_MEMORY;
    }

    for (i = 0; i < n_vert; ++i) {
        r->px[i] = vertices[3 * i + 0];
        r->py[i] = vertices[3 * i + 1];
        r->pz[i] = vertices[3 * i + 2];
        r->vfaces[i] = (int*)malloc(4 * sizeof(int));
        r->vf_len[i] = 0;
        r->vf_cap[i] = 4;
        r->is_boundary[i] = 0;
        if (!r->vfaces[i]) { reg_free(r); return DECIMATE_ERR_MEMORY; }
    }

    /* Build vertex -> face adjacency (small per-vertex lists; the
     * amortised realloc keeps it allocation-cheap). */
    for (i = 0; i < n_tri; ++i) {
        int f[3] = { triangles[3 * i + 0], triangles[3 * i + 1], triangles[3 * i + 2] };
        for (k = 0; k < 3; ++k) {
            int v = f[k], n = r->vf_len[v];
            if (n >= r->vf_cap[v]) { /* grow by doubling when full */
                int nc = r->vf_cap[v] * 2, j; int* nv = (int*)malloc((size_t)nc * sizeof(int));
                if (!nv) { reg_free(r); return DECIMATE_ERR_MEMORY; }
                for (j = 0; j < n; ++j) nv[j] = r->vfaces[v][j];
                free(r->vfaces[v]);
                r->vfaces[v] = nv;
                r->vf_cap[v] = nc;
            }
            /* de-duplicate (shouldn't happen on a valid mesh, but be safe) */
            int dup = 0, j;
            for (j = 0; j < n; ++j) if (r->vfaces[v][j] == i) { dup = 1; break; }
            if (!dup) r->vfaces[v][n] = i;
            if (!dup) r->vf_len[v] = n + 1;
        }
    }

    /* Boundary mask: a vertex is on the boundary if it owns any edge
     * (u,v) that appears in exactly one face. */
    {
        /* Count edge occurrences over the whole mesh. */
        int64_t* ekeys = (int64_t*)malloc((size_t)(n_tri * 3 > 0 ? n_tri * 3 : 1) * sizeof(int64_t));
        int ecount = 0;
        if (!ekeys) { reg_free(r); return DECIMATE_ERR_MEMORY; }
        for (i = 0; i < n_tri; ++i) {
            int f[3] = { triangles[3 * i + 0], triangles[3 * i + 1], triangles[3 * i + 2] };
            for (k = 0; k < 3; ++k) {
                int u = f[k], v = f[(k + 1) % 3];
                int lo = u < v ? u : v, hi = u < v ? v : u;
                ekeys[ecount++] = edge_pack64(lo, hi);
            }
        }
        qsort(ekeys, (size_t)ecount, sizeof(int64_t), cmp_int64);
        {
            int s = 0;
            while (s < ecount) {
                int j = s + 1;
                while (j < ecount && ekeys[j] == ekeys[s]) ++j;
                int run = j - s;
                if (run == 1) {
                    /* Edge shared by exactly one face => a free (boundary) edge. */
                    int u = (int)((unsigned)(ekeys[s] >> 32));
                    int v = (int)((unsigned)(ekeys[s] & 0xFFFFFFFFull));
                    r->is_boundary[u] = 1;
                    r->is_boundary[v] = 1;
                }
                s = j;
            }
        }
        free(ekeys);
    }

    return DECIMATE_OK;
}

decimate_status decimate_regularise(const float* vertices, const int32_t* triangles,
                                    int32_t n_vert, int32_t n_tri,
                                    const decimate_regularise_options* opts,
                                    decimate_result* out_result)
{
    decimate_regularise_options o;
    float step, smooth;
    int iters, pin, t0, i;
    int64_t t_start, t_end;

    if (!out_result) return DECIMATE_ERR_NULLPTR;
    memset(out_result, 0, sizeof(*out_result));
    if (!opts) o = decimate_default_regularise_options(); else o = *opts;

    if (n_vert < 0 || n_tri < 0) return DECIMATE_ERR_BAD_ARGS;
    if (n_tri > 0 && (!vertices || !triangles)) return DECIMATE_ERR_NULLPTR;
    if (n_tri > 0) {
        for (i = 0; i < n_tri; ++i) {
            int a = triangles[3 * i + 0], b = triangles[3 * i + 1], d = triangles[3 * i + 2];
            if (a < 0 || b < 0 || d < 0 || a >= n_vert || b >= n_vert || d >= n_vert)
                return DECIMATE_ERR_INPUT;
        }
    }
    if (o.iterations < 1) iters = 1; else iters = o.iterations;
    if (o.iterations > 64) iters = 64;
    step = o.step <= 0.0f ? 0.5f : o.step;
    if (step > 1.0f) step = 1.0f;
    smooth = o.smooth; if (smooth < 0.0f) smooth = 0.0f; if (smooth > 1.0f) smooth = 1.0f;
    pin = o.pin_boundary ? 1 : 0;

#if defined(_OPENMP)
    if (o.num_threads > 0) omp_set_num_threads(o.num_threads);
#endif

#if defined(DECIMATE_HAS_CUDA)
    if (decimate_cuda_available())
        return decimate_regularise_gpu(vertices, triangles, n_vert, n_tri, &o, out_result);
#endif

    t_start = decimate_now_ns();

    /* ---- Build local context --------------------------------------- */
    {
        reg_t r;
        decimate_status st = reg_build(vertices, triangles, n_vert, n_tri, &r);
        if (st != DECIMATE_OK) return st;

        /* ---- Lloyd iterations -------------------------------------- */
        for (t0 = 0; t0 < iters; ++t0) {
            float* npx = (float*)malloc((size_t)(n_vert ? n_vert : 1) * sizeof(float));
            float* npy = (float*)malloc((size_t)(n_vert ? n_vert : 1) * sizeof(float));
            float* npz = (float*)malloc((size_t)(n_vert ? n_vert : 1) * sizeof(float));
            if (!npx || !npy || !npz) { free(npx); free(npy); free(npz); reg_free(&r); return DECIMATE_ERR_MEMORY; }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (i = 0; i < n_vert; ++i) {
                if (r.vf_len[i] == 0 || (pin && r.is_boundary[i])) {
                    npx[i] = r.px[i]; npy[i] = r.py[i]; npz[i] = r.pz[i];
                    continue;
                }
                /* Per-vertex tangent-plane centroid of the incident faces.
                 * We recompute each incident face's centroid and normal from
                 * the *current* positions of its three corners. */
                const int n = r.vf_len[i];
                const float px = r.px[i], py = r.py[i], pz = r.pz[i];
                float cx = 0.0f, cy = 0.0f, cz = 0.0f;
                float nx = 0.0f, ny = 0.0f, nz = 0.0f;
                int t;
                for (t = 0; t < n; ++t) {
                    const int fi = r.vfaces[i][t];
                    const float ax = r.px[triangles[3 * fi + 0]], ay = r.py[triangles[3 * fi + 0]], az = r.pz[triangles[3 * fi + 0]];
                    const float bx = r.px[triangles[3 * fi + 1]], by = r.py[triangles[3 * fi + 1]], bz = r.pz[triangles[3 * fi + 1]];
                    const float dx = r.px[triangles[3 * fi + 2]], dy = r.py[triangles[3 * fi + 2]], dz = r.pz[triangles[3 * fi + 2]];
                    cx += (ax + bx + dx) * (1.0f / 3.0f);
                    cy += (ay + by + dy) * (1.0f / 3.0f);
                    cz += (az + bz + dz) * (1.0f / 3.0f);
                    {
                        const float v0x = bx - ax, v0y = by - ay, v0z = bz - az;
                        const float v1x = dx - ax, v1y = dy - ay, v1z = dz - az;
                        nx += v0y * v1z - v0z * v1y;
                        ny += v0z * v1x - v0x * v1z;
                        nz += v0x * v1y - v0y * v1x;
                    }
                }
                cx /= (float)n; cy /= (float)n; cz /= (float)n;

                /* Optional Laplacian smoothing target: the mean of the
                 * vertex's unique neighbors (the non-i corner of each incident
                 * face, de-duplicated).  This is a classic low-pass "neighbor
                 * average" that relaxes spurious per-vertex jitter; for a
                 * manifold it lies in the same tangent-plane family as the
                 * centroid above, and it stays correct for non-manifold input. */
                float sx = px, sy = py, sz = pz;
                if (smooth > 0.0f) {
                    int cand[96]; int nc = 0, m;
                    #define ADD_NBR(w) do { if ((w) != i) { int dd = 0; \
                        for (m = 0; m < nc; ++m) if (cand[m] == (w)) { dd = 1; break; } \
                        if (!dd && nc < 96) cand[nc++] = (w); } } while (0)
                    for (t = 0; t < n; ++t) {
                        const int fi = r.vfaces[i][t];
                        ADD_NBR(triangles[3 * fi + 0]);
                        ADD_NBR(triangles[3 * fi + 1]);
                        ADD_NBR(triangles[3 * fi + 2]);
                    }
                    #undef ADD_NBR
                    sx = 0.0f; sy = 0.0f; sz = 0.0f;
                    for (m = 0; m < nc; ++m) { sx += r.px[cand[m]]; sy += r.py[cand[m]]; sz += r.pz[cand[m]]; }
                    if (nc > 0) { sx /= (float)nc; sy /= (float)nc; sz /= (float)nc; }
                    else { sx = px; sy = py; sz = pz; }
                }

                /* Normalise the accumulated normal (fallback to a safe axis
                 * if degenerate). */
                float nlen = sqrtf(nx * nx + ny * ny + nz * nz);
                if (nlen < 1e-12f) { nx = 0.0f; ny = 0.0f; nz = 1.0f; nlen = 1.0f; }
                else { nx /= nlen; ny /= nlen; nz /= nlen; }

                /* Blend the Lloyd (CVT) target with the Laplacian smoothing
                 * target, each projected onto the tangent plane, then apply
                 * the damped move.  smooth=0 recovers the pure Lloyd pass. */
                {
                    float ldx = cx - px, ldy = cy - py, ldz = cz - pz;
                    float ld = ldx * nx + ldy * ny + ldz * nz;
                    ldx -= ld * nx; ldy -= ld * ny; ldz -= ld * nz;
                    if (smooth > 0.0f) {
                        float sdx = sx - px, sdy = sy - py, sdz = sz - pz;
                        float sd = sdx * nx + sdy * ny + sdz * nz;
                        sdx -= sd * nx; sdy -= sd * ny; sdz -= sd * nz;
                        ldx = (1.0f - smooth) * ldx + smooth * sdx;
                        ldy = (1.0f - smooth) * ldy + smooth * sdy;
                        ldz = (1.0f - smooth) * ldz + smooth * sdz;
                    }
                    npx[i] = px + step * ldx;
                    npy[i] = py + step * ldy;
                    npz[i] = pz + step * ldz;
                }
            }

            /* Swap in the new positions. */
            {
                float* tx = r.px; r.px = npx; npx = tx;
                float* ty = r.py; r.py = npy; npy = ty;
                float* tz = r.pz; r.pz = npz; npz = tz;
            }
        }

        /* ---- Emit --------------------------------------------------- */
        {
            float* ov = (float*)malloc((size_t)(n_vert ? n_vert : 1) * 3 * sizeof(float));
            int32_t* ot = (int32_t*)malloc((size_t)(n_tri ? n_tri : 1) * 3 * sizeof(int32_t));
            if (!ov || !ot) { free(ov); free(ot); reg_free(&r); return DECIMATE_ERR_MEMORY; }
            for (i = 0; i < n_vert; ++i) {
                ov[3 * i + 0] = r.px[i];
                ov[3 * i + 1] = r.py[i];
                ov[3 * i + 2] = r.pz[i];
            }
            for (i = 0; i < n_tri; ++i) {
                ot[3 * i + 0] = triangles[3 * i + 0];
                ot[3 * i + 1] = triangles[3 * i + 1];
                ot[3 * i + 2] = triangles[3 * i + 2];
            }
            reg_free(&r);
            t_end = decimate_now_ns();
            out_result->vertices = ov;
            out_result->triangles = ot;
            out_result->n_vert_out = n_vert;
            out_result->n_tri_out = n_tri;
            out_result->elapsed_ns = t_end - t_start;
            out_result->device_used = "cpu";
            return DECIMATE_OK;
        }
    }
}
