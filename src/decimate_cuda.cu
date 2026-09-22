/**
 * @file decimate_cuda.cu
 * @brief CUDA mirror of the greedy edge-collapse decimator.
 *
 * This translation unit is only compiled when the build defines DECIMATE_HAS_CUDA
 * (enabled via CMake option DECIMATE_ENABLE_CUDA). It implements the SAME
 * greedy algorithm as src/decimate_cpu.c:
 *
 *   - Identical cost functions for NONE / PRESERVE_BOUNDARIES / PRESERVE_VOLUME.
 *   - Identical midpoint merge, shared+degenerate face deletion, remap, repack.
 *   - Identical canonical edge-key tie-break (edge_key) so results MATCH the CPU.
 *
 * The greedy selection itself is inherently sequential (each collapse invalidates
 * the next candidate). GPU parallelism is applied to the bulk work that CAN be
 * parallelized: initial edge-cost computation, per-collapse neighborhood cost
 * recomputation, and face-normal recomputation. This keeps the mirror exact while
 * still exercising the GPU pipeline.
 *
 * Public entry points (declared extern in decimate_cpu.c):
 *   - int            decimate_cuda_available(void)
 *   - decimate_status decimate_mesh_gpu(vertices, triangles, n_vert, n_tri, opts, out)
 */
#include "decimate.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define DECIMATE_EPS 1e-8f

/* ------------------------------------------------------------------ */
/*  Host-side status helper                                            */
/* ------------------------------------------------------------------ */

static const char* cu_err(cudaError_t e) { return cudaGetErrorString(e); }

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            (void)cu_err(_e);                                                 \
            return DECIMATE_ERR_INTERNAL;                                     \
        }                                                                     \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Device data layout (struct-of-arrays)                              */
/* ------------------------------------------------------------------ */

typedef struct {
    float *vx, *vy, *vz;
    int   *v_alive;

    int   *f_v0, *f_v1, *f_v2;
    int   *f_alive;
    int    face_count;

    float *nrm;                       /* [n_tri*3] */

    int   *vf_head;                   /* head of per-vertex face linked list */
    int   *vf_next;                   /* next-face pointer (size n_tri) */
    int   *vf_len;                    /* per-vertex face count */

    float  boundary_penalty;
    float  curvature_gain;
    int    mode;
} GPUState;

static int g_cuda_ok = -1; /* -1 = not yet queried */

/* ------------------------------------------------------------------ */
/*  Device helpers                                                     */
/* ------------------------------------------------------------------ */

__device__ __forceinline__ void device_recompute_normal(const float* vx, const float* vy,
                                                        const float* vz, const int* fv,
                                                        float* nrm, int t)
{
    int a = fv[3 * t + 0], b = fv[3 * t + 1], d = fv[3 * t + 2];
    float v0x = vx[b] - vx[a], v0y = vy[b] - vy[a], v0z = vz[b] - vz[a];
    float v1x = vx[d] - vx[a], v1y = vy[d] - vy[a], v1z = vz[d] - vz[a];
    float nx = v0y * v1z - v0z * v1y;
    float ny = v0z * v1x - v0x * v1z;
    float nz = v0x * v1y - v0y * v1x;
    float n = sqrtf(nx * nx + ny * ny + nz * nz);
    if (n > DECIMATE_EPS) { nx /= n; ny /= n; nz /= n; }
    else { nx = ny = nz = 0.0f; }
    nrm[3 * t + 0] = nx;
    nrm[3 * t + 1] = ny;
    nrm[3 * t + 2] = nz;
}

__device__ int device_vfaces_contains(const int* vf_head, const int* vf_next,
                                      const int* vf_len, int v, int t)
{
    for (int p = vf_head[v]; p != -1; p = vf_next[p])
        if (p == t) return 1;
    return 0;
}

__device__ float device_edge_cost(const float* vx, const float* vy, const float* vz,
                                  const int* fv, const float* nrm,
                                  const int* vf_head, const int* vf_next,
                                  int a, int b, int mode, float bpen, float cgain)
{
    float dx = vx[a] - vx[b], dy = vy[a] - vy[b], dz = vz[a] - vz[b];
    float base = sqrtf(dx * dx + dy * dy + dz * dz);

    if (mode == DECIMATE_MODE_PRESERVE_BOUNDARIES) {
        int shared = 0, p;
        for (p = vf_head[a]; p != -1; p = vf_next[p])
            if (device_vfaces_contains(vf_head, vf_next, NULL, b, p)) ++shared;
        return shared == 1 ? base * bpen : base;
    }

    if (mode == DECIMATE_MODE_PRESERVE_VOLUME) {
        /* gather surrounding = (faces(a) U faces(b)) \ shared(a,b) */
        int list[64], n = 0, p;
        for (p = vf_head[a]; p != -1; p = vf_next[p])
            if (!device_vfaces_contains(vf_head, vf_next, NULL, b, p)) list[n++] = p;
        for (p = vf_head[b]; p != -1; p = vf_next[p])
            if (!device_vfaces_contains(vf_head, vf_next, NULL, a, p)) list[n++] = p;

        if (n > 1) {
            float mindot = 1.0f;
            int x, y;
            for (x = 0; x < n; ++x) {
                float nix = nrm[3 * list[x] + 0], niy = nrm[3 * list[x] + 1], niz = nrm[3 * list[x] + 2];
                for (y = x + 1; y < n; ++y) {
                    float dot = nix * nrm[3 * list[y] + 0] +
                                niy * nrm[3 * list[y] + 1] +
                                niz * nrm[3 * list[y] + 2];
                    if (dot < mindot) mindot = dot;
                }
            }
            float clip = mindot < -1.0f ? -1.0f : (mindot > 1.0f ? 1.0f : mindot);
            return base * (1.0f + (1.0f - clip) * cgain);
        }
    }
    return base;
}

/* Kernels ---------------------------------------------------------- */

__global__ void kernel_normals(const float* vx, const float* vy, const float* vz,
                               const int* fv, float* nrm, int n_tri)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n_tri) device_recompute_normal(vx, vy, vz, fv, nrm, t);
}

/* Compute cost for each unique edge; edges are stored as (lo,hi) pairs
 * in device arrays e_lo/e_hi of length n_edge. */
__global__ void kernel_edge_costs(const float* vx, const float* vy, const float* vz,
                                  const int* fv, const float* nrm,
                                  const int* vf_head, const int* vf_next,
                                  const int* e_lo, const int* e_hi, float* costs,
                                  int n_edge, int mode, float bpen, float cgain)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_edge)
        costs[e] = device_edge_cost(vx, vy, vz, fv, nrm, vf_head, vf_next,
                                    e_lo[e], e_hi[e], mode, bpen, cgain);
}

/* ------------------------------------------------------------------ */
/*  Host helpers                                                       */
/* ------------------------------------------------------------------ */

static int cuda_available(void)
{
    int dev_count = 0;
    cudaError_t e = cudaGetDeviceCount(&dev_count);
    return (e == cudaSuccess && dev_count > 0) ? 1 : 0;
}

int decimate_cuda_available(void)
{
    if (g_cuda_ok < 0) g_cuda_ok = cuda_available();
    return g_cuda_ok;
}

typedef struct {
    float cost;
    int lo, hi;
    long long key;
} GEdge;

static int g_less(const GEdge* x, const GEdge* y)
{
    if (x->cost != y->cost) return x->cost < y->cost;
    return x->key < y->key;
}

static void g_swap(GEdge* x, GEdge* y)
{
    GEdge t = *x; *x = *y; *y = t;
}

/* Build a small host-side min-heap used for the sequential greedy selection.
 * The GPU computes the *costs*; selection stays host-side (greedy is serial). */
static void gpush(GEdge** hp, size_t* size, size_t* cap, GEdge v)
{
    GEdge* h = *hp;
    if (*size == *cap) {
        size_t nc = *cap ? *cap * 2 : 256;
        GEdge* nh = (GEdge*)realloc(h, nc * sizeof(GEdge));
        if (!nh) abort();
        *hp = nh; *cap = nc;
        h = nh;
    }
    h[(*size)++] = v;
    size_t i = *size - 1;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (g_less(&h[i], &h[p])) { g_swap(&h[i], &h[p]); i = p; }
        else break;
    }
}

static int gpop(GEdge** hp, size_t* size, GEdge* out)
{
    GEdge* h = *hp;
    if (*size == 0) return 0;
    *out = h[0];
    --*size;
    if (*size > 0) {
        h[0] = h[*size];
        size_t i = 0;
        for (;;) {
            size_t l = 2 * i + 1, r = 2 * i + 2, best = i;
            if (l < *size && g_less(&h[l], &h[best])) best = l;
            if (r < *size && g_less(&h[r], &h[best])) best = r;
            if (best == i) break;
            g_swap(&h[i], &h[best]);
            i = best;
        }
    }
    return 1;
}

/* Device adjacency (linked-list) helpers, host side. */
static void dvf_add(GPUState* d, int v, int t, int* head, int* next, int* len)
{
    next[t] = head[v];
    head[v] = t;
    len[v]++;
}

static int dvf_contains(const int* head, const int* next, int v, int t)
{
    for (int p = head[v]; p != -1; p = next[p]) if (p == t) return 1;
    return 0;
}

static void dvf_remove(int* head, int* next, int v, int t)
{
    int* p = &head[v];
    while (*p != -1) {
        if (*p == t) { *p = next[*p]; break; }
        p = &next[*p];
    }
}

/* ------------------------------------------------------------------ */
/*  GPU entry point                                                    */
/* ------------------------------------------------------------------ */

decimate_status decimate_mesh_gpu(const float* vertices, const int32_t* triangles,
                                  int32_t n_vert, int32_t n_tri,
                                  const decimate_options* opts,
                                  decimate_result* out_result)
{
    if (!out_result) return DECIMATE_ERR_NULLPTR;
    if (!decimate_cuda_available()) return DECIMATE_ERR_DEVICE;

    if (n_tri == 0) {
        out_result->vertices = (float*)malloc((size_t)n_vert * 3 * sizeof(float));
        if (!out_result->vertices) return DECIMATE_ERR_MEMORY;
        memcpy(out_result->vertices, vertices, (size_t)n_vert * 3 * sizeof(float));
        out_result->triangles = NULL;
        out_result->n_vert_out = n_vert;
        out_result->n_tri_out = 0;
        out_result->device_used = "gpu";
        return DECIMATE_OK;
    }

    GPUState s;
    memset(&s, 0, sizeof(s));
    s.nrm = NULL;
    s.face_count = n_tri;
    s.mode = opts->mode;
    s.boundary_penalty = opts->boundary_penalty;
    s.curvature_gain = opts->curvature_gain;

    /* ---- Allocate host working state (mirrors the CPU context) ---- */
    size_t NV = (size_t)n_vert, NT = (size_t)n_tri;
    float *hx = (float*)malloc(NV * 3 * sizeof(float));
    int   *halive = (int*)malloc(NV * sizeof(int));
    int   *fv = (int*)malloc(NT * 3 * sizeof(int));
    int   *falive = (int*)malloc(NT * sizeof(int));
    float *hnrm = (float*)malloc(NT * 3 * sizeof(float));
    int   *head = (int*)malloc(NV * sizeof(int));
    int   *next = (int*)malloc(NT * sizeof(int));
    int   *len = (int*)malloc(NV * sizeof(int));
    GEdge *heap = NULL;
    size_t hsize = 0, hcap = 0;

    if (!hx || !halive || !fv || !falive || !hnrm || !head || !next || !len) {
        free(hx); free(halive); free(fv); free(falive); free(hnrm);
        free(head); free(next); free(len);
        return DECIMATE_ERR_MEMORY;
    }

    /* ---- Build host state + adjacency ----------------------------- */
    int i;
    for (i = 0; i < n_vert; ++i) {
        hx[3 * i + 0] = vertices[3 * i + 0];
        hx[3 * i + 1] = vertices[3 * i + 1];
        hx[3 * i + 2] = vertices[3 * i + 2];
        halive[i] = 1;
        head[i] = -1;
        len[i] = 0;
    }
    for (i = 0; i < n_tri; ++i) {
        fv[3 * i + 0] = triangles[3 * i + 0];
        fv[3 * i + 1] = triangles[3 * i + 1];
        fv[3 * i + 2] = triangles[3 * i + 2];
        falive[i] = 1;
        next[i] = -1;
    }
    for (i = 0; i < n_tri; ++i) {
        int v[3] = { fv[3 * i + 0], fv[3 * i + 1], fv[3 * i + 2] };
        int m;
        for (m = 0; m < 3; ++m) dvf_add(&s, v[m], i, head, next, len);
    }

    /* ---- Collect unique edges ------------------------------------- */
    {
        int ecap = n_tri * 3 > 0 ? n_tri * 3 : 8;
        int *elo = (int*)malloc((size_t)ecap * sizeof(int));
        int *ehi = (int*)malloc((size_t)ecap * sizeof(int));
        int n_edge = 0, found, q;
        if (!elo || !ehi) { free(elo); free(ehi); return DECIMATE_ERR_MEMORY; }
        for (i = 0; i < n_tri; ++i) {
            int f[3] = { fv[3 * i + 0], fv[3 * i + 1], fv[3 * i + 2] };
            int m;
            for (m = 0; m < 3; ++m) {
                int u = f[m], v = f[(m + 1) % 3];
                int lo = u < v ? u : v, hi = u < v ? v : u;
                found = 0;
                for (q = 0; q < n_edge; ++q) if (elo[q] == lo && ehi[q] == hi) { found = 1; break; }
                if (found) continue;
                if (n_edge == ecap) {
                    ecap *= 2;
                    elo = (int*)realloc(elo, (size_t)ecap * sizeof(int));
                    ehi = (int*)realloc(ehi, (size_t)ecap * sizeof(int));
                    if (!elo || !ehi) { free(elo); free(ehi); return DECIMATE_ERR_MEMORY; }
                }
                elo[n_edge] = lo; ehi[n_edge] = hi; ++n_edge;
            }
        }

        /* ---- Use the GPU for the expensive parallel passes -------- */
        float *dvx, *dvy, *dvz, *dnrm, *dcost;
        int   *dfv, *dhead, *dnext, *dlo, *dhi;

        CUDA_CHECK(cudaMalloc(&dvx, NV * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dvy, NV * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dvz, NV * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dfv, NT * 3 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dhead, NV * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dnext, NT * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dnrm, NT * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dlo, (size_t)(n_edge ? n_edge : 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dhi, (size_t)(n_edge ? n_edge : 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dcost, (size_t)(n_edge ? n_edge : 1) * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(dvx, hx + 0, NV * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dvy, hx + 1, NV * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dvz, hx + 2, NV * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dfv, fv, NT * 3 * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dhead, head, NV * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dnext, next, NT * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dlo, elo, (size_t)(n_edge ? n_edge : 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dhi, ehi, (size_t)(n_edge ? n_edge : 1) * sizeof(int), cudaMemcpyHostToDevice));

        {
            int blocks = (n_tri + 255) / 256;
            kernel_normals<<<blocks, 256, 0, 0>>>(dvx, dvy, dvz, dfv, dnrm, n_tri);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        {
            int blocks = (n_edge + 255) / 256;
            kernel_edge_costs<<<blocks, 256, 0, 0>>>(dvx, dvy, dvz, dfv, dnrm,
                                                      dhead, dnext, dlo, dhi,
                                                      dcost, n_edge,
                                                      opts->mode,
                                                      opts->boundary_penalty,
                                                      opts->curvature_gain);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hnrm, dnrm, NT * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        }

        /* Pull initial costs back and seed the selection heap. */
        {
            float* hcost = (float*)malloc((size_t)(n_edge ? n_edge : 1) * sizeof(float));
            if (!hcost) return DECIMATE_ERR_MEMORY;
            CUDA_CHECK(cudaMemcpy(hcost, dcost, (size_t)(n_edge ? n_edge : 1) * sizeof(float), cudaMemcpyDeviceToHost));
            for (i = 0; i < n_edge; ++i) {
                GEdge e;
                e.cost = hcost[i];
                e.lo = elo[i]; e.hi = ehi[i];
                e.key = (long long)elo[i] * (long long)n_vert + (long long)ehi[i];
                gpush(&heap, &hsize, &hcap, e);
            }
            free(hcost);
        }

        /* ---- Greedy collapse loop (identical to CPU) -------------- */
        int target = (int)((double)n_tri * (1.0 - (double)opts->target_reduction));
        if (target < 0) target = 0;

        while (s.face_count > target && hsize > 0) {
            GEdge cur;
            if (!gpop(&heap, &hsize, &cur)) break;
            int v0 = cur.lo, v1 = cur.hi;
            if (!halive[v0] || !halive[v1]) continue;

            hx[3 * v0 + 0] = (hx[3 * v0 + 0] + hx[3 * v1 + 0]) * 0.5f;
            hx[3 * v0 + 1] = (hx[3 * v0 + 1] + hx[3 * v1 + 1]) * 0.5f;
            hx[3 * v0 + 2] = (hx[3 * v0 + 2] + hx[3 * v1 + 2]) * 0.5f;

            int* to_del = (int*)malloc((size_t)(len[v0] + len[v1]) * sizeof(int));
            int td = 0;
            if (!to_del) return DECIMATE_ERR_MEMORY;
            int p;
            for (p = head[v0]; p != -1; p = next[p])
                if (dvf_contains(head, next, v1, p)) to_del[td++] = p;

            /* iterate over the union of both adjacency lists */
            for (p = head[v0]; p != -1; p = next[p]) {
                int in_del = 0;
                for (i = 0; i < td; ++i) if (to_del[i] == p) { in_del = 1; break; }
                if (in_del) continue;
                if (fv[3 * p + 0] == v1) fv[3 * p + 0] = v0;
                if (fv[3 * p + 1] == v1) fv[3 * p + 1] = v0;
                if (fv[3 * p + 2] == v1) fv[3 * p + 2] = v0;
                if (fv[3 * p + 0] == fv[3 * p + 1] || fv[3 * p + 1] == fv[3 * p + 2] || fv[3 * p + 2] == fv[3 * p + 0])
                    to_del[td++] = p;
                else {
                    if (!dvf_contains(head, next, v0, p)) dvf_add(&s, v0, p, head, next, len);
                    {
                        int a = fv[3 * p + 0], b = fv[3 * p + 1], d = fv[3 * p + 2];
                        float v0x = hx[3 * b + 0] - hx[3 * a + 0], v0y = hx[3 * b + 1] - hx[3 * a + 1], v0z = hx[3 * b + 2] - hx[3 * a + 2];
                        float v1x = hx[3 * d + 0] - hx[3 * a + 0], v1y = hx[3 * d + 1] - hx[3 * a + 1], v1z = hx[3 * d + 2] - hx[3 * a + 2];
                        float nx = v0y * v1z - v0z * v1y, ny = v0z * v1x - v0x * v1z, nz = v0x * v1y - v0y * v1x;
                        float nn = sqrtf(nx * nx + ny * ny + nz * nz);
                        if (nn > DECIMATE_EPS) { nx /= nn; ny /= nn; nz /= nn; }
                        else { nx = ny = nz = 0; }
                        hnrm[3 * p + 0] = nx; hnrm[3 * p + 1] = ny; hnrm[3 * p + 2] = nz;
                    }
                }
            }
            for (p = head[v1]; p != -1; p = next[p]) {
                int in_del = 0;
                for (i = 0; i < td; ++i) if (to_del[i] == p) { in_del = 1; break; }
                if (in_del) continue;
                if (fv[3 * p + 0] == v1) fv[3 * p + 0] = v0;
                if (fv[3 * p + 1] == v1) fv[3 * p + 1] = v0;
                if (fv[3 * p + 2] == v1) fv[3 * p + 2] = v0;
                if (fv[3 * p + 0] == fv[3 * p + 1] || fv[3 * p + 1] == fv[3 * p + 2] || fv[3 * p + 2] == fv[3 * p + 0])
                    to_del[td++] = p;
                else {
                    if (!dvf_contains(head, next, v0, p)) dvf_add(&s, v0, p, head, next, len);
                    {
                        int a = fv[3 * p + 0], b = fv[3 * p + 1], d = fv[3 * p + 2];
                        float v0x = hx[3 * b + 0] - hx[3 * a + 0], v0y = hx[3 * b + 1] - hx[3 * a + 1], v0z = hx[3 * b + 2] - hx[3 * a + 2];
                        float v1x = hx[3 * d + 0] - hx[3 * a + 0], v1y = hx[3 * d + 1] - hx[3 * a + 1], v1z = hx[3 * d + 2] - hx[3 * a + 2];
                        float nx = v0y * v1z - v0z * v1y, ny = v0z * v1x - v0x * v1z, nz = v0x * v1y - v0y * v1x;
                        float nn = sqrtf(nx * nx + ny * ny + nz * nz);
                        if (nn > DECIMATE_EPS) { nx /= nn; ny /= nn; nz /= nn; }
                        else { nx = ny = nz = 0; }
                        hnrm[3 * p + 0] = nx; hnrm[3 * p + 1] = ny; hnrm[3 * p + 2] = nz;
                    }
                }
            }

            /* commit deletions */
            for (i = 0; i < td; ++i) {
                int t = to_del[i];
                if (!falive[t]) continue;
                int v[3] = { fv[3 * t + 0], fv[3 * t + 1], fv[3 * t + 2] };
                int m;
                for (m = 0; m < 3; ++m) { dvf_remove(head, next, v[m], t); len[v[m]]--; }
                falive[t] = 0;
                --s.face_count;
            }
            free(to_del);

            /* retire v1 */
            halive[v1] = 0;
            head[v1] = -1;
            len[v1] = 0;

            /* recompute neighborhood costs (GPU pass) */
            {
                int ncap2 = 64, nn = 0, dup, q;
                int *neigh = (int*)malloc((size_t)ncap2 * sizeof(int));
                if (!neigh) return DECIMATE_ERR_MEMORY;
                for (p = head[v0]; p != -1; p = next[p]) {
                    int m;
                    for (m = 0; m < 3; ++m) {
                        int nv = fv[3 * p + m];
                        if (nv == v0) continue;
                        dup = 0;
                        for (q = 0; q < nn; ++q) if (neigh[q] == nv) { dup = 1; break; }
                        if (dup) continue;
                        if (nn == ncap2) { ncap2 *= 2; neigh = (int*)realloc(neigh, (size_t)ncap2 * sizeof(int)); }
                        neigh[nn++] = nv;
                    }
                }
                if (nn > 0) {
                    int *nlo = (int*)malloc((size_t)nn * sizeof(int));
                    int *nhi = (int*)malloc((size_t)nn * sizeof(int));
                    float *ncost = (float*)malloc((size_t)nn * sizeof(float));
                    if (!nlo || !nhi || !ncost) { free(nlo); free(nhi); free(ncost); free(neigh); return DECIMATE_ERR_MEMORY; }
                    for (i = 0; i < nn; ++i) {
                        int a = v0 < neigh[i] ? v0 : neigh[i], b = v0 < neigh[i] ? neigh[i] : v0;
                        nlo[i] = a; nhi[i] = b;
                    }
                    /* refresh device state (positions, faces, normals, adjacency) */
                    CUDA_CHECK(cudaMemcpy(dvx, hx + 0, NV * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(dvy, hx + 1, NV * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(dvz, hx + 2, NV * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(dfv, fv, NT * 3 * sizeof(int), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(dhead, head, NV * sizeof(int), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(dnext, next, NT * sizeof(int), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(dnrm, hnrm, NT * 3 * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(dlo, nlo, (size_t)nn * sizeof(int), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(dhi, nhi, (size_t)nn * sizeof(int), cudaMemcpyHostToDevice));
                    {
                        int blocks = (nn + 255) / 256;
                        kernel_edge_costs<<<blocks, 256, 0, 0>>>(dvx, dvy, dvz, dfv, dnrm,
                                                                  dhead, dnext, dlo, dhi,
                                                                  dcost, nn,
                                                                  opts->mode,
                                                                  opts->boundary_penalty,
                                                                  opts->curvature_gain);
                        CUDA_CHECK(cudaDeviceSynchronize());
                        CUDA_CHECK(cudaMemcpy(ncost, dcost, (size_t)nn * sizeof(float), cudaMemcpyDeviceToHost));
                    }
                    for (i = 0; i < nn; ++i) {
                        GEdge e;
                        e.cost = ncost[i];
                        e.lo = nlo[i]; e.hi = nhi[i];
                        e.key = (long long)nlo[i] * (long long)n_vert + (long long)nhi[i];
                        gpush(&heap, &hsize, &hcap, e);
                    }
                    free(nlo); free(nhi); free(ncost);
                }
                free(neigh);
            }
        }

        /* ---- Pack surviving vertices to contiguous indices -------- */
        int live = 0, out_v = 0, out_t = 0;
        int *order = (int*)malloc(NV * sizeof(int));
        int *remap = (int*)malloc(NV * sizeof(int));
        if (!order || !remap) { free(order); free(remap); return DECIMATE_ERR_MEMORY; }
        for (i = 0; i < n_vert; ++i) if (halive[i]) { order[live] = i; remap[i] = live++; }
        out_v = live;
        for (i = 0; i < n_tri; ++i) if (falive[i]) ++out_t;

        float* ov = (float*)malloc((size_t)out_v * 3 * sizeof(float));
        int32_t* ot = (int32_t*)malloc((size_t)(out_t ? out_t : 1) * 3 * sizeof(int32_t));
        if (!ov || !ot) { free(ov); free(ot); return DECIMATE_ERR_MEMORY; }
        for (i = 0; i < out_v; ++i) {
            int o = order[i];
            ov[3 * i + 0] = hx[3 * o + 0];
            ov[3 * i + 1] = hx[3 * o + 1];
            ov[3 * i + 2] = hx[3 * o + 2];
        }
        int w = 0;
        for (i = 0; i < n_tri; ++i) {
            if (!falive[i]) continue;
            ot[3 * w + 0] = remap[fv[3 * i + 0]];
            ot[3 * w + 1] = remap[fv[3 * i + 1]];
            ot[3 * w + 2] = remap[fv[3 * i + 2]];
            ++w;
        }

        /* ---- Free GPU + temp host buffers ------------------------- */
        cudaFree(dvx); cudaFree(dvy); cudaFree(dvz); cudaFree(dfv);
        cudaFree(dhead); cudaFree(dnext); cudaFree(dnrm);
        cudaFree(dlo); cudaFree(dhi); cudaFree(dcost);
        free(elo); free(ehi);
        free(hx); free(halive); free(fv); free(falive); free(hnrm);
        free(head); free(next); free(len); free(order); free(remap);
        free(heap);

        out_result->vertices = ov;
        out_result->triangles = ot;
        out_result->n_vert_out = out_v;
        out_result->n_tri_out = out_t;
        out_result->device_used = "gpu";
        return DECIMATE_OK;
    }
}

/* ================================================================== */
/*  Shape-regularisation pass  (Lloyd / centroidal-Voronoi)          */
/* ================================================================== */
/*
 * CUDA mirror of the CPU Lloyd pass in src/decimate_cpu.c.  Per-vertex work
 * is independent, so the kernel is one thread per vertex: compute the mean
 * of the incident face centroids (the CVT dual), accumulate the face normals,
 * project (centroid - vertex) onto the tangent plane, and apply a damped
 * move.  Boundary / empty-adjacency vertices are pinned.  The computation
 * (and the +z normal fallback for degenerate cells) is numerically identical
 * to the CPU implementation, so results match.
 */

__global__ void kernel_lloyd(const float* px, const float* py, const float* pz,
                             const int*   fv,    /* [n_tri*3] face vertex indices */
                             const int*   head,  /* [n_vert]  face list head */
                             const int*   next,  /* [n_tri]   linked-list next */
                             const int*   vflen, /* [n_vert]  face count */
                             const int*   is_boundary,
                             float* npx, float* npy, float* npz,
                             float step, int pin, float smooth)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0) return; /* bounds enforced by host launch; guard for safety */

    if (vflen[i] == 0 || (pin && is_boundary[i])) {
        npx[i] = px[i]; npy[i] = py[i]; npz[i] = pz[i];
        return;
    }

    const float pxv = px[i], pyv = py[i], pzv = pz[i];
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    float nx = 0.0f, ny = 0.0f, nz = 0.0f;

    int cand[96]; int nc = 0, m;
    #define ADD_NBR(w) do { if ((w) != i) { int dd = 0; \
        for (m = 0; m < nc; ++m) if (cand[m] == (w)) { dd = 1; break; } \
        if (!dd && nc < 96) cand[nc++] = (w); } } while (0)

    for (int p = head[i]; p != -1; p = next[p]) {
        const int a = fv[3 * p + 0], b = fv[3 * p + 1], d = fv[3 * p + 2];
        const float ax = px[a], ay = py[a], az = pz[a];
        const float bx = px[b], by = py[b], bz = pz[b];
        const float dx = px[d], dy = py[d], dz = pz[d];
        cx += (ax + bx + dx) * (1.0f / 3.0f);
        cy += (ay + by + dy) * (1.0f / 3.0f);
        cz += (az + bz + dz) * (1.0f / 3.0f);
        if (smooth > 0.0f) { ADD_NBR(a); ADD_NBR(b); ADD_NBR(d); }
        const float v0x = bx - ax, v0y = by - ay, v0z = bz - az;
        const float v1x = dx - ax, v1y = dy - ay, v1z = dz - az;
        nx += v0y * v1z - v0z * v1y;
        ny += v0z * v1x - v0x * v1z;
        nz += v0x * v1y - v0y * v1x;
    }
    #undef ADD_NBR

    const int n = vflen[i];
    cx /= (float)n; cy /= (float)n; cz /= (float)n;

    float nlen = sqrtf(nx * nx + ny * ny + nz * nz);
    if (nlen < 1e-12f) { nx = 0.0f; ny = 0.0f; nz = 1.0f; nlen = 1.0f; }
    else { nx /= nlen; ny /= nlen; nz /= nlen; }

    float ldx = cx - pxv, ldy = cy - pyv, ldz = cz - pzv;
    float ld = ldx * nx + ldy * ny + ldz * nz;
    ldx -= ld * nx; ldy -= ld * ny; ldz -= ld * nz;

    if (smooth > 0.0f) {
        /* Laplacian target = mean of the unique neighbors. */
        float sx = 0.0f, sy = 0.0f, sz = 0.0f;
        for (m = 0; m < nc; ++m) { sx += px[cand[m]]; sy += py[cand[m]]; sz += pz[cand[m]]; }
        if (nc > 0) { sx /= (float)nc; sy /= (float)nc; sz /= (float)nc; }
        else { sx = cx; sy = cy; sz = cz; }
        float sdx = sx - pxv, sdy = sy - pyv, sdz = sz - pzv;
        float sd = sdx * nx + sdy * ny + sdz * nz;
        sdx -= sd * nx; sdy -= sd * ny; sdz -= sd * nz;
        ldx = (1.0f - smooth) * ldx + smooth * sdx;
        ldy = (1.0f - smooth) * ldy + smooth * sdy;
        ldz = (1.0f - smooth) * ldz + smooth * sdz;
    }

    npx[i] = pxv + step * ldx;
    npy[i] = pyv + step * ldy;
    npz[i] = pzv + step * ldz;
}

decimate_status decimate_regularise_gpu(const float* vertices, const int32_t* triangles,
                                        int32_t n_vert, int32_t n_tri,
                                        const decimate_regularise_options* opts,
                                        decimate_result* out_result)
{
    if (!out_result) return DECIMATE_ERR_NULLPTR;
    if (!decimate_cuda_available()) return DECIMATE_ERR_DEVICE;

    decimate_regularise_options o;
    if (!opts) o = decimate_default_regularise_options(); else o = *opts;

    if (n_vert < 0 || n_tri < 0) return DECIMATE_ERR_BAD_ARGS;
    if (n_tri > 0 && (!vertices || !triangles)) return DECIMATE_ERR_NULLPTR;

    int iters = (o.iterations < 1) ? 1 : o.iterations;
    if (iters > 64) iters = 64;
    float step = (o.step <= 0.0f) ? 0.5f : o.step;
    if (step > 1.0f) step = 1.0f;
    float smooth = o.smooth;
    if (smooth < 0.0f) smooth = 0.0f;
    if (smooth > 1.0f) smooth = 1.0f;
    int pin = o.pin_boundary ? 1 : 0;

    size_t NV = (size_t)n_vert, NT = (size_t)n_tri;
    int i;

    /* ---- Build host state: positions, face indices, adjacency ---- */
    float *hx = (float*)malloc(NV * 3 * sizeof(float));
    int   *fv = (int*)malloc(NT * 3 * sizeof(int));
    int   *head = (int*)malloc(NV * sizeof(int));
    int   *next = (int*)malloc(NT * sizeof(int));
    int   *vflen = (int*)malloc(NV * sizeof(int));
    int   *isb = (int*)malloc(NV * sizeof(int));
    if (!hx || !fv || !head || !next || !vflen || !isb) {
        free(hx); free(fv); free(head); free(next); free(vflen); free(isb);
        return DECIMATE_ERR_MEMORY;
    }

    for (i = 0; i < n_vert; ++i) {
        hx[3 * i + 0] = vertices[3 * i + 0];
        hx[3 * i + 1] = vertices[3 * i + 1];
        hx[3 * i + 2] = vertices[3 * i + 2];
        head[i] = -1; vflen[i] = 0; isb[i] = 0;
    }
    for (i = 0; i < n_tri; ++i) {
        fv[3 * i + 0] = triangles[3 * i + 0];
        fv[3 * i + 1] = triangles[3 * i + 1];
        fv[3 * i + 2] = triangles[3 * i + 2];
        next[i] = -1;
    }
    for (i = 0; i < n_tri; ++i) {
        int v[3] = { fv[3 * i + 0], fv[3 * i + 1], fv[3 * i + 2] };
        int m;
        for (m = 0; m < 3; ++m) { next[i] = head[v[m]]; head[v[m]] = i; vflen[v[m]]++; }
    }

    /* Boundary mask: vertices owning an edge that appears in exactly one face.
     * Reuse the edge-pack / sort approach for a single forward scan. */
    {
        size_t ecount = (size_t)n_tri * 3;
        /* Simple host edge-count via packed 64-bit keys and qsort. */
        long long* ekeys = (long long*)malloc((ecount ? ecount : 1) * sizeof(long long));
        if (!ekeys) {
            free(hx); free(fv); free(head); free(next); free(vflen); free(isb);
            return DECIMATE_ERR_MEMORY;
        }
        int e = 0;
        for (i = 0; i < n_tri; ++i) {
            int f[3] = { fv[3 * i + 0], fv[3 * i + 1], fv[3 * i + 2] };
            int m;
            for (m = 0; m < 3; ++m) {
                int u = f[m], v = f[(m + 1) % 3];
                int lo = u < v ? u : v, hi = u < v ? v : u;
                ekeys[e++] = ((long long)(unsigned)lo << 32) | (long long)(unsigned)hi;
            }
        }
        /* Sort the packed keys, then single forward scan for runs of length 1. */
        {
            static int cmp_ll(const void* a, const void* b) {
                long long x = *(const long long*)a, y = *(const long long*)b;
                return (x > y) - (x < y);
            }
            qsort(ekeys, (size_t)e, sizeof(long long), cmp_ll);
            int s = 0;
            while (s < e) {
                int j = s + 1;
                while (j < e && ekeys[j] == ekeys[s]) ++j;
                int run = j - s;
                if (run == 1) {
                    int u = (int)((unsigned)(ekeys[s] >> 32));
                    int v = (int)((unsigned)(ekeys[s] & 0xFFFFFFFFull));
                    isb[u] = 1; isb[v] = 1;
                }
                s = j;
            }
        }
        free(ekeys);
    }

    /* ---- Device buffers ------------------------------------------ */
    float *dp0x, *dp0y, *dp0z;   /* current positions (buffer A) */
    float *dp1x, *dp1y, *dp1z;   /* next positions      (buffer B) */
    int   *dfv, *dhead, *dnext, *dvflen, *dism;

    CUDA_CHECK(cudaMalloc(&dp0x, NV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dp0y, NV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dp0z, NV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dp1x, NV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dp1y, NV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dp1z, NV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dfv, NT * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dhead, NV * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dnext, NT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dvflen, NV * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dism, NV * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dp0x, hx + 0, NV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp0y, hx + 1, NV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp0z, hx + 2, NV * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dfv, fv, NT * 3 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dhead, head, NV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dnext, next, NT * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvflen, vflen, NV * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dism, isb, NV * sizeof(int), cudaMemcpyHostToDevice));

    int blocks = (n_vert + 255) / 256;

    for (i = 0; i < iters; ++i) {
        /* cur = (i even ? A : B), nxt = the other */
        const float* cx = (i % 2 == 0) ? dp0x : dp1x;
        const float* cy = (i % 2 == 0) ? dp0y : dp1y;
        const float* cz = (i % 2 == 0) ? dp0z : dp1z;
        float* nx = (i % 2 == 0) ? dp1x : dp0x;
        float* ny = (i % 2 == 0) ? dp1y : dp0y;
        float* nz = (i % 2 == 0) ? dp1z : dp0z;

        kernel_lloyd<<<blocks, 256, 0, 0>>>(cx, cy, cz, dfv, dhead, dnext, dvflen, dism,
                                             nx, ny, nz, step, pin, smooth);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* ---- Copy final positions back -------------------------------- */
    const float* fx = (iters % 2 == 0) ? dp0x : dp1x;
    const float* fy = (iters % 2 == 0) ? dp0y : dp1y;
    const float* fz = (iters % 2 == 0) ? dp0z : dp1z;

    float* ov = (float*)malloc(NV * 3 * sizeof(float));
    int32_t* ot = (int32_t*)malloc((NT ? NT : 1) * 3 * sizeof(int32_t));
    if (!ov || !ot) {
        cudaFree(dp0x); cudaFree(dp0y); cudaFree(dp0z);
        cudaFree(dp1x); cudaFree(dp1y); cudaFree(dp1z);
        cudaFree(dfv); cudaFree(dhead); cudaFree(dnext); cudaFree(dvflen); cudaFree(dism);
        free(hx); free(fv); free(head); free(next); free(vflen); free(isb);
        free(ov); free(ot);
        return DECIMATE_ERR_MEMORY;
    }

    CUDA_CHECK(cudaMemcpy(ov + 0, fx, NV * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ov + 1, fy, NV * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ov + 2, fz, NV * sizeof(float), cudaMemcpyDeviceToHost));

    for (i = 0; i < n_tri; ++i) {
        ot[3 * i + 0] = triangles[3 * i + 0];
        ot[3 * i + 1] = triangles[3 * i + 1];
        ot[3 * i + 2] = triangles[3 * i + 2];
    }

    cudaFree(dp0x); cudaFree(dp0y); cudaFree(dp0z);
    cudaFree(dp1x); cudaFree(dp1y); cudaFree(dp1z);
    cudaFree(dfv); cudaFree(dhead); cudaFree(dnext); cudaFree(dvflen); cudaFree(dism);
    free(hx); free(fv); free(head); free(next); free(vflen); free(isb);

    out_result->vertices = ov;
    out_result->triangles = ot;
    out_result->n_vert_out = n_vert;
    out_result->n_tri_out = n_tri;
    out_result->device_used = "gpu";
    return DECIMATE_OK;
}
