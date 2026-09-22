/**
 * @file decimate.h
 * @brief Public C API for fast, multi-platform triangle-mesh decimation.
 *
 * This library implements a greedy edge-collapse mesh simplifier that mirrors
 * the reference Python prototype (Euclidean edge length + mode-specific
 * penalties). It provides:
 *
 *   - A high-performance CPU implementation parallelized with OpenMP.
 *   - An equivalent CUDA implementation for NVIDIA GPUs (optional build).
 *
 * The C ABI below is intentionally C89/C99 friendly and stable so it can be
 * consumed directly with `ctypes` (Python), C, C++, Rust, Go, etc.
 *
 * All array inputs are row-major, contiguous, little-endian.
 *   vertices : float32[N][3]
 *   triangles: int32 [M][3]
 */
#ifndef DECIMATE_H
#define DECIMATE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Export macro                                                       */
/* ------------------------------------------------------------------ */
/*
 * On Windows shared builds the public symbols are exported explicitly.
 * On other platforms (GNU ld / Mach-O) symbols are exported by default,
 * so the macro expands to nothing there.
 */
#if defined(_WIN32) || defined(_WIN64)
  #if defined(DECIMATE_BUILD_SHARED) || defined(DECIMATE_EXPORTS)
    #define DECIMATE_API __declspec(dllexport)
  #else
    #define DECIMATE_API __declspec(dllimport)
  #endif
#else
  #define DECIMATE_API
#endif

/* ------------------------------------------------------------------ */
/*  Enumerations                                                       */
/* ------------------------------------------------------------------ */

/** Decimation quality / cost mode. Mirrors the Python `mode` argument. */
typedef enum {
    DECIMATE_MODE_NONE               = 0, /**< Plain Euclidean edge length. */
    DECIMATE_MODE_PRESERVE_BOUNDARIES = 1, /**< Penalize boundary collapses. */
    DECIMATE_MODE_PRESERVE_VOLUME    = 2  /**< Penalize curvature (normal deviation). */
} decimate_mode;

/** Target compute device. */
typedef enum {
    DECIMATE_DEVICE_AUTO = 0, /**< Use GPU if available and built, else CPU. */
    DECIMATE_DEVICE_CPU  = 1, /**< Force CPU (OpenMP). */
    DECIMATE_DEVICE_GPU  = 2  /**< Force CUDA; error if unavailable. */
} decimate_device;

/* ------------------------------------------------------------------ */
/*  Status codes                                                       */
/* ------------------------------------------------------------------ */

typedef enum {
    DECIMATE_OK                  = 0,
    DECIMATE_ERR_NULLPTR         = -1, /**< A required pointer was NULL. */
    DECIMATE_ERR_BAD_ARGS        = -2, /**< Invalid dimension / count. */
    DECIMATE_ERR_MODE            = -3, /**< Unknown decimate_mode value. */
    DECIMATE_ERR_MEMORY          = -4, /**< Host allocation failure. */
    DECIMATE_ERR_DEVICE          = -5, /**< Requested device unavailable. */
    DECIMATE_ERR_INPUT           = -6, /**< Degenerate / inconsistent mesh. */
    DECIMATE_ERR_TARGET          = -7, /**< target_reduction out of range. */
    DECIMATE_ERR_INTERNAL        = -8
} decimate_status;

/* ------------------------------------------------------------------ */
/*  Options                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    decimate_mode   mode;            /**< Cost / quality mode.              */
    decimate_device device;          /**< Which device to run on.           */
    float           target_reduction;/**< Faces to remove, in [0.0, 1.0).  */
    int32_t         num_threads;     /**< OpenMP threads; <=0 => hardware.  */
    float           boundary_penalty;/**< Scale for boundary penalty (def 15). */
    float           curvature_gain;  /**< Scale for curvature penalty (def 5). */
    uint64_t        seed;            /**< Reserved (deterministic tiebreak). */
} decimate_options;

/** Return a copy of the default options (mode=NONE, AUTO, threads=hw). */
DECIMATE_API decimate_options decimate_default_options(void);

/* ------------------------------------------------------------------ */
/*  Regularisation pass (Lloyd / centroidal-Voronoi)                   */
/* ------------------------------------------------------------------ */

/**
 * Options for the post-decimation shape-regularisation pass.
 *
 * The pass is a topology-preserving optimisation: it keeps the same vertices,
 * faces, and boundary, and only moves *interior* vertices so that the
 * surrounding triangles become more regular (better size uniformity and
 * aspect ratio). It is the standard Lloyd / centroidal-Voronoi iteration
 * (Du-Faber-Gunzburger), which in 2D converges to a regular (near-equilateral)
 * tiling while preserving total area up to a small, controllable error.
 *
 * A small optional Laplacian (neighbor-averaging) smoothing term is blended
 * into the move, which relaxes spurious per-vertex jitter so the overall
 * lattice is smoother as well as better shaped. It is a weighted sum of the
 * vertex's connected neighbors, kept in the tangent plane like the Lloyd term.
 */
typedef struct {
    int32_t  iterations;        /**< Lloyd iterations to run (def 6, clamped >=1). */
    float    step;              /**< Damping in (0,1]; 1 = full move (def 0.8). */
    int32_t  pin_boundary;      /**< 1 = keep boundary vertices fixed (def 1). */
    int32_t  preserve_area;     /**< 1 = damped move keeps total area ~constant (def 1). */
    float    smooth;            /**< Mild Laplacian smoothing weight in [0,1] (def 0.2). */
    int32_t  num_threads;       /**< OpenMP threads; <=0 => hardware. */
    uint64_t seed;              /**< Reserved (deterministic). */
} decimate_regularise_options;

/** Return a copy of the default regularise options. */
DECIMATE_API decimate_regularise_options decimate_default_regularise_options(void);

/* ------------------------------------------------------------------ */
/*  Result / output descriptor                                         */
/* ------------------------------------------------------------------ */

/**
 * Output buffer. The library allocates `vertices` (float32) and
 * `triangles` (int32) here. Call `decimate_result_free` to release.
 */
typedef struct {
    float*  vertices;    /**< [n_vert_out * 3] float32                    */
    int32_t* triangles;  /**< [n_tri_out * 3]  int32                      */
    int32_t  n_vert_out; /**< Number of output vertices                   */
    int32_t  n_tri_out;  /**< Number of output triangles                  */
    int64_t  elapsed_ns; /**< Wall-clock execution time in nanoseconds    */
    const char* device_used; /**< "cpu" or "gpu"                         */
} decimate_result;

/* ------------------------------------------------------------------ */
/*  Core API                                                           */
/* ------------------------------------------------------------------ */

/**
 * Decimate a triangle mesh.
 *
 * @param vertices    Input vertices, row-major [n_vert][3] float32.
 * @param triangles   Input triangles, row-major [n_tri][3] int32.
 * @param n_vert      Number of input vertices (>= 0).
 * @param n_tri       Number of input triangles (>= 0).
 * @param opts        Options (see decimate_default_options).
 * @param out_result  On success, points to an allocated decimate_result.
 *                    Caller frees with decimate_result_free.
 * @return            DECIMATE_OK on success, or a DECIMATE_ERR_* code.
 */
DECIMATE_API decimate_status decimate_mesh(const float* vertices,
                              const int32_t* triangles,
                              int32_t n_vert,
                              int32_t n_tri,
                              const decimate_options* opts,
                              decimate_result* out_result);

/**
 * Post-decimation shape-regularisation pass (Lloyd / centroidal-Voronoi).
 *
 * Topology is preserved: the returned mesh has the same number of vertices,
 * the same faces (identical index array), and an identical boundary. Only the
 * coordinates of interior vertices are moved toward the centroid of their
 * local (tangent-plane) Voronoi cell, which makes the surrounding triangles
 * more uniform in size and aspect ratio while keeping total area (and, up to
 * second order, volume) nearly constant.
 *
 * This is the recommended final pass after decimate_mesh() to relieve the
 * "irregular triangles" artefact that greedy edge-collapse can leave behind.
 *
 * @param vertices    input float32[N][3]
 * @param triangles   input int32[M][3] (returned unchanged in the result)
 * @param n_vert      number of input vertices (>= 1)
 * @param n_tri       number of input triangles (>= 1)
 * @param opts        regularise options, or NULL for defaults
 * @param out_result  on success, allocated decimate_result (free with
 *                    decimate_result_free)
 * @return            DECIMATE_OK on success, or a DECIMATE_ERR_* code.
 */
DECIMATE_API decimate_status decimate_regularise(const float* vertices,
                                    const int32_t* triangles,
                                    int32_t n_vert,
                                    int32_t n_tri,
                                    const decimate_regularise_options* opts,
                                    decimate_result* out_result);

/** Release the buffers owned by a decimate_result (safe to call twice / on NULL). */
DECIMATE_API void decimate_result_free(decimate_result* result);

/** Human readable string for a status code (never NULL). */
DECIMATE_API const char* decimate_status_str(decimate_status s);

/**
 * Query backend capability without allocating a mesh.
 * @param gpu_available out: 1 if a CUDA device was detected & built, else 0.
 * @param cpu_threads   out: recommended OpenMP thread count (>=1).
 */
DECIMATE_API decimate_status decimate_query(int32_t* gpu_available, int32_t* cpu_threads);

/** Library version string (e.g. "1.3.0"). Never NULL. */
DECIMATE_API const char* decimate_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DECIMATE_H */
