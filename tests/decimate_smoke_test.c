/*
 * decimate_smoke_test.c
 *
 * Native C smoke test that validates the public C ABI of libdecimate.
 * Exercises the core entry point, option defaults, status strings, and the
 * result lifecycle (alloc/free) on a small, known mesh.
 */
#include "decimate.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A tiny 8-vertex cube (12 triangles). */
static const float CUBE_VERTS[] = {
    0,0,0, 1,0,0, 1,1,0, 0,1,0,
    0,0,1, 1,0,1, 1,1,1, 0,1,1
};
static const int32_t CUBE_TRIS[] = {
    0,1,2, 0,2,3,   /* bottom */
    4,6,5, 4,7,6,   /* top   */
    0,1,5, 0,5,4,   /* front */
    1,2,6, 1,6,5,   /* right */
    2,3,7, 2,7,6,   /* back  */
    3,0,4, 3,4,7    /* left  */
};
#define CUBE_NV 8
#define CUBE_NT 12

/* A skewed flat quad: 4 boundary vertices + 1 interior vertex (index 4).
 * All faces are coplanar (z=0) with +z normal.  The face-centroid mean is
 *   C = (5/3, 5/3, 0) and the neighbor (Laplacian) mean is S = (2, 3/2, 0).
 * One iteration (iterations=1, step=0.8, smooth=0.2) blends the Lloyd move
 * with the Laplacian move and sends the interior vertex
 *   p0 = (1, 2, 0)  ->  p0 + 0.8*(0.8*(C-p0) + 0.2*(S-p0)) = (119/75, 128/75, 0)
 *                        = (1.5867..., 1.7067..., 0). */
static const float QUAD_VERTS[] = {
    0, 0, 0,   /* 0 bottom-left   (boundary) */
    4, 0, 0,   /* 1 bottom-right  (boundary) */
    4, 3, 0,   /* 2 top-right     (boundary) */
    0, 3, 0,   /* 3 top-left      (boundary) */
    1, 2, 0    /* 4 interior      (free)     */
};
static const int32_t QUAD_TRIS[] = {
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
    3, 0, 4
};
#define QUAD_NV 5
#define QUAD_NT 4

static int float_eq(float a, float b)
{
    float d = a - b;
    if (d < 0.0f) d = -d;
    return d < 1e-4f;
}

static int run_regularise_case(void)
{
    decimate_regularise_options o = decimate_default_regularise_options();
    o.iterations = 1;   /* exactly one deterministic Lloyd+Laplacian step */

    decimate_result r;
    memset(&r, 0, sizeof(r));
    decimate_status st = decimate_regularise(QUAD_VERTS, QUAD_TRIS,
                                             QUAD_NV, QUAD_NT, &o, &r);
    if (st != DECIMATE_OK) {
        fprintf(stderr, "  [regularise-quad] FAILED: %s\n", decimate_status_str(st));
        return 1;
    }

    int fail = 0;
    if (r.n_vert_out != QUAD_NV || r.n_tri_out != QUAD_NT) {
        fprintf(stderr, "  [regularise-quad] size changed: %d verts / %d tris\n",
                r.n_vert_out, r.n_tri_out);
        fail = 1;
    }

    /* Boundary vertices (0..3) must be unchanged. */
    for (int i = 0; i < 4 && !fail; ++i) {
        for (int k = 0; k < 3; ++k) {
            if (!float_eq(r.vertices[3 * i + k], QUAD_VERTS[3 * i + k])) {
                fprintf(stderr, "  [regularise-quad] boundary vert %d moved to (%.4f,%.4f,%.4f)\n",
                        i, r.vertices[3*i], r.vertices[3*i+1], r.vertices[3*i+2]);
                fail = 1;
                break;
            }
        }
    }

    /* Interior vertex (4) moves via a Lloyd step blended with a Laplacian
     * neighbor-average (smooth=0.2, step=0.8):
     *   C = (5/3, 5/3, 0)  (face-centroid mean)
     *   S = (2, 3/2, 0)    (mean of neighbors 0..3)
     *   new = p0 + 0.8*[0.8*(C-p0) + 0.2*(S-p0)] = (119/75, 128/75, 0)
     *       = (1.5867, 1.7067, 0). */
    if (!fail) {
        if (!float_eq(r.vertices[12], 119.0f / 75.0f) ||
            !float_eq(r.vertices[13], 128.0f / 75.0f) ||
            !float_eq(r.vertices[14], 0.0f)) {
            fprintf(stderr, "  [regularise-quad] interior vertex at (%.4f,%.4f,%.4f), want (1.5867,1.7067,0)\n",
                    r.vertices[12], r.vertices[13], r.vertices[14]);
            fail = 1;
        }
    }

    decimate_result_free(&r);
    if (!fail) printf("  [regularise-quad] OK: interior -> (1.5867,1.7067,0), boundary fixed\n");
    return fail;
}

static int run_regularise_cube_case(void)
{
    decimate_regularise_options o = decimate_default_regularise_options();
    o.iterations = 2;

    decimate_result r;
    memset(&r, 0, sizeof(r));
    decimate_status st = decimate_regularise(CUBE_VERTS, CUBE_TRIS,
                                             CUBE_NV, CUBE_NT, &o, &r);
    if (st != DECIMATE_OK) {
        fprintf(stderr, "  [regularise-cube] FAILED: %s\n", decimate_status_str(st));
        return 1;
    }

    /* A cube is a *closed* mesh (every edge is shared by two faces), so it
     * has no boundary: all vertices are interior and free to move.  We only
     * check structural validity and that the (damped) moves keep every
     * vertex finite and close to the original unit cube. */
    int fail = 0;
    if (r.n_vert_out != CUBE_NV || r.n_tri_out != CUBE_NT) {
        fprintf(stderr, "  [regularise-cube] size changed: %d verts / %d tris\n",
                r.n_vert_out, r.n_tri_out);
        fail = 1;
    }
    if (!fail) {
        for (int i = 0; i < CUBE_NV && !fail; ++i) {
            for (int k = 0; k < 3; ++k) {
                float val = r.vertices[3 * i + k];
                if (!(val == val) || val < -1.0f || val > 2.0f) {
                    fprintf(stderr, "  [regularise-cube] vert %d.%d = %f out of range\n",
                            i, k, (double)val);
                    fail = 1;
                    break;
                }
            }
        }
    }

    decimate_result_free(&r);
    if (!fail) printf("  [regularise-cube] OK: closed mesh, sizes preserved, moves bounded\n");
    return fail;
}

static int run_case(decimate_mode mode, float target, const char* label)
{
    decimate_options o = decimate_default_options();
    o.mode = mode;
    o.target_reduction = target;
    o.device = DECIMATE_DEVICE_CPU;

    decimate_result r;
    memset(&r, 0, sizeof(r));
    decimate_status st = decimate_mesh(CUBE_VERTS, CUBE_TRIS,
                                       CUBE_NV, CUBE_NT, &o, &r);
    if (st != DECIMATE_OK) {
        fprintf(stderr, "  [%s] FAILED: %s\n", label, decimate_status_str(st));
        return 1;
    }

    int expected = (int)((float)CUBE_NT * (1.0f - target));
    printf("  [%s] in=%d tris -> out=%d tris, %d verts, device=%s (%lld ns)\n",
           label, CUBE_NT, r.n_tri_out, r.n_vert_out,
           r.device_used ? r.device_used : "?", (long long)r.elapsed_ns);

    /* Structural sanity: every output face references a valid vertex. */
    for (int i = 0; i < r.n_tri_out; ++i) {
        for (int k = 0; k < 3; ++k) {
            int idx = r.triangles[3 * i + k];
            if (idx < 0 || idx >= r.n_vert_out) {
                fprintf(stderr, "  [%s] BAD INDEX %d\n", label, idx);
                decimate_result_free(&r);
                return 1;
            }
        }
    }

    decimate_result_free(&r);
    return 0;
}

int main(void)
{
    int rc = 0;

    printf("decimate_smoke_test (version %s)\n", decimate_version());

    int32_t gpu = 0, threads = 0;
    decimate_status qs = decimate_query(&gpu, &threads);
    printf("  query: status=%s gpu=%d threads=%d\n",
           decimate_status_str(qs), gpu, threads);

    {
        decimate_options o = decimate_default_options();
        printf("  default options: mode=%d device=%d red=%.2f bpen=%.1f cgain=%.1f\n",
               o.mode, o.device, o.target_reduction,
               o.boundary_penalty, o.curvature_gain);
    }

    rc |= run_case(DECIMATE_MODE_NONE, 0.5f, "none");
    rc |= run_case(DECIMATE_MODE_PRESERVE_BOUNDARIES, 0.5f, "boundaries");
    rc |= run_case(DECIMATE_MODE_PRESERVE_VOLUME, 0.5f, "volume");
    rc |= run_case(DECIMATE_MODE_NONE, 0.0f, "none-0%");
    rc |= run_case(DECIMATE_MODE_NONE, 0.9f, "none-90%");

    /* Shape regularisation (Lloyd / CVT) pass. */
    rc |= run_regularise_case();
    rc |= run_regularise_cube_case();

    /* NULL-safety: free(NULL) must not crash. */
    decimate_result_free(NULL);

    if (rc == 0) {
        printf("SMOKE TEST PASSED\n");
        return 0;
    }
    printf("SMOKE TEST FAILED\n");
    return 1;
}
