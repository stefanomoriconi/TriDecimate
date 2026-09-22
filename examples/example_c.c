/*
 * example_c.c
 *
 * Minimal end-to-end example of the decimate C API: build a small UV-sphere,
 * decimate it by 70% in each mode, and print before/after stats plus timing.
 * Also reports a shape-regularity metric (mean shape-quality and min angle)
 * for each mode so the shape quality of the result is visible.
 *
 * Build (after configuring CMake):
 *   cmake --build build --target example_c
 * Run:
 *   ./build/examples/example_c
 */
#include "decimate.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Build a UV sphere (stacks latitude rings + 2 poles). */
static int build_uv_sphere(float** verts, int32_t** tris, int stacks, int slices)
{
    int nv = (stacks + 1) * (slices + 1);
    float* v = (float*)malloc(sizeof(float) * 3 * nv);
    if (!v) return -1;

    int i = 0;
    for (int lat = 0; lat <= stacks; ++lat) {
        float theta = (float)lat / stacks * (float)M_PI;
        float y = cosf(theta);
        float r = sinf(theta);
        for (int lon = 0; lon <= slices; ++lon) {
            float phi = (float)lon / slices * 2.0f * (float)M_PI;
            v[3 * i]     = r * cosf(phi);
            v[3 * i + 1] = y;
            v[3 * i + 2] = r * sinf(phi);
            i++;
        }
    }

    int nt = stacks * slices * 2;
    int32_t* t = (int32_t*)malloc(sizeof(int32_t) * 3 * nt);
    if (!t) { free(v); return -1; }

    int k = 0;
    for (int lat = 0; lat < stacks; ++lat) {
        for (int lon = 0; lon < slices; ++lon) {
            int a  = lat * (slices + 1) + lon;
            int b  = a + 1;
            int c  = a + (slices + 1);
            int d  = c + 1;

            if (lat == 0) {           /* top pole */
                t[k++] = b; t[k++] = a; t[k++] = d;
            } else if (lat == stacks - 1) {  /* bottom pole */
                t[k++] = a; t[k++] = b; t[k++] = c;
            } else {
                t[k++] = a; t[k++] = b; t[k++] = d;
                t[k++] = a; t[k++] = d; t[k++] = c;
            }
        }
    }

    *verts = v;
    *tris = t;
    return nt;
}

/* Compute per-triangle shape quality SQ = 4*sqrt(3)*Area / (e1^2+e2^2+e3^2)
 * in [0,1] (1 = equilateral), plus the minimum triangle angle in degrees.
 * Both are computed on the decimated output. Returns 0 on success. */
static int shape_report(const float* v, const int32_t* t, int nt,
                        float* mean_sq, float* min_angle_deg)
{
    if (nt <= 0) { *mean_sq = 0.0f; *min_angle_deg = 180.0f; return 0; }
    double sum_sq = 0.0;
    float min_ang = 180.0f;
    for (int i = 0; i < nt; ++i) {
        const float* a = &v[3 * t[3 * i + 0]];
        const float* b = &v[3 * t[3 * i + 1]];
        const float* c = &v[3 * t[3 * i + 2]];
        float ux = b[0]-a[0], uy = b[1]-a[1], uz = b[2]-a[2];
        float wx = c[0]-a[0], wy = c[1]-a[1], wz = c[2]-a[2];
        float cx = uy*wz - uz*wy, cy = uz*wx - ux*wz, cz = ux*wy - uy*wx;
        float area = 0.5f * sqrtf(cx*cx + cy*cy + cz*cz);
        float e1 = (float)((b[0]-a[0])*(b[0]-a[0]) + (b[1]-a[1])*(b[1]-a[1]) + (b[2]-a[2])*(b[2]-a[2]));
        float e2 = (float)((c[0]-b[0])*(c[0]-b[0]) + (c[1]-b[1])*(c[1]-b[1]) + (c[2]-b[2])*(c[2]-b[2]));
        float e3 = (float)((c[0]-a[0])*(c[0]-a[0]) + (c[1]-a[1])*(c[1]-a[1]) + (c[2]-a[2])*(c[2]-a[2]));
        float sum = e1 + e2 + e3;
        if (sum > 1e-8f) sum_sq += (double)(4.0f * 1.7320508075688772f * area / sum);
        /* angles via dot products */
        float l1 = sqrtf(e1), l2 = sqrtf(e2), l3 = sqrtf(e3);
        if (l1 > 1e-9f && l2 > 1e-9f) {
            float dot = (b[0]-a[0])*(c[0]-a[0]) + (b[1]-a[1])*(c[1]-a[1]) + (b[2]-a[2])*(c[2]-a[2]);
            float cosv = dot / (l1 * l3);
            cosv = cosv > 1.0f ? 1.0f : (cosv < -1.0f ? -1.0f : cosv);
            float ang = (float)(acos((double)cosv) * 57.29577951308232);
            if (ang < min_ang) min_ang = ang;
        }
    }
    *mean_sq = (float)(sum_sq / (double)nt);
    *min_angle_deg = min_ang;
    return 0;
}

/* Run decimation in `mode` and print stats + shape report. */
static void run(decimate_mode mode, const char* label, float red,
                const float* v, const int32_t* t, int nv, int nt)
{
    decimate_options o = decimate_default_options();
    o.mode = mode;
    o.target_reduction = red;
    o.device = DECIMATE_DEVICE_AUTO;

    decimate_result r;
    decimate_status st = decimate_mesh(v, t, nv, nt, &o, &r);
    if (st != DECIMATE_OK) {
        printf("  %-12s FAILED: %s\n", label, decimate_status_str(st));
        return;
    }
    float msq = 0.0f, mangle = 0.0f;
    shape_report(r.vertices, r.triangles, r.n_tri_out, &msq, &mangle);
    printf("  %-12s %6d -> %5d tris | %5d -> %4d verts | %10lld ns | %s | meanSQ=%.3f minAng=%.1f deg\n",
           label, nt, r.n_tri_out, nv, r.n_vert_out,
           (long long)r.elapsed_ns, r.device_used ? r.device_used : "?",
           msq, mangle);
    decimate_result_free(&r);
}

/* Decimate (NONE) then apply the Lloyd/CVT shape-regularisation pass,
 * reporting the shape-quality gain from the regularisation. */
static void run_regularise_demo(float red, int iterations,
                                const float* v, const int32_t* t, int nv, int nt)
{
    decimate_options o = decimate_default_options();
    o.mode = DECIMATE_MODE_NONE;
    o.target_reduction = red;
    o.device = DECIMATE_DEVICE_CPU;

    decimate_result r;
    if (decimate_mesh(v, t, nv, nt, &o, &r) != DECIMATE_OK) {
        printf("  %-12s FAILED to decimate\n", "regularise");
        return;
    }

    float msq0 = 0.0f, ma0 = 0.0f;
    shape_report(r.vertices, r.triangles, r.n_tri_out, &msq0, &ma0);

    decimate_regularise_options ro = decimate_default_regularise_options();
    ro.iterations = iterations;
    ro.step = 0.8f;
    ro.pin_boundary = 1;

    decimate_result rr;
    decimate_status st = decimate_regularise(r.vertices, r.triangles,
                                             r.n_vert_out, r.n_tri_out, &ro, &rr);
    if (st != DECIMATE_OK) {
        printf("  %-12s FAILED: %s\n", "regularise", decimate_status_str(st));
        decimate_result_free(&r);
        return;
    }

    float msq1 = 0.0f, ma1 = 0.0f;
    shape_report(rr.vertices, rr.triangles, rr.n_tri_out, &msq1, &ma1);

    printf("  %-12s %6d -> %5d tris | %5d -> %4d verts | %10lld ns | %s\n",
           "regularise", r.n_tri_out, rr.n_tri_out, r.n_vert_out, rr.n_vert_out,
           (long long)rr.elapsed_ns, rr.device_used ? rr.device_used : "?");
    printf("             meanSQ %.4f -> %.4f  (delta %+0.4f)   minAng %.2f -> %.2f deg\n",
           msq0, msq1, msq1 - msq0, ma0, ma1);

    decimate_result_free(&rr);
    decimate_result_free(&r);
}

int main(void)
{
    float* v = NULL;
    int32_t* t = NULL;
    int stacks = 24, slices = 24;
    int nt = build_uv_sphere(&v, &t, stacks, slices);
    if (nt < 0) { fprintf(stderr, "alloc failed\n"); return 1; }
    int nv = (stacks + 1) * (slices + 1);

    printf("UV sphere: %d verts, %d tris (target_reduction = 0.7)\n", nv, nt);
    printf("decimate version %s\n\n", decimate_version());

    int32_t gpu = 0, threads = 0;
    decimate_query(&gpu, &threads);
    printf("GPU available: %s | CPU threads: %d\n\n", gpu ? "yes" : "no", threads);

    run(DECIMATE_MODE_NONE, "none", 0.7f, v, t, nv, nt);
    run(DECIMATE_MODE_PRESERVE_BOUNDARIES, "boundaries", 0.7f, v, t, nv, nt);
    run(DECIMATE_MODE_PRESERVE_VOLUME, "volume", 0.7f, v, t, nv, nt);

    printf("\nShape regularisation (Lloyd/CVT) applied after decimation:\n");
    run_regularise_demo(0.7f, 8, v, t, nv, nt);

    free(v);
    free(t);
    return 0;
}
