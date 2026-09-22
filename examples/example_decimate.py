"""
Python example: decimate a UV-sphere in each of the three cost modes.

Mirrors ``examples/example_c.c``. Also prints a shape-regularity metric
(mean shape-quality and minimum triangle angle) so the shape quality of the
result is visible.

Requires the native library to be built and discoverable (set ``DECIMATE_LIB``
or ``DECIMATE_LIB_DIR`` if it is not on the default search path).

Run:
    python examples/example_decimate.py
"""
import numpy as np

from decimate import (
    decimate_mesh,
    regularise_mesh,
    Mode,
    Device,
    query,
    version,
)


def _shape_report(v, t):
    """Return (mean shape-quality in [0,1], min angle in degrees) for a mesh."""
    if len(t) == 0:
        return 0.0, 180.0
    a = v[t[:, 0]]; b = v[t[:, 1]]; c = v[t[:, 2]]
    e1 = np.sum((b - a) ** 2, axis=1)
    e2 = np.sum((c - b) ** 2, axis=1)
    e3 = np.sum((c - a) ** 2, axis=1)
    ssum = e1 + e2 + e3
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    sq = np.zeros_like(area)
    m = ssum > 1e-8
    sq[m] = 4.0 * 1.7320508075688772 * area[m] / ssum[m]
    l1 = np.sqrt(e1); l2 = np.sqrt(e2); l3 = np.sqrt(e3)
    ang = []
    lm = (l1 > 1e-9) & (l2 > 1e-9) & (l3 > 1e-9)
    if np.any(lm):
        cosv = np.clip(
            (np.sum((b - a) * (c - a), axis=1)[lm] / (l1[lm] * l3[lm])), -1.0, 1.0)
        ang = np.degrees(np.arccos(cosv))
        min_angle = float(ang.min())
    else:
        min_angle = 180.0
    return float(sq.mean()), min_angle


def uv_sphere(stacks: int = 24, slices: int = 24):
    v = []
    for i in range(stacks + 1):
        phi = np.pi * i / stacks
        for j in range(slices):
            theta = 2.0 * np.pi * j / slices
            v.append(
                [
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi),
                ]
            )
    v = np.asarray(v, dtype=np.float32)

    t = []
    for i in range(stacks):
        for j in range(slices):
            cur = i * slices + j
            cur_next = i * slices + (j + 1) % slices
            nxt = (i + 1) * slices + (j + 1) % slices
            nxt_next = (i + 1) * slices + j
            if i > 0:
                t.append([cur, cur_next, nxt_next])
            t.append([cur_next, nxt, nxt_next])
    t = np.asarray(t, dtype=np.int32)
    return v, t


def main() -> None:
    print(f"decimate {version()}  |  GPU available: {query()[0]}")

    v, t = uv_sphere()
    print(f"\nInput UV-sphere: {v.shape[0]} verts, {t.shape[0]} tris")

    modes = [
        ("NONE", Mode.NONE),
        ("PRESERVE_BOUNDARIES", Mode.PRESERVE_BOUNDARIES),
        ("PRESERVE_VOLUME", Mode.PRESERVE_VOLUME),
    ]

    for name, mode in modes:
        res = decimate_mesh(v, t, target_reduction=0.7, mode=mode, device=Device.AUTO)
        msq, mangle = _shape_report(res.vertices, res.triangles)
        print(
            f"  mode={name:<22} "
            f"-> {res.triangles.shape[0]:5d} tris, {res.vertices.shape[0]:5d} verts | "
            f"{res.elapsed_ns / 1e6:8.3f} ms on {res.device_used} | "
            f"meanSQ={msq:.3f} minAng={mangle:.1f} deg"
        )
        res.free()

    # Shape regularisation (Lloyd/CVT) applied after decimation.
    print("\nShape regularisation (Lloyd/CVT) applied after decimation:")
    dec = decimate_mesh(v, t, target_reduction=0.7, mode=Mode.NONE, device=Device.CPU)
    dv, dt = dec.copies()
    dec.free()
    msq0, ma0 = _shape_report(dv, dt)
    with regularise_mesh(dv, dt, iterations=8, step=0.8, pin_boundary=True) as reg:
        msq1, ma1 = _shape_report(reg.vertices, reg.triangles)
        n_tri = reg.triangles.shape[0]
        n_vert = reg.vertices.shape[0]
    print(
        f"  regularise             "
        f"-> {n_tri:5d} tris, {n_vert:5d} verts | "
        f"meanSQ={msq0:.4f} -> {msq1:.4f} (delta {msq1 - msq0:+.4f}) | "
        f"minAng={ma0:.1f} -> {ma1:.1f} deg"
    )


if __name__ == "__main__":
    main()
