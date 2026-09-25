"""
Generate the before/after decimation and regularisation figures used in the
top-level README. Run against a built library:

    DECIMATE_LIB_DIR=../build_cpu python docs/generate_figures.py

Uses only meshes/parameters already exercised by the test suite
(tests/test_decimate.py's make_uv_sphere) so the figures are a faithful,
reproducible illustration of real, tested behavior -- not cherry-picked demo
data.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "python"))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from decimate import decimate_mesh, regularise_mesh, Mode, Device  # noqa: E402
from test_decimate import make_uv_sphere  # noqa: E402

OUT_DIR = os.path.join(_HERE, "images")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_mesh(ax, v, t, title, color):
    tris = v[t]
    coll = Poly3DCollection(tris, facecolor=color, edgecolor="k", linewidths=0.2, alpha=0.9)
    ax.add_collection3d(coll)
    ax.set_title(f"{title}\n{len(v)} verts, {len(t)} tris", fontsize=10, pad=14)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev=20, azim=35)
    ax.set_axis_off()


def main():
    v, t = make_uv_sphere(stacks=24, slices=24)

    dec = decimate_mesh(v, t, target_reduction=0.85, mode=Mode.NONE,
                         device=Device.CPU, regularise=False)
    dv, dt = dec.copies()
    dec.free()

    with regularise_mesh(dv, dt, iterations=8, step=0.8) as reg:
        rv, rt = reg.vertices.copy(), reg.triangles.copy()

    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    plot_mesh(ax1, v, t, "Input (UV sphere)", "#4C72B0")
    plot_mesh(ax2, dv, dt, "Decimated (target_reduction=0.85)", "#DD8452")
    plot_mesh(ax3, rv, rt, "+ Shape regularisation (Lloyd/CVT)", "#55A868")
    fig.subplots_adjust(top=0.85, wspace=0.05)
    fig.savefig(os.path.join(OUT_DIR, "decimate_before_after.png"), dpi=140, bbox_inches="tight")
    print("wrote", os.path.join(OUT_DIR, "decimate_before_after.png"))


if __name__ == "__main__":
    main()
