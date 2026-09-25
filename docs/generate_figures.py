"""
Generate the before/after decimation and regularisation figures used in the
top-level README. Run against a built library:

    DECIMATE_LIB_DIR=../build_cpu python docs/generate_figures.py

Uses the Stanford Bunny (docs/sample_meshes/bunny.obj -- the classic Stanford
Computer Graphics Laboratory scanning-repository test mesh, widely
redistributed for research/testing use) as a recognizable, realistic input,
run through the exact same decimate_mesh/regularise_mesh API exercised by
tests/test_decimate.py -- not cherry-picked demo data.
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

from decimate import decimate_mesh, regularise_mesh, Mode, Device  # noqa: E402

OUT_DIR = os.path.join(_HERE, "images")
BUNNY_OBJ = os.path.join(_HERE, "sample_meshes", "bunny.obj")
os.makedirs(OUT_DIR, exist_ok=True)


def load_obj(path):
    verts, tris = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
                tris.append(idx)
    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.int32)


def plot_mesh(ax, v, t, title, color):
    tris = v[t]
    coll = Poly3DCollection(tris, facecolor=color, edgecolor="k", linewidths=0.2, alpha=0.9)
    ax.add_collection3d(coll)
    ax.set_title(f"{title}\n{len(v)} verts, {len(t)} tris", fontsize=10, pad=14)
    ax.set_box_aspect((1, 1, 1))
    lo, hi = v.min(axis=0), v.max(axis=0)
    ctr, half = (lo + hi) / 2.0, (hi - lo).max() / 2.0
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.view_init(elev=15, azim=110)
    ax.set_axis_off()


def main():
    v, t = load_obj(BUNNY_OBJ)

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
    plot_mesh(ax1, v, t, "Input (Stanford Bunny)", "#4C72B0")
    plot_mesh(ax2, dv, dt, "Decimated (target_reduction=0.85)", "#DD8452")
    plot_mesh(ax3, rv, rt, "+ Shape regularisation (Lloyd/CVT)", "#55A868")
    fig.subplots_adjust(top=0.85, wspace=0.05)
    fig.savefig(os.path.join(OUT_DIR, "decimate_before_after.png"), dpi=140, bbox_inches="tight")
    print("wrote", os.path.join(OUT_DIR, "decimate_before_after.png"))


if __name__ == "__main__":
    main()
