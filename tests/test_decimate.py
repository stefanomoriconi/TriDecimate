"""
Pytest suite for the ``decimate`` ctypes wrapper.

These tests exercise the native shared library through the Python wrapper.
They are run under CTest via ``python -m pytest`` (see ``tests/CMakeLists.txt``)
or directly from a shell after the library has been built.

The tests are intentionally independent of any single mesh file: each mesh is
built in-process. GPU tests are skipped automatically when the library was not
built with CUDA or no device is present.
"""
import sys
import os
import platform

import numpy as np
import pytest

# Make the wrapper importable regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

import decimate  # noqa: E402
from decimate import (  # noqa: E402
    decimate_mesh,
    regularise_mesh,
    Mode,
    Device,
    Status,
    DecimateError,
    query,
    version,
)


# ---------------------------------------------------------------------------
#  Mesh builders (VALID meshes only -- never reuse the prototype's broken rig)
# ---------------------------------------------------------------------------
def make_cube() -> "tuple[np.ndarray, np.ndarray]":
    """Unit cube: 8 vertices, 12 triangles, all wound consistently."""
    v = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float32,
    )
    # 2 triangles per face, consistent outward winding.
    t = np.array(
        [
            # bottom (z=0)
            [0, 2, 1], [0, 3, 2],
            # top (z=1)
            [4, 5, 6], [4, 6, 7],
            # front (y=0)
            [0, 1, 5], [0, 5, 4],
            # back (y=1)
            [2, 3, 7], [2, 7, 6],
            # left (x=0)
            [0, 4, 7], [0, 7, 3],
            # right (x=1)
            [1, 2, 6], [1, 6, 5],
        ],
        dtype=np.int32,
    )
    return v, t


def make_uv_sphere(stacks: int = 8, slices: int = 8) -> "tuple[np.ndarray, np.ndarray]":
    """Small UV sphere, valid and closed."""
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
            # skip degenerate cap triangles at the poles
            if i > 0:
                t.append([cur, cur_next, nxt_next])
            t.append([cur_next, nxt, nxt_next])
    t = np.asarray(t, dtype=np.int32)
    return v, t


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _assert_valid(v, t, n_vert_in, n_tri_in):
    assert v.dtype == np.float32 and v.ndim == 2 and v.shape[1] == 3
    assert t.dtype == np.int32 and t.ndim == 2 and t.shape[1] == 3
    assert v.shape[0] > 0
    assert t.shape[0] >= 0
    if t.shape[0] > 0:
        assert int(t.min()) >= 0
        assert int(t.max()) < v.shape[0]
        assert v.shape[0] <= n_vert_in
    assert not np.isnan(v).any()


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------
def test_version():
    ver = version()
    assert isinstance(ver, str)
    assert len(ver) > 0


def test_query():
    gpu, threads = query()
    assert isinstance(gpu, bool)
    assert threads >= 1


@pytest.mark.parametrize(
    "mode", [Mode.NONE, Mode.PRESERVE_BOUNDARIES, Mode.PRESERVE_VOLUME]
)
def test_modes_run(mode):
    v, t = make_cube()
    res = decimate_mesh(v, t, target_reduction=0.5, mode=mode, device=Device.CPU)
    _assert_valid(res.vertices, res.triangles, v.shape[0], t.shape[0])
    # 50% reduction should roughly halve the face count (within tolerance).
    target = int(np.floor(t.shape[0] * (1.0 - 0.5)))
    assert abs(res.triangles.shape[0] - target) <= max(2, target // 4)
    res.free()


def test_zero_reduction_is_identity():
    v, t = make_cube()
    res = decimate_mesh(v, t, target_reduction=0.0, mode=Mode.NONE, device=Device.CPU)
    # No reduction requested -> face count is preserved.
    assert res.triangles.shape[0] == t.shape[0]
    res.free()


def test_determinism_cpu():
    v, t = make_uv_sphere()
    r1 = decimate_mesh(v, t, target_reduction=0.7, mode=Mode.PRESERVE_VOLUME, device=Device.CPU)
    r2 = decimate_mesh(v, t, target_reduction=0.7, mode=Mode.PRESERVE_VOLUME, device=Device.CPU)
    assert r1.triangles.shape == r2.triangles.shape
    assert np.array_equal(r1.triangles, r2.triangles)
    assert np.allclose(r1.vertices, r2.vertices)
    r1.free()
    r2.free()


@pytest.mark.skipif(not query()[0], reason="no CUDA backend available")
def test_gpu_matches_cpu():
    v, t = make_uv_sphere()
    ref = decimate_mesh(v, t, target_reduction=0.7, mode=Mode.NONE, device=Device.CPU)
    got = decimate_mesh(v, t, target_reduction=0.7, mode=Mode.NONE, device=Device.GPU)
    assert ref.triangles.shape == got.triangles.shape
    assert np.array_equal(ref.triangles, got.triangles)
    assert np.allclose(ref.vertices, got.vertices, atol=1e-5)
    ref.free()
    got.free()


def test_context_manager():
    v, t = make_cube()
    with decimate_mesh(v, t, target_reduction=0.5, mode=Mode.NONE, device=Device.CPU) as res:
        assert res.triangles.shape[0] > 0
    # after the block, the native memory must be freed (no exception expected)


def test_dtypes_are_coerced():
    """The wrapper coerces float64/int64 inputs to float32/int32 for the C ABI."""
    v, t = make_cube()
    v64 = v.astype(np.float64)
    t64 = t.astype(np.int64)
    res = decimate_mesh(v64, t64, target_reduction=0.5, mode=Mode.NONE, device=Device.CPU)
    _assert_valid(res.vertices, res.triangles, v.shape[0], t.shape[0])
    res.free()


def test_invalid_target_rejected():
    v, t = make_cube()
    with pytest.raises((DecimateError, ValueError)):
        decimate_mesh(v, t, target_reduction=-0.2, mode=Mode.NONE, device=Device.CPU)
    with pytest.raises((DecimateError, ValueError)):
        decimate_mesh(v, t, target_reduction=1.5, mode=Mode.NONE, device=Device.CPU)


def test_invalid_device_rejected():
    v, t = make_cube()
    with pytest.raises((DecimateError, ValueError)):
        decimate_mesh(v, t, target_reduction=0.5, mode=Mode.NONE, device=99)


# ---------------------------------------------------------------------------
#  Shape regularisation (Lloyd / CVT) pass
# ---------------------------------------------------------------------------
def make_skewed_quad() -> "tuple[np.ndarray, np.ndarray]":
    """A flat skewed quad: 4 boundary verts + 1 interior vert (index 4)."""
    v = np.array(
        [[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0], [1, 2, 0]],
        dtype=np.float32,
    )
    t = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int32)
    return v, t


def test_regularise_preserves_topology_and_boundary():
    v, t = make_skewed_quad()
    res = regularise_mesh(v, t, iterations=1, step=0.8)
    assert res.vertices.shape == v.shape
    assert res.triangles.shape == t.shape
    # Topology is unchanged.
    assert np.array_equal(res.triangles, t)
    # Boundary vertices (0..3) are pinned exactly.
    assert np.array_equal(res.vertices[:4], v[:4])
    # Interior vertex moves to p0 + 0.8*(0.8*(C-p0) + 0.2*(S-p0)),
    # C=(5/3,5/3,0), S=(2,3/2,0)  ->  (119/75, 128/75, 0).
    expected = np.array([119 / 75, 128 / 75, 0], dtype=np.float32)
    assert np.allclose(res.vertices[4], expected, atol=1e-4)
    res.free()


def test_regularise_is_deterministic():
    v, t = make_uv_sphere(stacks=10, slices=10)
    r1 = regularise_mesh(v, t, iterations=4, step=0.8)
    r2 = regularise_mesh(v, t, iterations=4, step=0.8)
    assert np.array_equal(r1.triangles, r2.triangles)
    assert np.allclose(r1.vertices, r2.vertices)
    r1.free()
    r2.free()


def _mean_sq(v, t):
    """Mean square-quality: 4*sqrt(3)*Area / (e1^2+e2^2+e3^2)."""
    a = v[t[:, 0]]
    b = v[t[:, 1]]
    c = v[t[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    e2 = (
        np.sum((b - a) ** 2, axis=1)
        + np.sum((c - b) ** 2, axis=1)
        + np.sum((a - c) ** 2, axis=1)
    )
    sq = 4.0 * np.sqrt(3.0) * area / np.maximum(e2, 1e-12)
    return float(np.mean(sq))


def test_regularise_improves_shape_quality():
    """Decimating a sphere produces slivers; the CVT pass should raise the
    mean square-quality (or at least not degrade it)."""
    v, t = make_uv_sphere(stacks=16, slices=16)
    dec = decimate_mesh(v, t, target_reduction=0.5, mode=Mode.NONE, device=Device.CPU)
    dv, dt = dec.copies()  # independent copies: safe to free dec immediately
    dec.free()

    sq_before = _mean_sq(dv, dt)
    with regularise_mesh(dv, dt, iterations=8, step=0.8) as reg:
        sq_after = _mean_sq(reg.vertices, reg.triangles)
    assert np.isfinite(sq_after)
    assert sq_after >= sq_before * 0.99  # no degradation; typically an improvement


def test_regularise_rejects_bad_input():
    v, t = make_cube()
    # Out-of-range index must be rejected.
    bad_t = t.copy()
    bad_t[0, 0] = v.shape[0]
    with pytest.raises((DecimateError, ValueError)):
        regularise_mesh(v, bad_t, iterations=1)
