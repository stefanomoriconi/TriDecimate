"""
decimate
========

High-performance triangle-mesh decimation with a stable C ABI, exposed to
Python via :mod:`ctypes`.

The heavy lifting is done by a compiled shared library (``decimate_tri_mesh``)
built from the C source in ``src/``. It implements a deterministic greedy
edge-collapse algorithm with three cost modes:

    * ``MODE_NONE``                 -- plain edge-length cost
    * ``MODE_PRESERVE_BOUNDARIES``  -- penalise boundary edges
    * ``MODE_PRESERVE_VOLUME``      -- penalise curvature / high-mindot regions

Two backends may be available:

    * CPU (OpenMP) -- always present, deterministic
    * GPU (CUDA)   -- optional, built only when the library was compiled with
                      CUDA; it mirrors the CPU result exactly

This wrapper is the recommended way to call the library from Python. It
mirrors the original prototype's ``decimate_mesh`` signature, so existing
call sites keep working.

Example
-------
>>> import numpy as np
>>> from decimate import decimate_mesh, Mode
>>> v = np.zeros((1000, 3), dtype=np.float32)
>>> t = np.zeros((2000, 3), dtype=np.int32)
>>> res = decimate_mesh(v, t, target_reduction=0.6, mode=Mode.NONE)
>>> res.vertices.shape, res.triangles.shape
((400, 3), (800, 3))
"""
from __future__ import annotations

import ctypes as _ct
import os
import platform
from typing import Optional, Tuple

import numpy as np
import numpy.ctypeslib as _ctypeslib

__all__ = [
    "decimate_mesh",
    "regularise_mesh",
    "Result",
    "Mode",
    "Device",
    "Status",
    "DecimateError",
    "query",
    "version",
]

__version__ = "1.3.0"


# ---------------------------------------------------------------------------
#  Constants (kept in sync with include/decimate.h)
# ---------------------------------------------------------------------------
class Mode:
    """Cost modes for the decimation algorithm."""
    NONE = 0
    PRESERVE_BOUNDARIES = 1
    PRESERVE_VOLUME = 2


class Device:
    """Compute-device selection."""
    AUTO = 0
    CPU = 1
    GPU = 2


class Status:
    """Return-status codes (mirror of ``decimate_status``)."""
    OK = 0
    ERR_NULLPTR = -1
    ERR_BAD_ARGS = -2
    ERR_MODE = -3
    ERR_MEMORY = -4
    ERR_DEVICE = -5
    ERR_INPUT = -6
    ERR_TARGET = -7
    ERR_INTERNAL = -8

    _NAMES = {
        0: "OK",
        -1: "ERR_NULLPTR",
        -2: "ERR_BAD_ARGS",
        -3: "ERR_MODE",
        -4: "ERR_MEMORY",
        -5: "ERR_DEVICE",
        -6: "ERR_INPUT",
        -7: "ERR_TARGET",
        -8: "ERR_INTERNAL",
    }

    @classmethod
    def name(cls, code: int) -> str:
        return cls._NAMES.get(code, f"UNKNOWN({code})")


class DecimateError(RuntimeError):
    """Raised when the native library returns a non-OK status."""

    def __init__(self, status: int, message: Optional[str] = None):
        self.status = status
        super().__init__(message or f"decimate failed with status {Status.name(status)}")


# ---------------------------------------------------------------------------
#  Library loading
# ---------------------------------------------------------------------------
def _candidate_filenames() -> list:
    system = platform.system()
    if system == "Windows":
        return ["decimate_tri_mesh.dll", "decimate.dll"]
    if system == "Darwin":
        return ["libdecimate_tri_mesh.dylib", "libdecimate.dylib"]
    return ["libdecimate_tri_mesh.so", "libdecimate.so"]


def _candidate_dirs() -> list:
    dirs = []

    explicit = os.environ.get("DECIMATE_LIB") or os.environ.get("DECIMATE_LIB_DIR")
    if explicit:
        if os.path.isdir(explicit):
            dirs.append(explicit)
        else:
            dirs.append(os.path.dirname(explicit))

    # 1. Next to this package (…/python/decimate.py -> …/build or …/)
    here = os.path.dirname(os.path.abspath(__file__))
    dirs.append(here)
    parent = os.path.dirname(here)
    dirs.append(parent)
    for sub in ("build", "build/lib", "build/Release", "dist", "decimate.libs"):
        p = os.path.join(parent, sub)
        if os.path.isdir(p):
            dirs.append(p)

    # 2. CMake install locations
    for base in (
        r"C:\Program Files\decimate\bin",
        r"C:\Program Files\decimate\lib",
        "/usr/local/lib",
        "/usr/local/bin",
        "/usr/lib",
        "/usr/lib/x86_64-linux-gnu",
    ):
        dirs.append(base)

    # 3. Windows: PATH entries
    if platform.system() == "Windows":
        path_var = os.environ.get("PATH", "")
        dirs.extend(os.path.splitdrive(p)[1] for p in path_var.split(os.pathsep) if p)

    # De-duplicate while preserving order.
    seen = set()
    out = []
    for d in dirs:
        d = os.path.normpath(d)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _load_library() -> _ct.CDLL:
    """Locate and load the native library, raising a clear error if missing."""
    names = _candidate_filenames()
    searched = []
    for d in _candidate_dirs():
        for n in names:
            full = os.path.join(d, n)
            searched.append(full)
            if os.path.isfile(full):
                try:
                    return _ct.CDLL(full)
                except OSError as e:  # pragma: no cover - platform dependent
                    raise DecimateError(Status.ERR_DEVICE, f"could not load {full}: {e}")
    raise OSError(
        "Could not locate the decimate native library. "
        "Build it first (see README) or set DECIMATE_LIB / DECIMATE_LIB_DIR. "
        f"Search tried: {searched[:8]}..."
    )


_lib = _load_library()


# ---------------------------------------------------------------------------
#  C struct / function prototypes
# ---------------------------------------------------------------------------
_int32 = _ct.c_int32
_uint64 = _ct.c_uint64
_float = _ct.c_float
_int64 = _ct.c_int64
_cstr = _ct.c_char_p


class _Options(_ct.Structure):
    _fields_ = [
        ("mode", _int32),
        ("device", _int32),
        ("target_reduction", _float),
        ("num_threads", _int32),
        ("boundary_penalty", _float),
        ("curvature_gain", _float),
        ("seed", _uint64),
    ]


class _RegulariseOptions(_ct.Structure):
    _fields_ = [
        ("iterations", _int32),
        ("step", _float),
        ("pin_boundary", _int32),
        ("preserve_area", _int32),
        ("smooth", _float),
        ("num_threads", _int32),
        ("seed", _uint64),
    ]


class _Result(_ct.Structure):
    _fields_ = [
        ("vertices", _ct.POINTER(_float)),
        ("triangles", _ct.POINTER(_int32)),
        ("n_vert_out", _int32),
        ("n_tri_out", _int32),
        ("elapsed_ns", _int64),
        ("device_used", _cstr),
    ]


# ---- function signatures ---------------------------------------------------
# NOTE: decimate_default_options returns the struct BY VALUE (see header).
_lib.decimate_default_options.restype = _Options
_lib.decimate_default_options.argtypes = []

_lib.decimate_mesh.restype = _int32
_lib.decimate_mesh.argtypes = [
    _ct.POINTER(_float),   # vertices
    _ct.POINTER(_int32),   # triangles
    _int32,                # n_vert
    _int32,                # n_tri
    _ct.POINTER(_Options), # options
    _ct.POINTER(_Result),  # out result
]

_lib.decimate_result_free.restype = None
_lib.decimate_result_free.argtypes = [_ct.POINTER(_Result)]

_lib.decimate_status_str.restype = _cstr
_lib.decimate_status_str.argtypes = [_int32]

_lib.decimate_query.restype = _int32
_lib.decimate_query.argtypes = [_ct.POINTER(_int32), _ct.POINTER(_int32)]

_lib.decimate_version.restype = _cstr
_lib.decimate_version.argtypes = []

_lib.decimate_default_regularise_options.restype = _RegulariseOptions
_lib.decimate_default_regularise_options.argtypes = []

_lib.decimate_regularise.restype = _int32
_lib.decimate_regularise.argtypes = [
    _ct.POINTER(_float),   # vertices
    _ct.POINTER(_int32),   # triangles
    _int32,                # n_vert
    _int32,                # n_tri
    _ct.POINTER(_RegulariseOptions),  # options
    _ct.POINTER(_Result),  # out result
]


# ---------------------------------------------------------------------------
#  Public high-level API
# ---------------------------------------------------------------------------
def version() -> str:
    """Return the native library version string."""
    return _lib.decimate_version().decode("utf-8", "replace")


def query() -> Tuple[int, int]:
    """Return ``(gpu_available: bool, cpu_threads: int)``."""
    gpu = _int32(0)
    threads = _int32(0)
    st = _lib.decimate_query(_ct.byref(gpu), _ct.byref(threads))
    if st != Status.OK:
        raise DecimateError(st)
    return bool(gpu.value), int(threads.value)


def _default_options() -> _Options:
    """Return a copy of the library's default options struct (returned by value)."""
    src = _lib.decimate_default_options()
    dst = _Options()
    for field, _ in _Options._fields_:
        setattr(dst, field, getattr(src, field))
    return dst


def _build_options(
    mode: int,
    device: int,
    target_reduction: float,
    num_threads: int,
    boundary_penalty: float,
    curvature_gain: float,
    seed: int,
) -> _Options:
    opts = _Options()
    opts.mode = int(mode)
    opts.device = int(device)
    opts.target_reduction = float(target_reduction)
    opts.num_threads = int(num_threads)
    opts.boundary_penalty = float(boundary_penalty)
    opts.curvature_gain = float(curvature_gain)
    opts.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    return opts


class Result:
    """Owns the native result buffers and exposes them as numpy arrays.

    The buffers are released either explicitly via :meth:`free` or
    automatically when the object is garbage-collected / used as a
    context manager.
    """

    __slots__ = ("_r", "_owned", "elapsed_ns", "device_used")

    def __init__(self, r: _Result, owned: bool = True):
        self._r = r
        self._owned = owned
        self.elapsed_ns = int(r.elapsed_ns)
        self.device_used = r.device_used.decode("utf-8", "replace") if r.device_used else None

    # ---- numpy views (do NOT copy) -----------------------------------------
    @property
    def vertices(self) -> np.ndarray:
        n = self._r.n_vert_out
        arr = _ctypeslib.as_array(self._r.vertices, shape=(n, 3))
        return arr

    @property
    def triangles(self) -> np.ndarray:
        n = self._r.n_tri_out
        arr = _ctypeslib.as_array(self._r.triangles, shape=(n, 3))
        return arr

    @property
    def n_vert(self) -> int:
        return int(self._r.n_vert_out)

    @property
    def n_tri(self) -> int:
        return int(self._r.n_tri_out)

    def copies(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return independent owned copies of vertices and triangles."""
        return self.vertices.copy(), self.triangles.copy()

    # ---- lifecycle ---------------------------------------------------------
    def free(self) -> None:
        if self._owned and self._r:
            _lib.decimate_result_free(_ct.byref(self._r))
            self._owned = False

    def __del__(self):  # pragma: no cover - GC timing is non-deterministic
        try:
            self.free()
        except Exception:
            pass

    def __enter__(self) -> "Result":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.free()

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Result(n_vert={self.n_vert}, n_tri={self.n_tri}, "
                f"device={self.device_used!r}, {self.elapsed_ns} ns)")


def _ensure_contig(arr: np.ndarray, dtype: np.dtype, name: str) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        arr = np.asanyarray(arr)
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def decimate_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    target_reduction: float = 0.5,
    mode: int = Mode.NONE,
    device: int = Device.AUTO,
    *,
    num_threads: int = 0,
    boundary_penalty: float = 15.0,
    curvature_gain: float = 5.0,
    seed: int = 0,
    regularise: bool = True,
    reg_iterations: int = 6,
    reg_step: float = 0.8,
    reg_pin_boundary: bool = True,
    reg_preserve_area: bool = True,
    reg_smooth: float = 0.2,
) -> Result:
    """Decimate a triangle mesh, then optionally apply shape regularisation.

    The shape-regularisation pass (Lloyd / centroidal-Voronoi) is applied
    automatically after decimation when ``regularise`` is ``True`` (the
    default).  It re-positions interior vertices toward their cell centroid
    in the tangent plane, improving triangle shape quality **without**
    changing topology or boundary.  Set ``regularise=False`` to get the raw
    decimation output only.

    Parameters
    ----------
    vertices : (n_vert, 3) array, float32
        Input vertex positions.
    triangles : (n_tri, 3) array, int32
        Input face indices.
    target_reduction : float in [0, 1)
        Fraction of faces to remove (0.5 => keep half the faces).
    mode : int
        One of :class:`Mode`.
    device : int
        One of :class:`Device`. ``AUTO`` prefers the GPU if built & present.
    num_threads : int
        CPU OpenMP thread count (0 = library default / hardware).
    boundary_penalty : float
        Penalty multiplier for boundary edges (MODE_PRESERVE_BOUNDARIES).
    curvature_gain : float
        Curvature amplification (MODE_PRESERVE_VOLUME).
    seed : int
        Reserved (kept for API stability; the greedy selection is deterministic).
    regularise : bool
        Apply the Lloyd / CVT shape-regularisation pass after decimation
        (default ``True``).
    reg_iterations : int
        Number of Lloyd iterations (default 6, clamped to [1, 64]).
    reg_step : float
        Damping factor in (0, 1] for the regularisation move (default 0.8).
    reg_pin_boundary : bool
        Keep boundary vertices fixed during regularisation (default ``True``).
    reg_preserve_area : bool
        Constrain moves to the tangent plane so local area is preserved to
        first order (default ``True``).
    reg_smooth : float
        Mild Laplacian (neighbor-averaging) smoothing weight in [0, 1]
        (default 0.2). 0.0 = pure Lloyd; higher values relax per-vertex
        jitter toward the mean of connected neighbors.

    Returns
    -------
    Result
        Owns the native buffers; see :class:`Result`.
    """
    v = _ensure_contig(vertices, np.float32, "vertices")
    t = _ensure_contig(triangles, np.int32, "triangles")

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (n_vert, 3)")
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError("triangles must have shape (n_tri, 3)")

    n_vert = int(v.shape[0])
    n_tri = int(t.shape[0])

    opts = _build_options(
        mode, device, target_reduction,
        num_threads, boundary_penalty, curvature_gain, seed,
    )

    # Pin the numpy buffers for the lifetime of the call so ctypes can read them.
    v_buf = v.ctypes.data_as(_ct.POINTER(_float))
    t_buf = t.ctypes.data_as(_ct.POINTER(_int32))

    r = _Result()
    st = _lib.decimate_mesh(
        v_buf, t_buf, _int32(n_vert), _int32(n_tri),
        _ct.byref(opts), _ct.byref(r),
    )
    if st != Status.OK:
        msg = _lib.decimate_status_str(_int32(st))
        raise DecimateError(st, msg.decode("utf-8", "replace") if isinstance(msg, bytes) else str(msg))

    res = Result(r, owned=True)

    if not regularise:
        return res

    # Run the shape-regularisation pass on the decimated output.
    # Copy the native decimate buffers into freshly-owned arrays (a real copy,
    # NOT a view: ascontiguousarray would return a view over the ctypes buffer,
    # which we then free below), then release the decimate result so the
    # regularise path owns exactly one set of buffers.
    dec_verts = np.array(res.vertices, dtype=np.float32, copy=True)
    dec_tris  = np.array(res.triangles, dtype=np.int32, copy=True)
    res.free()
    return regularise_mesh(
        dec_verts, dec_tris,
        iterations=reg_iterations,
        step=reg_step,
        pin_boundary=reg_pin_boundary,
        preserve_area=reg_preserve_area,
        smooth=reg_smooth,
        num_threads=num_threads,
        seed=seed,
    )


def regularise_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    iterations: int = 6,
    step: float = 0.8,
    pin_boundary: bool = True,
    preserve_area: bool = True,
    smooth: float = 0.2,
    num_threads: int = 0,
    seed: int = 0,
) -> Result:
    """Run the optional shape-regularisation pass (Lloyd / CVT) on a mesh.

    Re-positions interior vertices toward the centroid of their incident
    face-centroid (tangent-plane Voronoi) cell, improving triangle shape
    quality without changing topology or boundary. A damped move keeps the
    total surface area approximately constant.

    Parameters
    ----------
    vertices : (n_vert, 3) array, float32
        Input vertex positions.
    triangles : (n_tri, 3) array, int32
        Input face indices.
    iterations : int, optional
        Lloyd iterations to run (default 6, clamped to >= 1).
    step : float, optional
        Damping in (0, 1]; 1.0 = full move (default 0.8).
    pin_boundary : bool, optional
        Keep boundary vertices fixed (default True).
    preserve_area : bool, optional
        Honour the area-preserving damped move (default True).
    smooth : float, optional
        Mild Laplacian (neighbor-averaging) smoothing weight in [0, 1]
        (default 0.2). 0.0 = pure Lloyd; higher values relax per-vertex
        jitter toward the mean of connected neighbors.
    num_threads : int, optional
        CPU OpenMP thread count (0 = library default / hardware).
    seed : int, optional
        Reserved (kept for API stability; the pass is deterministic).

    Returns
    -------
    Result
        Owns the native buffers; see :class:`Result`.
    """
    v = _ensure_contig(vertices, np.float32, "vertices")
    t = _ensure_contig(triangles, np.int32, "triangles")

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (n_vert, 3)")
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError("triangles must have shape (n_tri, 3)")

    n_vert = int(v.shape[0])
    n_tri = int(t.shape[0])

    opts = _lib.decimate_default_regularise_options()
    opts.iterations = _int32(int(iterations))
    opts.step = _float(float(step))
    opts.pin_boundary = _int32(1 if pin_boundary else 0)
    opts.preserve_area = _int32(1 if preserve_area else 0)
    opts.smooth = _float(float(smooth))
    opts.num_threads = _int32(int(num_threads))
    opts.seed = _uint64(int(seed))

    # Pin the numpy buffers for the lifetime of the call so ctypes can read them.
    v_buf = v.ctypes.data_as(_ct.POINTER(_float))
    t_buf = t.ctypes.data_as(_ct.POINTER(_int32))

    r = _Result()
    st = _lib.decimate_regularise(
        v_buf, t_buf, _int32(n_vert), _int32(n_tri),
        _ct.byref(opts), _ct.byref(r),
    )
    if st != Status.OK:
        msg = _lib.decimate_status_str(_int32(st))
        raise DecimateError(st, msg.decode("utf-8", "replace") if isinstance(msg, bytes) else str(msg))

    return Result(r, owned=True)


# Keep a reference to numpy so it can't be unloaded before the C callbacks.
_np = np
