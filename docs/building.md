# Building

## Requirements

- **CMake** >= 3.18
- A **C99** compiler:
  - Windows: MSVC (Visual Studio Build Tools) or MinGW-w64 gcc/clang
  - Linux: gcc or clang
  - macOS: Apple clang (or Homebrew clang)
- **OpenMP** (recommended): bundled with MSVC; `libgomp` on Linux (gcc);
  Apple clang uses a bundled OMP runtime. The build still succeeds without
  OpenMP and simply runs single-threaded.
- **CUDA toolkit + nvcc** (only for the GPU build), NVIDIA driver, GPU with
  compute capability >= 6.0.

## Options

| CMake option | Default | Description |
|---|---|---|
| `DECIMATE_ENABLE_CUDA` | `OFF` | Build the CUDA mirror (`src/decimate_cuda.cu`), link `cudart`, define `DECIMATE_HAS_CUDA`. |
| `DECIMATE_BUILD_TESTS` | `ON` | Build the native smoke test + wire pytest. |
| `DECIMATE_BUILD_EXAMPLES` | `ON` | Build `example_c`. |

## CPU-only build (default)

```bash
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build
```

The shared library lands in `build/`:

| OS | Artifact |
|---|---|
| Windows | `build/Release/decimate_tri_mesh.dll` (or `build/decimate_tri_mesh.dll` single-config) |
| Linux | `build/libdecimate_tri_mesh.so` |
| macOS | `build/libdecimate_tri_mesh.dylib` |

### Windows (MSVC, single-config)

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release
```

On Windows the public symbols are exported via `__declspec(dllexport)` through
the `DECIMATE_API` macro in `include/decimate.h`, enabled by the `DECIMATE_EXPORTS`
compile definition set by CMake. No per-symbol linker flags are needed.

### Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

### macOS

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

The dylib is installed with an `@rpath`-relative `install_name`, so downstream
`find_package` consumers can link it via rpath.

## GPU build (opt-in)

```bash
cmake -S . -B build -DDECIMATE_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="75;80;86"
cmake --build build --config Release
ctest --test-dir build
```

Notes:

- `CMAKE_CUDA_ARCHITECTURES` overrides the default `60;70;75;80;86;90`; set it
  to the capability of your GPU to shrink the binary.
- `DECIMATE_HAS_CUDA` is defined only for this target, so the CPU source only
  references the GPU entry points when compiled with the flag.
- `decimate_query` reports `gpu_available = 1` on a CUDA build that sees a
  device; otherwise `0`.
- The shape-regularisation pass is also mirrored to the GPU
  (`decimate_regularise_gpu` / `kernel_lloyd` in `src/decimate_cuda.cu`). It
  is compiled only when `DECIMATE_HAS_CUDA` is defined, and is **not**
  exercised on this machine (no local `nvcc`). On a CUDA build with a device
  present, `decimate_regularise` dispatches to the GPU path automatically;
  otherwise it runs on the CPU (OpenMP).

## Install

```bash
cmake --install build --prefix /your/prefix
```

Installs:

- the shared library (`decimate_tri_mesh.dll` / `libdecimate_tri_mesh.so` / `libdecimate_tri_mesh.dylib`),
- `include/decimate.h`,
- a CMake package `decimate-config.cmake` exposing `decimate::decimate`.

### Consume from another CMake project

```cmake
find_package(decimate REQUIRED CONFIG)
target_link_libraries(myapp PRIVATE decimate::decimate)
```

## Building the Python package

The Python wrapper is pure Python; it loads the compiled library at runtime.

```bash
# 1) build the native lib (as above)
cmake -S . -B build && cmake --build build --config Release

# 2) point the wrapper at the artifact
export DECIMATE_LIB_DIR=$PWD/build          # Linux/macOS
# set  DECIMATE_LIB_DIR=C:\path\to\build    # Windows

# 3) install the module
pip install .

# 4) test
pip install .[test]
DECIMATE_LIB_DIR=$PWD/build pytest
```

`pip install .` uses `pyproject.toml` (setuptools) and installs the `decimate`
module from `python/`.

## Cross-platform & CUDA verification status

| Platform | Build | Tests | Notes |
|---|---|---|---|
| Windows (MSVC, x64) | ✅ | ✅ C smoke + Python wrapper + `REGULAR` quality + Lloyd regularisation | local, verified |
| Linux (GCC) | ✅ (CI) | ✅ C smoke + Python wrapper | via `cpu-linux` job |
| macOS (Apple clang) | ✅ (CI) | ✅ C smoke + Python wrapper | serial path (no OpenMP runtime) |
| CUDA | compile + link ✅ (CI) | GPU runtime **not** exercised locally | no local `nvcc`/device |

The CUDA backend (`src/decimate_cuda.cu`) is a faithful mirror of the CPU
backend — same SoA layout, same four cost modes, the same `REGULAR`
threshold-gated penalty, and the same deterministic `edge_key` tie-break — and
**compiles and links** in the `cuda-compile` CI job. Because no local `nvcc`
or GPU device was available in the development environment, end-to-end GPU
execution (including the bit-for-bit CPU↔GPU equivalence on ties) is asserted
by construction and by the CI compile+link check, but has not been run on
real GPU hardware. Add a GPU runner to exercise the CUDA execution path.
