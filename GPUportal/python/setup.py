"""setup.py — builds the gpuportal_torch pybind11 extension
(spec_gpuportal.md section 9.1/11 item 2).

Deliberately NOT wired into the C CMake build (GPUportal/CMakeLists.txt):
requiring libtorch headers at `cmake configure` time for the whole project
would impose exactly the kind of hard, non-optional dependency this
project's dlopen/optional-probe discipline exists to avoid (see
GPUBridge/CLAUDE.md's "Conventions for future work" section, and
spec_gpuportal.md section 11 item 2 for the fuller rationale). Run this
manually, only when a Python environment with `torch` installed is
available:

    cd GPUportal/python
    python setup.py build_ext --inplace

Prerequisite: the C libraries must already be built with
GPUBRIDGE_BUILD_PORTAL=ON — see gpubridge-runtime/CMakeLists.txt and
GPUBridge/CLAUDE.md's "Build & run" section for the exact cmake invocation.
This script links the resulting static libraries in directly (extra_objects)
rather than recompiling gpubridge-runtime/GPUportal's C sources itself —
"reuse, don't fork" (spec_gpuportal.md section 2), applied to the build
step too.

Two env vars locate that prior build, following this project's existing
"env var override, sensible default" convention (e.g. GPUBRIDGE_CLBLAST_LIBRARY
in gpubridge-runtime/CMakeLists.txt):
  GPUBRIDGE_RUNTIME_BUILD_DIR   default: ../../gpubridge-runtime/build
                                 (must contain libgpubridge_runtime.a)
  GPUBRIDGE_PORTAL_BUILD_DIR    default: ${GPUBRIDGE_RUNTIME_BUILD_DIR}/GPUportal
                                 (must contain libgpuportal.a, libgpuportal_blas.a
                                 — that's where add_subdirectory() puts them,
                                 see gpubridge-runtime/CMakeLists.txt)
"""
import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

_HERE = os.path.dirname(os.path.abspath(__file__))
_GPUPORTAL_ROOT = os.path.dirname(_HERE)                       # GPUportal/
_GPUBRIDGE_ROOT = os.path.dirname(_GPUPORTAL_ROOT)              # GPUBridge/

_runtime_build_dir = os.environ.get(
    "GPUBRIDGE_RUNTIME_BUILD_DIR",
    os.path.join(_GPUBRIDGE_ROOT, "gpubridge-runtime", "build"),
)
_portal_build_dir = os.environ.get(
    "GPUBRIDGE_PORTAL_BUILD_DIR",
    os.path.join(_runtime_build_dir, "GPUportal"),
)

_runtime_lib = os.path.join(_runtime_build_dir, "libgpubridge_runtime.a")
_portal_lib = os.path.join(_portal_build_dir, "libgpuportal.a")
_blas_lib = os.path.join(_portal_build_dir, "libgpuportal_blas.a")

for _lib, _hint in (
    (_runtime_lib, "build gpubridge-runtime first (see GPUBridge/CLAUDE.md)"),
    (_portal_lib, "build with -DGPUBRIDGE_BUILD_PORTAL=ON (see GPUBridge/CLAUDE.md)"),
    (_blas_lib, "build with -DGPUBRIDGE_BUILD_PORTAL=ON (see GPUBridge/CLAUDE.md)"),
):
    if not os.path.isfile(_lib):
        sys.exit(
            f"gpuportal_torch setup.py: required static library not found: {_lib}\n"
            f"  -> {_hint}\n"
            f"  -> override the search path with GPUBRIDGE_RUNTIME_BUILD_DIR / "
            f"GPUBRIDGE_PORTAL_BUILD_DIR if your build directory is elsewhere."
        )

setup(
    name="gpuportal_torch",
    ext_modules=[
        CppExtension(
            name="gpuportal",
            sources=["gpuportal_torch.cpp"],
            include_dirs=[
                os.path.join(_GPUPORTAL_ROOT, "include"),
                os.path.join(_GPUBRIDGE_ROOT, "gpubridge-runtime", "include"),
            ],
            # Static-link the already-built C libraries directly, in
            # dependency order (gpuportal -> gpuportal_blas ->
            # gpubridge_runtime) — no -L/-l against them as shared libraries,
            # since none of the three are built as .so (CMakeLists.txt: all
            # STATIC). -ldl for the dlopen/dlsym calls gpubridge_runtime
            # itself makes at runtime against libOpenCL/libclblast/
            # libopenblas/libvulkan (CMAKE_DL_LIBS in the C build).
            extra_objects=[_portal_lib, _blas_lib, _runtime_lib],
            libraries=["dl"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
