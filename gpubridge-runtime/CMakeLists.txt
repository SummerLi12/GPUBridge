# CMakeLists.txt — build definition for the GPUBridge Runtime Prototype
# (spec1.md Milestone 0-1).
#
# Produces two targets:
#   gpubridge_runtime      static library: the whole runtime + both backends
#   gpubridge_vecadd_test  the correctness test executable (tests/test_vecadd.c),
#                   linked against gpubridge_runtime
#
# See CLAUDE.md for the full rationale behind the OpenCL-detection strategy
# below; the short version is in the comments at each step.

cmake_minimum_required(VERSION 3.10)
project(gpubridge_runtime C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# GPUportal (GPUBridge/GPUportal/, spec_gpuportal.md): a separately-scoped
# sub-project living outside this directory, off by default. See the
# add_subdirectory() call near the bottom of this file for why it's added
# there (after every gpubridge_runtime target/probe result it depends on
# already exists) rather than here.
option(GPUBRIDGE_BUILD_PORTAL "Build GPUportal (PyTorch/cuBLAS-shaped OpenCL bridge, GPUBridge/GPUportal/spec_gpuportal.md)" OFF)

# GPUportal's static libraries (libgpuportal.a/libgpuportal_blas.a) end up
# statically linked into gpuportal_torch's .so (GPUportal/python/setup.py,
# spec_gpuportal.md section 11 item 2) — and so, transitively, does this
# gpubridge_runtime static library. A shared object cannot contain
# non-PIC object code, so every object file in that link chain needs
# -fPIC. Scoped to GPUBRIDGE_BUILD_PORTAL=ON only (not this project's
# always-static-executable default) since it is otherwise unnecessary
# overhead for a project that never otherwise touches a .so build product.
if(GPUBRIDGE_BUILD_PORTAL)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# The runtime core (src/) plus both backend implementations, built as one
# static library. backends/opencl/gpubridge_backend_opencl.c is always included
# here regardless of whether OpenCL headers are found below — its own
# #ifdef GPUBRIDGE_HAVE_OPENCL guard (see that file) makes it compile to a
# does-nothing stub in that case, so there is no need for conditional
# source-file inclusion at the CMake level.
add_library(gpubridge_runtime STATIC
    src/gpubridge_runtime.c
    src/gpubridge_selector.c
    src/gpubridge_env.c
    src/gpubridge_diagnostics.c
    src/gpubridge_ir.c
    src/gpubridge_memory.c
    src/gpubridge_pool.c
    src/gpubridge_scheduling.c                   # Phase 4 (spec4.md)
    backends/cpu/gpubridge_backend_cpu.c
    backends/cpu/gpubridge_cpu_openblas.c        # Milestone 3 (spec3.md)
    backends/opencl/gpubridge_backend_opencl.c
    backends/opencl/gpubridge_opencl_clblast.c   # Milestone 3 (spec3.md)
    backends/vulkan/gpubridge_backend_vulkan.c   # Phase 4.5 (spec4.5.md)
)

# include/ holds the public headers (gpubridge_runtime.h, gpubridge_backend.h, gpubridge_ir.h)
# that tests/test_vecadd.c also needs, hence PUBLIC. src/ holds
# gpubridge_internal.h, needed only by the .c files inside gpubridge_runtime itself,
# hence PRIVATE.
target_include_directories(gpubridge_runtime PUBLIC include)
target_include_directories(gpubridge_runtime PRIVATE src)

# --- OpenCL: headers only, no link-time dependency ---------------------
# We deliberately do NOT use find_package(OpenCL) or link against libOpenCL.
# The OpenCL backend dlopen()s libOpenCL.so.1 at runtime instead, so the
# executable has zero hard dependency on OpenCL and still starts/runs
# GPUBRIDGE_BACKEND=cpu (and GPUBRIDGE_BACKEND=auto without a GPU) on a machine with no
# OpenCL installed at all. See CLAUDE.md for the full rationale.
#
# find_path just locates the *directory* containing CL/cl.h (e.g.
# /usr/include on this machine) — it does not add any link dependency by
# itself, unlike find_package(OpenCL), which would also produce an
# IMPORTED library target that pulls in -lOpenCL if linked.
find_path(GPUBRIDGE_OPENCL_INCLUDE_DIR NAMES CL/cl.h)
if(GPUBRIDGE_OPENCL_INCLUDE_DIR)
    message(STATUS "Found CL/cl.h in ${GPUBRIDGE_OPENCL_INCLUDE_DIR} — OpenCL backend enabled")

    # GPUBRIDGE_HAVE_OPENCL=1 switches on the real implementation in
    # gpubridge_backend_opencl.c (vs. its always-false stub). The CL_TARGET_*/
    # CL_USE_DEPRECATED_* defines pin us to the OpenCL 1.2 API surface
    # (specifically clCreateCommandQueue, which is deprecated in 2.0+ but
    # simpler than the properties-array-based replacement and universally
    # supported, including by this machine's NVIDIA driver) and silence the
    # deprecation warnings that would otherwise fire for using it.
    target_compile_definitions(gpubridge_runtime PRIVATE
        GPUBRIDGE_HAVE_OPENCL=1
        CL_TARGET_OPENCL_VERSION=120
        CL_USE_DEPRECATED_OPENCL_1_2_APIS
    )
    target_include_directories(gpubridge_runtime PRIVATE ${GPUBRIDGE_OPENCL_INCLUDE_DIR})

    # Embed backends/opencl/vector_add_f32.cl into the binary as a byte
    # array, rather than reading it from disk at runtime (which would add a
    # working-directory-dependent failure mode unrelated to actual OpenCL
    # availability) or hand-copying its text into a C string literal inside
    # gpubridge_backend_opencl.c (which would create a second copy of the kernel
    # source that could silently drift out of sync with the real .cl file).
    #
    # file(READ ... HEX) reads the file as one long hex string; the
    # REGEX REPLACE turns every 2 hex digits into a "0xNN," C initializer
    # element; the trailing "0x00" null-terminates it (gpubridge_backend_opencl.c
    # treats this array as a C string via strlen()). configure_file()
    # substitutes the resulting @CL_KERNEL_HEX@ into
    # cmake/vector_add_f32_cl.h.in to produce a real header under the build
    # directory, which gpubridge_backend_opencl.c #includes as
    # "generated/vector_add_f32_cl.h".
    set(CL_KERNEL_SRC ${CMAKE_SOURCE_DIR}/backends/opencl/vector_add_f32.cl)
    file(READ ${CL_KERNEL_SRC} CL_KERNEL_HEX HEX)
    string(REGEX REPLACE "(..)" "0x\\1," CL_KERNEL_HEX "${CL_KERNEL_HEX}")
    set(CL_KERNEL_HEX "${CL_KERNEL_HEX}0x00")
    configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/vector_add_f32_cl.h.in
        ${CMAKE_BINARY_DIR}/generated/vector_add_f32_cl.h
        @ONLY
    )
    # The generated header lives at ${CMAKE_BINARY_DIR}/generated/..., so
    # adding ${CMAKE_BINARY_DIR} (not .../generated) as an include dir is
    # what makes the backend's #include "generated/vector_add_f32_cl.h"
    # resolve correctly.
    target_include_directories(gpubridge_runtime PRIVATE ${CMAKE_BINARY_DIR})

    # Milestone 2 (spec2.md section 13 item 1): the four new tensor-op
    # kernels, embedded with the exact same HEX pattern as vector_add_f32
    # above. Looped (rather than four hand-copied blocks) since all four
    # follow an identical <name>.cl -> cmake/<name>_cl.h.in ->
    # generated/<name>_cl.h mapping; the per-file logic itself is unchanged
    # from the vector_add_f32 block.
    set(GPUBRIDGE_TENSOR_KERNELS matmul_f32 reduce_sum_f32 broadcast_add_f32 relu_f32)
    foreach(kernel ${GPUBRIDGE_TENSOR_KERNELS})
        set(_kernel_src ${CMAKE_SOURCE_DIR}/backends/opencl/${kernel}.cl)
        file(READ ${_kernel_src} CL_KERNEL_HEX HEX)
        string(REGEX REPLACE "(..)" "0x\\1," CL_KERNEL_HEX "${CL_KERNEL_HEX}")
        set(CL_KERNEL_HEX "${CL_KERNEL_HEX}0x00")
        configure_file(
            ${CMAKE_SOURCE_DIR}/cmake/${kernel}_cl.h.in
            ${CMAKE_BINARY_DIR}/generated/${kernel}_cl.h
            @ONLY
        )
    endforeach()

    # --- Milestone 3 (spec3.md section 11): CLBlast, optional GEMM
    # acceleration for the OpenCL backend. Nested inside this
    # GPUBRIDGE_OPENCL_INCLUDE_DIR block: CLBlast's own header depends on
    # OpenCL's types, so probing for it only makes sense once CL/cl.h
    # itself was already found. Same headers-only-at-compile-time,
    # dlopen-at-runtime rule as OpenCL itself: no find_package(CLBlast),
    # no -lclblast.
    find_path(GPUBRIDGE_CLBLAST_INCLUDE_DIR NAMES clblast_c.h)
    if(GPUBRIDGE_CLBLAST_INCLUDE_DIR)
        message(STATUS "Found clblast_c.h in ${GPUBRIDGE_CLBLAST_INCLUDE_DIR} — CLBlast GEMM path enabled")
        target_compile_definitions(gpubridge_runtime PRIVATE GPUBRIDGE_HAVE_CLBLAST=1)
        target_include_directories(gpubridge_runtime PRIVATE ${GPUBRIDGE_CLBLAST_INCLUDE_DIR})
    else()
        message(STATUS "clblast_c.h not found — OpenCL matmul_f32 will always use "
                        "the hand-written kernel. Install with your distro's "
                        "CLBlast package, or build CLBlast from source, to enable it.")
    endif()
else()
    # Not a fatal error: gpubridge_backend_opencl.c's #else stub branch handles
    # this by making the OpenCL backend permanently report itself
    # unavailable, so GPUBRIDGE_BACKEND=cpu/auto still work on a machine without
    # these headers installed. Just point the user at how to fix it.
    message(STATUS "CL/cl.h not found — OpenCL backend will report unavailable "
                    "at runtime. Install headers with: "
                    "sudo apt-get install opencl-headers ocl-icd-opencl-dev")
endif()

# --- Milestone 3 (spec3.md section 11): OpenBLAS, optional GEMM
# acceleration for the CPU backend — same headers-only/dlopen-at-runtime
# rule, independent of OpenCL.
find_path(GPUBRIDGE_OPENBLAS_INCLUDE_DIR NAMES cblas.h)
if(GPUBRIDGE_OPENBLAS_INCLUDE_DIR)
    message(STATUS "Found cblas.h in ${GPUBRIDGE_OPENBLAS_INCLUDE_DIR} — CPU BLAS GEMM path enabled")
    target_compile_definitions(gpubridge_runtime PRIVATE GPUBRIDGE_HAVE_OPENBLAS=1)
    target_include_directories(gpubridge_runtime PRIVATE ${GPUBRIDGE_OPENBLAS_INCLUDE_DIR})
else()
    message(STATUS "cblas.h not found — CPU matmul_f32 will always use the "
                    "naive loop. Install libopenblas-dev (or equivalent) to enable it.")
endif()

# --- Vulkan: headers only, no link-time dependency (spec4.5.md section 6) ---
# Same headers-only-at-compile-time, dlopen-at-runtime rule as OpenCL/CLBlast/
# OpenBLAS above: no find_package(Vulkan), no -lvulkan. find_path only
# locates the *directory* containing vulkan/vulkan.h; it adds no link
# dependency by itself.
find_path(GPUBRIDGE_VULKAN_INCLUDE_DIR NAMES vulkan/vulkan.h)
if(GPUBRIDGE_VULKAN_INCLUDE_DIR)
    message(STATUS "Found vulkan/vulkan.h in ${GPUBRIDGE_VULKAN_INCLUDE_DIR} — Vulkan backend enabled")
    target_compile_definitions(gpubridge_runtime PRIVATE GPUBRIDGE_HAVE_VULKAN=1)
    target_include_directories(gpubridge_runtime PRIVATE ${GPUBRIDGE_VULKAN_INCLUDE_DIR})

    # Embed the pre-compiled backends/vulkan/vector_add_f32.spv (spec4.5.md
    # section 7.2/9.2) — identical file(READ ... HEX) technique as every .cl
    # kernel above, applied to a binary file instead of source text (no
    # trailing 0x00 terminator: SPIR-V is measured by byte count, not treated
    # as a C string — see cmake/vector_add_f32_spv.h.in).
    set(VULKAN_SPV_SRC ${CMAKE_SOURCE_DIR}/backends/vulkan/vector_add_f32.spv)
    file(READ ${VULKAN_SPV_SRC} VULKAN_SPV_HEX HEX)
    string(REGEX REPLACE "(..)" "0x\\1," VULKAN_SPV_HEX "${VULKAN_SPV_HEX}")
    configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/vector_add_f32_spv.h.in
        ${CMAKE_BINARY_DIR}/generated/vector_add_f32_spv.h
        @ONLY
    )

    # The four Milestone 2 tensor ops, ported to Vulkan beyond spec4.5.md's
    # own vector_add_f32-only scope (CLAUDE.md's "Phase 4.5" section records
    # this deviation) — identical file(READ ... HEX) embedding, looped the
    # same way the OpenCL block above loops GPUBRIDGE_TENSOR_KERNELS.
    set(GPUBRIDGE_VULKAN_TENSOR_KERNELS matmul_f32 reduce_sum_f32 broadcast_add_f32 relu_f32)
    foreach(kernel ${GPUBRIDGE_VULKAN_TENSOR_KERNELS})
        set(_vulkan_kernel_src ${CMAKE_SOURCE_DIR}/backends/vulkan/${kernel}.spv)
        file(READ ${_vulkan_kernel_src} VULKAN_SPV_HEX HEX)
        string(REGEX REPLACE "(..)" "0x\\1," VULKAN_SPV_HEX "${VULKAN_SPV_HEX}")
        configure_file(
            ${CMAKE_SOURCE_DIR}/cmake/${kernel}_spv.h.in
            ${CMAKE_BINARY_DIR}/generated/${kernel}_spv.h
            @ONLY
        )
    endforeach()

    # ${CMAKE_BINARY_DIR} (not .../generated) is already added as an include
    # dir by the OpenCL block above when OpenCL is also found; add it here
    # too so this still works standalone if OpenCL headers are absent.
    target_include_directories(gpubridge_runtime PRIVATE ${CMAKE_BINARY_DIR})
else()
    # Not fatal: gpubridge_backend_vulkan.c's #else stub branch handles this
    # by making the Vulkan backend permanently report itself unavailable.
    message(STATUS "vulkan/vulkan.h not found — Vulkan backend will report "
                    "unavailable at runtime. Install headers with: "
                    "sudo apt-get install libvulkan-dev")
endif()

# ${CMAKE_DL_LIBS} is the portable way to link whatever provides
# dlopen/dlsym (on this machine's glibc 2.35, dlopen is already inside
# libc, so this typically resolves to an empty/no-op link requirement, but
# it's linked unconditionally so the build stays portable to older glibc /
# other libc implementations where a separate libdl is still needed).
target_link_libraries(gpubridge_runtime PRIVATE ${CMAKE_DL_LIBS})

# --- Test executable -----------------------------------------------------
# The single correctness test from spec1.md sections 14-16. Linked against
# `m` for libm (fabs()) in addition to gpubridge_runtime itself.
add_executable(gpubridge_vecadd_test tests/test_vecadd.c)
target_link_libraries(gpubridge_vecadd_test PRIVATE gpubridge_runtime m)

# Milestone 2 (spec2.md section 13 item 2): one isolated-debugging test per
# new op, plus the dense-forward capstone. Same shape as gpubridge_vecadd_test
# above.
add_executable(gpubridge_matmul_test tests/test_matmul.c)
target_link_libraries(gpubridge_matmul_test PRIVATE gpubridge_runtime m)

add_executable(gpubridge_reduce_sum_test tests/test_reduce_sum.c)
target_link_libraries(gpubridge_reduce_sum_test PRIVATE gpubridge_runtime m)

add_executable(gpubridge_broadcast_add_test tests/test_broadcast_add.c)
target_link_libraries(gpubridge_broadcast_add_test PRIVATE gpubridge_runtime m)

add_executable(gpubridge_relu_test tests/test_relu.c)
target_link_libraries(gpubridge_relu_test PRIVATE gpubridge_runtime m)

add_executable(gpubridge_dense_forward_test tests/test_dense_forward.c)
target_link_libraries(gpubridge_dense_forward_test PRIVATE gpubridge_runtime m)

# Milestone 1.5 (spec1.5.md section 14): a pure-function unit test for
# gpuBridgeChooseMemoryStrategy (no libm needed — no floating-point math), and
# a profiling correctness test with the same shape as gpubridge_vecadd_test.
add_executable(gpubridge_memory_strategy_test tests/test_memory_strategy.c)
target_link_libraries(gpubridge_memory_strategy_test PRIVATE gpubridge_runtime)

add_executable(gpubridge_profile_test tests/test_profile.c)
target_link_libraries(gpubridge_profile_test PRIVATE gpubridge_runtime m)

# Milestone 2.5 (spec2.5.md section 12): pooled-allocator and
# host/device-mirror correctness tests, same shape as gpubridge_profile_test
# above (both need a real backend, so both link `m` even though neither
# does floating-point math beyond what gpubridge_runtime itself needs).
add_executable(gpubridge_tensor_pool_test tests/test_tensor_pool.c)
target_link_libraries(gpubridge_tensor_pool_test PRIVATE gpubridge_runtime m)

add_executable(gpubridge_tensor_mirror_test tests/test_tensor_mirror.c)
target_link_libraries(gpubridge_tensor_mirror_test PRIVATE gpubridge_runtime m)

# Phase 2 (milestone.md section 22 deliverable 4): pure-function unit test
# for gpuBridgeIrValidateKernel/gpuBridgeIrValidateTensorKernel — no libm needed,
# same shape as gpubridge_memory_strategy_test.
add_executable(gpubridge_ir_validate_test tests/test_ir_validate.c)
target_link_libraries(gpubridge_ir_validate_test PRIVATE gpubridge_runtime)

# Phase 4 (spec4.md section 12): edge-inference-vocabulary tests. edge_vocabulary
# is pure-function, no backend (same shape as memory_strategy/ir_validate
# above); edge_profile needs a real backend, same shape as profile_test.
add_executable(gpubridge_edge_vocabulary_test tests/test_edge_vocabulary.c)
target_link_libraries(gpubridge_edge_vocabulary_test PRIVATE gpubridge_runtime)

add_executable(gpubridge_edge_profile_test tests/test_edge_profile.c)
target_link_libraries(gpubridge_edge_profile_test PRIVATE gpubridge_runtime m)

# Milestone 3 (spec3.md section 9): GEMM benchmark, same shape as every
# other test executable above. tests/bench_common.c (Phase 5, spec5.md
# section 5) is test-harness-only plumbing (median/percentile/iteration
# parsing) shared by this and the four new *_benchmark_test executables
# below — compiled directly into each benchmark executable rather than
# into libgpubridge_runtime.a, since it has no runtime API surface.
add_executable(gpubridge_matmul_benchmark_test tests/test_matmul_benchmark.c tests/bench_common.c)
target_link_libraries(gpubridge_matmul_benchmark_test PRIVATE gpubridge_runtime m)

# spec3.md section 9.4's "build type" report line: a plain configure-time
# string, not a generator expression. This project only targets
# single-config generators (Makefiles/Ninja) on Linux, so CMAKE_BUILD_TYPE
# is already known at configure time — a $<CONFIG>-style multi-config
# expression would be unnecessary complexity here.
if(CMAKE_BUILD_TYPE)
    set(GPUBRIDGE_BUILD_TYPE_STR "${CMAKE_BUILD_TYPE}")
else()
    set(GPUBRIDGE_BUILD_TYPE_STR "unspecified")
endif()
target_compile_definitions(gpubridge_matmul_benchmark_test PRIVATE
    GPUBRIDGE_BUILD_TYPE="${GPUBRIDGE_BUILD_TYPE_STR}")

# Phase 5 (spec5.md section 6): the other four ops' cross-backend
# benchmarks, generalizing test_matmul_benchmark.c's methodology to
# single-path (no vendor/naive alternation) ops via tests/bench_common.c.
add_executable(gpubridge_vecadd_benchmark_test tests/test_vecadd_benchmark.c tests/bench_common.c)
target_link_libraries(gpubridge_vecadd_benchmark_test PRIVATE gpubridge_runtime m)

add_executable(gpubridge_reduce_sum_benchmark_test tests/test_reduce_sum_benchmark.c tests/bench_common.c)
target_link_libraries(gpubridge_reduce_sum_benchmark_test PRIVATE gpubridge_runtime m)

add_executable(gpubridge_broadcast_add_benchmark_test tests/test_broadcast_add_benchmark.c tests/bench_common.c)
target_link_libraries(gpubridge_broadcast_add_benchmark_test PRIVATE gpubridge_runtime m)

add_executable(gpubridge_relu_benchmark_test tests/test_relu_benchmark.c tests/bench_common.c)
target_link_libraries(gpubridge_relu_benchmark_test PRIVATE gpubridge_runtime m)

# `ctest` entries covering the two combinations that must always succeed on
# any machine (cpu always works; auto falls back to cpu if no GPU backend
# is available, or picks OpenCL if one is). GPUBRIDGE_BACKEND=opencl/sycl are
# hardware-dependent (opencl needs a real GPU, sycl always fails in this
# milestone) and are exercised manually — see CLAUDE.md's "Build & run"
# section — rather than wired into ctest, since sycl failing is *expected*
# and shouldn't show up as a red ctest failure.
enable_testing()
add_test(NAME vecadd_cpu COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=cpu $<TARGET_FILE:gpubridge_vecadd_test>)
add_test(NAME vecadd_auto COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=auto $<TARGET_FILE:gpubridge_vecadd_test>)

# Phase 4.5 (spec4.5.md section 16): explicit Vulkan request, reusing this
# same unmodified-in-its-correctness-logic gpubridge_vecadd_test binary.
# Passes if a Vulkan-capable device is present, skips cleanly (exit 77) via
# test_vecadd.c's new GPUBRIDGE_TEST_SKIP branch otherwise.
add_test(NAME vecadd_vulkan COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=vulkan $<TARGET_FILE:gpubridge_vecadd_test>)
set_tests_properties(vecadd_vulkan PROPERTIES SKIP_RETURN_CODE 77)

# Milestone 2 (spec2.md section 13 item 3): same cpu/auto pair, one per new
# test executable, mirroring vecadd_cpu/vecadd_auto exactly.
add_test(NAME matmul_cpu COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=cpu $<TARGET_FILE:gpubridge_matmul_test>)
add_test(NAME matmul_auto COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=auto $<TARGET_FILE:gpubridge_matmul_test>)

# Vulkan tensor-op port (beyond spec4.5.md's own scope, see CLAUDE.md's
# "Phase 4.5" section): same explicit-request-skips-cleanly shape as
# vecadd_vulkan above, reusing each existing test binary unmodified in its
# correctness logic.
add_test(NAME matmul_vulkan COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=vulkan $<TARGET_FILE:gpubridge_matmul_test>)
set_tests_properties(matmul_vulkan PROPERTIES SKIP_RETURN_CODE 77)

# Milestone 3 (spec3.md section 10.1): 5 more entries reusing this same,
# unmodified-in-its-correctness-logic gpubridge_matmul_test binary.
# matmul_cpu_novendor/matmul_auto_novendor force the naive path via
# GPUBRIDGE_VENDOR_GEMM_DISABLE=1. matmul_cpu_vendor/matmul_opencl_vendor
# require the relevant vendor library to actually be loaded
# (GPUBRIDGE_TEST_REQUIRE_VENDOR=1), skipping cleanly (exit 77) otherwise —
# do NOT rely on matmul_auto to prove CLBlast works, since auto prefers
# OpenCL only when available and silently resolves to CPU otherwise.
# matmul_opencl_novendor forces GPUBRIDGE_BACKEND=opencl explicitly (not
# auto) so the naive OpenCL kernel is independently proven correct too,
# skipping if this machine has no OpenCL device at all.
add_test(NAME matmul_cpu_novendor    COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=cpu    GPUBRIDGE_VENDOR_GEMM_DISABLE=1 $<TARGET_FILE:gpubridge_matmul_test>)
add_test(NAME matmul_auto_novendor   COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=auto   GPUBRIDGE_VENDOR_GEMM_DISABLE=1 $<TARGET_FILE:gpubridge_matmul_test>)
add_test(NAME matmul_cpu_vendor      COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=cpu    GPUBRIDGE_TEST_REQUIRE_VENDOR=1 $<TARGET_FILE:gpubridge_matmul_test>)
add_test(NAME matmul_opencl_vendor   COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=opencl GPUBRIDGE_TEST_REQUIRE_VENDOR=1 $<TARGET_FILE:gpubridge_matmul_test>)
add_test(NAME matmul_opencl_novendor COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=opencl GPUBRIDGE_VENDOR_GEMM_DISABLE=1 $<TARGET_FILE:gpubridge_matmul_test>)
set_tests_properties(matmul_cpu_vendor matmul_opencl_vendor matmul_opencl_novendor
    PROPERTIES SKIP_RETURN_CODE 77)

# Milestone 3 (spec3.md section 10.2): the GEMM benchmark, labeled
# "benchmark" so a CI config can exclude it from a fast default run
# (ctest -LE benchmark) or run only it (ctest -L benchmark) — explicit
# opt-in/opt-out, not a built-in skip.
add_test(NAME matmul_benchmark_cpu  COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=cpu  $<TARGET_FILE:gpubridge_matmul_benchmark_test>)
add_test(NAME matmul_benchmark_auto COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=auto $<TARGET_FILE:gpubridge_matmul_benchmark_test>)
set_tests_properties(matmul_benchmark_cpu matmul_benchmark_auto PROPERTIES LABELS "benchmark")

# Phase 5 (spec5.md section 9): three ctest entries per new benchmark
# (cpu/auto/vulkan), all labeled "benchmark" (so `ctest -LE benchmark`
# still excludes every Phase 5 entry from a fast default run, same as
# matmul_benchmark_cpu/auto above), with SKIP_RETURN_CODE 77 on the
# *_vulkan entries specifically (same clean-skip-if-unavailable treatment
# every other explicitly-requested-backend ctest entry in this project
# already gets — vecadd_vulkan, matmul_vulkan, reduce_sum_vulkan, etc.).
foreach(op vecadd reduce_sum broadcast_add relu)
    add_test(NAME ${op}_benchmark_cpu    COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=cpu    $<TARGET_FILE:gpubridge_${op}_benchmark_test>)
    add_test(NAME ${op}_benchmark_auto   COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=auto   $<TARGET_FILE:gpubridge_${op}_benchmark_test>)
    add_test(NAME ${op}_benchmark_vulkan COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=vulkan $<TARGET_FILE:gpubridge_${op}_benchmark_test>)
    set_tests_properties(${op}_benchmark_cpu ${op}_benchmark_auto ${op}_benchmark_vulkan PROPERTIES LABELS "benchmark")
    set_tests_properties(${op}_benchmark_vulkan PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
add_test(NAME reduce_sum_cpu COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=cpu $<TARGET_FILE:gpubridge_reduce_sum_test>)
add_test(NAME reduce_sum_auto COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=auto $<TARGET_FILE:gpubridge_reduce_sum_test>)
add_test(NAME reduce_sum_vulkan COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=vulkan $<TARGET_FILE:gpubridge_reduce_sum_test>)
set_tests_properties(reduce_sum_vulkan PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME broadcast_add_cpu COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=cpu $<TARGET_FILE:gpubridge_broadcast_add_test>)
add_test(NAME broadcast_add_auto COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=auto $<TARGET_FILE:gpubridge_broadcast_add_test>)
add_test(NAME broadcast_add_vulkan COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=vulkan $<TARGET_FILE:gpubridge_broadcast_add_test>)
set_tests_properties(broadcast_add_vulkan PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME relu_cpu COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=cpu $<TARGET_FILE:gpubridge_relu_test>)
add_test(NAME relu_auto COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=auto $<TARGET_FILE:gpubridge_relu_test>)
add_test(NAME relu_vulkan COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=vulkan $<TARGET_FILE:gpubridge_relu_test>)
set_tests_properties(relu_vulkan PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME dense_forward_cpu COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=cpu $<TARGET_FILE:gpubridge_dense_forward_test>)
add_test(NAME dense_forward_auto COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=auto $<TARGET_FILE:gpubridge_dense_forward_test>)

# Milestone 1.5 (spec1.5.md section 14): gpubridge_memory_strategy_test needs
# no backend (pure function, no gpuBridgeInit()), so it gets a single ctest
# entry rather than a cpu/auto pair. gpubridge_profile_test gets the usual
# cpu/auto pair, plus GPUBRIDGE_PROFILE=1 so the "GPUBridge profile:" block
# actually prints during ctest (not required for pass/fail — that comes from
# gpuBridgeGetProfileStats() assertions inside the test itself — but useful
# for eyeballing ctest --output-on-failure/--verbose output).
add_test(NAME memory_strategy COMMAND $<TARGET_FILE:gpubridge_memory_strategy_test>)
add_test(NAME profile_cpu COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=cpu GPUBRIDGE_PROFILE=1 $<TARGET_FILE:gpubridge_profile_test>)
add_test(NAME profile_auto COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=auto GPUBRIDGE_PROFILE=1 $<TARGET_FILE:gpubridge_profile_test>)

# Phase 2: same no-backend-needed shape as memory_strategy above.
add_test(NAME ir_validate COMMAND $<TARGET_FILE:gpubridge_ir_validate_test>)

# Milestone 2.5 (spec2.5.md section 12): both new tests need a real backend
# (gpuBridgeInit()), so both get the usual cpu/auto pair, same as
# profile_cpu/profile_auto above.
add_test(NAME tensor_pool_cpu COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=cpu $<TARGET_FILE:gpubridge_tensor_pool_test>)
add_test(NAME tensor_pool_auto COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=auto $<TARGET_FILE:gpubridge_tensor_pool_test>)
add_test(NAME tensor_mirror_cpu COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=cpu $<TARGET_FILE:gpubridge_tensor_mirror_test>)
add_test(NAME tensor_mirror_auto COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=auto $<TARGET_FILE:gpubridge_tensor_mirror_test>)

# Phase 4 (spec4.md section 13): edge_vocabulary needs no backend, same
# single-entry shape as memory_strategy/ir_validate above. edge_profile_cpu/
# edge_profile_auto get the usual cpu/auto pair plus GPUBRIDGE_PROFILE=1 so the
# new profile lines actually print during ctest (same reasoning as
# profile_cpu/profile_auto above).
add_test(NAME edge_vocabulary COMMAND $<TARGET_FILE:gpubridge_edge_vocabulary_test>)
add_test(NAME edge_profile_cpu  COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=cpu  GPUBRIDGE_PROFILE=1 $<TARGET_FILE:gpubridge_edge_profile_test>)
add_test(NAME edge_profile_auto COMMAND ${CMAKE_COMMAND} -E env GPUBRIDGE_BACKEND=auto GPUBRIDGE_PROFILE=1 $<TARGET_FILE:gpubridge_edge_profile_test>)

# --- GPUportal (optional sub-project, off by default) ---------------------
# Added here, after enable_testing() (a directory's add_test() calls are
# only collected into ctest if enable_testing() was already called in an
# ancestor directory scope by the time add_subdirectory() processes it —
# adding this any earlier in this file silently drops GPUportal's own
# add_test() entries from `ctest -N`, which is exactly what happened the
# first time this block was placed right after the gpubridge_runtime target
# definition above; moved here after catching that in this session's own
# verification pass) and after gpubridge_runtime is fully defined, since
# GPUportal/CMakeLists.txt links against it as an already-complete target
# and re-probes nothing (spec_gpuportal.md section 11 item 1). A binary dir
# is given explicitly because GPUportal/ is a sibling of this directory, not
# a child of it.
if(GPUBRIDGE_BUILD_PORTAL)
    add_subdirectory(${CMAKE_SOURCE_DIR}/../GPUportal ${CMAKE_BINARY_DIR}/GPUportal)
endif()
