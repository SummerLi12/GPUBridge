/*
 * gpubridge_backend_cpu.c — the CPU fallback backend.
 *
 * Implements the GpuBridgeBackend vtable (gpubridge_backend.h) using nothing but plain C
 * heap memory and a for-loop. This backend must always work — it is the
 * guaranteed fallback when no GPU backend is available (spec1.md's
 * GPUBRIDGE_BACKEND=auto policy, and the AUTO case in gpubridge_selector.c), so every
 * function here is written to be effectively infallible.
 *
 * Memory model: unlike the OpenCL backend, there is no separate device
 * address space here. A "device" pointer returned by cpu_malloc_device() is
 * just a normal malloc()'d host pointer. copy_to_device/copy_to_host are
 * still real memcpy()s (not no-ops), because the source and destination are
 * always two *distinct* allocations (the caller's own host array vs. the
 * buffer this backend allocated) even though both happen to live in the
 * same host RAM.
 */
#include "gpubridge_backend.h"

#include <stdlib.h>
#include <string.h>

/* Milestone 3 additions (spec3.md sections 6, 8.5): optional OpenBLAS GEMM
 * acceleration, and gpuBridgeEnvVendorGemmDisable() for the dispatch logic
 * below. gpubridge_internal.h's "not included by any file outside src/"
 * comment is deliberately crossed here for the same reason
 * gpubridge_backend_opencl.c crosses it — an optional runtime-library
 * integration, not the original backend-selection plumbing. */
#include "gpubridge_cpu_openblas.h"
#include "gpubridge_internal.h"

#include <stdio.h>

/* Milestone 3 addition: this backend previously had no failure modes worth
 * reporting (see cpu_last_error()'s original comment), but
 * cpu_launch_matmul_f32's OpenBLAS dimension-bounds check (spec3.md
 * section 6.1) can now genuinely fail — needs somewhere to record why. */
static char g_last_error[256] = "";

static void set_error(const char* msg)
{
    snprintf(g_last_error, sizeof(g_last_error), "%s", msg);
}

static bool cpu_available(void)
{
    /* The CPU backend has no external dependency to probe for — it can
     * always run. */
    return true;
}

static bool cpu_init(void)
{
    /* Nothing else to set up: no context, no queue, no compiled kernel.
     * Milestone 3 (spec3.md section 6.2): optional OpenBLAS probe, result
     * intentionally ignored — OpenBLAS is acceleration, not a requirement,
     * so it must never cause cpu_init() itself to fail (CPU must always
     * succeed). */
    gpuBridgeCpuOpenblasTryInit();
    return true;
}

static void cpu_shutdown(void)
{
    /* Nothing to tear down, symmetric with cpu_init(). */
}

static void* cpu_malloc_device(size_t bytes)
{
    /* "Device" memory for this backend is just heap memory. Returns NULL on
     * allocation failure, matching the GpuBridgeBackend contract. */
    return malloc(bytes);
}

static void cpu_free_device(void* ptr)
{
    free(ptr);
}

static bool cpu_copy_to_device(void* dst, const void* src, size_t bytes)
{
    /* dst is a buffer previously returned by cpu_malloc_device(); src is the
     * caller's host array. This is a real copy between two distinct
     * buffers, even though there's no real host/device boundary to cross. */
    memcpy(dst, src, bytes);
    return true;
}

static bool cpu_copy_to_host(void* dst, const void* src, size_t bytes)
{
    /* Mirror image of copy_to_device: src is the "device" buffer, dst is
     * the caller's host array. */
    memcpy(dst, src, bytes);
    return true;
}

static bool cpu_launch_vector_add_f32(
    const float* a, const float* b, float* c, size_t n)
{
    /* The reference implementation of vector_add_f32 (spec1.md section 12,
     * verbatim): every other backend's result is checked against what this
     * loop would produce. */
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
    return true;
}

static bool cpu_sync(void)
{
    /* Every operation above is already synchronous (a plain function call
     * blocks until it returns), so there is nothing to wait for. */
    return true;
}

static const char* cpu_last_error(void)
{
    /* Milestone 3: g_last_error is set only by cpu_launch_matmul_f32's
     * OpenBLAS dimension-bounds check (spec3.md section 6.1) — every other
     * function in this backend remains infallible, matching this file's
     * original design (malloc failure aside, vanishingly unlikely for the
     * sizes this milestone uses), so this is "" unless that specific check
     * has ever fired. Never returns NULL. */
    return g_last_error;
}

/* --- Milestone 2 additions (spec2.md section 7): CPU reference
 * implementations of the four new tensor ops. Each is the literal reference
 * loop described in its spec2.md subsection — these are what the OpenCL
 * kernels (and the capstone test's tolerance check) are validated against. */

static bool cpu_launch_matmul_f32(
    const float* a, const float* b, float* c, size_t m, size_t k, size_t n)
{
    /* Milestone 3 (spec3.md section 6.2): try the optional OpenBLAS fast
     * path first. Unlike the OpenCL/CLBlast dispatch, cblas_sgemm has no
     * error-return convention — the only failure gpuBridgeCpuOpenblasSgemmF32
     * can report is the dimension-bounds check (m/n/k > INT_MAX), which is
     * a genuine fault, not silently retried on the naive loop below even
     * though the naive loop could technically handle the size — consistent,
     * predictable behavior over a clever save. */
    if (gpuBridgeCpuOpenblasAvailable() && !gpuBridgeEnvVendorGemmDisable()) {
        char err_buf[128];
        if (gpuBridgeCpuOpenblasSgemmF32(a, b, c, m, k, n, err_buf, sizeof(err_buf))) {
            return true;
        }
        set_error(err_buf);
        return false;
    }

    /* C[m,n] = sum_k( A[m,k] * B[k,n] ), naive triple-nested loop (spec2.md
     * 7.1) — no tiling, no blocking, matching this milestone's "correctness
     * only" performance non-goal. */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    return true;
}

static bool cpu_launch_reduce_sum_f32(
    const float* a, float* result, size_t n)
{
    /* Linear accumulation loop (spec2.md 7.2). Accumulates in float, same
     * precision profile the OpenCL kernel's naive sequential accumulation
     * will use — the 1e-3 relative tolerance (section 7.2) accounts for the
     * resulting fp32 rounding error. */
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += a[i];
    }
    *result = sum;
    return true;
}

static bool cpu_launch_broadcast_add_f32(
    float* y, const float* bias, size_t rows, size_t cols)
{
    /* Y[i,j] += b[j], in-place on y's buffer (spec2.md 7.3). */
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            y[i * cols + j] += bias[j];
        }
    }
    return true;
}

static bool cpu_launch_relu_f32(const float* x, float* y, size_t count)
{
    /* y[i] = max(0, x[i]), elementwise over the flattened element count
     * (spec2.md 7.4) — works identically for rank 1 or rank 2 since this op
     * has no cross-element structure. */
    for (size_t i = 0; i < count; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
    return true;
}

/* --- Milestone 1.5 addition (spec1.5.md section 12) --- */

static bool cpu_get_caps(GpuBridgeBackendCaps* out_caps)
{
    /* Every capability is false: copy_to_device/copy_to_host above are real
     * memcpy()s between two distinct allocations, not a shared-address/SVM
     * mechanism, so this backend has no "shared memory" to offer despite
     * both sides living in host RAM. No pinning, no async transfers (every
     * call above is a plain blocking function call), no events. This makes
     * gpuBridgeChooseMemoryStrategy() fall through to GPUBRIDGE_MEMORY_HOST_ONLY
     * for this backend, which accurately describes it. */
    *out_caps = (GpuBridgeBackendCaps){ 0 };
    /* Milestone 3 (spec3.md section 8.2): the one non-zero field, honest
     * "loaded" reporting only. */
    out_caps->vendor_gemm_loaded = gpuBridgeCpuOpenblasAvailable();
    return true;
}

/* The CPU backend's vtable instance. `static const` because exactly one
 * instance ever exists and nothing outside this file should be able to
 * mutate it; gpuBridgeBackendCpuGet() hands out its address, never a copy. */
static const GpuBridgeBackend g_cpu_backend = {
    .name = "cpu",
    .available = cpu_available,
    .init = cpu_init,
    .shutdown = cpu_shutdown,
    .malloc_device = cpu_malloc_device,
    .free_device = cpu_free_device,
    .copy_to_device = cpu_copy_to_device,
    .copy_to_host = cpu_copy_to_host,
    .launch_vector_add_f32 = cpu_launch_vector_add_f32,
    .sync = cpu_sync,
    .last_error = cpu_last_error,
    .launch_matmul_f32 = cpu_launch_matmul_f32,
    .launch_reduce_sum_f32 = cpu_launch_reduce_sum_f32,
    .launch_broadcast_add_f32 = cpu_launch_broadcast_add_f32,
    .launch_relu_f32 = cpu_launch_relu_f32,
    .get_caps = cpu_get_caps,
};

const GpuBridgeBackend* gpuBridgeBackendCpuGet(void)
{
    return &g_cpu_backend;
}
