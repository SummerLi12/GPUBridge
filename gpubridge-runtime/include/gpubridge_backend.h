/*
 * gpubridge_backend.h — the backend vtable every execution backend must implement.
 *
 * spec1.md section 9 defines this interface verbatim. Every backend (CPU in
 * backends/cpu/gpubridge_backend_cpu.c, OpenCL in
 * backends/opencl/gpubridge_backend_opencl.c) implements every one of these
 * function pointers and exposes a single `const GpuBridgeBackend*` through its own
 * accessor (gpuBridgeBackendCpuGet / gpuBridgeBackendOpenclGet below). gpubridge_selector.c is
 * the only code that calls these accessors; gpubridge_runtime.c only ever talks to
 * whichever GpuBridgeBackend* the selector chose — it never knows or cares whether
 * that's the CPU or OpenCL implementation. This is what makes the runtime
 * "backend-neutral": adding a new backend later (SYCL, HIP, ...) means
 * writing a new file that fills in this same struct, with zero changes to
 * gpubridge_runtime.c.
 *
 * Memory model note: `malloc_device`/`copy_to_device`/`copy_to_host` and the
 * `const float*`/`float*` arguments of `launch_vector_add_f32` are all plain
 * pointers, because that is the fixed shape of this interface (per
 * spec1.md). Backends whose real device-memory handle isn't naturally a
 * pointer (OpenCL's cl_mem, in this milestone) work around this by casting
 * their handle to/from void* — see gpubridge_backend_opencl.c for exactly how and
 * why that's safe here.
 */
#ifndef GPUBRIDGE_BACKEND_H
#define GPUBRIDGE_BACKEND_H

#include <stdbool.h>
#include <stddef.h>

#include "gpubridge_memory.h"

typedef struct GpuBridgeBackend {
    /* Human-readable, lowercase backend name ("cpu", "opencl", ...). Used
     * only for diagnostics/printing, never for dispatch logic. */
    const char* name;

    /* Cheap, idempotent probe: can this backend run on this machine right
     * now? Must be safe to call repeatedly and before init(). For CPU this
     * is always true; for OpenCL this actually queries for a GPU device. */
    bool (*available)(void);

    /* One-time setup: acquire whatever context/queue/compiled-kernel state
     * the backend needs before it can execute work. Returns false and sets
     * an error retrievable via last_error() on failure. Only called after
     * available() has returned true. */
    bool (*init)(void);

    /* Tear down whatever init() set up. Safe to call even if init() was
     * never called or failed (the CPU backend's shutdown is a no-op; the
     * OpenCL backend releases only the handles that were actually created). */
    void (*shutdown)(void);

    /* Allocate `bytes` of device-resident storage and return an opaque
     * handle disguised as a pointer (see the memory-model note above).
     * Returns NULL on failure. */
    void* (*malloc_device)(size_t bytes);

    /* Free storage previously returned by malloc_device. No-op-safe on
     * NULL is not guaranteed by every backend; callers should avoid passing
     * NULL. */
    void (*free_device)(void* ptr);

    /* Copy `bytes` bytes from a host buffer `src` into a device buffer
     * `dst` (dst came from malloc_device). Blocking: must not return until
     * the data has actually landed, so callers never need to insert an
     * extra sync before using it. */
    bool (*copy_to_device)(void* dst, const void* src, size_t bytes);

    /* Copy `bytes` bytes from a device buffer `src` (from malloc_device)
     * back into a host buffer `dst`. Also blocking, for the same reason. */
    bool (*copy_to_host)(void* dst, const void* src, size_t bytes);

    /* Execute vector_add_f32: c[i] = a[i] + b[i] for i in [0, n). a, b, c
     * are device buffers (from malloc_device); n is the element count. This
     * is the only compute operation this milestone's interface supports —
     * later milestones would generalize this to a generic
     * launch_kernel(name, args, ...) once more than one op kind exists. */
    bool (*launch_vector_add_f32)(
        const float* a,
        const float* b,
        float* c,
        size_t n
    );

    /* Block until all previously issued work on this backend has completed.
     * For CPU (which is always synchronous already) this is a trivial true;
     * for OpenCL this is clFinish() on the command queue. */
    bool (*sync)(void);

    /* Return a pointer to a backend-owned, null-terminated string describing
     * the most recent failure on this backend. The string is only valid
     * until the next call into this backend; callers must copy it out (as
     * gpubridge_runtime.c does) before making another backend call. Returns an
     * empty string, never NULL, if nothing has failed yet. */
    const char* (*last_error)(void);

    /* --- Milestone 2 additions (spec2.md section 9) ---
     *
     * Four new tensor ops. All arguments are device pointers (from
     * malloc_device, or a tensor's device_ptr after gpuBridgeTensorAlloc — the two
     * are interchangeable, see spec2.md section 8.2). Same convention as
     * launch_vector_add_f32: plain pointers + size_t dimensions, bool
     * success return, no generic launch_kernel(op_kind, args) dispatch.
     */

    /* C[m,n] = sum_k( A[m,k] * B[k,n] ), naive triple-nested-loop reference
     * on CPU / one-work-item-per-output-element on OpenCL (spec2.md 7.1). */
    bool (*launch_matmul_f32)(
        const float* a,
        const float* b,
        float* c,
        size_t m,
        size_t k,
        size_t n
    );

    /* *result = sum_i( a[i] ) for i in [0, n) (spec2.md 7.2). */
    bool (*launch_reduce_sum_f32)(
        const float* a,
        float* result,
        size_t n
    );

    /* y[i,j] += bias[j] for i in [0,rows), j in [0,cols) — in-place on `y`,
     * matching spec2.md 7.3's `Y += b` semantics. */
    bool (*launch_broadcast_add_f32)(
        float* y,
        const float* bias,
        size_t rows,
        size_t cols
    );

    /* y[i] = max(0, x[i]) for i in [0, count), elementwise over a flattened
     * rank-1-or-2 tensor (spec2.md 7.4). */
    bool (*launch_relu_f32)(
        const float* x,
        float* y,
        size_t count
    );

    /* --- Milestone 1.5 addition (spec1.5.md section 7) ---
     *
     * Report what this backend's current implementation actually supports
     * (honest present-tense reporting, not hardware capability probing —
     * see spec1.5.md section 12 for the exact value each backend returns
     * and why). Returns false only if caps cannot be determined at all;
     * every implemented backend (including the OpenCL "unavailable" stub)
     * always returns true with *out_caps filled in. */
    bool (*get_caps)(GpuBridgeBackendCaps* out_caps);
} GpuBridgeBackend;

/* One accessor per implemented backend. No gpuBridgeBackendSyclGet(): SYCL has no
 * backend object in this milestone, gpubridge_selector.c special-cases it as
 * always-unavailable. */
const GpuBridgeBackend* gpuBridgeBackendCpuGet(void);
const GpuBridgeBackend* gpuBridgeBackendOpenclGet(void);
const GpuBridgeBackend* gpuBridgeBackendVulkanGet(void);   /* new, Phase 4.5 */

#endif /* GPUBRIDGE_BACKEND_H */
