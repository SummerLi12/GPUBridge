/*
 * gpubridge_blas.h — GpuBridge-BLAS: a small, cuBLAS-*shaped* C API
 * (spec_gpuportal.md section 6.1).
 *
 * Mirrors cuBLAS's own handle-based shape (cublasHandle_t/cublasCreate/
 * cublasSgemm) for familiarity, but is explicitly NOT ABI- or
 * source-compatible with real <cublas_v2.h> (spec_gpuportal.md section 0)
 * and computes plain C = A×B only — no transpose, no alpha/beta scaling
 * beyond the fixed 1/0 case (section 6.3). Backed entirely by the existing,
 * unmodified gpubridge-runtime OpenCL matmul_f32 path (spec2.md), including
 * its CLBlast fast path (spec3.md) when installed — this file adds no new
 * kernel and no new GpuBridgeBackend.
 *
 * The handle pins gpubridge-runtime's process-global session to
 * GPUBRIDGE_BACKEND_OPENCL (section 2: OpenCL only, no auto/cpu fallback in
 * this sub-project) and fails clearly, never silently substituting CPU, if
 * no OpenCL device is enumerable.
 */
#ifndef GPUBRIDGE_BLAS_H
#define GPUBRIDGE_BLAS_H
#include <stddef.h>

/* gpubridge_blas.c is compiled as C (GPUportal/CMakeLists.txt); guard for
 * C++ consumers (e.g. a future direct include from gpuportal_torch.cpp) the
 * same way gpubridge_portal.h does, so this header can't silently produce a
 * name-mangling link error if it's ever included from a .cpp file. */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuBridgeBlasContext *GpuBridgeBlasHandle;

typedef enum {
    GPUBRIDGE_BLAS_SUCCESS = 0,
    GPUBRIDGE_BLAS_NOT_INITIALIZED,
    GPUBRIDGE_BLAS_BACKEND_UNAVAILABLE,   /* no OpenCL device enumerable */
    GPUBRIDGE_BLAS_UNSUPPORTED_PARAM,     /* transpose != none, or alpha/beta != 1/0 */
    GPUBRIDGE_BLAS_INVALID_VALUE,         /* dimension mismatch, null pointer, etc. */
    GPUBRIDGE_BLAS_EXECUTION_FAILED       /* underlying gpubridge-runtime call failed */
} GpuBridgeBlasStatus;

typedef enum {
    GPUBRIDGE_BLAS_OP_N = 0   /* no-transpose only, this spec (section 6.3) */
} GpuBridgeBlasOperation;

/* Pins the handle's internal gpubridge-runtime context to
 * GPUBRIDGE_BACKEND_OPENCL. Fails with GPUBRIDGE_BLAS_BACKEND_UNAVAILABLE,
 * not a silent CPU fallback, if no OpenCL device is enumerable (section 2).
 *
 * gpubridge-runtime's session state is process-global (not per-handle), so
 * only one GpuBridgeBlasHandle may be live at a time — creating a second one
 * while the first is still open fails with GPUBRIDGE_BLAS_NOT_INITIALIZED
 * rather than silently stealing/resetting the first handle's session
 * (a GPUportal-local safety addition beyond the spec's literal text; see
 * GPUportal/CLAUDE.md). */
GpuBridgeBlasStatus gpuBridgeBlasCreate(GpuBridgeBlasHandle *handle);
GpuBridgeBlasStatus gpuBridgeBlasDestroy(GpuBridgeBlasHandle handle);

/* Same parameter shape as cublasSgemm (column-major m/n/k/lda/ldb/ldc
 * convention, matched deliberately for familiarity), operating on HOST
 * pointers (section 2 — no device-resident tensor bridge this round).
 * transa/transb must be GPUBRIDGE_BLAS_OP_N; *alpha must be exactly 1.0f and
 * *beta must be exactly 0.0f, else GPUBRIDGE_BLAS_UNSUPPORTED_PARAM is
 * returned before any GPU work is dispatched.
 *
 * lda/ldb/ldc must equal k/n/n respectively (matmul_f32's existing
 * row-major, zero-padding tensor model has no leading-dimension/stride
 * concept — section 7.2 already fixes this same convention for the portal
 * dispatch path). Any other value is rejected with
 * GPUBRIDGE_BLAS_INVALID_VALUE rather than silently computing a wrong
 * answer against an unsupported stride. */
GpuBridgeBlasStatus gpuBridgeBlasSgemm(
    GpuBridgeBlasHandle handle,
    GpuBridgeBlasOperation transa, GpuBridgeBlasOperation transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc);

const char *gpuBridgeBlasStatusString(GpuBridgeBlasStatus status);

#ifdef __cplusplus
}
#endif

#endif
