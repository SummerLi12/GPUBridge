/*
 * gpubridge_blas.c — implementation of GpuBridge-BLAS (spec_gpuportal.md
 * section 6.2).
 *
 * gpuBridgeBlasSgemm's steps map directly onto section 6.2's four items:
 *   1. reject unsupported transa/transb/alpha/beta before touching the GPU
 *      (section 6.3);
 *   2. reject basic m/n/k/lda/ldb/ldc inconsistency, also before touching
 *      the GPU;
 *   3. gpuBridgeTensorAlloc / gpuBridgeMemcpy / gpuBridgeLaunchTensorKernel
 *      (matmul_f32, spec2.md's builder reused verbatim) / gpuBridgeMemcpy /
 *      gpuBridgeTensorFree;
 *   4. map any gpubridge-runtime failure to GPUBRIDGE_BLAS_EXECUTION_FAILED,
 *      surfacing gpuBridgeGetLastErrorString() through a handle-local
 *      last-error buffer.
 */
#include "gpubridge_blas.h"
#include "gpubridge_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Same GPUBRIDGE_VERBOSE=1 convention as gpubridge_portal.c and every
 * gpubridge-runtime backend — reused here at the BLAS layer so the caller
 * can see A/B/C actually leaving the host, hitting gpuBridgeTensorAlloc/
 * gpuBridgeLaunchTensorKernel (which itself prints gpubridge-runtime's own
 * "gemm path selected"/backend-selection lines when GPUBRIDGE_VERBOSE=1 is
 * set), and coming back. */
static bool blas_verbose(void)
{
    const char* val = getenv("GPUBRIDGE_VERBOSE");
    return val != NULL && strcmp(val, "1") == 0;
}

#define GPUBRIDGE_BLAS_ERROR_BUF_LEN 256

struct GpuBridgeBlasContext {
    char last_error[GPUBRIDGE_BLAS_ERROR_BUF_LEN];
};

/* gpubridge-runtime's session state is process-global (gpubridge_runtime.c's
 * static g_initialized/g_selection), not per-handle — GpuBridgeBlasHandle is
 * a thin, cuBLAS-shaped wrapper around that shared state (spec_gpuportal.md
 * non-goal #11: single-threaded only, same standing non-goal the rest of
 * this project already states). This one static tracks whether a handle is
 * currently live, so a second gpuBridgeBlasCreate() call while the first
 * handle is still open fails loudly instead of silently re-pinning/
 * resetting the shared session out from under the first handle — a
 * GPUportal-local safety addition, documented in GPUportal */
static bool g_handle_live = false;

static const char* g_status_strings[] = {
    "success",
    "not initialized",
    "backend unavailable (no OpenCL device enumerable)",
    "unsupported parameter",
    "invalid value",
    "execution failed"
};

const char *gpuBridgeBlasStatusString(GpuBridgeBlasStatus status)
{
    size_t idx = (size_t)status;
    if (status < 0 || idx >= sizeof(g_status_strings) / sizeof(g_status_strings[0])) {
        return "unknown GpuBridgeBlasStatus";
    }
    return g_status_strings[idx];
}

GpuBridgeBlasStatus gpuBridgeBlasCreate(GpuBridgeBlasHandle *handle)
{
    bool verbose = blas_verbose();
    if (verbose) {
        printf("we are in function: gpuBridgeBlasCreate\n"
            "purpose: allocate a cuBLAS-shaped session handle and pin "
            "gpubridge-runtime's shared session to GPUBRIDGE_BACKEND_OPENCL\n");
    }

    if (handle == NULL) {
        if (verbose) {
            printf("leaving function: gpuBridgeBlasCreate, "
                "result: failed (handle == NULL)\n\n");
        }
        return GPUBRIDGE_BLAS_INVALID_VALUE;
    }
    *handle = NULL;

    if (g_handle_live) {
        /* See the g_handle_live comment above: this is the GPUportal-local
         * "only one handle at a time" safety guard, not a spec_gpuportal.md
         * literal requirement. */
        if (verbose) {
            printf("leaving function: gpuBridgeBlasCreate, "
                "result: failed (a GpuBridgeBlasHandle is already live)\n\n");
        }
        return GPUBRIDGE_BLAS_NOT_INITIALIZED;
    }

    struct GpuBridgeBlasContext *ctx = calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        if (verbose) {
            printf("leaving function: gpuBridgeBlasCreate, "
                "result: failed (calloc)\n\n");
        }
        return GPUBRIDGE_BLAS_EXECUTION_FAILED;
    }

    /* Section 2: OpenCL only, no auto/cpu fallback. gpuBridgeSetBackend
     * explicitly pins the shared gpubridge-runtime session to
     * GPUBRIDGE_BACKEND_OPENCL, failing clearly (never silently
     * substituting CPU) if no OpenCL device is enumerable — the same "fail
     * loudly, never silently substitute" rule spec4.5.md's Vulkan selector
     * case already follows. */
     //0=success, failure=nonzero
     //it also pins the shared runtime session to OpenCL. So ctx isn't just raw memory — its existence also represents "an OpenCL session is live."
    if (gpuBridgeSetBackend(GPUBRIDGE_BACKEND_OPENCL) != 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
            "OpenCL backend unavailable: %s", gpuBridgeGetLastErrorString());
        if (verbose) {
            printf("leaving function: gpuBridgeBlasCreate, "
                "result: failed (%s)\n\n", ctx->last_error);
        }
        free(ctx);
        return GPUBRIDGE_BLAS_BACKEND_UNAVAILABLE;
    }

    g_handle_live = true;//not a second handle can be created while this one is live
    *handle = ctx;
    if (verbose) {
        printf("leaving function: gpuBridgeBlasCreate, "
            "result: success (session pinned to opencl)\n\n");
    }
    return GPUBRIDGE_BLAS_SUCCESS;
}

GpuBridgeBlasStatus gpuBridgeBlasDestroy(GpuBridgeBlasHandle handle)
{
    if (handle == NULL) {
        return GPUBRIDGE_BLAS_NOT_INITIALIZED;
    }
    gpuBridgeShutdown();
    free(handle);
    g_handle_live = false;
    return GPUBRIDGE_BLAS_SUCCESS;// no error
}

GpuBridgeBlasStatus gpuBridgeBlasSgemm(
    GpuBridgeBlasHandle handle,
    GpuBridgeBlasOperation transa, GpuBridgeBlasOperation transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc)
{
    bool verbose = blas_verbose();
    if (verbose) {
        printf("we are in function: gpuBridgeBlasSgemm\n"
            "purpose: cuBLAS-shaped GEMM entry point -- validate params, "
            "then alloc/upload/launch/sync/download through gpubridge-runtime's "
            "tensor API (m=%d n=%d k=%d)\n", m, n, k);
    }

    if (handle == NULL) {
        if (verbose) {
            printf("leaving function: gpuBridgeBlasSgemm, "
                "result: failed (handle == NULL)\n\n");
        }
        return GPUBRIDGE_BLAS_NOT_INITIALIZED;
    }
    //sets just the first byte of that buffer to the null terminator — in C, a '\0' at position 0 makes the whole string read as empty (""), even though the rest of the buffer's old bytes are still sitting there physically. It's the cheap idiom for "clear this C string" without needing memset on the whole buffer.
    handle->last_error[0] = '\0';

    if (verbose) {
        printf("GpuBridge-BLAS: Sgemm called: m=%d n=%d k=%d lda=%d ldb=%d "
            "ldc=%d alpha=%g beta=%g\n",
            m, n, k, lda, ldb, ldc, alpha ? (double)*alpha : 0.0,
            beta ? (double)*beta : 0.0);
    }

    /* Step 1 (section 6.3): honest rejection of unsupported cuBLAS
     * parameters, checked before any GPU work is dispatched. */
    if (transa != GPUBRIDGE_BLAS_OP_N || transb != GPUBRIDGE_BLAS_OP_N) {
        snprintf(handle->last_error, sizeof(handle->last_error),
            "unsupported parameter: transa/transb must be GPUBRIDGE_BLAS_OP_N "
            "(transposed GEMM is not implemented this round)");
        if (verbose) {
            printf("leaving function: gpuBridgeBlasSgemm, "
                "result: failed (%s)\n\n", handle->last_error);
        }
        return GPUBRIDGE_BLAS_UNSUPPORTED_PARAM;
    }
    if (alpha == NULL || beta == NULL) {
        snprintf(handle->last_error, sizeof(handle->last_error),
            "invalid value: alpha/beta must not be NULL");
        if (verbose) {
            printf("leaving function: gpuBridgeBlasSgemm, "
                "result: failed (%s)\n\n", handle->last_error);
        }
        return GPUBRIDGE_BLAS_INVALID_VALUE;
    }
    if (*alpha != 1.0f) {
        snprintf(handle->last_error, sizeof(handle->last_error),
            "unsupported parameter: alpha must be exactly 1.0f (got %g; "
            "arbitrary scaling is not implemented this round)", (double)*alpha);
        if (verbose) {
            printf("leaving function: gpuBridgeBlasSgemm, "
                "result: failed (%s)\n\n", handle->last_error);
        }
        return GPUBRIDGE_BLAS_UNSUPPORTED_PARAM;
    }
    if (*beta != 0.0f) {
        snprintf(handle->last_error, sizeof(handle->last_error),
            "unsupported parameter: beta must be exactly 0.0f (got %g; "
            "accumulation into C is not implemented this round)", (double)*beta);
        if (verbose) {
            printf("leaving function: gpuBridgeBlasSgemm, "
                "result: failed (%s)\n\n", handle->last_error);
        }
        return GPUBRIDGE_BLAS_UNSUPPORTED_PARAM;
    }

    /* Step 2: basic m/n/k/lda/ldb/ldc consistency, also before touching the
     * GPU. matmul_f32's tensor model (spec2.md) is tightly-packed row-major
     * with no leading-dimension/stride concept, so lda/ldb/ldc must equal
     * k/n/n exactly — the same row-major-consistent convention section 7.2
     * fixes for the portal dispatch path. Rejecting any other value here
     * (rather than silently misinterpreting a strided/padded caller buffer)
     * follows spec3.md section 2's "must never silently fail into a slow or
     * incorrect path" rule, which this spec's section 2 explicitly reuses. */
    if (A == NULL || B == NULL || C == NULL) {
        snprintf(handle->last_error, sizeof(handle->last_error),
            "invalid value: A/B/C must not be NULL");
        if (verbose) {
            printf("leaving function: gpuBridgeBlasSgemm, "
                "result: failed (%s)\n\n", handle->last_error);
        }
        return GPUBRIDGE_BLAS_INVALID_VALUE;
    }
    if (m <= 0 || n <= 0 || k <= 0) {
        snprintf(handle->last_error, sizeof(handle->last_error),
            "invalid value: m/n/k must be positive (got m=%d n=%d k=%d)", m, n, k);
        if (verbose) {
            printf("leaving function: gpuBridgeBlasSgemm, "
                "result: failed (%s)\n\n", handle->last_error);
        }
        return GPUBRIDGE_BLAS_INVALID_VALUE;
    }
    if (lda != k || ldb != n || ldc != n) {
        snprintf(handle->last_error, sizeof(handle->last_error),
            "invalid value: lda/ldb/ldc must equal k/n/n respectively "
            "(row-major, tightly packed — got lda=%d want %d, ldb=%d want %d, "
            "ldc=%d want %d)", lda, k, ldb, n, ldc, n);
        if (verbose) {
            printf("leaving function: gpuBridgeBlasSgemm, "
                "result: failed (%s)\n\n", handle->last_error);
        }
        return GPUBRIDGE_BLAS_INVALID_VALUE;
    }

    /* Step 3: alloc/copy-to-device/launch/copy-back/free, entirely through
     * the existing, unmodified gpubridge-runtime tensor API. Zero-initialized
     * so gpuBridgeTensorFree() below is a safe no-op on whichever of these
     * three never successfully allocated. */
    GpuBridgeTensorDesc a_dev = { 0 };
    GpuBridgeTensorDesc b_dev = { 0 };
    GpuBridgeTensorDesc c_dev = { 0 };
    const size_t a_shape[2] = { (size_t)m, (size_t)k };
    const size_t b_shape[2] = { (size_t)k, (size_t)n };
    const size_t c_shape[2] = { (size_t)m, (size_t)n };

    GpuBridgeBlasStatus status = GPUBRIDGE_BLAS_SUCCESS;//no error

    if (verbose) {
        printf("GpuBridge-BLAS: params validated OK, allocating device tensors "
            "A[%d,%d] B[%d,%d] C[%d,%d] via gpuBridgeTensorAlloc "
            "(gpubridge-runtime, currently pinned to GPUBRIDGE_BACKEND_OPENCL)\n",
            m, k, k, n, m, n);
    }
    //requests a device tensor
    if (gpuBridgeTensorAlloc(&a_dev, GPUBRIDGE_TYPE_F32, 2, a_shape) != 0 ||
        gpuBridgeTensorAlloc(&b_dev, GPUBRIDGE_TYPE_F32, 2, b_shape) != 0 ||
        gpuBridgeTensorAlloc(&c_dev, GPUBRIDGE_TYPE_F32, 2, c_shape) != 0) {
        snprintf(handle->last_error, sizeof(handle->last_error),
            "gpuBridgeTensorAlloc failed: %s", gpuBridgeGetLastErrorString());
        status = GPUBRIDGE_BLAS_EXECUTION_FAILED;
        goto cleanup;
    }

    if (verbose) {
        printf("GpuBridge-BLAS: copying A (%zu bytes) and B (%zu bytes) "
            "host -> device via gpuBridgeMemcpy\n",
            (size_t)m * (size_t)k * sizeof(float),
            (size_t)k * (size_t)n * sizeof(float));
    }
    if (gpuBridgeMemcpy(a_dev.device_ptr, A, (size_t)m * (size_t)k * sizeof(float),
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0 ||
        gpuBridgeMemcpy(b_dev.device_ptr, B, (size_t)k * (size_t)n * sizeof(float),
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0) {
        snprintf(handle->last_error, sizeof(handle->last_error),
            "gpuBridgeMemcpy (host to device) failed: %s", gpuBridgeGetLastErrorString());
        status = GPUBRIDGE_BLAS_EXECUTION_FAILED;
        goto cleanup;
    }

    {
        GpuBridgeTensorKernelIR kernel;//describes what op to run and its shape (matmul_f32 with dimensions m×k times k×n).
        GpuBridgeTensorArg kernel_args[3];
        //build low level tensor IR for matmul_f32, using the shapes of the device tensors we just allocated
        //no device pointers yet at this point, just shape/type info.
        gpuBridgeIrInitMatmulF32(&kernel, kernel_args, (size_t)m, (size_t)k, (size_t)n);

        if (verbose) {
            printf("GpuBridge-BLAS: launching matmul_f32 via "
                "gpuBridgeLaunchTensorKernel on backend=%s (set "
                "GPUBRIDGE_VERBOSE=1 to also see gpubridge-runtime's own "
                "backend-selection / gemm-path-selected lines below)\n",
                gpuBridgeBackendKindToString(gpuBridgeGetSelectedBackend()));
        }
        //3 device pointers, in the order the kernel expects them (A, B, C)
        void *launch_args[3] = { a_dev.device_ptr, b_dev.device_ptr, c_dev.device_ptr };
        if (gpuBridgeLaunchTensorKernel(&kernel, launch_args, 3) != 0) {
            snprintf(handle->last_error, sizeof(handle->last_error),
                "gpuBridgeLaunchTensorKernel failed: %s", gpuBridgeGetLastErrorString());
            status = GPUBRIDGE_BLAS_EXECUTION_FAILED;
            goto cleanup;
        }
    }

    if (gpuBridgeDeviceSynchronize() != 0) {
        snprintf(handle->last_error, sizeof(handle->last_error),
            "gpuBridgeDeviceSynchronize failed: %s", gpuBridgeGetLastErrorString());
        status = GPUBRIDGE_BLAS_EXECUTION_FAILED;
        goto cleanup;
    }

    if (verbose) {
        printf("GpuBridge-BLAS: copying C (%zu bytes) device -> host via "
            "gpuBridgeMemcpy\n", (size_t)m * (size_t)n * sizeof(float));
    }
    if (gpuBridgeMemcpy(C, c_dev.device_ptr, (size_t)m * (size_t)n * sizeof(float),
            GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) != 0) {
        snprintf(handle->last_error, sizeof(handle->last_error),
            "gpuBridgeMemcpy (device to host) failed: %s", gpuBridgeGetLastErrorString());
        status = GPUBRIDGE_BLAS_EXECUTION_FAILED;
    }

    if (verbose && status == GPUBRIDGE_BLAS_SUCCESS) {
        printf("GpuBridge-BLAS: Sgemm complete, freeing device tensors\n");
    }

cleanup:
    gpuBridgeTensorFree(&a_dev);
    gpuBridgeTensorFree(&b_dev);
    gpuBridgeTensorFree(&c_dev);

    if (verbose) {
        if (status == GPUBRIDGE_BLAS_SUCCESS) {
            printf("leaving function: gpuBridgeBlasSgemm, "
                "result: success (C[%d,%d] computed)\n\n", m, n);
        } else {
            printf("leaving function: gpuBridgeBlasSgemm, "
                "result: failed (%s)\n\n", handle->last_error);
        }
    }
    return status;
}
