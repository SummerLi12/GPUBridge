/*
 * gpubridge_portal.c — implementation of GPUportal's dispatch behavior
 * (spec_gpuportal.md section 7.2).
 *
 * For each op in graph->ops (in order), switches on op->kind. The only
 * implemented case, GPUBRIDGE_PORTAL_OP_MATMUL, calls gpuBridgeBlasSgemm()
 * with transa=transb=GPUBRIDGE_BLAS_OP_N, alpha=1.0f, beta=0.0f, lda=k,
 * ldb=n, ldc=n — the row-major-consistent defaults matching matmul_f32's
 * existing row-major convention (a deliberate, documented difference from
 * cuBLAS's own column-major default). Any other kind value returns
 * GPUBRIDGE_PORTAL_UNSUPPORTED_OP, reachable in practice today only via a
 * hand-corrupted GpuBridgePortalOp.
 */
#include "gpubridge_portal.h"
#include "gpubridge_blas.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

/* Reuses gpubridge-runtime's own GPUBRIDGE_VERBOSE=1 convention (the
 * internal gpuBridgeEnvVerbose() helper is not part of that project's
 * public header surface, so this GPUportal-local file re-checks getenv()
 * directly, same "1"-only strict match every other *_VERBOSE/_DISABLE env
 * var in this codebase uses). Shows the tensor-IR op this graph carries
 * before it is handed down to GpuBridge-BLAS. */
static bool portal_verbose(void)
{
    const char* val = getenv("GPUBRIDGE_VERBOSE");
    return val != NULL && strcmp(val, "1") == 0;
}

/* One GpuBridgeBlasHandle, created lazily on first gpuBridgePortalSubmit()
 * call and destroyed by gpuBridgePortalShutdown() — matches
 * GpuBridgeBlasHandle's own "process-global runtime session, single
 * live handle" shape (gpubridge_blas.c), so GPUportal never tries to hold
 * more than one handle open at a time either. */
static GpuBridgeBlasHandle g_blas_handle = NULL;

static const char *g_status_strings[] = {
    "ok",
    "empty graph",
    "unsupported op",
    "blas error"
};
//bound checking for the status string array, to avoid out-of-bounds access in gpuBridgePortalStatusString().
const char *gpuBridgePortalStatusString(GpuBridgePortalStatus status)
{
    size_t idx = (size_t)status;
    if (status < 0 || idx >= sizeof(g_status_strings) / sizeof(g_status_strings[0])) {
        return "unknown GpuBridgePortalStatus";
    }
    return g_status_strings[idx];
}
//This is GPUportal's single dispatch entry point — every op in the graph gets executed against GpuBridge-BLAS, in order, synchronously, before this function returns.
GpuBridgePortalStatus gpuBridgePortalSubmit(const GpuBridgePortalGraph *graph)
{
    bool verbose = portal_verbose();

    if (verbose) {
        printf("we are in function: gpuBridgePortalSubmit\n"
            "purpose: GPUportal's single dispatch entry point -- walk every "
            "op in the caller's high-level tensor-IR graph and run each one "
            "against GpuBridge-BLAS, in order, synchronously\n");
    }

    if (graph == NULL || graph->ops == NULL || graph->op_count == 0) {
        if (verbose) {
            printf("leaving function: gpuBridgePortalSubmit, "
                "result: failed (empty graph)\n\n");
        }
        return GPUBRIDGE_PORTAL_EMPTY_GRAPH;
    }

    if (verbose) {
        /* Line-buffer stdout for the rest of this process: when stdout is
         * redirected to a file (e.g. "> out.txt"), C's stdio and Python's
         * own print() each keep independent, fully-buffered-by-default
         * buffers on the same fd, so their output otherwise lands in
         * whichever buffer happens to flush first (typically at process
         * exit) rather than in true call order — this is purely a
         * demo/debug readability fix, harmless to correctness. */
        setvbuf(stdout, NULL, _IOLBF, BUFSIZ);
        printf("GPUportal: tensor IR graph received, op_count=%zu\n", graph->op_count);
        /* Dump each op's raw GpuBridgePortalOp struct fields, one per line for
         * readability — kind/m/n/k are plain scalars, input_a/input_b/output
         * are host pointer addresses only (no tensor data has been read or
         * copied yet at this point). */
        printf("High Level tensor IR\n");
        for (size_t dbg_i = 0; dbg_i < graph->op_count; dbg_i++) {
            const GpuBridgePortalOp *dbg_op = &graph->ops[dbg_i];
            printf("GPUportal tensor IR op[%zu]:\n", dbg_i);
            printf("GPUportal tensor IR   kind: %d (matmul)\n", (int)dbg_op->kind);
            printf("GPUportal tensor IR   m: %d\n", dbg_op->m);
            printf("GPUportal tensor IR   n: %d\n", dbg_op->n);
            printf("GPUportal tensor IR   k: %d\n", dbg_op->k);
            printf("GPUportal tensor IR   input_a: %p\n", (const void*)dbg_op->input_a);
            printf("GPUportal tensor IR   input_b: %p\n", (const void*)dbg_op->input_b);
            printf("GPUportal tensor IR   output: %p\n", (void*)dbg_op->output);
        }
    }

    if (g_blas_handle == NULL) {
        if (verbose) {
            printf("GPUportal: no GpuBridgeBlasHandle live yet, creating one ");
            printf("pins gpubridge-runtime session to GPUBRIDGE_BACKEND_OPENCL\n");
        }
        if (gpuBridgeBlasCreate(&g_blas_handle) != GPUBRIDGE_BLAS_SUCCESS) {
            if (verbose) {
                printf("leaving function: gpuBridgePortalSubmit, "
                    "result: failed (gpuBridgeBlasCreate error)\n\n");
            }
            return GPUBRIDGE_PORTAL_BLAS_ERROR;
        }
    }

    for (size_t i = 0; i < graph->op_count; i++) {
        const GpuBridgePortalOp *op = &graph->ops[i];

        switch (op->kind) {
        case GPUBRIDGE_PORTAL_OP_MATMUL: {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            if (verbose) {
                printf("GPUportal: op[%zu] kind=matmul: A[%d,%d] x B[%d,%d] -> "
                    "C[%d,%d] (tensor-IR shapes as received from the caller)\n",
                    i, op->m, op->k, op->k, op->n, op->m, op->n);
                printf("GPUportal: dispatching op[%zu] to GpuBridge-BLAS Sgemm("
                    "transa=N, transb=N, m=%d, n=%d, k=%d, alpha=1.0, "
                    "lda=%d, ldb=%d, beta=0.0, ldc=%d)\n",
                    i, op->m, op->n, op->k, op->k, op->n, op->n);
            }
            GpuBridgeBlasStatus status = gpuBridgeBlasSgemm(
                g_blas_handle,
                GPUBRIDGE_BLAS_OP_N, GPUBRIDGE_BLAS_OP_N,
                op->m, op->n, op->k,
                &alpha,
                op->input_a, op->k,//lda=k: leadig dimension of A is k, because A is m x k
                op->input_b, op->n,//ldb=n
                &beta,
                op->output, op->n);//ldc=n
            if (status != GPUBRIDGE_BLAS_SUCCESS) {
                if (verbose) {
                    printf("leaving function: gpuBridgePortalSubmit, "
                        "result: failed (op[%zu] gpuBridgeBlasSgemm error: %s)\n\n",
                        i, gpuBridgeBlasStatusString(status));
                }
                return GPUBRIDGE_PORTAL_BLAS_ERROR;
            }
            if (verbose) {
                printf("GPUportal: op[%zu] completed, status=%s\n", i,
                    gpuBridgeBlasStatusString(status));
            }
            break;
        }
        default:
            if (verbose) {
                printf("leaving function: gpuBridgePortalSubmit, "
                    "result: failed (op[%zu] unsupported op kind=%d)\n\n",
                    i, (int)op->kind);
            }
            return GPUBRIDGE_PORTAL_UNSUPPORTED_OP;
        }
    }

    if (verbose) {
        printf("leaving function: gpuBridgePortalSubmit, "
            "result: success (all %zu op(s) completed)\n\n", graph->op_count);
    }
    return GPUBRIDGE_PORTAL_OK;
}

//frees the handle — and resets the global back to NULL.
//If it's already NULL (never created, or already shut down), it's a safe no-op.
void gpuBridgePortalShutdown(void)
{
    if (g_blas_handle != NULL) {
        gpuBridgeBlasDestroy(g_blas_handle);
        g_blas_handle = NULL;
    }
}
