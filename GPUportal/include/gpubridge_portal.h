/*
 * gpubridge_portal.h — GPUportal: the single dispatch entry-point library
 * (spec_gpuportal.md section 7).
 *
 * GpuBridgePortalGraph/GpuBridgePortalOp is a frontend-agnostic tensor-IR
 * graph, one layer above gpubridge-runtime's existing low-level
 * GpuBridgeTensorKernelIR (gpubridge_ir.h) — this spec does not replace or
 * duplicate that type (section 5). Today a graph always has exactly one
 * node (op_count == 1); the walk in gpuBridgePortalSubmit() is not
 * hard-coded to that, so a future multi-op graph does not require a
 * signature change.
 */
#ifndef GPUBRIDGE_PORTAL_H
#define GPUBRIDGE_PORTAL_H
#include <stddef.h>

/* gpubridge_blas.c/gpubridge_portal.c are compiled as C (GPUportal/CMakeLists.txt,
 * same C11 project language as gpubridge-runtime itself), so their symbols
 * have C linkage; gpuportal_torch.cpp (section 9.1) is the one C++
 * consumer of this header. Without this guard, a C++ translation unit
 * would declare these functions with C++ (name-mangled) linkage instead,
 * producing an "undefined symbol" link error against the real .a files. */
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GPUBRIDGE_PORTAL_OP_MATMUL = 0
    /* future op kinds (broadcast_add, relu, reduce_sum) are new enum
       values here + a new switch arm in gpuBridgePortalSubmit() — additive,
       not a redesign (non-goal #7). */
} GpuBridgePortalOpKind;

typedef struct {
    GpuBridgePortalOpKind kind;
    int m, n, k;              /* matmul only, this spec; a future op kind
                                  would add its own shape fields here */
    const float *input_a;
    const float *input_b;
    float *output;
} GpuBridgePortalOp;

typedef struct {
    GpuBridgePortalOp *ops;
    size_t op_count;           /* == 1 for every graph this spec produces */
} GpuBridgePortalGraph;

typedef enum {
    GPUBRIDGE_PORTAL_OK = 0,
    GPUBRIDGE_PORTAL_EMPTY_GRAPH,
    GPUBRIDGE_PORTAL_UNSUPPORTED_OP,
    GPUBRIDGE_PORTAL_BLAS_ERROR
} GpuBridgePortalStatus;

/* Owns one GpuBridgeBlasHandle internally (created lazily on first call,
   destroyed by gpuBridgePortalShutdown()). Walks graph->ops in order —
   trivial for op_count == 1, but not hard-coded to exactly one, so a
   future multi-op graph does not require a signature change. */
GpuBridgePortalStatus gpuBridgePortalSubmit(const GpuBridgePortalGraph *graph);
void gpuBridgePortalShutdown(void);
const char *gpuBridgePortalStatusString(GpuBridgePortalStatus status);

#ifdef __cplusplus
}
#endif

#endif
