/*
 * gpubridge_ir.c — implements the one IR-builder helper declared in gpubridge_ir.h.
 *
 * See gpubridge_ir.h for why this function exists (it's a small addition beyond
 * spec1.md's literal text, added to avoid duplicating GpuBridgeKernelIR/
 * GpuBridgeBufferArg construction between callers).
 */
#include "gpubridge_ir.h"

#include <stdio.h>

void gpuBridgeIrInitVectorAddF32(GpuBridgeKernelIR* out, GpuBridgeBufferArg out_args[3], size_t n)
{
    /* Input buffer a: read-only, n elements of f32. */
    out_args[0].name = "a";
    out_args[0].type = GPUBRIDGE_TYPE_F32;
    out_args[0].length = n;
    out_args[0].is_input = true;
    out_args[0].is_output = false;

    /* Input buffer b: read-only, n elements of f32. */
    out_args[1].name = "b";
    out_args[1].type = GPUBRIDGE_TYPE_F32;
    out_args[1].length = n;
    out_args[1].is_input = true;
    out_args[1].is_output = false;

    /* Output buffer c: write-only, n elements of f32. */
    out_args[2].name = "c";
    out_args[2].type = GPUBRIDGE_TYPE_F32;
    out_args[2].length = n;
    out_args[2].is_input = false;
    out_args[2].is_output = true;

    /* Tie the three buffer descriptors together into one kernel launch
     * description. global_size = n because vector_add_f32 launches exactly
     * one parallel lane per output element (see the OpenCL kernel's
     * get_global_id(0) and the CPU backend's for-loop bound). */
    out->kernel_name = "vector_add_f32";
    out->op_kind = GPUBRIDGE_OP_VECTOR_ADD;
    out->args = out_args;
    out->arg_count = 3;
    out->global_size = n;
}

/* --- Milestone 2 additions (spec2.md section 10) --- */

void gpuBridgeIrInitMatmulF32(GpuBridgeTensorKernelIR* out, GpuBridgeTensorArg out_args[3],
    size_t m, size_t k, size_t n)
{
    /* Input a: rank 2, shape [m,k]. */
    out_args[0].name = "a";
    out_args[0].type = GPUBRIDGE_TYPE_F32;
    out_args[0].rank = 2;
    out_args[0].shape[0] = m;
    out_args[0].shape[1] = k;
    out_args[0].is_input = true;
    out_args[0].is_output = false;

    /* Input b: rank 2, shape [k,n]. */
    out_args[1].name = "b";
    out_args[1].type = GPUBRIDGE_TYPE_F32;
    out_args[1].rank = 2;
    out_args[1].shape[0] = k;
    out_args[1].shape[1] = n;
    out_args[1].is_input = true;
    out_args[1].is_output = false;

    /* Output c: rank 2, shape [m,n]. */
    out_args[2].name = "c";
    out_args[2].type = GPUBRIDGE_TYPE_F32;
    out_args[2].rank = 2;
    out_args[2].shape[0] = m;
    out_args[2].shape[1] = n;
    out_args[2].is_input = false;
    out_args[2].is_output = true;

    out->kernel_name = "matmul_f32";
    out->op_kind = GPUBRIDGE_OP_MATMUL;
    out->args = out_args;
    out->arg_count = 3;
    /* One lane per output element C[row,col] (spec2.md 7.1). */
    out->global_size[0] = m;
    out->global_size[1] = n;
}

void gpuBridgeIrInitReduceSumF32(GpuBridgeTensorKernelIR* out, GpuBridgeTensorArg out_args[2],
    size_t n)
{
    /* Input a: rank 1, shape [n]. */
    out_args[0].name = "a";
    out_args[0].type = GPUBRIDGE_TYPE_F32;
    out_args[0].rank = 1;
    out_args[0].shape[0] = n;
    out_args[0].shape[1] = 0;
    out_args[0].is_input = true;
    out_args[0].is_output = false;

    /* Output result: rank 1, shape [1]. */
    out_args[1].name = "result";
    out_args[1].type = GPUBRIDGE_TYPE_F32;
    out_args[1].rank = 1;
    out_args[1].shape[0] = 1;
    out_args[1].shape[1] = 0;
    out_args[1].is_input = false;
    out_args[1].is_output = true;

    out->kernel_name = "reduce_sum_f32";
    out->op_kind = GPUBRIDGE_OP_REDUCE_SUM;
    out->args = out_args;
    out->arg_count = 2;
    /* Single work-group sequential accumulation over n elements (spec2.md
     * 7.2) — a rank-1 launch, so only global_size[0] is used. */
    out->global_size[0] = n;
    out->global_size[1] = 0;
}

void gpuBridgeIrInitBroadcastAddF32(GpuBridgeTensorKernelIR* out, GpuBridgeTensorArg out_args[2],
    size_t rows, size_t cols)
{
    /* y: rank 2, shape [rows,cols], both read and written — in-place on y's
     * buffer, matching spec2.md 7.3's `Y += b` semantics. */
    out_args[0].name = "y";
    out_args[0].type = GPUBRIDGE_TYPE_F32;
    out_args[0].rank = 2;
    out_args[0].shape[0] = rows;
    out_args[0].shape[1] = cols;
    out_args[0].is_input = true;
    out_args[0].is_output = true;

    /* Input bias: rank 1, shape [cols]. */
    out_args[1].name = "bias";
    out_args[1].type = GPUBRIDGE_TYPE_F32;
    out_args[1].rank = 1;
    out_args[1].shape[0] = cols;
    out_args[1].shape[1] = 0;
    out_args[1].is_input = true;
    out_args[1].is_output = false;

    out->kernel_name = "broadcast_add_f32";
    out->op_kind = GPUBRIDGE_OP_BROADCAST_ADD;
    out->args = out_args;
    out->arg_count = 2;
    /* One lane per output element, 2D global size (rows, cols) (spec2.md
     * 7.3). */
    out->global_size[0] = rows;
    out->global_size[1] = cols;
}

void gpuBridgeIrInitReluF32(GpuBridgeTensorKernelIR* out, GpuBridgeTensorArg out_args[2],
    int rank, const size_t shape[2])
{
    /* Input x: same rank/shape as the caller's tensor. */
    out_args[0].name = "x";
    out_args[0].type = GPUBRIDGE_TYPE_F32;
    out_args[0].rank = rank;
    out_args[0].shape[0] = shape[0];
    out_args[0].shape[1] = (rank == 2) ? shape[1] : 0;
    out_args[0].is_input = true;
    out_args[0].is_output = false;

    /* Output y: same rank/shape as x — element count in equals element
     * count out (spec2.md 7.4). */
    out_args[1].name = "y";
    out_args[1].type = GPUBRIDGE_TYPE_F32;
    out_args[1].rank = rank;
    out_args[1].shape[0] = shape[0];
    out_args[1].shape[1] = (rank == 2) ? shape[1] : 0;
    out_args[1].is_input = false;
    out_args[1].is_output = true;

    out->kernel_name = "relu_f32";
    out->op_kind = GPUBRIDGE_OP_RELU;
    out->args = out_args;
    out->arg_count = 2;
    /* Flattened global size: launch dispatch (gpubridge_runtime.c) treats relu as
     * purely elementwise over shape[0]*shape[1] (or just shape[0] for rank
     * 1) lanes, so global_size mirrors the tensor's own shape here and is
     * flattened at launch time rather than here. */
    out->global_size[0] = shape[0];
    out->global_size[1] = (rank == 2) ? shape[1] : 0;
}

/* --- Phase 2 addition (milestone.md section 22 deliverable 4) --- */

bool gpuBridgeIrValidateKernel(const GpuBridgeKernelIR* kernel, size_t arg_count,
    char* err_buf, size_t err_buf_size)
{
    if (kernel == NULL) {
        snprintf(err_buf, err_buf_size, "kernel must not be NULL");
        return false;
    }
    if (kernel->op_kind != GPUBRIDGE_OP_VECTOR_ADD) {
        snprintf(err_buf, err_buf_size,
            "gpuBridgeLaunchKernel only supports GPUBRIDGE_OP_VECTOR_ADD in this milestone");
        return false;
    }
    if (kernel->args == NULL || kernel->arg_count != 3 || arg_count != 3) {
        snprintf(err_buf, err_buf_size,
            "vector_add_f32 requires exactly 3 args (a, b, c)");
        return false;
    }
    if (kernel->global_size == 0) {
        snprintf(err_buf, err_buf_size, "global_size must be greater than 0");
        return false;
    }

    /* a, b are inputs; c is output — matching gpuBridgeIrInitVectorAddF32()'s
     * layout exactly. Every arg must be f32 and have one element per lane
     * (length == global_size), since vector_add_f32 launches exactly one
     * lane per output element. */
    static const bool expect_input[3]  = { true,  true,  false };
    static const bool expect_output[3] = { false, false, true  };
    for (size_t i = 0; i < 3; i++) {
        const GpuBridgeBufferArg* a = &kernel->args[i];
        if (a->type != GPUBRIDGE_TYPE_F32) {
            snprintf(err_buf, err_buf_size, "arg %zu: only f32 is supported", i);
            return false;
        }
        if (a->length != kernel->global_size) {
            snprintf(err_buf, err_buf_size,
                "arg %zu: length (%zu) does not match global_size (%zu)",
                i, a->length, kernel->global_size);
            return false;
        }
        if (a->is_input != expect_input[i] || a->is_output != expect_output[i]) {
            snprintf(err_buf, err_buf_size,
                "arg %zu: is_input/is_output does not match vector_add_f32's a,b,c layout", i);
            return false;
        }
    }

    return true;
}

bool gpuBridgeIrValidateTensorKernel(const GpuBridgeTensorKernelIR* kernel, size_t arg_count,
    char* err_buf, size_t err_buf_size)
{
    if (kernel == NULL || kernel->args == NULL) {
        snprintf(err_buf, err_buf_size, "kernel/args must not be NULL");
        return false;
    }

    /* Checks shared by every op: f32 only, rank 1 or 2, every dimension
     * actually used by that rank must be nonzero. */
    for (size_t i = 0; i < kernel->arg_count; i++) {
        const GpuBridgeTensorArg* a = &kernel->args[i];
        if (a->type != GPUBRIDGE_TYPE_F32) {
            snprintf(err_buf, err_buf_size, "arg %zu (%s): only f32 is supported", i, a->name);
            return false;
        }
        if (a->rank != 1 && a->rank != 2) {
            snprintf(err_buf, err_buf_size, "arg %zu (%s): rank must be 1 or 2", i, a->name);
            return false;
        }
        if (a->shape[0] == 0 || (a->rank == 2 && a->shape[1] == 0)) {
            snprintf(err_buf, err_buf_size, "arg %zu (%s): shape dimensions must be > 0",
                i, a->name);
            return false;
        }
    }

    switch (kernel->op_kind) {
    case GPUBRIDGE_OP_MATMUL:
        if (kernel->arg_count != 3 || arg_count != 3) {
            snprintf(err_buf, err_buf_size, "matmul_f32 requires exactly 3 args");
            return false;
        }
        if (kernel->args[0].rank != 2 || kernel->args[1].rank != 2 ||
            kernel->args[2].rank != 2) {
            snprintf(err_buf, err_buf_size, "matmul_f32: a, b, c must all be rank 2");
            return false;
        }
        /* A[m,k] * B[k,n] requires A's inner dimension to match B's outer
         * dimension — never checked before this validator existed. */
        if (kernel->args[0].shape[1] != kernel->args[1].shape[0]) {
            snprintf(err_buf, err_buf_size,
                "matmul_f32: inner dimensions do not match (a[.,%zu] vs b[%zu,.])",
                kernel->args[0].shape[1], kernel->args[1].shape[0]);
            return false;
        }
        if (kernel->args[2].shape[0] != kernel->args[0].shape[0] ||
            kernel->args[2].shape[1] != kernel->args[1].shape[1]) {
            snprintf(err_buf, err_buf_size,
                "matmul_f32: output shape [%zu,%zu] does not match a/b ([%zu,.]/[.,%zu])",
                kernel->args[2].shape[0], kernel->args[2].shape[1],
                kernel->args[0].shape[0], kernel->args[1].shape[1]);
            return false;
        }
        break;

    case GPUBRIDGE_OP_REDUCE_SUM:
        if (kernel->arg_count != 2 || arg_count != 2) {
            snprintf(err_buf, err_buf_size, "reduce_sum_f32 requires exactly 2 args");
            return false;
        }
        if (kernel->args[1].shape[0] != 1) {
            snprintf(err_buf, err_buf_size, "reduce_sum_f32: result must have shape [1]");
            return false;
        }
        break;

    case GPUBRIDGE_OP_BROADCAST_ADD:
        if (kernel->arg_count != 2 || arg_count != 2) {
            snprintf(err_buf, err_buf_size, "broadcast_add_f32 requires exactly 2 args");
            return false;
        }
        if (kernel->args[0].rank != 2) {
            snprintf(err_buf, err_buf_size, "broadcast_add_f32: y must be rank 2");
            return false;
        }
        if (kernel->args[1].shape[0] != kernel->args[0].shape[1]) {
            snprintf(err_buf, err_buf_size,
                "broadcast_add_f32: bias length (%zu) does not match y's column count (%zu)",
                kernel->args[1].shape[0], kernel->args[0].shape[1]);
            return false;
        }
        break;

    case GPUBRIDGE_OP_RELU:
        if (kernel->arg_count != 2 || arg_count != 2) {
            snprintf(err_buf, err_buf_size, "relu_f32 requires exactly 2 args");
            return false;
        }
        if (kernel->args[0].rank != kernel->args[1].rank ||
            kernel->args[0].shape[0] != kernel->args[1].shape[0] ||
            kernel->args[0].shape[1] != kernel->args[1].shape[1]) {
            snprintf(err_buf, err_buf_size, "relu_f32: x and y must have the same rank/shape");
            return false;
        }
        break;

    default:
        /* GPUBRIDGE_OP_VECTOR_ADD (or any future value) is not a tensor op —
         * gpuBridgeLaunchKernel() is the entry point for that. */
        snprintf(err_buf, err_buf_size,
            "gpuBridgeLaunchTensorKernel: unsupported op_kind");
        return false;
    }

    return true;
}

/* --- Phase 4 addition (spec4.md section 7) --- */

void gpuBridgeIrDescribeQuantizedTensor(GpuBridgeTensorDesc* out,
    GpuBridgeScalarType quant_type, int rank, const size_t shape[2],
    GpuBridgeQuantDesc quant)
{
    out->type = quant_type;
    out->rank = rank;
    out->shape[0] = shape[0];
    out->shape[1] = (rank == 2) ? shape[1] : 0;
    /* Describes intent only — never backed by a real allocation (spec4.md
     * section 7). */
    out->device_ptr = NULL;
    out->device_resident = false;
    out->memory_strategy = GPUBRIDGE_MEMORY_AUTO;
    out->usage = GPUBRIDGE_TENSOR_USAGE_GENERIC;
    out->quant = quant;
}
