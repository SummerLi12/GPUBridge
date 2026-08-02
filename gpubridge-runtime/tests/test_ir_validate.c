/*
 * test_ir_validate.c — Phase 2 unit test for gpuBridgeIrValidateKernel /
 * gpuBridgeIrValidateTensorKernel (milestone.md section 22, "Common GPUBridge
 * GPU/Tensor IR" deliverable 4: "IR validator").
 *
 * Pure function test: no gpuBridgeInit() / backend involved. Each case builds
 * a GpuBridgeKernelIR/GpuBridgeTensorKernelIR (valid ones via the existing
 * gpuBridgeIrInit*F32 builders, invalid ones hand-constructed) and checks the
 * validator's bool result — and, for the reject cases, that it actually
 * wrote a non-empty reason into err_buf rather than just returning false.
 *
 * Exit code doubles as the pass/fail signal for `ctest`: 0 on PASS
 * (every case), 1 on FAIL.
 */
#include "gpubridge_runtime.h"

#include <stdio.h>
#include <string.h>

static bool g_all_pass = true;

static void check(const char* name, bool expected, bool actual, const char* err_buf)
{
    bool pass = (actual == expected);
    g_all_pass = g_all_pass && pass;
    printf("  %-40s expected=%-5s actual=%-5s %s%s%s\n",
        name,
        expected ? "true" : "false",
        actual ? "true" : "false",
        pass ? "PASS" : "FAIL",
        (!actual && err_buf[0] != '\0') ? " reason=" : "",
        (!actual && err_buf[0] != '\0') ? err_buf : "");

    /* A rejection with an empty reason string is itself a bug in the
     * validator (callers rely on err_buf being populated whenever it
     * returns false) — fail the case even if the bool result was correct. */
    if (!expected && actual == false && err_buf[0] == '\0') {
        printf("  %-40s FAIL (rejected but err_buf was left empty)\n", name);
        g_all_pass = false;
    }
}

int main(void)
{
    char err[256];

    printf("GPUBridge ir_validate test\n");

    /* --- gpuBridgeIrValidateKernel (vector_add_f32) --- */

    GpuBridgeKernelIR vecadd_kernel;
    GpuBridgeBufferArg vecadd_args[3];
    gpuBridgeIrInitVectorAddF32(&vecadd_kernel, vecadd_args, 1024);
    err[0] = '\0';
    check("vecadd: builder output accepted", true,
        gpuBridgeIrValidateKernel(&vecadd_kernel, 3, err, sizeof(err)), err);

    err[0] = '\0';
    check("vecadd: NULL kernel rejected", false,
        gpuBridgeIrValidateKernel(NULL, 3, err, sizeof(err)), err);

    err[0] = '\0';
    check("vecadd: wrong arg_count rejected", false,
        gpuBridgeIrValidateKernel(&vecadd_kernel, 2, err, sizeof(err)), err);

    GpuBridgeKernelIR bad_length_kernel = vecadd_kernel;
    GpuBridgeBufferArg bad_length_args[3];
    memcpy(bad_length_args, vecadd_args, sizeof(bad_length_args));
    bad_length_args[0].length = 999; /* no longer matches global_size (1024) */
    bad_length_kernel.args = bad_length_args;
    err[0] = '\0';
    check("vecadd: arg length != global_size rejected", false,
        gpuBridgeIrValidateKernel(&bad_length_kernel, 3, err, sizeof(err)), err);

    /* --- gpuBridgeIrValidateTensorKernel: matmul_f32 --- */

    GpuBridgeTensorKernelIR matmul_kernel;
    GpuBridgeTensorArg matmul_args[3];
    gpuBridgeIrInitMatmulF32(&matmul_kernel, matmul_args, 16, 32, 8);
    err[0] = '\0';
    check("matmul: builder output accepted", true,
        gpuBridgeIrValidateTensorKernel(&matmul_kernel, 3, err, sizeof(err)), err);

    /* Corrupt B's shape so its row count (16) no longer matches A's column
     * count (32) — a mismatch the old inline checks never caught. */
    GpuBridgeTensorKernelIR bad_matmul_kernel = matmul_kernel;
    GpuBridgeTensorArg bad_matmul_args[3];
    memcpy(bad_matmul_args, matmul_args, sizeof(bad_matmul_args));
    bad_matmul_args[1].shape[0] = 16;
    bad_matmul_kernel.args = bad_matmul_args;
    err[0] = '\0';
    check("matmul: mismatched inner dimensions rejected", false,
        gpuBridgeIrValidateTensorKernel(&bad_matmul_kernel, 3, err, sizeof(err)), err);

    /* GPUBRIDGE_OP_VECTOR_ADD is gpuBridgeLaunchKernel's op, not a tensor op — an
     * otherwise-valid tensor kernel IR with that op_kind must still be
     * rejected here. Built from the valid matmul IR (not a cast of
     * GpuBridgeKernelIR — the two structs have unrelated layouts). */
    GpuBridgeTensorKernelIR wrong_op_kernel = matmul_kernel;
    wrong_op_kernel.op_kind = GPUBRIDGE_OP_VECTOR_ADD;
    err[0] = '\0';
    check("matmul: GPUBRIDGE_OP_VECTOR_ADD rejected as tensor kernel", false,
        gpuBridgeIrValidateTensorKernel(&wrong_op_kernel, 3, err, sizeof(err)), err);

    /* --- gpuBridgeIrValidateTensorKernel: broadcast_add_f32 --- */

    GpuBridgeTensorKernelIR ba_kernel;
    GpuBridgeTensorArg ba_args[2];
    gpuBridgeIrInitBroadcastAddF32(&ba_kernel, ba_args, 16, 8);
    err[0] = '\0';
    check("broadcast_add: builder output accepted", true,
        gpuBridgeIrValidateTensorKernel(&ba_kernel, 2, err, sizeof(err)), err);

    /* Corrupt the bias length so it no longer matches y's column count. */
    GpuBridgeTensorKernelIR bad_ba_kernel = ba_kernel;
    GpuBridgeTensorArg bad_ba_args[2];
    memcpy(bad_ba_args, ba_args, sizeof(bad_ba_args));
    bad_ba_args[1].shape[0] = 999;
    bad_ba_kernel.args = bad_ba_args;
    err[0] = '\0';
    check("broadcast_add: mismatched bias length rejected", false,
        gpuBridgeIrValidateTensorKernel(&bad_ba_kernel, 2, err, sizeof(err)), err);

    /* --- gpuBridgeIrValidateTensorKernel: relu_f32 --- */

    GpuBridgeTensorKernelIR relu_kernel;
    GpuBridgeTensorArg relu_args[2];
    size_t relu_shape[2] = { 16, 8 };
    gpuBridgeIrInitReluF32(&relu_kernel, relu_args, 2, relu_shape);
    err[0] = '\0';
    check("relu: builder output accepted", true,
        gpuBridgeIrValidateTensorKernel(&relu_kernel, 2, err, sizeof(err)), err);

    /* Corrupt y's shape so it no longer matches x's. */
    GpuBridgeTensorKernelIR bad_relu_kernel = relu_kernel;
    GpuBridgeTensorArg bad_relu_args[2];
    memcpy(bad_relu_args, relu_args, sizeof(bad_relu_args));
    bad_relu_args[1].shape[1] = 3;
    bad_relu_kernel.args = bad_relu_args;
    err[0] = '\0';
    check("relu: mismatched x/y shape rejected", false,
        gpuBridgeIrValidateTensorKernel(&bad_relu_kernel, 2, err, sizeof(err)), err);

    /* --- gpuBridgeIrValidateTensorKernel: reduce_sum_f32 --- */

    GpuBridgeTensorKernelIR rs_kernel;
    GpuBridgeTensorArg rs_args[2];
    gpuBridgeIrInitReduceSumF32(&rs_kernel, rs_args, 1024);
    err[0] = '\0';
    check("reduce_sum: builder output accepted", true,
        gpuBridgeIrValidateTensorKernel(&rs_kernel, 2, err, sizeof(err)), err);

    printf("result: %s\n", g_all_pass ? "PASS" : "FAIL");
    return g_all_pass ? 0 : 1;
}
