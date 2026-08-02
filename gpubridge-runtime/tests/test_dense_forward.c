/*
 * test_dense_forward.c — the Milestone 2 capstone test (spec2.md section
 * 11): a single dense-layer forward pass,
 *
 *   Y = matmul(X, W) + B
 *   Z = relu(Y)
 *
 * chained entirely on device-resident tensors with exactly one H2D copy (for
 * X, W, B) before the chain and exactly one D2H copy (for Z) after it — no
 * gpuBridgeMemcpy call is issued between the matmul_f32 / broadcast_add_f32 /
 * relu_f32 launches, proving spec2.md section 8.1's device-residency
 * property.
 *
 * Exit code doubles as ctest's pass/fail signal (dense_forward_cpu /
 * dense_forward_auto in CMakeLists.txt): 0 on PASS, 1 otherwise.
 */
#include "gpubridge_runtime.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    const size_t M = 16;
    const size_t K = 32;
    const size_t N = 8;

    if (gpuBridgeInit() != 0) {
        fprintf(stderr, "gpuBridgeInit failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    /* Host arrays X[M,K], W[K,N], B[N], deterministic values chosen to
     * exercise both the positive and negative side of relu. */
    float* x = malloc(M * K * sizeof(float));
    float* w = malloc(K * N * sizeof(float));
    float* b = malloc(N * sizeof(float));
    float* z = malloc(M * N * sizeof(float));
    for (size_t i = 0; i < M * K; i++) {
        x[i] = (float)(i % 9) * 0.1f - 0.4f;
    }
    for (size_t i = 0; i < K * N; i++) {
        w[i] = (float)(i % 5) * 0.2f - 0.4f;
    }
    for (size_t j = 0; j < N; j++) {
        b[j] = (float)j * 0.05f - 0.1f;
    }

    /* Device tensors: X_dev, W_dev, B_dev, Y_dev. */
    GpuBridgeTensorDesc x_dev, w_dev, b_dev, y_dev;
    const size_t x_shape[2] = { M, K };
    const size_t w_shape[2] = { K, N };
    const size_t b_shape[2] = { N, 0 };
    const size_t y_shape[2] = { M, N };
    if (gpuBridgeTensorAlloc(&x_dev, GPUBRIDGE_TYPE_F32, 2, x_shape) != 0 ||
        gpuBridgeTensorAlloc(&w_dev, GPUBRIDGE_TYPE_F32, 2, w_shape) != 0 ||
        gpuBridgeTensorAlloc(&b_dev, GPUBRIDGE_TYPE_F32, 1, b_shape) != 0 ||
        gpuBridgeTensorAlloc(&y_dev, GPUBRIDGE_TYPE_F32, 2, y_shape) != 0) {
        fprintf(stderr, "gpuBridgeTensorAlloc failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    /* The only H2D copies in this test — one each for X, W, B. */
    if (gpuBridgeMemcpy(x_dev.device_ptr, x, M * K * sizeof(float),
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0 ||
        gpuBridgeMemcpy(w_dev.device_ptr, w, K * N * sizeof(float),
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0 ||
        gpuBridgeMemcpy(b_dev.device_ptr, b, N * sizeof(float),
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy H2D failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    /* --- The chained, device-resident section: matmul -> broadcast_add ->
     * relu. No gpuBridgeMemcpy call appears between these three launches — this is
     * the property spec2.md section 8.1 requires and this test proves. --- */

    GpuBridgeTensorKernelIR matmul_kernel;
    GpuBridgeTensorArg matmul_args[3];
    gpuBridgeIrInitMatmulF32(&matmul_kernel, matmul_args, M, K, N);
    void* matmul_launch_args[3] =
        { x_dev.device_ptr, w_dev.device_ptr, y_dev.device_ptr };
    if (gpuBridgeLaunchTensorKernel(&matmul_kernel, matmul_launch_args, 3) != 0) {
        fprintf(stderr, "matmul_f32 launch failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeTensorKernelIR badd_kernel;
    GpuBridgeTensorArg badd_args[2];
    gpuBridgeIrInitBroadcastAddF32(&badd_kernel, badd_args, M, N);
    void* badd_launch_args[2] = { y_dev.device_ptr, b_dev.device_ptr };
    if (gpuBridgeLaunchTensorKernel(&badd_kernel, badd_launch_args, 2) != 0) {
        fprintf(stderr, "broadcast_add_f32 launch failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeTensorKernelIR relu_kernel;
    GpuBridgeTensorArg relu_args[2];
    gpuBridgeIrInitReluF32(&relu_kernel, relu_args, 2, y_shape);
    /* In-place: relu reads and writes the same device buffer, matching
     * "Z = relu(Y)" reusing Y's storage per spec2.md section 7.3's
     * in-place precedent. */
    void* relu_launch_args[2] = { y_dev.device_ptr, y_dev.device_ptr };
    if (gpuBridgeLaunchTensorKernel(&relu_kernel, relu_launch_args, 2) != 0) {
        fprintf(stderr, "relu_f32 launch failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }

    /* --- End of the chained section. --- */

    if (gpuBridgeDeviceSynchronize() != 0) {
        fprintf(stderr, "gpuBridgeDeviceSynchronize failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }

    /* The only D2H copy in this test. */
    if (gpuBridgeMemcpy(z, y_dev.device_ptr, M * N * sizeof(float),
            GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy D2H failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    /* CPU double-precision reference computing the same three steps
     * (spec2.md section 11 step 10). */
    double max_rel_error = 0.0;
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double acc = 0.0;
            for (size_t p = 0; p < K; p++) {
                acc += (double)x[i * K + p] * (double)w[p * N + j];
            }
            acc += (double)b[j];
            double expected = acc > 0.0 ? acc : 0.0;
            double actual = (double)z[i * N + j];
            double error = fabs(actual - expected);
            double rel = error / fmax(1.0, fabs(expected));
            if (rel > max_rel_error) {
                max_rel_error = rel;
            }
        }
    }

    /* Tolerance per section 11: matmul-derived, 1e-3 relative. */
    bool pass = max_rel_error <= 1e-3;

    printf("GPUBridge dense_forward test (matmul -> broadcast_add -> relu)\n");
    printf("backend: %s\n", gpuBridgeBackendKindToString(gpuBridgeGetSelectedBackend()));
    printf("shape: X[%zu,%zu] W[%zu,%zu] B[%zu] -> Z[%zu,%zu]\n", M, K, K, N,
        N, M, N);
    printf("result: %s\n", pass ? "PASS" : "FAIL");
    printf("max relative error: %f\n", max_rel_error);

    gpuBridgeTensorFree(&x_dev);
    gpuBridgeTensorFree(&w_dev);
    gpuBridgeTensorFree(&b_dev);
    gpuBridgeTensorFree(&y_dev);
    free(x);
    free(w);
    free(b);
    free(z);
    gpuBridgeShutdown();

    return pass ? 0 : 1;
}
