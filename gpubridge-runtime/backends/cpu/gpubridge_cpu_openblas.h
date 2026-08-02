/*
 * gpubridge_cpu_openblas.h — optional OpenBLAS-accelerated GEMM for the CPU
 * backend (spec3.md section 6). Same shape as gpubridge_opencl_clblast.h,
 * but simpler: no queue/context concept, no async event.
 *
 * Same rules: no link-time dependency (dlopen/dlsym only), whole real
 * implementation is #ifdef GPUBRIDGE_HAVE_OPENBLAS (set by CMakeLists.txt
 * only when cblas.h was found), #else stub otherwise.
 *
 * Internal, not under include/, not part of the public API — only
 * #include'd by gpubridge_backend_cpu.c.
 */
#ifndef GPUBRIDGE_CPU_OPENBLAS_H
#define GPUBRIDGE_CPU_OPENBLAS_H

#include <stdbool.h>
#include <stddef.h>

bool gpuBridgeCpuOpenblasTryInit(void);
bool gpuBridgeCpuOpenblasAvailable(void);

/* C[m,n] = A[m,k] * B[k,n], row-major, via cblas_sgemm. Precondition:
 * gpuBridgeCpuOpenblasAvailable() == true. Returns false (with a message in
 * err_buf) only for the dimension-bounds check below — cblas_sgemm itself
 * has no error-return convention in the CBLAS ABI. */
bool gpuBridgeCpuOpenblasSgemmF32(const float* a, const float* b, float* c,
    size_t m, size_t k, size_t n, char* err_buf, size_t err_buf_len);

#endif /* GPUBRIDGE_CPU_OPENBLAS_H */
