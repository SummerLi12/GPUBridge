/*
 * gpubridge_opencl_clblast.h — optional CLBlast-accelerated GEMM for the
 * OpenCL backend (spec3.md section 5).
 *
 * CLBlast operates directly on this backend's already-open
 * cl_context/cl_command_queue/cl_mem handles (milestone.md section 8.5's
 * table lists it as the OpenCL GEMM library) — it is not a separate device
 * or execution context, so this is purely an internal helper, not a new
 * GpuBridgeBackend vtable slot or backend kind (spec3.md section 2).
 *
 * Same rules as gpubridge_backend_opencl.c itself:
 *   1. No link-time dependency on libclblast — every entry point is
 *      resolved via dlopen()/dlsym() at gpuBridgeOpenclClblastTryInit() time.
 *   2. The whole real implementation is #ifdef GPUBRIDGE_HAVE_CLBLAST (set by
 *      CMakeLists.txt only when clblast_c.h *and* CL/cl.h were both found —
 *      CLBlast's own header depends on OpenCL's types). The #else stub makes
 *      this file compile and report "unavailable" on any other machine.
 *
 * Internal, not under include/, not part of the public API — only
 * #include'd by gpubridge_backend_opencl.c.
 */
#ifndef GPUBRIDGE_OPENCL_CLBLAST_H
#define GPUBRIDGE_OPENCL_CLBLAST_H

#include <CL/cl.h>
#include <stdbool.h>
#include <stddef.h>

/* Attempts to dlopen CLBlast (GPUBRIDGE_CLBLAST_LIBRARY override first if
 * set, spec3.md section 7, else the default candidate list) and resolve
 * CLBlastSgemm exactly once. Call from opencl_init(), after g.context/
 * g.queue exist. Idempotent: only probes on the first call, caches the
 * result. Never causes opencl_init() to fail — CLBlast is acceleration,
 * not a requirement. Not thread-safe — see spec3.md section 8.4. */
bool gpuBridgeOpenclClblastTryInit(void);

/* Cheap, cached accessor: was the library *loaded* (spec3.md section 8.1's
 * term)? Does not consult GPUBRIDGE_VENDOR_GEMM_DISABLE — callers check
 * that separately (section 8.1's "enabled" term). */
bool gpuBridgeOpenclClblastAvailable(void);

/* C[m,n] = A[m,k] * B[k,n], row-major, via CLBlastSgemm on `queue`. a/b/c
 * are cl_mem handles already resident on that queue's device. Returns
 * false and writes a message into err_buf on a genuine CLBlast runtime
 * error. Precondition: gpuBridgeOpenclClblastAvailable() == true. */
bool gpuBridgeOpenclClblastSgemmF32(cl_command_queue queue, cl_mem a, cl_mem b,
    cl_mem c, size_t m, size_t k, size_t n, char* err_buf, size_t err_buf_len);

#endif /* GPUBRIDGE_OPENCL_CLBLAST_H */
