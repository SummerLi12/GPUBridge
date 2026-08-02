/*
 * gpubridge_opencl_clblast.c — see gpubridge_opencl_clblast.h for the design
 * rationale (spec3.md section 5). This module is scoped purely to GEMM
 * acceleration — device-identity reporting (spec3.md section 8.5) lives in
 * gpubridge_backend_opencl.c instead, not here.
 */
#include "gpubridge_opencl_clblast.h"

#ifdef GPUBRIDGE_HAVE_CLBLAST

#include <clblast_c.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpubridge_internal.h" /* for gpuBridgeEnvVerbose() — spec3.md section 8.5's
                                   note on this already-established header-boundary
                                   deviation */

/* Mirrors clblast_c.h's CLBlastSgemm prototype exactly, so dlsym()'s
 * untyped result can be called normally — same pattern
 * gpubridge_backend_opencl.c already uses for every OpenCL entry point. */
typedef CLBlastStatusCode (*PFN_CLBlastSgemm)(
    const CLBlastLayout, const CLBlastTranspose, const CLBlastTranspose,
    const size_t, const size_t, const size_t, const float,
    const cl_mem, const size_t, const size_t,
    const cl_mem, const size_t, const size_t, const float,
    cl_mem, const size_t, const size_t,
    cl_command_queue*, cl_event*);

static struct {
    void* dl_handle;
    bool probed;
    bool available;
    PFN_CLBlastSgemm CLBlastSgemm;
} g;

static void* try_open(const char* path)
{
    return path ? dlopen(path, RTLD_NOW) : NULL;
}

bool gpuBridgeOpenclClblastTryInit(void)
{
    if (g.probed) { return g.available; }
    g.probed = true;
    g.available = false;

    /* spec3.md section 7: GPUBRIDGE_CLBLAST_LIBRARY, if set, is tried first
     * and is authoritative — if it fails to open, this probe does NOT fall
     * through to the default candidate list (an explicit override that
     * fails is a configuration problem worth surfacing, not silently
     * masking with a different library than the one requested). */
    const char* override_path = getenv("GPUBRIDGE_CLBLAST_LIBRARY");
    if (override_path != NULL) {
        g.dl_handle = try_open(override_path);
        if (g.dl_handle == NULL && gpuBridgeEnvVerbose()) {
            fprintf(stderr,
                "GPUBridge: GPUBRIDGE_CLBLAST_LIBRARY=%s failed to load "
                "(dlerror: %s) - not falling back to the default search "
                "path; CLBlast will be treated as unavailable.\n",
                override_path, dlerror());
        }
    } else {
        g.dl_handle = try_open("libclblast.so");
        if (g.dl_handle == NULL) { g.dl_handle = try_open("libclblast.so.1"); }
    }
    if (g.dl_handle == NULL) {
        return false; /* not installed / override invalid - not an error */
    }

    g.CLBlastSgemm = (PFN_CLBlastSgemm)dlsym(g.dl_handle, "CLBlastSgemm");
    if (g.CLBlastSgemm == NULL) {
        /* dlopen succeeded but the expected symbol is missing (wrong
         * library at that path, ABI mismatch, stripped build) - clean up
         * fully rather than leaking a handle we'll never use. */
        dlclose(g.dl_handle);
        g.dl_handle = NULL;
        return false;
    }
    g.available = true;
    return true;
}

bool gpuBridgeOpenclClblastAvailable(void) { return g.probed && g.available; }

bool gpuBridgeOpenclClblastSgemmF32(cl_command_queue queue, cl_mem a, cl_mem b,
    cl_mem c, size_t m, size_t k, size_t n, char* err_buf, size_t err_buf_len)
{
    if (gpuBridgeEnvVerbose()) {
        printf("we are in function: gpuBridgeOpenclClblastSgemmF32\n"
            "purpose: build/dispatch CLBlast's own tuned GEMM kernel on the "
            "GPU (m=%zu k=%zu n=%zu), reading/writing the device cl_mem "
            "buffers already allocated+uploaded by the caller\n", m, k, n);
    }

    /* Row-major, no transpose, C = 1.0*A*B + 0.0*C - matches
     * cpu_launch_matmul_f32/matmul_f32.cl's exact semantics: C[m,n] =
     * sum_k(A[m,k]*B[k,n]). Leading dimensions equal each matrix's row
     * length since layout is unpadded, row-major, offset 0. */
    CLBlastStatusCode status = g.CLBlastSgemm(
        CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo,
        m, n, k, 1.0f,
        a, 0, k,
        b, 0, n, 0.0f,
        c, 0, n,
        &queue, NULL);

    if (status != CLBlastSuccess) {
        snprintf(err_buf, err_buf_len, "CLBlastSgemm failed (status %d)", (int)status);
        if (gpuBridgeEnvVerbose()) {
            printf("leaving function: gpuBridgeOpenclClblastSgemmF32, "
                "result: failed (%s)\n\n", err_buf);
        }
        return false;
    }
    if (gpuBridgeEnvVerbose()) {
        printf("leaving function: gpuBridgeOpenclClblastSgemmF32, "
            "result: success (C[%zu,%zu] computed on GPU via CLBlast)\n\n", m, n);
    }
    return true;
}

#else /* !GPUBRIDGE_HAVE_CLBLAST */

bool gpuBridgeOpenclClblastTryInit(void) { return false; }
bool gpuBridgeOpenclClblastAvailable(void) { return false; }
bool gpuBridgeOpenclClblastSgemmF32(cl_command_queue queue, cl_mem a, cl_mem b,
    cl_mem c, size_t m, size_t k, size_t n, char* err_buf, size_t err_buf_len)
{
    (void)queue; (void)a; (void)b; (void)c; (void)m; (void)k; (void)n;
    (void)err_buf; (void)err_buf_len;
    return false;
}

#endif /* GPUBRIDGE_HAVE_CLBLAST */
