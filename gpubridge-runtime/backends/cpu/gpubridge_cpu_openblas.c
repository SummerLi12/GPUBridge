/*
 * gpubridge_cpu_openblas.c — see gpubridge_cpu_openblas.h for the design
 * rationale (spec3.md section 6).
 */
#include "gpubridge_cpu_openblas.h"

#ifdef GPUBRIDGE_HAVE_OPENBLAS

#include <cblas.h>
#include <dlfcn.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include "gpubridge_internal.h" /* for gpuBridgeEnvVerbose() — spec3.md section 8.5's
                                   note on this already-established header-boundary
                                   deviation */

typedef void (*PFN_cblas_sgemm)(
    const enum CBLAS_ORDER, const enum CBLAS_TRANSPOSE, const enum CBLAS_TRANSPOSE,
    const int, const int, const int, const float,
    const float*, const int, const float*, const int, const float, float*, const int);

static struct {
    void* dl_handle;
    bool probed;
    bool available;
    PFN_cblas_sgemm cblas_sgemm;
} g;

static void* try_open(const char* path)
{
    return path ? dlopen(path, RTLD_NOW) : NULL;
}

bool gpuBridgeCpuOpenblasTryInit(void)
{
    if (g.probed) { return g.available; }
    g.probed = true;
    g.available = false;

    /* spec3.md section 7: GPUBRIDGE_OPENBLAS_LIBRARY override, same
     * authoritative (no-fallback-on-failure) behavior as CLBlast's
     * override (gpubridge_opencl_clblast.c). */
    const char* override_path = getenv("GPUBRIDGE_OPENBLAS_LIBRARY");
    if (override_path != NULL) {
        g.dl_handle = try_open(override_path);
        if (g.dl_handle == NULL && gpuBridgeEnvVerbose()) {
            fprintf(stderr,
                "GPUBridge: GPUBRIDGE_OPENBLAS_LIBRARY=%s failed to load "
                "(dlerror: %s) - not falling back to the default search "
                "path; OpenBLAS will be treated as unavailable.\n",
                override_path, dlerror());
        }
    } else {
        g.dl_handle = try_open("libopenblas.so.0");
        if (g.dl_handle == NULL) { g.dl_handle = try_open("libopenblas.so"); }
    }
    if (g.dl_handle == NULL) { return false; }

    g.cblas_sgemm = (PFN_cblas_sgemm)dlsym(g.dl_handle, "cblas_sgemm");
    if (g.cblas_sgemm == NULL) {
        dlclose(g.dl_handle);
        g.dl_handle = NULL;
        return false;
    }
    g.available = true;
    return true;
}

bool gpuBridgeCpuOpenblasAvailable(void) { return g.probed && g.available; }

bool gpuBridgeCpuOpenblasSgemmF32(const float* a, const float* b, float* c,
    size_t m, size_t k, size_t n, char* err_buf, size_t err_buf_len)
{
    /* cblas_sgemm's ABI takes `int` dimensions (OpenBLAS's LP64 `blasint`);
     * GPUBridge uses size_t throughout. A silent size_t->int truncation
     * would corrupt the multiply for m/n/k >= 2^31 rather than fail
     * loudly, so check first (spec3.md section 6.1). */
    if (m > (size_t)INT_MAX || n > (size_t)INT_MAX || k > (size_t)INT_MAX) {
        snprintf(err_buf, err_buf_len,
            "cblas_sgemm: dimension exceeds INT_MAX (m=%zu n=%zu k=%zu)", m, n, k);
        return false;
    }
    g.cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)m, (int)n, (int)k, 1.0f, a, (int)k, b, (int)n, 0.0f, c, (int)n);
    return true;
}

#else /* !GPUBRIDGE_HAVE_OPENBLAS */

bool gpuBridgeCpuOpenblasTryInit(void) { return false; }
bool gpuBridgeCpuOpenblasAvailable(void) { return false; }
bool gpuBridgeCpuOpenblasSgemmF32(const float* a, const float* b, float* c,
    size_t m, size_t k, size_t n, char* err_buf, size_t err_buf_len)
{
    (void)a; (void)b; (void)c; (void)m; (void)k; (void)n;
    (void)err_buf; (void)err_buf_len;
    return false;
}

#endif /* GPUBRIDGE_HAVE_OPENBLAS */
