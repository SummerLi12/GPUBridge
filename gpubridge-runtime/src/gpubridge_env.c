/*
 * gpubridge_env.c — parses the two environment variables this runtime reads:
 * GPUBRIDGE_BACKEND (which backend to use) and GPUBRIDGE_VERBOSE (whether to print
 * diagnostics). See spec1.md sections 10-11 for the exact expected
 * behavior. Nothing here touches any backend or runtime state; this file
 * is pure environment-variable-to-enum translation.
 */
#include "gpubridge_internal.h"

#include <stdlib.h>
#include <strings.h>

GpuBridgeBackendKind gpuBridgeEnvGetRequestedBackend(bool* out_valid)
{
    *out_valid = true;

    const char* val = getenv("GPUBRIDGE_BACKEND");
    if (val == NULL) {
        /* Unset GPUBRIDGE_BACKEND means "auto", per spec1.md section 10. */
        return GPUBRIDGE_BACKEND_AUTO;
    }

    /* Case-insensitive match, so GPUBRIDGE_BACKEND=OpenCL / GPUBRIDGE_BACKEND=OPENCL work
     * the same as GPUBRIDGE_BACKEND=opencl. */
    if (strcasecmp(val, "auto") == 0) {
        return GPUBRIDGE_BACKEND_AUTO;
    }
    if (strcasecmp(val, "cpu") == 0) {
        return GPUBRIDGE_BACKEND_CPU;
    }
    if (strcasecmp(val, "opencl") == 0) {
        return GPUBRIDGE_BACKEND_OPENCL;
    }
    if (strcasecmp(val, "sycl") == 0) {
        return GPUBRIDGE_BACKEND_SYCL;
    }
    if (strcasecmp(val, "vulkan") == 0) {
        return GPUBRIDGE_BACKEND_VULKAN;
    }

    /* Unrecognized value (e.g. a typo like "opncl"): report invalid rather
     * than silently falling back to auto. spec1.md success criterion #8
     * ("wrong backend selection fails clearly") is best served by treating
     * this as a hard error in gpuBridgeInit(), not a silent substitution. */
    *out_valid = false;
    return GPUBRIDGE_BACKEND_AUTO;
}

bool gpuBridgeEnvVerbose(void)
{
    const char* val = getenv("GPUBRIDGE_VERBOSE");
    /* Strict match against the literal string "1" (spec1.md's
     * GPUBRIDGE_VERBOSE=1 example) — unset, "0", "true", empty, etc. are all
     * treated as non-verbose. */
    return val != NULL && val[0] == '1' && val[1] == '\0';
}

/* --- Milestone 1.5 addition (spec1.5.md section 9.1) --- */
bool gpuBridgeEnvProfile(void)
{
    const char* val = getenv("GPUBRIDGE_PROFILE");
    /* Same strict "1"-only match as gpuBridgeEnvVerbose above. */
    return val != NULL && val[0] == '1' && val[1] == '\0';
}

/* --- Milestone 2.5 addition (spec2.5.md section 4 item 5) --- */
bool gpuBridgeEnvPoolDisable(void)
{
    const char* val = getenv("GPUBRIDGE_POOL_DISABLE");
    /* Same strict "1"-only match as gpuBridgeEnvVerbose/gpuBridgeEnvProfile above. */
    return val != NULL && val[0] == '1' && val[1] == '\0';
}

/* --- Milestone 3 addition (spec3.md section 7) --- */
bool gpuBridgeEnvVendorGemmDisable(void)
{
    const char* val = getenv("GPUBRIDGE_VENDOR_GEMM_DISABLE");
    /* Same strict "1"-only match as every other GPUBRIDGE_*_DISABLE/VERBOSE/
     * PROFILE flag above. Checked per-call (not cached) at each matmul_f32
     * launch — cheap, unlike the vendor-library dlopen/dlsym probe itself. */
    return val != NULL && val[0] == '1' && val[1] == '\0';
}
