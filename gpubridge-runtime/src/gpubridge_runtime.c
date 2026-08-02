/*
 * gpubridge_runtime.c — implements the public API declared in gpubridge_runtime.h.
 *
 * This file owns all persistent runtime state (which backend is active,
 * what the user originally requested, the last error string) and is the
 * only file that touches that state directly. Every public function
 * follows the same shape: clear any stale error, check that gpuBridgeInit()
 * succeeded, delegate to the active backend (via the GpuBridgeBackend vtable) or
 * to gpubridge_selector.c, and translate failures into gpuBridgeGetLastErrorString().
 *
 * It never knows *which* backend (CPU or OpenCL) is active — it only ever
 * calls through g_selection.selected, a `const GpuBridgeBackend*` handed to it by
 * gpubridge_selector.c. That indirection is what makes this file backend-neutral.
 */
/* Needed for clock_gettime/CLOCK_MONOTONIC (Milestone 1.5 profiling, see
 * below) regardless of whether CMAKE_C_EXTENSIONS happens to be on. */
#define _POSIX_C_SOURCE 199309L

#include "gpubridge_internal.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h> /* Phase 4: sysconf(_SC_PHYS_PAGES/_SC_PAGE_SIZE) for gpuBridgeGetDeviceMemoryLimit's CPU path */

/* Whether gpuBridgeInit() has succeeded and not yet been undone by gpuBridgeShutdown(). */
static bool g_initialized = false;
/* The full result of the most recent successful backend selection:
 * availability flags for every backend plus the one actually chosen. */
static GpuBridgeSelection g_selection;
/* What GPUBRIDGE_BACKEND actually asked for (e.g. GPUBRIDGE_BACKEND_AUTO), kept
 * separately from g_selection.selected_kind (what was *resolved*), because
 * diagnostics need to print both ("requested backend: auto" /
 * "selected backend: opencl" are two different lines). */
static GpuBridgeBackendKind g_requested_kind = GPUBRIDGE_BACKEND_AUTO;
/* Human-readable reason for the most recent failure that happened *above*
 * any single backend (invalid GPUBRIDGE_BACKEND value, "not initialized", an
 * unsupported kernel shape, ...). Takes priority over a backend's own
 * last_error() in gpuBridgeGetLastErrorString() — see the precedence rule there. */
static char g_runtime_error[256] = "";
/* --- Milestone 1.5 addition (spec1.5.md section 9) ---
 * Cumulative profiling counters for the current session. Reset in
 * gpuBridgeInit()/gpuBridgeSetBackend(), printed and cleared by gpuBridgeShutdown(). */
static GpuBridgeProfileStats g_profile;
/* --- gpubridge_performance_spec.md section 5 gap-fix (2026-07-14) ---
 * Wall-clock timestamp of the current session's gpuBridgeInit()/gpuBridgeSetBackend(),
 * used only to compute total_runtime_ms in gpuBridgeShutdown() (passed to
 * gpuBridgePrintProfile alongside g_profile, not stored inside it — a
 * timestamp isn't a cumulative counter, unlike everything else in
 * GpuBridgeProfileStats). */
static struct timespec g_session_start;
/* --- Phase 4 addition (spec4.md section 5.2) ---
 * Not-nestable model-load bracket: g_model_load_active tracks whether a
 * gpuBridgeBeginModelLoad() is currently open; g_model_load_start is its
 * timestamp. A second Begin before a matching End just restarts the bracket
 * (documented, not an error — single-threaded, no stack). */
static struct timespec g_model_load_start;
static bool g_model_load_active = false;

const char* gpuBridgeBackendKindToString(GpuBridgeBackendKind kind)
{
    /* Canonical, lowercase name table shared by diagnostics
     * (gpubridge_diagnostics.c) and the test program — see gpubridge_runtime.h's comment
     * on why lowercase was chosen for every backend name. */
    switch (kind) {
    case GPUBRIDGE_BACKEND_AUTO:   return "auto";
    case GPUBRIDGE_BACKEND_CPU:    return "cpu";
    case GPUBRIDGE_BACKEND_OPENCL: return "opencl";
    case GPUBRIDGE_BACKEND_SYCL:   return "sycl";
    case GPUBRIDGE_BACKEND_VULKAN: return "vulkan";
    default:                return "unknown";
    }
}

/* Every public entry point calls this first so that a stale error message
 * from an earlier, unrelated failure never leaks into a later successful
 * call's gpuBridgeGetLastErrorString() result. */
static void clear_runtime_error(void)
{
    g_runtime_error[0] = '\0';
}

/* --- Milestone 1.5 addition (spec1.5.md section 9.3) ---
 * Host-side wall-clock elapsed time in milliseconds between two
 * CLOCK_MONOTONIC timestamps. Both backends are already fully synchronous
 * (blocking transfers; every test explicitly syncs after a launch), so
 * bracketing their calls with this is accurate for this milestone's
 * purposes without needing GPU-side profiling events (deferred, see
 * spec1.5.md section 5 item 10). */
static double elapsed_ms(const struct timespec* t0, const struct timespec* t1)
{
    double sec = (double)(t1->tv_sec - t0->tv_sec);
    double nsec = (double)(t1->tv_nsec - t0->tv_nsec);
    return sec * 1000.0 + nsec / 1e6;
}

/*
 * Shared helper used by both gpuBridgeInit() and gpuBridgeSetBackend(): asks
 * gpubridge_selector.c to resolve `requested` into a concrete backend, then calls
 * that backend's init(). On success, commits the result into the global
 * g_selection so every other public function can use it. On failure,
 * copies whichever error message is more specific (the selector's, or the
 * backend's own last_error() if selection succeeded but init() itself
 * failed) into g_runtime_error and leaves g_selection untouched.
 */
static bool select_and_init(GpuBridgeBackendKind requested)
{
    GpuBridgeSelection selection;
    char err[256];

    if (!gpuBridgeSelectBackend(requested, &selection, err, sizeof(err))) {
        snprintf(g_runtime_error, sizeof(g_runtime_error), "%s", err);
        return false;
    }

    if (!selection.selected->init()) {
        snprintf(g_runtime_error, sizeof(g_runtime_error), "%s",
            selection.selected->last_error());
        return false;
    }

    g_selection = selection;
    return true;
}

int gpuBridgeInit(void)
{
    clear_runtime_error();

    /* Calling gpuBridgeInit() again after a prior success is a cheap no-op rather
     * than re-selecting/re-initializing a backend. */
    if (g_initialized) {
        return 0;
    }

    /* Fresh session: start profiling counters from zero (spec1.5.md
     * section 9.3). Only on the real-init path, matching g_selection. */
    memset(&g_profile, 0, sizeof(g_profile));
    clock_gettime(CLOCK_MONOTONIC, &g_session_start);
    g_model_load_active = false; /* Phase 4: no stale bracket carried into a fresh session */

    bool valid = true;
    GpuBridgeBackendKind requested = gpuBridgeEnvGetRequestedBackend(&valid);
    if (!valid) {
        /* gpuBridgeEnvGetRequestedBackend() only tells us *that* the value was
         * bad, not what it was; re-read it here purely to quote it back in
         * the error message. */
        const char* raw = getenv("GPUBRIDGE_BACKEND");
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "GPUBRIDGE_BACKEND has unrecognized value '%s' (expected "
            "auto|cpu|opencl|sycl)", raw ? raw : "");
        return 1;
    }

    /* Remember what was actually requested (before resolution) so
     * diagnostics can report it later, even after AUTO resolves to a
     * concrete backend. */
    g_requested_kind = requested;
    /* --- Phase 4 addition (spec4.md section 5.1) ---
     * Brackets the existing select_and_init() call in place, without
     * restructuring gpuBridgeInit()'s early-return shape: this is the
     * runtime's own cold-start cost (backend selection + that backend's
     * init()), a documented *subset* of cpu_orchestration_time_ms (already
     * folded into that bucket, not double-subtracted — see
     * GpuBridgeProfileStats.runtime_init_time_ms's own comment). */
    struct timespec init_t0, init_t1;
    clock_gettime(CLOCK_MONOTONIC, &init_t0);
    if (!select_and_init(requested)) {
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &init_t1);
    g_profile.runtime_init_time_ms = elapsed_ms(&init_t0, &init_t1);

    g_initialized = true;
    return 0;
}

int gpuBridgeSetBackend(GpuBridgeBackendKind backend)
{
    clear_runtime_error();

    /* Cleanly tear down whatever backend was active before switching, so we
     * never have two backends simultaneously holding device resources.
     * gpuBridgePoolDrain() (Milestone 2.5, spec2.5.md section 7.3) must run
     * first, while g_selection.selected still names the outgoing backend —
     * a pooled device_ptr from that backend is meaningless to whatever
     * backend `backend` resolves to. */
    if (g_initialized && g_selection.selected != NULL) {
        gpuBridgePoolDrain(g_selection.selected);
        g_selection.selected->shutdown();
    }
    g_initialized = false;

    /* New session under the new backend: start profiling counters from
     * zero rather than mixing two backends' work into one report. */
    memset(&g_profile, 0, sizeof(g_profile));
    clock_gettime(CLOCK_MONOTONIC, &g_session_start);
    g_model_load_active = false; /* Phase 4: same reason as gpuBridgeInit() above */

    g_requested_kind = backend;
    /* Phase 4 (spec4.md section 5.1): identical bracket to gpuBridgeInit()'s. */
    struct timespec init_t0, init_t1;
    clock_gettime(CLOCK_MONOTONIC, &init_t0);
    if (!select_and_init(backend)) {
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &init_t1);
    g_profile.runtime_init_time_ms = elapsed_ms(&init_t0, &init_t1);

    g_initialized = true;
    return 0;
}

int gpuBridgeShutdown(void)
{
    clear_runtime_error();

    if (g_initialized && g_selection.selected != NULL) {
        /* Print the GPUBRIDGE_PROFILE=1 summary (self-gated, no-op unless the
         * env var is set) for the session that's about to end, before
         * tearing down the backend or clearing the counters — spec1.5.md
         * section 9.3. Guarded on g_initialized so a shutdown() that never
         * had a successful init (or a double-shutdown) doesn't print a
         * misleading all-zero block. total_runtime_ms (gpubridge_performance_spec.md
         * section 5 gap-fix) is wall-clock since g_session_start, computed
         * here rather than stored in GpuBridgeProfileStats (see its definition). */
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double total_runtime_ms = elapsed_ms(&g_session_start, &now);

        /* --- Phase 4 addition (spec4.md section 9.2) ---
         * Device memory limit/hardware tier are session-static once a
         * backend is selected — querying them on every launch would add
         * avoidable per-launch OpenCL-query overhead for information that
         * never changes between launches. Queried once here, gated behind
         * gpuBridgeEnvProfile() (zero extra cost when GPUBRIDGE_PROFILE isn't
         * set), immediately before the one gpuBridgePrintProfile() call site. */
        GpuBridgeBackendKind selected_backend = GPUBRIDGE_BACKEND_AUTO;
        char device_name_buf[128] = "";
        size_t mem_limit = 0;
        GpuBridgeHardwareTier tier = GPUBRIDGE_HARDWARE_TIER_UNKNOWN;
        if (gpuBridgeEnvProfile()) {
            selected_backend = g_selection.selected_kind;
            gpuBridgeGetSelectedDeviceName(device_name_buf, sizeof(device_name_buf));
            gpuBridgeGetDeviceMemoryLimit(&mem_limit);
            gpuBridgeGetHardwareTier(&tier);
        }
        gpuBridgePrintProfile(&g_profile, total_runtime_ms, selected_backend,
            device_name_buf, mem_limit, tier);
        /* Milestone 2.5 (spec2.5.md section 7.3): drain the tensor pool
         * (actually free_device() every pooled block) before the backend
         * itself is shut down — a pooled device_ptr from this backend must
         * never be handed to a different backend after a later gpuBridgeInit(). */
        gpuBridgePoolDrain(g_selection.selected);
        g_selection.selected->shutdown();
    }
    g_initialized = false;
    /* Zero the selection so a stray use-after-shutdown reads NULL/false
     * fields instead of a stale backend pointer. */
    memset(&g_selection, 0, sizeof(g_selection));
    memset(&g_profile, 0, sizeof(g_profile));
    g_model_load_active = false; /* Phase 4: same reason as gpuBridgeInit()/gpuBridgeSetBackend() above */
    return 0;
}

int gpuBridgeGetDeviceCount(int* count)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        *count = 0;
        return 1;
    }

    /* Exactly one usable device in this milestone, whichever backend is
     * active: one CPU "device", or the single OpenCL GPU device found
     * during selection. */
    *count = 1;
    return 0;
}

int gpuBridgeMalloc(void** ptr, size_t bytes)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }

    /* Delegate straight to the active backend. For OpenCL this returns a
     * cl_mem cast to void*; for CPU it's a plain malloc() pointer. Either
     * way, this file treats it as fully opaque. */
    *ptr = g_selection.selected->malloc_device(bytes);
    if (*ptr == NULL) {
        snprintf(g_runtime_error, sizeof(g_runtime_error), "%s",
            g_selection.selected->last_error());
        return 1;
    }
    return 0;
}

int gpuBridgeFree(void* ptr)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }

    g_selection.selected->free_device(ptr);
    return 0;
}

int gpuBridgeMemcpy(void* dst, const void* src, size_t bytes, GpuBridgeMemcpyKind kind)
{
    clear_runtime_error();

    if (gpuBridgeEnvVerbose()) {
        static const char* kind_names[] = {
            "host_to_device", "device_to_host", "host_to_host", "device_to_device"
        };
        printf("we are in function: gpuBridgeMemcpy\n"
            "purpose: move %zu bytes (%s) through the selected backend's "
            "copy_to_device/copy_to_host — the actual host<->device data "
            "transfer step\n", bytes, kind_names[(int)kind]);
    }

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        if (gpuBridgeEnvVerbose()) {
            printf("leaving function: gpuBridgeMemcpy, "
                "result: failed (not initialized)\n\n");
        }
        return 1;
    }

    bool ok = false;
    struct timespec t0, t1;
    switch (kind) {
    case GPUBRIDGE_MEMCPY_HOST_TO_DEVICE:
        /* dst is a device handle (from gpuBridgeMalloc), src is a host pointer. */
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ok = g_selection.selected->copy_to_device(dst, src, bytes);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        /* Only count bytes/time that actually moved (spec1.5.md section
         * 9.3) — a failed copy didn't transfer anything. */
        if (ok) {
            g_profile.host_to_device_bytes += bytes;
            g_profile.host_to_device_time_ms += elapsed_ms(&t0, &t1);
            g_profile.host_to_device_count++;
        }
        break;
    case GPUBRIDGE_MEMCPY_DEVICE_TO_HOST:
        /* dst is a host pointer, src is a device handle. */
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ok = g_selection.selected->copy_to_host(dst, src, bytes);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (ok) {
            g_profile.device_to_host_bytes += bytes;
            g_profile.device_to_host_time_ms += elapsed_ms(&t0, &t1);
            g_profile.device_to_host_count++;
        }
        break;
    case GPUBRIDGE_MEMCPY_HOST_TO_HOST:
        /* Neither side involves the backend at all — just a normal copy
         * between two host buffers. */
        memcpy(dst, src, bytes);
        ok = true;
        break;
    case GPUBRIDGE_MEMCPY_DEVICE_TO_DEVICE:
        /* Out of scope for this milestone: the GpuBridgeBackend interface has no
         * device-to-device copy slot (spec1.md section 9), and the vecadd
         * test never exercises this path. Fail clearly rather than
         * mis-copying or crashing. */
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "GPUBRIDGE_MEMCPY_DEVICE_TO_DEVICE is not supported in this milestone");
        ok = false;
        break;
    }

    /* Only pull the backend's own last_error() if this path didn't already
     * set a more specific g_runtime_error (the DEVICE_TO_DEVICE case above
     * already did). */
    if (!ok && g_runtime_error[0] == '\0') {
        snprintf(g_runtime_error, sizeof(g_runtime_error), "%s",
            g_selection.selected->last_error());
    }
    if (gpuBridgeEnvVerbose()) {
        printf("leaving function: gpuBridgeMemcpy, result: %s\n\n",
            ok ? "success" : g_runtime_error);
    }
    return ok ? 0 : 1;
}

int gpuBridgeLaunchKernel(const GpuBridgeKernelIR* kernel, void** args, size_t arg_count)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }

    if (args == NULL) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpuBridgeLaunchKernel: args must not be NULL");
        return 1;
    }

    /* This milestone's runtime only understands one kernel shape:
     * vector_add_f32 with exactly 3 args (a, b, c). gpuBridgeIrValidateKernel
     * (Phase 2, milestone.md section 22 deliverable 4) rejects anything
     * else up front with a clear message instead of it being passed down
     * to a backend that has no way to interpret it. */
    if (!gpuBridgeIrValidateKernel(kernel, arg_count, g_runtime_error, sizeof(g_runtime_error))) {
        return 1;
    }

    /* This is the only point in the runtime where both the resolved
     * backend selection and the specific operation/element-count being
     * launched are simultaneously available, so it's where the
     * GPUBRIDGE_VERBOSE=1 block gets printed (see gpubridge_diagnostics.c). Uses
     * g_requested_kind (not g_selection.selected_kind) so "requested
     * backend:" reflects what GPUBRIDGE_BACKEND was actually set to. */
    gpuBridgePrintDiagnostics(g_requested_kind, &g_selection,
        kernel->kernel_name, kernel->global_size);

    /* args[0]/args[1]/args[2] are the device pointers for a/b/c, in the
     * same order gpuBridgeIrInitVectorAddF32() placed them in kernel->args. */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    bool ok = g_selection.selected->launch_vector_add_f32(
        (const float*)args[0], (const float*)args[1], (float*)args[2],
        kernel->global_size);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (ok) {
        g_profile.kernel_time_ms += elapsed_ms(&t0, &t1);
        g_profile.kernel_launch_count++;
    }

    if (!ok) {
        snprintf(g_runtime_error, sizeof(g_runtime_error), "%s",
            g_selection.selected->last_error());
        return 1;
    }
    return 0;
}

int gpuBridgeDeviceSynchronize(void)
{
    clear_runtime_error();

    if (gpuBridgeEnvVerbose()) {
        printf("we are in function: gpuBridgeDeviceSynchronize\n"
            "purpose: block until the selected backend's device queue has "
            "finished all pending work (needed since kernel launches like "
            "OpenCL's clEnqueueNDRangeKernel are non-blocking)\n");
    }

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        if (gpuBridgeEnvVerbose()) {
            printf("leaving function: gpuBridgeDeviceSynchronize, "
                "result: failed (not initialized)\n\n");
        }
        return 1;
    }

    /* Counts every synchronization *request*, regardless of outcome — it
     * measures how often sync was asked for, not how much work completed
     * (spec1.5.md section 9.3), so this increments before the call below
     * can fail. synchronization_time_ms (gpubridge_performance_spec.md section
     * 5 gap-fix) times the call itself regardless of outcome too, for the
     * same reason: a failed sync still spent wall-clock time blocked in it. */
    g_profile.synchronization_count++;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    bool ok = g_selection.selected->sync();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_profile.synchronization_time_ms += elapsed_ms(&t0, &t1);

    if (!ok) {
        snprintf(g_runtime_error, sizeof(g_runtime_error), "%s",
            g_selection.selected->last_error());
        if (gpuBridgeEnvVerbose()) {
            printf("leaving function: gpuBridgeDeviceSynchronize, "
                "result: failed (%s)\n\n", g_runtime_error);
        }
        return 1;
    }
    if (gpuBridgeEnvVerbose()) {
        printf("leaving function: gpuBridgeDeviceSynchronize, result: success\n\n");
    }
    return 0;
}

const char* gpuBridgeGetLastErrorString(void)
{
    /* Precedence: a runtime-level error (invalid GPUBRIDGE_BACKEND, "not
     * initialized", bad kernel shape, selector failure) always wins over
     * whatever the backend's own last_error() currently holds, because it
     * is strictly more specific to what just failed. */
    if (g_runtime_error[0] != '\0') {
        return g_runtime_error;
    }
    if (g_initialized && g_selection.selected != NULL) {
        return g_selection.selected->last_error();
    }
    return "gpubridge runtime not initialized";
}

GpuBridgeBackendKind gpuBridgeGetSelectedBackend(void)
{
    if (!g_initialized) {
        return GPUBRIDGE_BACKEND_AUTO;
    }
    return g_selection.selected_kind;
}

/* --- Milestone 1.5 additions (spec1.5.md section 10) --- */

int gpuBridgeGetBackendCaps(GpuBridgeBackendCaps* out_caps)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }
    if (out_caps == NULL) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpuBridgeGetBackendCaps: out_caps must not be NULL");
        return 1;
    }

    if (!g_selection.selected->get_caps(out_caps)) {
        snprintf(g_runtime_error, sizeof(g_runtime_error), "%s",
            g_selection.selected->last_error());
        return 1;
    }
    return 0;
}

int gpuBridgeGetProfileStats(GpuBridgeProfileStats* out_stats)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }
    if (out_stats == NULL) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpuBridgeGetProfileStats: out_stats must not be NULL");
        return 1;
    }

    *out_stats = g_profile;
    return 0;
}

/* --- Milestone 3 addition (spec3.md section 8.5) --- */

int gpuBridgeGetSelectedDeviceName(char* buf, size_t buf_len)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }
    if (buf == NULL || buf_len == 0) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpuBridgeGetSelectedDeviceName: buf must not be NULL/zero-length");
        return 1;
    }

    /* buf is always written once initialized, regardless of the underlying
     * query's own success (CPU has nothing to query; OpenCL's query
     * failing writes "unknown", not an error) — the only failure this
     * function itself reports is "not initialized", matching every other
     * function in this header's 0/1 convention. */
    if (g_selection.selected_kind == GPUBRIDGE_BACKEND_OPENCL) {
        gpuBridgeOpenclGetDeviceName(buf, buf_len);
    } else if (g_selection.selected_kind == GPUBRIDGE_BACKEND_VULKAN) {
        gpuBridgeVulkanGetDeviceName(buf, buf_len);
    } else {
        snprintf(buf, buf_len, "CPU (host)");
    }
    return 0;
}

/* --- Milestone 2 additions (spec2.md sections 8.2, 10) --- */

/* Number of elements described by a rank-1-or-2 shape, used to turn a
 * GpuBridgeTensorDesc's shape into a byte count for malloc_device(). shape[1] is
 * ignored for rank 1, matching GpuBridgeTensorDesc's documented convention. */
static size_t tensor_element_count(int rank, const size_t shape[2])
{
    return rank == 1 ? shape[0] : shape[0] * shape[1];
}

/*
 * --- Phase 4 addition (spec4.md section 6.1) ---
 *
 * Shared body for gpuBridgeTensorAlloc()/gpuBridgeTensorAllocKvCache(). The
 * only two differences between the two public entry points are whether the
 * pool is unconditionally bypassed (force_pool_bypass) and what usage tag
 * the resulting descriptor carries — everything else (type/rank rejection,
 * byte-size formula, pool-acquire-or-miss, memory_strategy derivation) is
 * identical, so this is the single place that logic lives.
 */
static int tensor_alloc_impl(GpuBridgeTensorDesc* desc, GpuBridgeScalarType type, int rank,
    const size_t shape[2], bool force_pool_bypass, GpuBridgeTensorUsage usage)
{
    clear_runtime_error();

    if (gpuBridgeEnvVerbose()) {
        printf("we are in function: tensor_alloc_impl\n"
            "purpose: turn a caller's shape request into a real device-"
            "resident GpuBridgeTensorDesc — pool-acquire-or-miss, then the "
            "backend's malloc_device() on a miss (rank=%d, usage=%s)\n",
            rank, usage == GPUBRIDGE_TENSOR_USAGE_KV_CACHE ? "kv_cache" : "generic");
    }

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        if (gpuBridgeEnvVerbose()) {
            printf("leaving function: tensor_alloc_impl, "
                "result: failed (not initialized)\n\n");
        }
        return 1;
    }

    /* --- Phase 4 (spec4.md section 7): quantized types are descriptive-only
     * this phase — decisive rejection, not a silent fallback. The shared
     * byte-size formula below is built around 4-byte f32 elements;
     * accepting a quantized type "for storage only" would require either a
     * correct packed byte count (non-trivial for int4) that nothing then
     * reads correctly, or silently allocating f32-sized storage that
     * misrepresents what's actually in it — exactly the "silently fail into
     * a slow or incorrect path" gpubridge_performance_spec.md section 3.10
     * forbids. Checked before the generic type/rank rejection below so this
     * specific, more actionable message wins. */
    if (desc != NULL &&
        (type == GPUBRIDGE_TYPE_INT8_QUANTIZED || type == GPUBRIDGE_TYPE_INT4_QUANTIZED)) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpuBridgeTensorAlloc: %s is descriptive-only in this phase (Phase 4 "
            "quantization-aware metadata) — no allocation or execution support "
            "exists yet; describe a quantized layout via "
            "gpuBridgeIrDescribeQuantizedTensor() instead",
            type == GPUBRIDGE_TYPE_INT8_QUANTIZED
                ? "GPUBRIDGE_TYPE_INT8_QUANTIZED" : "GPUBRIDGE_TYPE_INT4_QUANTIZED");
        if (gpuBridgeEnvVerbose()) {
            printf("leaving function: tensor_alloc_impl, "
                "result: failed (quantized type is descriptive-only)\n\n");
        }
        return 1;
    }

    /* Only f32 exists otherwise (gpubridge_ir.h), and only rank 1/2 tensors
     * are in scope (spec2.md non-goal #8) — reject anything else up front
     * rather than letting a backend misinterpret it. Same restriction
     * applies to KV-cache tensors (spec4.md section 6.1): no quantized KV
     * cache yet. */

     //rank != 1 && rank != 2 — only vectors (rank 1) and matrices (rank 2) are supported;
    //tensor system only actually implements float32 storage.
     if (desc == NULL || type != GPUBRIDGE_TYPE_F32 || (rank != 1 && rank != 2)) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpuBridgeTensorAlloc: unsupported type/rank (only f32, rank 1 or 2, "
            "is supported in this milestone)");
        if (gpuBridgeEnvVerbose()) {
            printf("leaving function: tensor_alloc_impl, "
                "result: failed (unsupported type/rank)\n\n");
        }
        return 1;
    }

    /* --- Milestone 2.5 (spec2.5.md section 7.2): pooled allocation --- *
     * Reuses the active backend's existing malloc_device() on a pool miss
     * (or whenever GPUBRIDGE_POOL_DISABLE=1, or force_pool_bypass — Phase 4's
     * KV-cache path reuses this exact existing bypass mechanism rather than
     * inventing new logic) — still no new backend allocator slot for
     * tensors (spec2.md section 8.2); the pool sits entirely above that
     * vtable call. gpuBridgeMalloc/gpuBridgeFree (vector_add_f32's path)
     * never consult the pool and are completely unaffected. */
    size_t bytes = tensor_element_count(rank, shape) * sizeof(float);//around 4-byte elements
    bool pool_disabled = gpuBridgeEnvPoolDisable() || force_pool_bypass;
    void* ptr = NULL;
    int reuse_count = 1; /* spec2.5.md section 7.2's disabled-path default */

    /* tensor_alloc_time_ms (gpubridge_performance_spec.md section 5 gap-fix)
     * brackets the whole acquire-or-miss-then-malloc_device path, whichever
     * branch actually runs — this is "allocation time" in the sense the
     * performance spec's area 4 asks for, not just the backend call. */
    struct timespec alloc_t0, alloc_t1;
    clock_gettime(CLOCK_MONOTONIC, &alloc_t0);

    if (!pool_disabled) {
        ptr = gpuBridgePoolAcquire(bytes, &reuse_count);
        if (ptr != NULL) {
            /* Pool hit: no new backend memory was consumed, so
             * peak_device_bytes (section 9.1) does not grow here. */
            g_profile.tensor_pool_hit_count++;
        }
    }
    if (ptr == NULL) {
        ptr = g_selection.selected->malloc_device(bytes);
        if (ptr != NULL) {
            /* A real backend allocation happened — grows the session's
             * high-water mark (spec2.5.md section 9.1's peak_device_bytes;
             * monotonic, since a pooled block is never truly freed until a
             * drain, so this figure is also the current footprint). */
            g_profile.peak_device_bytes += bytes;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &alloc_t1);
    if (ptr != NULL) {
        g_profile.tensor_alloc_time_ms += elapsed_ms(&alloc_t0, &alloc_t1);
    }
    if (ptr == NULL) {
        snprintf(g_runtime_error, sizeof(g_runtime_error), "%s",
            g_selection.selected->last_error());
        if (gpuBridgeEnvVerbose()) {
            printf("leaving function: tensor_alloc_impl, "
                "result: failed (%s)\n\n", g_runtime_error);
        }
        return 1;
    }

    /* memory_strategy (spec2.5.md section 6) uses the pool's real
     * reuse_count for this byte size instead of spec1.5.md's illustrative
     * constant. Not fatal if get_caps() ever failed (never happens for the
     * CPU/OpenCL backends implemented so far) — falls back to AUTO. */
    GpuBridgeBackendCaps caps;
    GpuBridgeMemoryStrategy strategy = GPUBRIDGE_MEMORY_AUTO;
    if (g_selection.selected->get_caps(&caps)) {
        strategy = gpuBridgeChooseMemoryStrategy(bytes, reuse_count, false, &caps);
    }

    desc->type = type;
    desc->rank = rank;
    desc->shape[0] = shape[0];
    desc->shape[1] = (rank == 2) ? shape[1] : 0;
    desc->device_ptr = ptr;
    desc->device_resident = true;
    desc->memory_strategy = strategy;
    /* --- Phase 4 (spec4.md section 6): both explicitly assigned, since
     * gpuBridgeTensorAlloc() does not zero-initialize *desc beforehand. */
    desc->usage = usage;
    desc->quant = (GpuBridgeQuantDesc){0};

    /* Low-level IR-to-device-descriptor trace: the caller only ever hands
     * tensor_alloc_impl() a scalar type + rank + shape[2] (whatever a
     * GpuBridgeTensorArg/GpuBridgeIrInit*F32 builder produced); this is the
     * one place that reshapes that request into the actual device-resident
     * GpuBridgeTensorDesc (pool hit-or-miss pointer, computed memory
     * strategy, byte size) — not printed anywhere else, unlike the
     * pre-launch "shapes:" line in gpuBridgePrintTensorDiagnostics, which
     * only echoes the IR's own shape fields back. */
    if (gpuBridgeEnvVerbose()) {
        printf("GPUBridge: tensor_alloc_impl -> GpuBridgeTensorDesc { rank=%d, "
            "shape=[%zu,%zu], bytes=%zu, device_ptr=%p, memory_strategy=%s, "
            "usage=%s, pool_reuse_count=%d }\n",
            desc->rank, desc->shape[0], desc->shape[1], bytes, desc->device_ptr,
            gpuBridgeMemoryStrategyToString(desc->memory_strategy),
            usage == GPUBRIDGE_TENSOR_USAGE_KV_CACHE ? "kv_cache" : "generic",
            reuse_count);
        printf("leaving function: tensor_alloc_impl, "
            "result: success (device_ptr=%p)\n\n", desc->device_ptr);
    }

    /* Counted regardless of pool involvement (spec2.5.md section 4 item 4) —
     * even a GPUBRIDGE_POOL_DISABLE=1 allocation is still a real tensor
     * allocation. */
    g_profile.tensor_allocation_count++;
    return 0;
}

int gpuBridgeTensorAlloc(GpuBridgeTensorDesc* desc, GpuBridgeScalarType type, int rank,
    const size_t shape[2])
{
    return tensor_alloc_impl(desc, type, rank, shape, false, GPUBRIDGE_TENSOR_USAGE_GENERIC);
}

int gpuBridgeTensorAllocKvCache(GpuBridgeTensorDesc* desc, GpuBridgeScalarType type, int rank,
    const size_t shape[2])
{
    int rc = tensor_alloc_impl(desc, type, rank, shape, true, GPUBRIDGE_TENSOR_USAGE_KV_CACHE);
    if (rc == 0) {
        size_t bytes = tensor_element_count(rank, shape) * sizeof(float);
        g_profile.kv_cache_allocation_count++;
        g_profile.kv_cache_bytes += bytes;
        if (gpuBridgeEnvVerbose()) {
            printf("GPUBridge: tensor usage=kv_cache, pool bypassed "
                "(long-lived, device-resident), bytes=%zu\n", bytes);
        }
    }
    return rc;
}

int gpuBridgeTensorFree(GpuBridgeTensorDesc* desc)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }

    /* Freeing a descriptor that was never (or already) allocated is a
     * no-op, not an error — mirrors gpuBridgeFree's tolerance and keeps
     * tests/callers from needing to track allocation state separately. */
    if (desc == NULL || !desc->device_resident) {
        return 0;
    }

    /* --- Milestone 2.5 (spec2.5.md section 7.2/7.4): return to the pool --- *
     * Ownership rule (spec2.5.md section 7.4): once this returns,
     * desc->device_ptr (cleared below to NULL) and any copy of it the
     * caller cached separately must not be used again — the pool may
     * silently reissue this exact block to a different, still-live tensor
     * on a later gpuBridgeTensorAlloc() of the same size. */
    struct timespec free_t0, free_t1;
    clock_gettime(CLOCK_MONOTONIC, &free_t0);
    /* --- Phase 4 (spec4.md section 6.2): free-side symmetry --- *
     * A KV-cache tensor bypassed the pool on allocation (tensor_alloc_impl's
     * force_pool_bypass), so it must bypass it here too — a pooled block
     * from a KV-cache allocation would otherwise be silently reissued to an
     * ordinary short-lived tensor. */
    if (gpuBridgeEnvPoolDisable() || desc->usage == GPUBRIDGE_TENSOR_USAGE_KV_CACHE) {
        g_selection.selected->free_device(desc->device_ptr);
    } else {
        size_t bytes = tensor_element_count(desc->rank, desc->shape) * sizeof(float);
        gpuBridgePoolRelease(desc->device_ptr, bytes);
    }
    clock_gettime(CLOCK_MONOTONIC, &free_t1);
    g_profile.tensor_free_time_ms += elapsed_ms(&free_t0, &free_t1);

    g_profile.tensor_free_count++;
    desc->device_ptr = NULL;
    desc->device_resident = false;
    return 0;
}

int gpuBridgeLaunchTensorKernel(const GpuBridgeTensorKernelIR* kernel, void** args,
    size_t arg_count)
{
    clear_runtime_error();

    if (gpuBridgeEnvVerbose()) {
        printf("we are in function: gpuBridgeLaunchTensorKernel\n"
            "purpose: validate the tensor-op IR, then dispatch to the "
            "selected backend's launch_%s vtable slot (this is the runtime's "
            "single dispatch point for all four tensor ops)\n",
            kernel != NULL ? kernel->kernel_name : "?");
    }

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        if (gpuBridgeEnvVerbose()) {
            printf("leaving function: gpuBridgeLaunchTensorKernel, "
                "result: failed (not initialized)\n\n");
        }
        return 1;
    }

    if (args == NULL) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpuBridgeLaunchTensorKernel: args must not be NULL");
        if (gpuBridgeEnvVerbose()) {
            printf("leaving function: gpuBridgeLaunchTensorKernel, "
                "result: failed (args == NULL)\n\n");
        }
        return 1;
    }

    /* gpuBridgeIrValidateTensorKernel (Phase 2, milestone.md section 22
     * deliverable 4) checks kernel != NULL, op_kind, arg_count, and full
     * shape/rank consistency (e.g. matmul's inner dimensions actually
     * matching) up front, so every case below can trust kernel->args'
     * shapes without re-checking them. */
    if (!gpuBridgeIrValidateTensorKernel(kernel, arg_count, g_runtime_error, sizeof(g_runtime_error))) {
        if (gpuBridgeEnvVerbose()) {
            printf("leaving function: gpuBridgeLaunchTensorKernel, "
                "result: failed (IR validation: %s)\n\n", g_runtime_error);
        }
        return 1;
    }

    /* Dimensions are pulled from kernel->args' shapes (filled in by the
     * gpuBridgeIrInit<Op>F32 builders in gpubridge_ir.c) rather than accepted separately,
     * so the IR is the single source of truth for shape information. */
    bool ok = false;
    struct timespec t0, t1;

    /* Full field-by-field IR dump (gpuBridgePrintTensorIR — 2026-07-29
     * addition), once per launch regardless of op_kind, right alongside the
     * existing per-case gpuBridgePrintTensorDiagnostics compact-line calls
     * below. Placed before the switch since it doesn't need any op-specific
     * unpacking (m/k/n, rows/cols, ...) the way the backend dispatch calls
     * below do — it just walks kernel->args/kernel itself. */
    gpuBridgePrintTensorIR(kernel);

    switch (kernel->op_kind) {
    case GPUBRIDGE_OP_MATMUL: {
        size_t m = kernel->args[0].shape[0];
        size_t k = kernel->args[0].shape[1];
        size_t n = kernel->args[1].shape[1];
        //before actually launching the kernel, print the tensor diagnostics for this kernel launch
        //already includes the kernel name, op kind, and tensor shapes
        //it's called before the actual launch — everything it prints (backend selection, shapes, predicted gemm path) is knowable in advance, from state already set up during gpuBridgeInit/gpuBridgeSetBackend plus the IR just validated.
        gpuBridgePrintTensorDiagnostics(g_requested_kind, &g_selection,
            kernel->kernel_name, kernel->args, arg_count);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        //dispatch call to the selected backend's matmul implementation
        ok = g_selection.selected->launch_matmul_f32(
            (const float*)args[0], (const float*)args[1], (float*)args[2],
            m, k, n);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        break;
    }
    case GPUBRIDGE_OP_REDUCE_SUM: {
        size_t n = kernel->args[0].shape[0];
        gpuBridgePrintTensorDiagnostics(g_requested_kind, &g_selection,
            kernel->kernel_name, kernel->args, arg_count);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ok = g_selection.selected->launch_reduce_sum_f32(
            (const float*)args[0], (float*)args[1], n);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        break;
    }
    case GPUBRIDGE_OP_BROADCAST_ADD: {
        size_t rows = kernel->args[0].shape[0];
        size_t cols = kernel->args[0].shape[1];
        gpuBridgePrintTensorDiagnostics(g_requested_kind, &g_selection,
            kernel->kernel_name, kernel->args, arg_count);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ok = g_selection.selected->launch_broadcast_add_f32(
            (float*)args[0], (const float*)args[1], rows, cols);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        break;
    }
    case GPUBRIDGE_OP_RELU: {
        /* Flattened element count — relu is purely elementwise, so a rank-2
         * [rows,cols] tensor launches rows*cols lanes, same as rank 1. */
        size_t count = tensor_element_count(kernel->args[0].rank,
            kernel->args[0].shape);
        gpuBridgePrintTensorDiagnostics(g_requested_kind, &g_selection,
            kernel->kernel_name, kernel->args, arg_count);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ok = g_selection.selected->launch_relu_f32(
            (const float*)args[0], (float*)args[1], count);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        break;
    }
    default:
        /* Unreachable in practice: gpuBridgeIrValidateTensorKernel above already
         * rejects GPUBRIDGE_OP_VECTOR_ADD (or any future value) with the same
         * message before this switch is ever reached. Kept as a defensive
         * fallback, not a live code path. */
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpuBridgeLaunchTensorKernel: unsupported op_kind");
        if (gpuBridgeEnvVerbose()) {
            printf("leaving function: gpuBridgeLaunchTensorKernel, "
                "result: failed (unsupported op_kind)\n\n");
        }
        return 1;
    }

    if (ok) {
        g_profile.kernel_time_ms += elapsed_ms(&t0, &t1);
        g_profile.kernel_launch_count++;
    }

    if (!ok) {
        snprintf(g_runtime_error, sizeof(g_runtime_error), "%s",
            g_selection.selected->last_error());
        if (gpuBridgeEnvVerbose()) {
            printf("leaving function: gpuBridgeLaunchTensorKernel, "
                "result: failed (%s)\n\n", g_runtime_error);
        }
        return 1;
    }
    if (gpuBridgeEnvVerbose()) {
        printf("leaving function: gpuBridgeLaunchTensorKernel, "
            "result: success (%s)\n\n", kernel->kernel_name);
    }
    return 0;
}

/* --- Milestone 2.5 additions (spec2.5.md section 8.2) --- */

int gpuBridgeTensorMirrorCreate(GpuBridgeTensorMirror* mirror, void* host_ptr,
    const GpuBridgeTensorDesc* desc, size_t bytes)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }
    if (mirror == NULL || host_ptr == NULL || desc == NULL || !desc->device_resident) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpuBridgeTensorMirrorCreate: mirror/host_ptr must not be NULL, and "
            "desc must already be device_resident (e.g. via gpuBridgeTensorAlloc)");
        return 1;
    }

    mirror->host_ptr = host_ptr;
    mirror->tensor = *desc;
    mirror->bytes = bytes;
    /* Common case: host data already exists (the caller's array), the
     * device copy does not exist yet (spec2.5.md section 8.2). */
    mirror->host_valid = true;
    mirror->device_valid = false;
    return 0;
}

int gpuBridgeTensorMirrorDestroy(GpuBridgeTensorMirror* mirror)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }
    if (mirror == NULL) {
        return 0;
    }

    /* Returns the underlying tensor to the pool via gpuBridgeTensorFree
     * (section 7) — does not touch mirror->host_ptr, which the caller owns. */
    int rc = gpuBridgeTensorFree(&mirror->tensor);
    mirror->host_ptr = NULL;
    mirror->bytes = 0;
    mirror->host_valid = false;
    mirror->device_valid = false;
    return rc;
}

int gpuBridgeTensorEnsureDeviceCurrent(GpuBridgeTensorMirror* mirror)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }
    if (mirror == NULL) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpuBridgeTensorEnsureDeviceCurrent: mirror must not be NULL");
        return 1;
    }

    /* Already current: no gpuBridgeMemcpy call issued (spec2.5.md section 8.2 —
     * the lazy-transfer property test_tensor_mirror.c checks for). */
    if (mirror->device_valid) {
        return 0;
    }

    if (gpuBridgeMemcpy(mirror->tensor.device_ptr, mirror->host_ptr, mirror->bytes,
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0) {
        return 1;
    }
    mirror->device_valid = true;
    return 0;
}

int gpuBridgeTensorEnsureHostCurrent(GpuBridgeTensorMirror* mirror)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }
    if (mirror == NULL) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpuBridgeTensorEnsureHostCurrent: mirror must not be NULL");
        return 1;
    }

    if (mirror->host_valid) {
        return 0;
    }

    if (gpuBridgeMemcpy(mirror->host_ptr, mirror->tensor.device_ptr, mirror->bytes,
            GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) != 0) {
        return 1;
    }
    mirror->host_valid = true;
    return 0;
}

int gpuBridgeTensorMarkHostModified(GpuBridgeTensorMirror* mirror)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }
    if (mirror == NULL) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpuBridgeTensorMarkHostModified: mirror must not be NULL");
        return 1;
    }

    mirror->host_valid = true;
    mirror->device_valid = false;
    return 0;
}

int gpuBridgeTensorMarkDeviceModified(GpuBridgeTensorMirror* mirror)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }
    if (mirror == NULL) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpuBridgeTensorMarkDeviceModified: mirror must not be NULL");
        return 1;
    }

    mirror->device_valid = true;
    mirror->host_valid = false;
    return 0;
}

/* --- Phase 4 additions (spec4.md sections 5.2, 9) --- */

int gpuBridgeBeginModelLoad(void)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }

    /* Not nestable: a second Begin before a matching End just restarts the
     * bracket from here (single-threaded, no stack) — documented, not an
     * error (spec4.md section 5.2). */
    clock_gettime(CLOCK_MONOTONIC, &g_model_load_start);
    g_model_load_active = true;
    if (gpuBridgeEnvVerbose()) {
        printf("GPUBridge: model load started\n");
    }
    return 0;
}

int gpuBridgeEndModelLoad(void)
{
    clear_runtime_error();

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }

    /* Tolerant no-op without a prior Begin, matching this project's
     * existing "double free"/"free of never-allocated" tolerance
     * conventions (spec4.md section 5.2). */
    if (!g_model_load_active) {
        return 0;
    }

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed = elapsed_ms(&g_model_load_start, &now);
    g_profile.model_load_time_ms += elapsed;
    g_model_load_active = false;
    if (gpuBridgeEnvVerbose()) {
        printf("GPUBridge: model load finished (%.2f ms)\n", elapsed);
    }
    return 0;
}

int gpuBridgeGetDeviceMemoryLimit(size_t* out_bytes)
{
    clear_runtime_error();
    *out_bytes = 0;

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }

    if (g_selection.selected_kind == GPUBRIDGE_BACKEND_OPENCL) {
        gpuBridgeOpenclGetDeviceMemoryLimit(out_bytes);
    } else if (g_selection.selected_kind == GPUBRIDGE_BACKEND_VULKAN) {
        gpuBridgeVulkanGetDeviceMemoryLimit(out_bytes);
    } else {
        /* CPU: total system RAM — POSIX-portable, no root needed. The CPU
         * "device" has no separate memory pool, so this is the accurate
         * figure (spec4.md section 9). */
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0) {
            *out_bytes = (size_t)pages * (size_t)page_size;
        }
    }
    return 0;
}

int gpuBridgeGetHardwareTier(GpuBridgeHardwareTier* out_tier)
{
    clear_runtime_error();
    *out_tier = GPUBRIDGE_HARDWARE_TIER_UNKNOWN;

    if (!g_initialized) {
        snprintf(g_runtime_error, sizeof(g_runtime_error),
            "gpubridge runtime not initialized");
        return 1;
    }

    if (g_selection.selected_kind == GPUBRIDGE_BACKEND_OPENCL) {
        char device_name[128];
        gpuBridgeOpenclGetDeviceName(device_name, sizeof(device_name));
        *out_tier = gpuBridgeClassifyHardwareTierFromDeviceName(device_name);
    } else if (g_selection.selected_kind == GPUBRIDGE_BACKEND_VULKAN) {
        char device_name[128];
        gpuBridgeVulkanGetDeviceName(device_name, sizeof(device_name));
        *out_tier = gpuBridgeClassifyHardwareTierFromDeviceName(device_name);
    } else {
        /* CPU backend -> always CPU_ONLY: deterministic, no heuristic
         * needed (spec4.md section 9). */
        *out_tier = GPUBRIDGE_HARDWARE_TIER_CPU_ONLY;
    }
    return 0;
}
