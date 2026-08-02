/*
 * gpubridge_internal.h — private declarations shared only among src/*.c files.
 *
 * Nothing in this header is part of the public API (gpubridge_runtime.h /
 * gpubridge_backend.h / gpubridge_ir.h). It is the glue between the four pieces that
 * together implement the public gpuBridgeInit/gpuBridgeMalloc/.../gpuBridgeLaunchKernel
 * functions defined in gpubridge_runtime.c:
 *
 *   gpubridge_env.c         parses GPUBRIDGE_BACKEND / GPUBRIDGE_VERBOSE from the environment
 *   gpubridge_selector.c    turns a requested GpuBridgeBackendKind into an actual
 *                    GpuBridgeBackend*, applying the availability/fallback policy
 *   gpubridge_diagnostics.c prints the GPUBRIDGE_VERBOSE=1 block
 *   gpubridge_runtime.c     the public API; owns all persistent runtime state and
 *                    calls into the three files above
 *
 * Not included by any file outside src/ — tests and backends never see this
 * header, only gpubridge_runtime.h/gpubridge_backend.h/gpubridge_ir.h.
 */
#ifndef GPUBRIDGE_INTERNAL_H
#define GPUBRIDGE_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>
#include "gpubridge_runtime.h"
#include "gpubridge_backend.h"

/* gpubridge_env.c */

/* Parses the GPUBRIDGE_BACKEND environment variable (case-insensitively) into a
 * GpuBridgeBackendKind. Unset -> GPUBRIDGE_BACKEND_AUTO with *out_valid = true. Any value
 * other than auto/cpu/opencl/sycl sets *out_valid = false (the return value
 * in that case is unspecified/GPUBRIDGE_BACKEND_AUTO and must not be used) so the
 * caller (gpuBridgeInit in gpubridge_runtime.c) can report a clear "unrecognized value"
 * error instead of silently guessing. */
GpuBridgeBackendKind gpuBridgeEnvGetRequestedBackend(bool* out_valid);

/* True only if GPUBRIDGE_VERBOSE is set to exactly the string "1" (matching
 * spec1.md's literal GPUBRIDGE_VERBOSE=1 example — not a general truthy parse of
 * "true"/"yes"/etc). */
bool gpuBridgeEnvVerbose(void);

/*
 * --- Milestone 1.5 addition (spec1.5.md section 9.1) ---
 * True only if GPUBRIDGE_PROFILE is set to exactly the string "1" (same
 * strict-match convention as gpuBridgeEnvVerbose above).
 */
bool gpuBridgeEnvProfile(void);

/*
 * --- Milestone 2.5 addition (spec2.5.md section 4 item 5) ---
 * True only if GPUBRIDGE_POOL_DISABLE is set to exactly the string "1" (same
 * strict-match convention as gpuBridgeEnvVerbose/gpuBridgeEnvProfile above).
 * When true, gpuBridgeTensorAlloc/gpuBridgeTensorFree bypass the pool
 * (gpubridge_pool.c) entirely and call malloc_device/free_device directly,
 * for debugging.
 */
bool gpuBridgeEnvPoolDisable(void);

/*
 * --- Milestone 3 addition (spec3.md section 7) ---
 * True only if GPUBRIDGE_VENDOR_GEMM_DISABLE is set to exactly the string "1"
 * (same strict-match convention as the flags above). When true,
 * opencl_launch_matmul_f32/cpu_launch_matmul_f32 skip their vendor-library
 * fast path (CLBlast/OpenBLAS) even if it was successfully loaded, and
 * always use the hand-written kernel/naive loop instead.
 */
bool gpuBridgeEnvVendorGemmDisable(void);

/* gpubridge_selector.c */

/*
 * The full result of one backend-selection decision: which backends were
 * found to be available at probe time, and which one was ultimately chosen
 * (`selected`/`selected_kind`). gpubridge_runtime.c stores one of these globally
 * after a successful gpuBridgeInit()/gpuBridgeSetBackend(), and gpubridge_diagnostics.c reads it
 * to print the GPUBRIDGE_VERBOSE block.
 */
typedef struct {
    bool cpu_available;
    bool opencl_available;
    bool sycl_available;
    bool vulkan_available;   /* new, Phase 4.5 */
    const GpuBridgeBackend* selected;
    GpuBridgeBackendKind selected_kind;
} GpuBridgeSelection;

/*
 * Applies spec1.md section 10's selection policy for `requested`:
 *   GPUBRIDGE_BACKEND_CPU    -> always selects CPU (cannot fail)
 *   GPUBRIDGE_BACKEND_OPENCL -> selects OpenCL if available, else fails with a
 *                        message written into err_buf
 *   GPUBRIDGE_BACKEND_SYCL   -> always fails (SYCL is not implemented in this
 *                        milestone); message written into err_buf
 *   GPUBRIDGE_BACKEND_AUTO   -> selects OpenCL if available, else CPU (CPU is
 *                        always available, so AUTO never fails)
 *
 * On success, *out is fully populated (including the availability flags for
 * every backend, needed for diagnostics regardless of which one was picked)
 * and the function returns true. On failure, *out's availability flags are
 * still populated but out->selected is NULL and a human-readable reason has
 * been snprintf'd into err_buf (a caller-owned buffer of err_buf_size
 * bytes); the function returns false.
 */
bool gpuBridgeSelectBackend(GpuBridgeBackendKind requested, GpuBridgeSelection* out,
                      char* err_buf, size_t err_buf_size);

/* gpubridge_diagnostics.c */

/*
 * Prints the GPUBRIDGE_VERBOSE=1 diagnostic block from spec1.md section 11,
 * exactly once per gpuBridgeLaunchKernel() call, if and only if gpuBridgeEnvVerbose() is
 * true. `requested` should be the *originally requested* backend kind (e.g.
 * GPUBRIDGE_BACKEND_AUTO), not the resolved one, so that the printed
 * "requested backend:" line matches what the user actually set GPUBRIDGE_BACKEND
 * to; `selection` supplies the availability flags and the resolved backend
 * for the "selected backend:" line. `operation`/`elements` describe the
 * kernel currently being launched (kernel_name / global_size from the
 * GpuBridgeKernelIR). No-op if verbose mode is off.
 */
void gpuBridgePrintDiagnostics(GpuBridgeBackendKind requested,
                         const GpuBridgeSelection* selection,
                         const char* operation, size_t elements);

/*
 * --- Milestone 2 addition (spec2.md section 12) ---
 *
 * Same preamble as gpuBridgePrintDiagnostics (requested/CPU/OpenCL/SYCL/selected
 * backend lines) and the same "operation:" line, but replaces the
 * "elements: N" line with a "shapes: A[..] B[..] -> C[..]" line built from
 * `args`/`arg_count` (a GpuBridgeTensorKernelIR's args), matching spec2.md section
 * 12's worked example exactly. Called once per gpuBridgeLaunchTensorKernel() call,
 * analogous to gpuBridgePrintDiagnostics's role in gpuBridgeLaunchKernel(). No-op unless
 * GPUBRIDGE_VERBOSE=1, same as gpuBridgePrintDiagnostics.
 */
void gpuBridgePrintTensorDiagnostics(GpuBridgeBackendKind requested,
                               const GpuBridgeSelection* selection,
                               const char* operation,
                               const GpuBridgeTensorArg* args, size_t arg_count);

/*
 * --- 2026-07-29 addition (not spec-numbered; small additive diagnostic,
 * same as tensor_alloc_impl()'s verbose trace line) ---
 *
 * Dumps a GpuBridgeTensorKernelIR field-by-field, exactly as the struct is
 * laid out in gpubridge_ir.h: every GpuBridgeTensorArg in kernel->args (name,
 * type, rank, shape, is_input/is_output), then the GpuBridgeTensorKernelIR
 * itself (kernel_name, op_kind, args, arg_count, global_size). This is a
 * strictly more verbose companion to gpuBridgePrintTensorDiagnostics's
 * compact "shapes: A[..] -> C[..]" line above — that line is derived from
 * the same args; this prints the args themselves, unabbreviated, for anyone
 * who wants to see the actual IR shape rather than a rendering of it. Called
 * once per gpuBridgeLaunchTensorKernel() call, right alongside
 * gpuBridgePrintTensorDiagnostics. No-op unless GPUBRIDGE_VERBOSE=1.
 */
void gpuBridgePrintTensorIR(const GpuBridgeTensorKernelIR* kernel);

/*
 * --- Milestone 1.5 addition (spec1.5.md section 11.2) ---
 *
 * Prints the GPUBRIDGE_PROFILE=1 "GPUBridge profile:" block from a snapshot of
 * profile counters (spec1.5.md section 9.2). No-op unless gpuBridgeEnvProfile()
 * is true (same self-gating pattern as gpuBridgePrintDiagnostics).
 * total_gpu_path_time_ms is computed here at print time as the sum of the
 * three *_time_ms fields — GpuBridgeProfileStats does not store it separately.
 * Called once by gpuBridgeShutdown(), not per-launch.
 *
 * --- gpubridge_performance_spec.md section 5 gap-fix (2026-07-14) ---
 * total_runtime_ms is wall-clock elapsed since gpuBridgeInit()/gpuBridgeSetBackend(),
 * measured by the caller (gpubridge_runtime.c owns g_session_start, a timestamp,
 * not a cumulative counter, so it does not belong in GpuBridgeProfileStats).
 * cpu_orchestration_time_ms is derived here, the same "computed at print
 * time" treatment as total_gpu_path_time_ms: total_runtime_ms minus GPU path
 * time minus synchronization time, floored at 0 to absorb clock jitter.
 *
 * --- Phase 4 addition (spec4.md section 11) ---
 * Its third parameter-list growth (Milestone 1.5, the 2026-07-14 gap-fix,
 * now this) — an already-accepted pattern for this specific internal
 * function, not a new precedent (spec4.md section 2). selected_backend/
 * device_name/device_memory_limit_bytes/hardware_tier are queried once by
 * gpuBridgeShutdown() (gated on gpuBridgeEnvProfile(), so zero extra cost when
 * GPUBRIDGE_PROFILE isn't set) rather than per-launch — they are session-static
 * once a backend is selected (spec4.md section 9.2). avg_kernel_latency_ms/
 * throughput_ops_per_sec are derived here from stats already collected, the
 * same "derived, not stored" treatment as cpu_orchestration_time_ms.
 */
void gpuBridgePrintProfile(const GpuBridgeProfileStats* stats, double total_runtime_ms,
    GpuBridgeBackendKind selected_backend, const char* device_name,
    size_t device_memory_limit_bytes, GpuBridgeHardwareTier hardware_tier);

/* gpubridge_pool.c */

/*
 * --- Milestone 2.5 addition (spec2.5.md section 7) ---
 *
 * A pooled device-memory allocator for tensor allocations only
 * (gpuBridgeTensorAlloc/gpuBridgeTensorFree in gpubridge_runtime.c) — not part
 * of the public API surface, and not consulted by gpuBridgeMalloc/gpuBridgeFree
 * (the vector_add_f32 path), which are completely unaffected by this
 * milestone. Internal to this runtime, so its declarations live here
 * alongside gpubridge_env.c/gpubridge_selector.c/gpubridge_diagnostics.c's, rather
 * than in a separate gpubridge_pool.h — every other purely-internal (not
 * publicly-typed) module in this project follows that same one-shared-header
 * pattern.
 *
 * Single-threaded only (spec2.5.md section 5 non-goal #9): no locking is
 * used, matching the fact that nothing anywhere in gpubridge-runtime/ spawns a
 * thread or calls into the runtime concurrently.
 */

/*
 * Looks up a free block of exactly `bytes` size. Returns NULL (a pool miss)
 * if none is available. On a hit, removes the block from the pool and
 * returns it. Either way, *out_reuse_count is set to how many times a block
 * of this exact byte size has now been served from the pool (0 if this is
 * the first time this size has ever been served, i.e. every miss until the
 * first hit for that size) — the real reuse_count signal
 * gpuBridgeChooseMemoryStrategy() uses (spec2.5.md section 7.2), replacing
 * spec1.5.md's illustrative constant.
 */
void* gpuBridgePoolAcquire(size_t bytes, int* out_reuse_count);

/* Returns `ptr` (of size `bytes`) to the pool instead of freeing it. */
void gpuBridgePoolRelease(void* ptr, size_t bytes);

/*
 * Frees every block currently held by the pool via `backend`'s
 * free_device(), then clears all pool state, including each size class's
 * reuse-count history. Must be called (with the backend that is about to be
 * torn down) before gpuBridgeShutdown()'s own shutdown() call and before
 * gpuBridgeSetBackend() tears down the previously-selected backend — a
 * device_ptr from one backend is meaningless to another (spec2.5.md section
 * 7.3). Safe to call with an empty pool (no-op).
 */
void gpuBridgePoolDrain(const GpuBridgeBackend* backend);

/* backends/opencl/gpubridge_backend_opencl.c */

/*
 * --- Milestone 3 addition (spec3.md section 8.5) ---
 *
 * Best-effort OpenCL device identity, defined in gpubridge_backend_opencl.c
 * (next to the g.device handle it reads) rather than in the CLBlast module
 * (gpubridge_opencl_clblast.c) — this is a property of the OpenCL backend
 * itself, independent of whether CLBlast happens to be loaded. Declared
 * here (rather than in a CLBlast-specific header) so gpubridge_runtime.c's
 * public gpuBridgeGetSelectedDeviceName() wrapper can call it. Writes a
 * null-terminated name into buf (clGetDeviceInfo(CL_DEVICE_NAME)) and
 * returns true, or writes "unknown" and returns false if the query itself
 * fails or OpenCL support wasn't compiled in (never fatal).
 */
bool gpuBridgeOpenclGetDeviceName(char* buf, size_t buf_len);

/*
 * --- Phase 4 addition (spec4.md section 9.1) ---
 *
 * Best-effort OpenCL device memory limit via
 * clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE), defined in
 * gpubridge_backend_opencl.c next to gpuBridgeOpenclGetDeviceName() above (same
 * "no vtable slot" precedent — spec4.md section 9.1 confirms this rather than
 * challenges it). Writes 0 and returns false if the query fails or OpenCL
 * support wasn't compiled in (never fatal).
 */
bool gpuBridgeOpenclGetDeviceMemoryLimit(size_t* out_bytes);

/* backends/vulkan/gpubridge_backend_vulkan.c */

/*
 * --- Phase 4.5 addition ---
 *
 * Not part of spec4.5.md's own scope, but needed for the same reason as
 * gpuBridgeOpenclGetDeviceName/gpuBridgeOpenclGetDeviceMemoryLimit above:
 * without these, gpuBridgeGetSelectedDeviceName()/
 * gpuBridgeGetDeviceMemoryLimit()/gpuBridgeGetHardwareTier()
 * (gpubridge_runtime.c, Milestone 3 / Phase 4) would silently misreport the
 * Vulkan backend as the CPU backend — their pre-existing
 * "if (selected_kind == OPENCL) ... else <CPU path>" shape predates this
 * backend's existence. Same "never fatal" convention: writes "unknown"/0
 * and returns false if the query fails or Vulkan support wasn't compiled
 * in.
 */
bool gpuBridgeVulkanGetDeviceName(char* buf, size_t buf_len);
bool gpuBridgeVulkanGetDeviceMemoryLimit(size_t* out_bytes);

#endif /* GPUBRIDGE_INTERNAL_H */
