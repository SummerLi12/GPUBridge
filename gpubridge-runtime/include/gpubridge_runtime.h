/*
 * gpubridge_runtime.h — the public GPUBridge Runtime API (spec1.md section 8).
 *
 * This is the only header a *user* of the runtime (e.g. tests/test_vecadd.c,
 * or eventually sac2c-generated host code) needs to include. It hides the
 * existence of backends (gpubridge_backend.h) and the selection policy
 * (gpubridge_selector.c) entirely: a caller just does gpuBridgeInit(), then gpuBridgeMalloc /
 * gpuBridgeMemcpy / gpuBridgeLaunchKernel / gpuBridgeDeviceSynchronize / gpuBridgeFree / gpuBridgeShutdown,
 * exactly as it would with a CUDA-runtime-style API, without ever knowing
 * whether CPU or OpenCL ended up doing the work.
 *
 * Convention used by every function below: return 0 on success, non-zero
 * (currently always 1) on failure. On failure, call gpuBridgeGetLastErrorString()
 * immediately afterward (before calling anything else into the runtime) to
 * get a human-readable reason.
 */
#ifndef GPUBRIDGE_RUNTIME_H
#define GPUBRIDGE_RUNTIME_H

#include <stddef.h>
#include "gpubridge_ir.h"
#include "gpubridge_memory.h"

/* Mirrors the GPUBRIDGE_BACKEND environment variable values (case-insensitive):
 * "auto", "cpu", "opencl", "sycl". GPUBRIDGE_BACKEND_AUTO is the default when the
 * env var is unset. See gpubridge_selector.c for exactly how each value resolves
 * to a backend. */
typedef enum {
    GPUBRIDGE_BACKEND_AUTO,
    GPUBRIDGE_BACKEND_CPU,
    GPUBRIDGE_BACKEND_OPENCL,
    GPUBRIDGE_BACKEND_SYCL,
    GPUBRIDGE_BACKEND_VULKAN   /* new, Phase 4.5 (spec4.5.md section 5) */
} GpuBridgeBackendKind;

/* Direction for gpuBridgeMemcpy. HOST_TO_DEVICE and DEVICE_TO_HOST are the two
 * directions actually used by tests/test_vecadd.c and implemented by both
 * backends. HOST_TO_HOST is implemented as a plain memcpy with no backend
 * involvement. DEVICE_TO_DEVICE is accepted by the enum (per spec1.md
 * section 8) but deliberately unimplemented in this milestone — see the
 * comment in gpubridge_runtime.c's gpuBridgeMemcpy for why. */
typedef enum {
    GPUBRIDGE_MEMCPY_HOST_TO_DEVICE,
    GPUBRIDGE_MEMCPY_DEVICE_TO_HOST,
    GPUBRIDGE_MEMCPY_DEVICE_TO_DEVICE,
    GPUBRIDGE_MEMCPY_HOST_TO_HOST
} GpuBridgeMemcpyKind;

/* Must be called before any other runtime function. Reads GPUBRIDGE_BACKEND from
 * the environment, probes backend availability, selects one per the policy
 * in gpubridge_selector.c, and calls that backend's init(). Returns 0 on success.
 * Calling gpuBridgeInit() again after a successful call is a cheap no-op (returns
 * 0 immediately) rather than re-initializing. */
int gpuBridgeInit(void);

/* Shuts down the currently selected backend (if any) and resets all runtime
 * state, so a later gpuBridgeInit() call starts fresh. Always returns 0. */
int gpuBridgeShutdown(void);

/* Writes the number of usable devices for the currently selected backend
 * into *count (always 1 in this milestone: one CPU "device" or one OpenCL
 * GPU). Fails if gpuBridgeInit() has not succeeded yet. */
int gpuBridgeGetDeviceCount(int* count);

/* Explicitly (re-)selects a backend at runtime, overriding whatever
 * GPUBRIDGE_BACKEND/gpuBridgeInit() chose. Shuts down the previous backend first if one
 * was active. Most callers only need gpuBridgeInit(); this exists for completeness
 * with spec1.md's API and isn't exercised by tests/test_vecadd.c. */
int gpuBridgeSetBackend(GpuBridgeBackendKind backend);

/* Allocates `bytes` of device-resident memory via the active backend and
 * writes the resulting handle into *ptr. The handle is opaque to the
 * caller — never dereference it directly, only ever pass it back into
 * gpuBridgeMemcpy/gpuBridgeLaunchKernel/gpuBridgeFree. */
int gpuBridgeMalloc(void** ptr, size_t bytes);

/* Frees memory previously returned by gpuBridgeMalloc. */
int gpuBridgeFree(void* ptr);

/* Copies `bytes` bytes between `src` and `dst` in the direction given by
 * `kind`. For HOST_TO_DEVICE/DEVICE_TO_HOST, one of src/dst must be a
 * device handle from gpuBridgeMalloc and the other a plain host pointer. Blocking:
 * completes before returning, so a caller never needs an extra sync just to
 * safely read the destination afterward. */
int gpuBridgeMemcpy(
    void* dst,
    const void* src,
    size_t bytes,
    GpuBridgeMemcpyKind kind
);

/* Executes the kernel described by `kernel` (see gpubridge_ir.h) using the device
 * pointers in `args` (in the same order as kernel->args), and prints
 * GPUBRIDGE_VERBOSE=1 diagnostics as a side effect. In this milestone the only
 * supported kernel is vector_add_f32 with exactly 3 args (a, b, c); any
 * other op_kind or arg_count fails immediately with a clear error. */
int gpuBridgeLaunchKernel(
    const GpuBridgeKernelIR* kernel,
    void** args,
    size_t arg_count
);

/* Blocks until all work previously launched on the active backend has
 * completed. Safe to call even when nothing is outstanding. */
int gpuBridgeDeviceSynchronize(void);

/* Returns a human-readable description of the most recent failure from any
 * runtime or backend call. Never returns NULL — returns a fixed
 * "gpubridge runtime not initialized" string if nothing has been attempted yet.
 * The returned pointer is only valid until the next runtime call. */
const char* gpuBridgeGetLastErrorString(void);

/*
 * Not part of spec1.md section 8's literal API. Added because section 14's
 * expected test output prints the *resolved* backend (e.g. what
 * GPUBRIDGE_BACKEND=auto actually picked), and the spec gives no getter for it.
 *
 * gpuBridgeGetSelectedBackend() returns whichever backend gpuBridgeInit()/gpuBridgeSetBackend()
 * actually resolved to (e.g. GPUBRIDGE_BACKEND_OPENCL even if the user asked for
 * GPUBRIDGE_BACKEND_AUTO). gpuBridgeBackendKindToString() renders any GpuBridgeBackendKind value
 * (requested or resolved) as its lowercase name ("auto"/"cpu"/"opencl"/
 * "sycl"), used by both diagnostics and the test program so there is a
 * single canonical name table instead of two ad hoc ones.
 */
GpuBridgeBackendKind gpuBridgeGetSelectedBackend(void);
const char* gpuBridgeBackendKindToString(GpuBridgeBackendKind kind);

/*
 * --- Milestone 2 additions (spec2.md sections 8.2, 10) ---
 */

/* Allocates device-resident storage for `desc` via the active backend's
 * existing malloc_device() (no new backend allocator — spec2.md section
 * 8.2) and sets desc->type/rank/shape/device_resident accordingly. `shape`
 * follows GpuBridgeTensorDesc's convention: shape[1] is ignored when rank == 1.
 * Returns 0 on success; on failure desc is left with device_resident=false. */
int gpuBridgeTensorAlloc(GpuBridgeTensorDesc* desc, GpuBridgeScalarType type, int rank,
    const size_t shape[2]);

/* Frees desc->device_ptr via the active backend's free_device() and clears
 * the descriptor (device_ptr=NULL, device_resident=false). Safe to call on
 * an already-freed/never-allocated descriptor (no-op in that case). */
int gpuBridgeTensorFree(GpuBridgeTensorDesc* desc);

/* Executes the tensor-op kernel described by `kernel` (see gpubridge_ir.h) using
 * the device pointers in `args` (same order as kernel->args), dispatching
 * by kernel->op_kind to the matching launch_* vtable slot. This is a second
 * entry point (rather than overloading gpuBridgeLaunchKernel) specifically so that
 * gpuBridgeLaunchKernel's existing signature and vector_add_f32 behavior stay
 * completely unchanged (spec2.md section 10). Prints one GPUBRIDGE_VERBOSE=1
 * diagnostic block per call, same as gpuBridgeLaunchKernel. */
int gpuBridgeLaunchTensorKernel(
    const GpuBridgeTensorKernelIR* kernel,
    void** args,
    size_t arg_count
);

/*
 * --- Milestone 2.5 additions (spec2.5.md section 8.2) ---
 *
 * GpuBridgeTensorMirror (type declared in gpubridge_ir.h) operations: lazy,
 * staleness-tracked host/device transfer for one tensor. All six return 0 on
 * success, 1 on failure (same convention as every other function in this
 * header) with gpuBridgeGetLastErrorString() describing why.
 */

/* Binds `host_ptr` (caller-owned, never freed by this API) to `desc`, an
 * already-allocated device tensor (desc->device_resident must be true —
 * e.g. freshly returned by gpuBridgeTensorAlloc). Initial state: host_valid
 * = true, device_valid = false (the common case: host data exists, no
 * device copy yet). Does not itself allocate or copy anything. */
int gpuBridgeTensorMirrorCreate(GpuBridgeTensorMirror* mirror, void* host_ptr,
    const GpuBridgeTensorDesc* desc, size_t bytes);

/* Frees the underlying tensor via gpuBridgeTensorFree (returning it to the
 * pool, per spec2.5.md section 7) and clears the mirror. Does not free
 * mirror->host_ptr — the caller owns it. */
int gpuBridgeTensorMirrorDestroy(GpuBridgeTensorMirror* mirror);

/* No-op (no gpuBridgeMemcpy call) if mirror->device_valid is already true.
 * Otherwise issues one HOST_TO_DEVICE gpuBridgeMemcpy and sets device_valid =
 * true. Call this before launching a kernel that reads the tensor. */
int gpuBridgeTensorEnsureDeviceCurrent(GpuBridgeTensorMirror* mirror);

/* Symmetric: no-op if mirror->host_valid is already true, otherwise issues
 * one DEVICE_TO_HOST gpuBridgeMemcpy and sets host_valid = true. Call this
 * before reading mirror->host_ptr after device-side writes. */
int gpuBridgeTensorEnsureHostCurrent(GpuBridgeTensorMirror* mirror);

/* Caller wrote directly into mirror->host_ptr (bypassing this API) — marks
 * host_valid = true, device_valid = false, so the next
 * gpuBridgeTensorEnsureDeviceCurrent() call actually re-transfers. */
int gpuBridgeTensorMarkHostModified(GpuBridgeTensorMirror* mirror);

/* Symmetric: caller's kernel wrote into the device tensor directly — marks
 * device_valid = true, host_valid = false. */
int gpuBridgeTensorMarkDeviceModified(GpuBridgeTensorMirror* mirror);

/*
 * --- Milestone 1.5 additions (spec1.5.md section 10) ---
 */

/* Writes the currently selected backend's capabilities (spec1.5.md section
 * 7) into *out_caps. Fails if gpuBridgeInit() has not succeeded yet. */
int gpuBridgeGetBackendCaps(GpuBridgeBackendCaps* out_caps);

/* Writes a snapshot of the current session's cumulative profiling counters
 * (spec1.5.md section 9) into *out_stats. Must be called before
 * gpuBridgeShutdown(), which prints (if GPUBRIDGE_PROFILE=1) and then clears
 * these counters. Fails if gpuBridgeInit() has not succeeded yet. */
int gpuBridgeGetProfileStats(GpuBridgeProfileStats* out_stats);

/*
 * --- Milestone 3 addition (spec3.md section 8.5) ---
 *
 * Writes a null-terminated, best-effort device name into buf (max buf_len
 * bytes) for whichever backend gpuBridgeInit()/gpuBridgeSetBackend() most
 * recently selected. CPU always writes "CPU (host)". OpenCL queries
 * clGetDeviceInfo(CL_DEVICE_NAME) on its already-open device handle; on any
 * query failure (never fatal) writes "unknown" and returns false. Fails
 * (returns 1, buf untouched) only if gpuBridgeInit() has not succeeded yet.
 */
int gpuBridgeGetSelectedDeviceName(char* buf, size_t buf_len);

/*
 * --- Phase 4 additions (spec4.md sections 5, 6, 9) ---
 */

/* Optional model-load phase marker (gpubridge_performance_spec.md Area 9).
 * GPUBridge itself never loads model weights — this brackets whatever
 * caller-side code (e.g. SaC-generated inference code built on top of
 * gpuBridgeTensorAlloc/gpuBridgeMemcpy) does to load them, so "cold start" can
 * be reported distinctly from "steady state" via
 * GpuBridgeProfileStats.model_load_time_ms. Not nestable: a second
 * gpuBridgeBeginModelLoad() before a matching End simply restarts the bracket
 * from that point (single-threaded, no stack) — documented, not an error.
 * gpuBridgeEndModelLoad() without a prior Begin is a tolerant no-op, matching
 * this project's existing "double free"/"free of never-allocated" tolerance
 * conventions. Both fail only if gpuBridgeInit() has not succeeded yet. */
int gpuBridgeBeginModelLoad(void);
int gpuBridgeEndModelLoad(void);

/* Allocates a device-resident, long-lived tensor tagged
 * GPUBRIDGE_TENSOR_USAGE_KV_CACHE (gpubridge_ir.h) that unconditionally bypasses
 * the tensor pool on both this call and the matching gpuBridgeTensorFree() —
 * a KV-cache tensor is meant to stay device-resident, not cycle through the
 * pool's exact-size-match free list alongside ordinary short-lived tensors.
 * Same type/rank restriction as gpuBridgeTensorAlloc() (only GPUBRIDGE_TYPE_F32,
 * rank 1/2 — no quantized KV cache yet). Increments
 * GpuBridgeProfileStats.kv_cache_allocation_count/kv_cache_bytes on success. */
int gpuBridgeTensorAllocKvCache(GpuBridgeTensorDesc* desc, GpuBridgeScalarType type, int rank,
    const size_t shape[2]);

/* Writes the current backend's device memory limit into *out_bytes. CPU:
 * total system RAM via sysconf(_SC_PHYS_PAGES)*sysconf(_SC_PAGE_SIZE)
 * (POSIX-portable, no root needed — the CPU "device" has no separate memory
 * pool). OpenCL: clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE). Writes 0 and
 * returns 1 only if gpuBridgeInit() has not succeeded yet; a query failure
 * inside an already-initialized backend writes 0 but still returns 0,
 * matching gpuBridgeGetSelectedDeviceName's "never fatal" convention. */
int gpuBridgeGetDeviceMemoryLimit(size_t* out_bytes);

/* Writes the current backend's best-effort hardware tier into *out_tier.
 * CPU backend -> always GPUBRIDGE_HARDWARE_TIER_CPU_ONLY (deterministic, no
 * heuristic needed). OpenCL backend ->
 * gpuBridgeClassifyHardwareTierFromDeviceName() (gpubridge_memory.h) applied to
 * gpuBridgeGetSelectedDeviceName()'s result — see that function's own comment
 * for its documented false-negative risk. Fails (returns 1) only if
 * gpuBridgeInit() has not succeeded yet. */
int gpuBridgeGetHardwareTier(GpuBridgeHardwareTier* out_tier);

#endif /* GPUBRIDGE_RUNTIME_H */
