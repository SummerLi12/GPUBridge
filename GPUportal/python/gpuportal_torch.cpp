// gpuportal_torch.cpp — the PyTorch frontend (spec_gpuportal.md section
// 9.1), a small pybind11 extension exposing exactly one function:
//
//   import gpuportal
//   c = gpuportal.matmul(a, b)   # a, b: torch.Tensor, float32, CPU, 2-D, contiguous
//
// An ordinary Python function call, not a registered PyTorch dispatch
// backend or autograd Function (non-goal #4) — every validation below
// raises a clear Python exception rather than silently coercing, copying,
// or falling back. Builds a one-node GpuBridgePortalGraph/GpuBridgePortalOp
// and calls the existing gpuBridgePortalSubmit() (gpubridge_portal.h); does
// not touch gpubridge-runtime or GpuBridge-BLAS directly.
#include <torch/extension.h>

#include "gpubridge_portal.h"

// gpubridge_runtime.h/gpubridge_memory.h (gpubridge-runtime, unlike
// GPUportal's own gpubridge_portal.h/gpubridge_blas.h) have no extern "C"
// guard of their own — they were never previously included from a .cpp
// file in this project. Wrapping the include here (rather than editing
// gpubridge-runtime's headers) avoids the exact C++ name-mangling link
// error GPUportal/CLAUDE.md already documents hitting once for
// gpubridge_portal.h/gpubridge_blas.h, without touching gpubridge-runtime's
// own source at all.
extern "C" {
#include "gpubridge_runtime.h"
#include "gpubridge_memory.h"
}

#include <stdexcept>
#include <string>

namespace {

torch::Tensor gpuportal_matmul(torch::Tensor a, torch::Tensor b)
{
    // 1. dtype — TypeError, not RuntimeError, matching section 9.1 item 1's
    //    distinction (a type mismatch, not a value/state problem).
    if (a.scalar_type() != torch::kFloat32 || b.scalar_type() != torch::kFloat32) {
        throw py::type_error("gpuportal.matmul: both tensors must be torch.float32");
    }

    // 2. device — only CPU tensors are accepted this round (section 2's
    //    "host memory only" principle / non-goal #9: no device-resident
    //    tensor bridge, no torch.device("gpubridge")).
    if (!a.device().is_cpu() || !b.device().is_cpu()) {
        throw std::runtime_error(
            "gpuportal.matmul: only CPU tensors are accepted this round "
            "(no device-resident tensor bridge yet)");
    }

    // 3. rank + inner-dimension match.
    if (a.dim() != 2 || b.dim() != 2) {
        throw std::runtime_error("gpuportal.matmul: both tensors must be 2-D");
    }
    if (a.size(1) != b.size(0)) {
        throw std::runtime_error(
            "gpuportal.matmul: dimension mismatch: a.shape[1] (" +
            std::to_string(a.size(1)) + ") != b.shape[0] (" +
            std::to_string(b.size(0)) + ")");
    }

    // 4.a and b has to be contiguous so that we can get a accurate result.a.contigupur reorganized the memory layout without changing the data.
    if (!a.is_contiguous() || !b.is_contiguous()) {
        throw std::runtime_error(
            "gpuportal.matmul: both tensors must be contiguous "
            "(call .contiguous() explicitly)");
    }

    // 5. autograd — not silently detached (non-goal #4).
    if (a.requires_grad() || b.requires_grad()) {
        throw std::runtime_error(
            "gpuportal.matmul: autograd is not supported this round "
            "(requires_grad tensors are rejected, not silently detached)");
    }
    //Read matrix dimensions
    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b.size(1);
    //this is allocate result C: [4, 3]
    torch::Tensor c = torch::empty({ m, n }, torch::kFloat32);

    GpuBridgePortalOp op;
    op.kind = GPUBRIDGE_PORTAL_OP_MATMUL;
    op.m = static_cast<int>(m);//int64_t->int
    op.n = static_cast<int>(n);
    op.k = static_cast<int>(k);
    op.input_a = a.data_ptr<float>();
    op.input_b = b.data_ptr<float>();
    op.output = c.data_ptr<float>();
    
    //Put the operation into a graph
    GpuBridgePortalGraph graph;
    graph.ops = &op;  //address of op
    graph.op_count = 1;

    //This passes the graph to Layer 3, the Portal layer.
    GpuBridgePortalStatus status = gpuBridgePortalSubmit(&graph);
    //assumes gpuBridgePortalSubmit() completes the operation and copies the result back before returning. 
    //If it were asynchronous, the local op and graph objects could go out of scope too early.
    if (status != GPUBRIDGE_PORTAL_OK) {
        throw std::runtime_error(
            std::string("gpuportal.matmul: ") + gpuBridgePortalStatusString(status));
    }

    return c;
}

// Not part of spec_gpuportal.md's own scope — a small, additive accessor
// added on top of gpubridge-runtime's existing gpuBridgeGetSelectedDeviceName/
// gpuBridgeGetDeviceMemoryLimit/gpuBridgeGetHardwareTier (Milestone 3 / Phase 4,
// gpubridge_runtime.h), which until now were only ever printed as a side
// effect of GPUBRIDGE_PROFILE=1's gpuBridgeShutdown()-time summary — a summary
// this PyTorch path never triggers, since gpuBridgeBlasDestroy()/
// gpuBridgeShutdown() are never called during a normal Python session (the
// process-global session stays open for the interpreter's lifetime). This
// gives direct, on-demand access to "which GPU is gpuportal actually
// running on" without depending on that shutdown-time ordering.


//describing which GPU backend gpuportal.matmul() actually ran on
py::dict gpuportal_device_info()
{
    char name_buf[256] = "";
    //0 means success, nonzero means failure (no gpubridge-runtime session is active yet)
    if (gpuBridgeGetSelectedDeviceName(name_buf, sizeof(name_buf)) != 0) {
        throw std::runtime_error(
            "gpuportal.device_info: no gpubridge-runtime session is active yet "
            "(call gpuportal.matmul() at least once first, so a backend is "
            "selected)");
    }

    size_t mem_limit_bytes = 0;
    gpuBridgeGetDeviceMemoryLimit(&mem_limit_bytes);

    GpuBridgeHardwareTier tier = GPUBRIDGE_HARDWARE_TIER_UNKNOWN;
    gpuBridgeGetHardwareTier(&tier);

    py::dict info;
    info["backend"] = std::string(gpuBridgeBackendKindToString(gpuBridgeGetSelectedBackend()));
    info["device_name"] = std::string(name_buf);
    info["memory_limit_bytes"] = mem_limit_bytes;
    info["hardware_tier"] = std::string(gpuBridgeHardwareTierToString(tier));
    return info;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul", &gpuportal_matmul,
        "GPUportal matmul: CPU torch.Tensor in/out, computed on the GPU via "
        "gpubridge-runtime's OpenCL backend (spec_gpuportal.md section 9.1)",
        py::arg("a"), py::arg("b"));
    m.def("device_info", &gpuportal_device_info,
        "Returns a dict (backend, device_name, memory_limit_bytes, "
        "hardware_tier) describing the GPU gpuportal.matmul() is actually "
        "running on. Call gpuportal.matmul() at least once first.");
}
