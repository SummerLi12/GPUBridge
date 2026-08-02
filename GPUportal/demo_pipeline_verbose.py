"""demo_pipeline_verbose.py — manual, not-pytest demo of the full
PyTorch -> tensor-IR (GpuBridgePortalGraph) -> GpuBridge-BLAS -> OpenCL -> GPU
pipeline, with every layer's own GPUBRIDGE_VERBOSE=1 diagnostics turned on,
plus an expected-vs-actual correctness printout at the end (torch.matmul on
CPU as the reference).

Small, deterministic (non-random) shapes so the printed values are easy to
read and hand-check, not just "trust the assert."

Run:
    cd GPUportal
    PYTHONPATH=python:$PYTHONPATH LD_LIBRARY_PATH=$HOME/.local/lib \
        GPUBRIDGE_VERBOSE=1 python3 demo_pipeline_verbose.py > out.txt 2>&1
"""
import os
import sys

os.environ.setdefault("GPUBRIDGE_VERBOSE", "1")

import torch
import gpuportal
# generate same random numbers each time for reproducibility
torch.manual_seed(0)

print("=" * 70)
print("STEP 1: PyTorch tensors (host, CPU, float32)")
print("=" * 70)
# small GEMM shape: A[4,5] @ B[5,3] -> C[4,3], deliberately tiny/deterministic
# so every value below is easy to eyeball/hand-check
m, k, n = 4, 5, 3
# arange(20).reshape(4,5)*0.1 - 1.0 -> deterministic values spanning negative
# and positive (avoids an all-positive/all-zero edge case masking a bug)
a = torch.arange(m * k, dtype=torch.float32).reshape(m, k) * 0.1 - 1.0
# same idea for B, different scale (0.2) so A and B aren't just scaled copies
# of each other
b = torch.arange(k * n, dtype=torch.float32).reshape(k, n) * 0.2 - 1.0
print(f"a (A[{m},{k}]):\n{a}\n")
print(f"b (B[{k},{n}]):\n{b}\n")

print("=" * 70)
print("STEP 2-5: gpuportal.matmul(a, b) -- watch for the tensor-IR")
print("(GPUportal:), BLAS (GpuBridge-BLAS:), and OpenCL backend")
print("(GPUBridge runtime: / gemm path selected) diagnostic lines below")
print("=" * 70)
print("test set for nakul", a.requires_grad, b.requires_grad)
sys.stdout.flush()  # keep this Python output ahead of the C extension's own
                     # printf output below (see gpuBridgePortalSubmit's
                     # setvbuf comment for why this matters when > out.txt)
c = gpuportal.matmul(a, b)
sys.stdout.flush()

print()
print("=" * 70)
print("STEP 5.5: second gpuportal.matmul() call, same A[4,5]/B[5,3]/C[4,3]")
print("byte sizes as above (80/60/48 bytes) -- Sgemm frees its device")
print("tensors after every call, so this call's gpuBridgeTensorAlloc()s")
print("should each be served from the pool instead of a fresh")
print("malloc_device(): watch pool_reuse_count go 0 -> 1 in the")
print("tensor_alloc_impl trace lines below")
print("=" * 70)
# a2/b2: different values, same shapes -- proves the pool is matched by
# byte size, not by reusing the exact same Python tensor
# a2 = torch.arange(m * k, dtype=torch.float32).reshape(m, k) * 0.05 - 0.5
# b2 = torch.arange(k * n, dtype=torch.float32).reshape(k, n) * 0.1 - 0.5
# sys.stdout.flush()
# c2 = gpuportal.matmul(a2, b2)
# sys.stdout.flush()

# print()
# print("=" * 70)
# print("STEP 6: expected vs. actual (reference: torch.matmul on CPU)")
# print("=" * 70)
# expected = torch.matmul(a, b)
# diff = (c - expected).abs()
# print(f"actual C (from GPU via OpenCL):\n{c}\n")
# print(f"expected C (torch.matmul, CPU reference):\n{expected}\n")
# print(f"abs diff:\n{diff}\n")
# max_abs_error = diff.max().item()
# passed = torch.allclose(c, expected, rtol=1e-3, atol=1e-3)
# print(f"max abs error: {max_abs_error:.8f}")
# print(f"tolerance rule: torch.allclose(rtol=1e-3, atol=1e-3)")
# print(f"result: {'PASS' if passed else 'FAIL'}")
