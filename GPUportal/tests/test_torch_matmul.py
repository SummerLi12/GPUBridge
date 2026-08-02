"""test_torch_matmul.py — PyTorch-facing correctness/rejection test for
gpuportal.matmul() (spec_gpuportal.md section 12 item 3).

Run manually via pytest, NOT wired into ctest — a Python test has no place
in a `ctest` C test suite (same reasoning gpubridge-runtime never mixes its
own C tests with a Python runner). Requires the gpuportal_torch extension to
already be built (`python GPUportal/python/setup.py build_ext --inplace`,
which itself requires the C libraries built with GPUBRIDGE_BUILD_PORTAL=ON
first — see GPUBridge/CLAUDE.md's "Build & run" section).

    cd GPUportal
    PYTHONPATH=python:$PYTHONPATH pytest tests/test_torch_matmul.py -v -s

Random float32 CPU tensors of a few shapes, gpuportal.matmul(a, b) compared
to torch.matmul(a, b) via torch.allclose; one case each for the five
rejection conditions in section 9.1, asserting the specific exception
type, not just "it raised something." The last test also records the
section 12 item 6 / section 14 real-hardware latency measurement
(gpuportal.matmul end-to-end vs. plain CPU torch.matmul), printed rather
than asserted on (a latency number is not a pass/fail condition).
"""
import time

import pytest
import torch

import gpuportal


SHAPES = [(4, 4, 4), (64, 128, 32), (256, 256, 256)]


@pytest.mark.parametrize("m,k,n", SHAPES)
def test_matmul_matches_torch(m, k, n):
    a = torch.rand(m, k, dtype=torch.float32)
    b = torch.rand(k, n, dtype=torch.float32)

    c = gpuportal.matmul(a, b)
    expected = torch.matmul(a, b)

    assert c.shape == (m, n)
    assert c.dtype == torch.float32
    assert c.device.type == "cpu"
    assert torch.allclose(c, expected, rtol=1e-3, atol=1e-3)


def test_rejects_non_float32_dtype():
    a = torch.rand(4, 4, dtype=torch.float64)
    b = torch.rand(4, 4, dtype=torch.float64)
    with pytest.raises(TypeError):
        gpuportal.matmul(a, b)


def test_rejects_non_cpu_device():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine to construct a non-CPU tensor")
    a = torch.rand(4, 4, dtype=torch.float32, device="cuda")
    b = torch.rand(4, 4, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        gpuportal.matmul(a, b)


def test_rejects_dimension_mismatch():
    a = torch.rand(4, 5, dtype=torch.float32)
    b = torch.rand(6, 4, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        gpuportal.matmul(a, b)


def test_rejects_non_contiguous():
    a = torch.rand(8, 8, dtype=torch.float32).t()  # transpose view: non-contiguous
    b = torch.rand(8, 8, dtype=torch.float32)
    assert not a.is_contiguous()
    with pytest.raises(RuntimeError):
        gpuportal.matmul(a, b)


def test_rejects_requires_grad():
    a = torch.rand(4, 4, dtype=torch.float32, requires_grad=True)
    b = torch.rand(4, 4, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        gpuportal.matmul(a, b)


def test_measured_latency():
    """Section 12 item 6 / section 14: an actual before/after number, not an
    unmeasured claim (gpubridge_performance_spec.md section 8's standing
    rule, referenced by this spec's own acceptance criteria)."""
    m, k, n = 512, 512, 512
    a = torch.rand(m, k, dtype=torch.float32)
    b = torch.rand(k, n, dtype=torch.float32)

    # Warm up both paths (first call pays one-time OpenCL context / kernel
    # build cost inside gpuBridgeBlasCreate, same reasoning
    # gpubridge-runtime's own test_matmul_benchmark.c warms up before timing).
    gpuportal.matmul(a, b)
    torch.matmul(a, b)

    iters = 10

    t0 = time.perf_counter()
    for _ in range(iters):
        gpuportal.matmul(a, b)
    gpuportal_ms = (time.perf_counter() - t0) / iters * 1000.0

    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    torch_cpu_ms = (time.perf_counter() - t0) / iters * 1000.0

    print(
        f"\n[measured] gpuportal.matmul {m}x{k}x{n}: "
        f"{gpuportal_ms:.3f} ms/call (end-to-end, incl. host-copy round trip) "
        f"vs. torch.matmul (CPU): {torch_cpu_ms:.3f} ms/call "
        f"(ratio: {gpuportal_ms / torch_cpu_ms:.2f}x)"
    )
