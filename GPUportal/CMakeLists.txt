# CMakeLists.txt — GPUportal sub-project (spec_gpuportal.md).
#
# Only ever add_subdirectory'd from ../gpubridge-runtime/CMakeLists.txt, and
# only when GPUBRIDGE_BUILD_PORTAL=ON (default OFF) — see that file's own
# comment at the add_subdirectory call. Never configured standalone: it
# depends on the gpubridge_runtime target already existing in the including
# project (spec_gpuportal.md section 11 item 1 — "no source duplication, no
# re-probing for OpenCL/CLBlast").
#
# Produces two static libraries:
#   gpuportal_blas   GpuBridge-BLAS (section 6): gpuBridgeBlasCreate/Destroy/Sgemm
#   gpuportal        GPUportal dispatch entry point (section 7):
#                     gpuBridgePortalSubmit/Shutdown, links against gpuportal_blas
# and two test executables (section 12, items 1-2). test_torch_matmul.py
# (section 12 item 3) is deliberately NOT wired in here — see python/setup.py
# and CLAUDE.md's "Build & run" section for that separate, manual step.

add_library(gpuportal_blas STATIC src/gpubridge_blas.c)
target_include_directories(gpuportal_blas PUBLIC include)
target_link_libraries(gpuportal_blas PUBLIC gpubridge_runtime)

add_library(gpuportal STATIC src/gpubridge_portal.c)
target_include_directories(gpuportal PUBLIC include)
target_link_libraries(gpuportal PUBLIC gpuportal_blas)

# --- Tests (section 12 items 1-2) -----------------------------------------
# Both require a real OpenCL device (section 2: OpenCL only, no cpu/auto
# fallback anywhere in GPUportal) — skip cleanly (exit 77) rather than fail
# outright on a machine with no OpenCL device, the same SKIP_RETURN_CODE
# convention ../gpubridge-runtime/CMakeLists.txt already uses for
# matmul_opencl_vendor/vecadd_vulkan.
add_executable(gpuportal_blas_sgemm_test tests/test_blas_sgemm.c)
target_link_libraries(gpuportal_blas_sgemm_test PRIVATE gpuportal_blas m)

add_executable(gpuportal_portal_graph_test tests/test_portal_graph.c)
target_link_libraries(gpuportal_portal_graph_test PRIVATE gpuportal m)

add_test(NAME gpuportal_blas_sgemm COMMAND $<TARGET_FILE:gpuportal_blas_sgemm_test>)
add_test(NAME gpuportal_portal_graph COMMAND $<TARGET_FILE:gpuportal_portal_graph_test>)
set_tests_properties(gpuportal_blas_sgemm gpuportal_portal_graph PROPERTIES SKIP_RETURN_CODE 77)
