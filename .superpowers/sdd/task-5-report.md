# Task 5 Report: Optional RK3568 RKNN Backend And Conversion

## Status

Implemented the optional RKNN backend, RK3568 conversion script, cross-toolchain file, and CLI selection. RKNN remains disabled by default and CPU builds do not require RKNN headers or libraries.

The shared backend contract now explicitly accepts backend-ready input. OpenCV receives normalized FP32 NCHW from `preprocess`; RKNN receives resized RGB uint8 NHWC from `preprocess_rknn`, leaving the conversion-configured mean/std normalization inside the RKNN model.

## RED Evidence

- Before the CMake option existed, `cmake -S . -B build-rknn -DENABLE_RKNN=ON` succeeded and warned that `ENABLE_RKNN` was unused. This violated the required missing-SDK failure behavior.
- After adding the RKNN preprocessing assertions but before its implementation, building `preprocess_test` failed with `use of undeclared identifier 'preprocess_rknn'`.

## GREEN Evidence

- `preprocess_test` builds and passes with assertions for a contiguous 320x320 `CV_8UC3` RGB tensor and unchanged coordinate scales.
- `cmake -S . -B build-rknn -DENABLE_RKNN=ON` now fails during configuration with `ENABLE_RKNN requires RKNN_SDK_ROOT`.
- `python3 -m py_compile scripts/convert_to_rknn.py` succeeds.
- `human_detection --backend rknn` in the CPU build exits with status 1 and prints `rknn backend is not available in this build`.
- Both RKNN translation units pass C++17 syntax-only compilation against Rockchip's current official `rknn_api.h`.

## CPU Regression Results

Commands run:

```text
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Result: 4/4 tests passed (`preprocess_test`, `postprocess_test`, `model_smoke_test`, and `help_test`).

## Unavailable SDK Boundary

`RKNN_SDK_ROOT`, `librknnrt.so`, and `aarch64-linux-gnu-g++` are unavailable in this environment. Therefore the RKNN target was not linked, the conversion script was not executed with RKNN-Toolkit2, and no RK3568 runtime, driver compatibility, output accuracy, latency, or camera behavior is claimed. Syntax compilation against the official header is not a substitute for SDK or board verification.

## Files Changed

- `.gitignore`
- `CMakeLists.txt`
- `cmake/toolchains/aarch64-linux-gnu.cmake`
- `scripts/convert_to_rknn.py`
- `src/inference_backend.hpp`
- `src/main.cpp`
- `src/preprocess.cpp`
- `src/preprocess.hpp`
- `src/rknn_backend.cpp`
- `src/rknn_backend.hpp`
- `tests/preprocess_test.cpp`
- `.superpowers/sdd/task-5-report.md`

## Self-Review

- Confirmed RKNN sources, include paths, compile definition, and runtime library are only added when `ENABLE_RKNN=ON`.
- Confirmed model bytes and RKNN output buffers are copied or released with exception-safe cleanup, and context ownership is non-copyable and movable.
- Confirmed all eight outputs become owned 3D `CV_32F` matrices in the OpenCV backend's output order and expected element counts.
- Confirmed the CLI selects `RknnBackend` only in enabled builds and defaults to the INT8 RKNN model when the default ONNX path was not overridden.
- Confirmed conversion normalization matches the existing OpenCV preprocessing numerically and is not applied twice.

## Concerns

- A real RKNN SDK and RK3568 board are still required to verify library discovery, link compatibility, tensor metadata for the converted model, inference correctness, and performance.
- Output ordering relies on RKNN preserving the ONNX export order, as expected by the conversion pipeline; target-board smoke testing should compare all eight output shapes and representative detections against OpenCV.
