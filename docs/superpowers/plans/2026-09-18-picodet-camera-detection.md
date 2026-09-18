# PicoDet Camera Human Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Current goal:** Build and accept a C++17 USB-camera person detector using PicoDet-S 320x320 and OpenCV DNN on macOS.

**Current architecture:** OpenCV handles capture, display, CPU inference, and drawing. OpenCV DNN produces the eight raw PicoDet tensors; shared code decodes distributional boxes at strides 8/16/32/64, filters COCO class 0, runs NMS, and maps coordinates to the source frame.

**Current tech stack:** C++17, CMake 3.16+, OpenCV 4 (`core`, `imgproc`, `highgui`, `videoio`, `dnn`), PaddleDetection PicoDet-S 320 ONNX, and POSIX shell on macOS.

> **Current scope (2026-09-18):** Only macOS with the OpenCV backend is implemented and subject to this plan's acceptance requirements. Intel Linux, ARM Linux CPU, RKNN conversion, cross-build, and RK3568 NPU support are deferred. The explicitly labeled historical section below preserves the original ideas for reference; it contains no current implementation steps, deliverables, or acceptance requirements.

## Global Constraints

- Input is fixed at `1x3x320x320` for the OpenCV backend.
- Official preprocessing is direct resize with `INTER_CUBIC`, RGB conversion, scale `1/255`, mean `[0.485, 0.456, 0.406]`, and std `[0.229, 0.224, 0.225]`.
- The model is `models/picodet_s_320_person.onnx`.
- Only COCO class `0` (`person`) is emitted.
- Only the OpenCV backend is currently supported; RKNN is not an implemented build option or acceptance target.
- Model binaries, build outputs, and calibration images are excluded from Git.
- Tracking, distance estimation, reversing-zone alarms, and vehicle-control integration are excluded.

---

### Task 1: Project Foundation And Preprocessing

**Files:**
- Create: `CMakeLists.txt`
- Create: `.gitignore`
- Create: `src/detection.hpp`
- Create: `src/preprocess.hpp`
- Create: `src/preprocess.cpp`
- Create: `tests/preprocess_test.cpp`

**Interfaces:**
- Produces: `PreprocessResult preprocess(const cv::Mat&)`, containing a `1x3x320x320` `CV_32F` blob plus `scale_x` and `scale_y`.
- Produces: `cv::Rect2f restore_box(const cv::Rect2f&, const PreprocessResult&, cv::Size)`.

- [ ] **Step 1: Add the minimal CMake project and failing preprocessing test**

```cmake
cmake_minimum_required(VERSION 3.16)
project(human_detection LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui videoio dnn)

add_library(human_detection_core src/preprocess.cpp)
target_include_directories(human_detection_core PUBLIC src)
target_link_libraries(human_detection_core PUBLIC ${OpenCV_LIBS})

enable_testing()
add_executable(preprocess_test tests/preprocess_test.cpp)
target_link_libraries(preprocess_test PRIVATE human_detection_core)
add_test(NAME preprocess_test COMMAND preprocess_test)
```

```cpp
// tests/preprocess_test.cpp
#include "preprocess.hpp"
#include <cassert>
#include <cmath>

int main() {
  cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(0, 0, 255));
  const auto result = preprocess(image);
  assert(result.blob.dims == 4);
  assert(result.blob.size[2] == 320 && result.blob.size[3] == 320);
  assert(std::abs(result.scale_x - 0.25f) < 1e-6f);
  assert(std::abs(result.scale_y - (320.0f / 720.0f)) < 1e-6f);
  const auto box = restore_box({32, 40, 160, 200}, result, image.size());
  assert(std::abs(box.x - 128.0f) < 1e-4f);
  assert(std::abs(box.y - 90.0f) < 1e-4f);
  assert(std::abs(box.width - 640.0f) < 1e-4f);
  assert(std::abs(box.height - 450.0f) < 1e-4f);
}
```

- [ ] **Step 2: Configure and run the test to verify failure**

Run: `cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure`

Expected: build fails because `src/preprocess.cpp` and `preprocess.hpp` do not exist.

- [ ] **Step 3: Implement official preprocessing and coordinate restoration**

```cpp
// src/preprocess.hpp
#pragma once
#include <opencv2/core.hpp>

struct PreprocessResult {
  cv::Mat blob;
  float scale_x;
  float scale_y;
};

PreprocessResult preprocess(const cv::Mat& bgr);
cv::Rect2f restore_box(const cv::Rect2f& model_box,
                       const PreprocessResult& prep,
                       cv::Size source_size);
```

```cpp
// src/preprocess.cpp
#include "preprocess.hpp"
#include <algorithm>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

PreprocessResult preprocess(const cv::Mat& bgr) {
  if (bgr.empty()) throw std::invalid_argument("input frame is empty");
  cv::Mat resized, rgb, float_image;
  cv::resize(bgr, resized, {320, 320}, 0, 0, cv::INTER_CUBIC);
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(float_image, CV_32F, 1.0 / 255.0);
  const cv::Scalar mean(0.485, 0.456, 0.406);
  const cv::Scalar inv_std(1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225);
  cv::subtract(float_image, mean, float_image);
  cv::multiply(float_image, inv_std, float_image);
  return {cv::dnn::blobFromImage(float_image),
          320.0f / static_cast<float>(bgr.cols),
          320.0f / static_cast<float>(bgr.rows)};
}

cv::Rect2f restore_box(const cv::Rect2f& box, const PreprocessResult& prep,
                       cv::Size source) {
  const float x1 = std::clamp(box.x / prep.scale_x, 0.0f, float(source.width));
  const float y1 = std::clamp(box.y / prep.scale_y, 0.0f, float(source.height));
  const float x2 = std::clamp((box.x + box.width) / prep.scale_x, 0.0f, float(source.width));
  const float y2 = std::clamp((box.y + box.height) / prep.scale_y, 0.0f, float(source.height));
  return {x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1)};
}
```

- [ ] **Step 4: Run the tests and commit**

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: `100% tests passed`.

```bash
git add CMakeLists.txt .gitignore src tests
git commit -m "feat: add PicoDet preprocessing"
```

### Task 2: Shared PicoDet Decoder And NMS

**Files:**
- Create: `src/picodet_postprocess.hpp`
- Create: `src/picodet_postprocess.cpp`
- Create: `tests/postprocess_test.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: eight `CV_32F` tensors ordered as score/box pairs for strides `8,16,32,64` with shapes `[1,N,80]` and `[1,N,32]`.
- Produces: `std::vector<Detection> decode_picodet(const std::vector<cv::Mat>&, float, float)`.

- [ ] **Step 1: Define the detection contract and a failing synthetic decode test**

```cpp
// src/detection.hpp
#pragma once
#include <opencv2/core.hpp>

struct Detection {
  cv::Rect2f box;
  float score;
};
```

Create synthetic score tensors filled with zero, set class `0` at stride 8 cell `(10,12)` to `0.9`, and set each of the four 8-bin distance distributions to peak at bin `2`. Assert one detection, score `0.9`, and a box centered at `((12.5)*8, (10.5)*8)` with approximately 16-pixel distances.

- [ ] **Step 2: Build to verify the decoder test fails**

Run: `cmake --build build --target postprocess_test`

Expected: compile failure because `decode_picodet` is undefined.

- [ ] **Step 3: Implement strict shape validation and distribution decoding**

```cpp
// src/picodet_postprocess.hpp
#pragma once
#include "detection.hpp"
#include <opencv2/core.hpp>
#include <vector>

std::vector<Detection> decode_picodet(const std::vector<cv::Mat>& outputs,
                                      float confidence, float nms_iou);
```

For each stride `{8,16,32,64}`, require respectively `{1600,400,100,25}` rows. Read person score at class index `0`. For every side, apply numerically stable softmax to its eight logits, compute `sum(probability * bin) * stride`, build XYXY around `(column + 0.5, row + 0.5) * stride`, and use `cv::dnn::NMSBoxes` to retain indices. Reject any tensor whose element count does not match the declared shape.

- [ ] **Step 4: Add tests for filtering, overlap suppression, and malformed shapes**

Add assertions that a score below threshold yields no detection, two identical candidates retain only the higher score, and replacing a `[1,1600,80]` score tensor with `[1,10,80]` throws `std::runtime_error`.

- [ ] **Step 5: Run and commit**

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: preprocessing and postprocessing tests pass.

```bash
git add CMakeLists.txt src tests
git commit -m "feat: decode PicoDet person detections"
```

### Task 3: Official Model Download And OpenCV Backend

**Files:**
- Create: `src/inference_backend.hpp`
- Create: `src/opencv_backend.hpp`
- Create: `src/opencv_backend.cpp`
- Create: `scripts/download_model.sh`
- Create: `tests/model_smoke_test.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `InferenceBackend::infer(const cv::Mat&) -> std::vector<cv::Mat>`.
- Produces: `OpenCvBackend(std::filesystem::path)` implementing the interface.

- [ ] **Step 1: Add a failing backend smoke test**

The test skips with exit code `0` when `models/picodet_s_320_person.onnx` is absent. When present, it loads the model, runs a zero blob, and asserts exactly eight outputs with element shapes `1600x80`, `1600x32`, `400x80`, `400x32`, `100x80`, `100x32`, `25x80`, and `25x32`.

- [ ] **Step 2: Verify the test fails before the backend exists**

Run: `cmake --build build --target model_smoke_test`

Expected: compile failure because `OpenCvBackend` is undefined.

- [ ] **Step 3: Implement the backend and stable output ordering**

```cpp
// src/inference_backend.hpp
#pragma once
#include <opencv2/core.hpp>
#include <vector>

class InferenceBackend {
 public:
  virtual ~InferenceBackend() = default;
  virtual std::vector<cv::Mat> infer(const cv::Mat& blob) = 0;
};
```

`OpenCvBackend` loads with `cv::dnn::readNetFromONNX`, sets `DNN_BACKEND_OPENCV` and `DNN_TARGET_CPU`, and requests these names in this order: `transpose_0.tmp_0` through `transpose_7.tmp_0`. Constructor errors include the model path; inference errors include the unexpected output count or shape.

- [ ] **Step 4: Add the official download script**

```sh
#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$root/models"
curl -fL --retry 3 \
  https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_s_320_coco_lcnet.onnx \
  -o "$root/models/picodet_s_320_person.onnx"
```

- [ ] **Step 5: Download, run the real model smoke test, and commit**

Run: `chmod +x scripts/download_model.sh && scripts/download_model.sh && cmake --build build && ctest --test-dir build --output-on-failure`

Expected: all tests pass and the smoke test reports eight valid outputs.

```bash
git add CMakeLists.txt src scripts tests .gitignore
git commit -m "feat: add OpenCV PicoDet backend"
```

### Task 4: Camera Preview Application

**Files:**
- Create: `src/main.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `preprocess`, `OpenCvBackend::infer`, `decode_picodet`, and `restore_box`.
- Produces: executable `human_detection` and CLI defined by the design specification.

- [ ] **Step 1: Add CLI parsing tests through a `--help` process check**

Register a CTest named `help_test` that runs `human_detection --help` and requires output containing `--backend`, `--camera`, `--confidence`, and `--nms`.

- [ ] **Step 2: Implement argument validation and camera loop**

Use defaults `backend=opencv`, `camera=0`, `confidence=0.40`, `nms=0.50`, `width=1280`, `height=720`, and `model=models/picodet_s_320_person.onnx`. Reject thresholds outside `[0,1]`, negative camera indices, and backend values other than `opencv` or `rknn`; accept `rknn` as a reserved interface value, then report that it is unavailable in this build.

Open `cv::VideoCapture(camera, cv::CAP_ANY)`, request width/height, preprocess each frame, infer, decode, restore boxes, draw green 2-pixel rectangles and `person %.2f`, calculate moving FPS over 30 frames, and exit on `q`, Escape, or `cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1`.

- [ ] **Step 3: Build and run non-hardware checks**

Run: `cmake --build build && ./build/human_detection --help && ./build/human_detection --camera -1`

Expected: help exits `0`; invalid camera argument exits nonzero with `camera index must be non-negative`.

- [ ] **Step 4: Run the interactive Mac camera check**

Run: `./build/human_detection --camera 0`

Expected: live preview, person boxes when a person is visible, FPS overlay, and clean exit with `q` or Escape. If the environment cannot access the desktop or camera, record that limitation in the final verification instead of claiming this check passed.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt src/main.cpp
git commit -m "feat: add realtime camera detection"
```

### Historical/Deferred Task 5: Optional RK3568 RKNN Backend And Conversion

> **Not active:** Everything in this Task 5 section is retained from the original plan as historical design detail only. None of its files, interfaces, steps, commands, expected results, or commit instructions are current deliverables or acceptance requirements. RKNN and RK3568 work must be separately respecified before implementation.

**Files:**
- Create: `src/rknn_backend.hpp`
- Create: `src/rknn_backend.cpp`
- Create: `scripts/convert_to_rknn.py`
- Create: `cmake/toolchains/aarch64-linux-gnu.cmake`
- Modify: `CMakeLists.txt`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: RKNN Runtime headers/library supplied through `RKNN_SDK_ROOT`.
- Produces: `RknnBackend` implementing `InferenceBackend` and eight dequantized `CV_32F` outputs in the same order as OpenCV.

- [ ] **Step 1: Add the disabled-by-default CMake option and configuration check**

```cmake
option(ENABLE_RKNN "Build RK3568 RKNN backend" OFF)
if(ENABLE_RKNN)
  if(NOT DEFINED RKNN_SDK_ROOT)
    message(FATAL_ERROR "ENABLE_RKNN requires RKNN_SDK_ROOT")
  endif()
endif()
```

Run: `cmake -S . -B build-rknn -DENABLE_RKNN=ON`

Expected: configuration fails with `ENABLE_RKNN requires RKNN_SDK_ROOT`.

- [ ] **Step 2: Implement RKNN lifecycle and tensor conversion**

`RknnBackend` reads the model bytes, calls `rknn_init`, queries input/output attributes, requires one `320x320x3` input and eight outputs, submits RGB `uint8` NHWC input with `rknn_inputs_set`, calls `rknn_run`, requests float outputs with `want_float=1`, copies each output into owned `cv::Mat`, calls `rknn_outputs_release`, and destroys the context in its destructor. Move construction is allowed; copying is deleted.

- [ ] **Step 3: Add deterministic INT8 conversion script**

```python
#!/usr/bin/env python3
import argparse
from rknn.api import RKNN

p = argparse.ArgumentParser()
p.add_argument("--onnx", default="models/picodet_s_320_person.onnx")
p.add_argument("--dataset", required=True)
p.add_argument("--output", default="models/picodet_s_320_person_int8.rknn")
a = p.parse_args()

rknn = RKNN(verbose=True)
assert rknn.config(target_platform="rk3568", mean_values=[[123.675, 116.28, 103.53]],
                   std_values=[[58.395, 57.12, 57.375]]) == 0
assert rknn.load_onnx(model=a.onnx) == 0
assert rknn.build(do_quantization=True, dataset=a.dataset) == 0
assert rknn.export_rknn(a.output) == 0
rknn.release()
```

The dataset argument is a text file containing one calibration image path per line, as required by RKNN-Toolkit2.

- [ ] **Step 4: Verify CPU builds remain independent and cross-build configuration is valid**

Run: `cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure`

Expected: all CPU tests pass without RKNN installed.

With an SDK and cross compiler available, run:

```bash
cmake -S . -B build-rknn \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu.cmake \
  -DENABLE_RKNN=ON -DRKNN_SDK_ROOT=/opt/rknpu2/runtime/Linux/librknn_api
cmake --build build-rknn
```

Expected: `human_detection` links against `librknnrt.so`.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt .gitignore src/rknn_backend.* scripts/convert_to_rknn.py cmake
git commit -m "feat: add optional RK3568 backend"
```

### Task 6: macOS/OpenCV Documentation And Final Verification

**Files:**
- Create: `README.md`
- Modify: `docs/superpowers/specs/2026-09-18-picodet-camera-detection-design.md`

**Interfaces:**
- Documents: macOS build/run, OpenCV backend usage, camera permissions, and known interactive verification boundaries.

- [ ] **Step 1: Write README commands that match the implemented CLI**

Include `brew install cmake opencv`, model download, CMake build, CTest, `./build/human_detection --camera 0`, model/threshold overrides, and macOS Privacy & Security camera permission. Do not present Linux, ARM, RKNN conversion, cross-build, or RK3568 board instructions as implemented or currently supported.

- [ ] **Step 2: Run formatting-free static checks and all available tests**

Run:

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/human_detection --help
rg -n 'TBD|TODO|FIXME' README.md src tests scripts CMakeLists.txt
git status --short
```

Expected: configure/build succeed, all tests pass, help succeeds, the placeholder scan prints nothing, and Git shows only intended documentation changes.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md docs/superpowers/specs/2026-09-18-picodet-camera-detection-design.md
git commit -m "docs: add build and deployment guide"
```

- [ ] **Step 4: Record verification boundaries**

The completion report must distinguish automated macOS/OpenCV checks from the unverified interactive camera check. Any mention of Intel Linux, ARM Linux, RKNN conversion, cross-builds, RK3568 boards, NPU latency, INT8 accuracy, or driver compatibility must identify that material as historical or deferred and outside current acceptance.
