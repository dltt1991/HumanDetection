# PicoDet Camera Human Detection Design

## Goal

Build a C++17 application that previews a USB reversing camera in real time, detects people, and draws boxes on the original image. Development and interactive verification happen on macOS. The same codebase must also support Intel Linux and ARM Linux CPU inference, plus RK3568-class NPU inference.

The first release performs detection only. Tracking, distance estimation, reversing-zone alarms, and vehicle-control integration are outside its scope.

## Model

Use PaddleDetection's PicoDet-S 320x320 LCNet model, exported to fixed-shape ONNX without post-processing or NMS. Runtime filtering keeps only the COCO `person` class.

The 320x320 input is fixed for every backend. This keeps output tensor layouts stable, lowers conversion risk, and is sufficient for the initial low-speed reversing use case. Model files are downloaded or generated locally and are not committed to Git.

Two model artifacts come from the same exported network:

- `picodet_s_320_person.onnx`: FP32 model for OpenCV DNN on macOS, Intel Linux, and ARM Linux CPUs.
- `picodet_s_320_person_int8.rknn`: INT8 model produced with RKNN-Toolkit2 for RK3568-class NPUs.

The RKNN calibration set must contain representative frames from the intended reversing camera, including daylight, low light, people near image edges, partial occlusion, and different person sizes.

## Architecture

OpenCV owns camera capture, image display, drawing, and the portable CPU inference path. RKNN Runtime is an optional build dependency enabled only for RK3568 builds.

The application consists of these focused units:

- `main.cpp`: arguments, camera lifecycle, frame loop, FPS, and exit handling.
- `preprocess.cpp`: aspect-preserving letterbox to 320x320 and backend-ready RGB input.
- `opencv_backend.cpp`: load and execute the ONNX model with OpenCV DNN.
- `rknn_backend.cpp`: conditionally compiled RKNN Runtime adapter for the INT8 model.
- `picodet_postprocess.cpp`: decode PicoDet outputs, keep `person`, apply confidence filtering and NMS, and map boxes back through letterbox coordinates.

A small inference-backend interface is justified because there are two real implementations. Both return the same named output tensors to the shared post-processor. RKNN support is disabled by default so a normal macOS or CPU build does not require the RKNN SDK.

## Data Flow

1. Parse arguments and select `opencv` or `rknn`.
2. Load the matching model and validate its input/output shapes.
3. Open the selected camera and request 1280x720 capture by default.
4. Letterbox each frame to 320x320 while recording scale and padding.
5. Run inference on the selected backend.
6. Decode the fixed PicoDet feature outputs, select class `person`, filter by confidence, and apply class-independent NMS.
7. Undo letterbox scaling, clamp boxes to the source frame, and draw confidence labels and FPS.
8. Display the frame until `q`, Escape, window closure, camera failure, or inference failure.

## Runtime Interface

The executable is named `human_detection` and supports:

```text
--backend opencv|rknn  inference backend; defaults to opencv
--model PATH           ONNX or RKNN model path
--camera INDEX         camera index; defaults to 0
--confidence VALUE     person threshold; defaults to 0.40
--nms VALUE            IoU threshold; defaults to 0.50
--width N              requested capture width; defaults to 1280
--height N             requested capture height; defaults to 720
--help                  print usage
```

The model path defaults to the matching file under `models/`. Selecting `rknn` in a build without RKNN support fails immediately with a clear message.

## Model Tooling

Scripts document and automate the reproducible model path:

1. Fetch the official PaddleDetection PicoDet-S 320 weights.
2. Export a fixed 320x320 Paddle inference model without post-processing and NMS.
3. Convert and simplify it to ONNX, preserving raw output names and shapes.
4. Convert the ONNX model to INT8 RKNN for target `rk3568` using a user-supplied calibration image directory.

PaddleDetection and RKNN-Toolkit2 remain external tool dependencies. The project does not vendor either repository.

## Error Handling

Configuration fails when OpenCV is unavailable. Startup fails with actionable messages for missing models, unsupported backend selection, invalid thresholds, incompatible tensor shapes, or a camera that cannot be opened. Runtime failures include a disconnected camera, empty frames, or backend inference errors and return a nonzero exit code.

The README includes macOS camera-permission instructions and RK3568 requirements for matching RKNN Runtime, NPU driver, and RKNN-Toolkit2 versions.

## Testing

One lightweight assertion-based test executable covers:

- Letterbox scaling and inverse coordinate mapping.
- Confidence and `person` class filtering.
- NMS behavior and frame-bound clipping.
- Rejection of unexpected output shapes.

Verification includes CMake configure/build, unit tests, ONNX inference on a known image, and interactive macOS USB-camera preview. The RKNN conversion script can be syntax-checked locally; NPU accuracy, latency, runtime/driver compatibility, and camera capture must be validated on the RK3568 board.

## Acceptance Criteria

- macOS displays a responsive live USB-camera preview with correctly aligned person boxes and FPS.
- The same source builds for Intel Linux and ARM64 Linux with the OpenCV backend.
- An RK3568 build can select the RKNN backend without changing application or post-processing code.
- Default inference input is fixed at 320x320.
- Missing models, invalid camera indices, unsupported backends, and tensor-contract mismatches produce actionable errors.
- `q` and Escape close the application cleanly.
