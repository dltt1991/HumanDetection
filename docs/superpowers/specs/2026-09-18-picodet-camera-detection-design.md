# PicoDet Camera Human Detection Design

## Goal

Build a C++17 application that previews a USB reversing camera in real time,
detects people, and draws boxes on the original image. The current acceptance
scope is macOS execution with OpenCV DNN. Intel Linux, ARM Linux CPU inference,
and RK3568-class NPU inference are deferred future work.

The first release performs detection only. Tracking, distance estimation,
reversing-zone alarms, and vehicle-control integration are outside its scope.

## Model

Use PaddleDetection's PicoDet-S 320x320 LCNet model, exported to fixed-shape
ONNX without post-processing or NMS. Runtime filtering keeps only the COCO
`person` class.

The 320x320 input is fixed. This keeps output tensor layouts stable, lowers
integration risk, and is sufficient for the initial low-speed reversing use
case. Model files are downloaded locally and are not committed to Git.

The current application uses one model artifact:

- `picodet_s_320_person.onnx`: FP32 model for OpenCV DNN on macOS.

Future non-macOS backends should derive their artifacts from the same exported
network so preprocessing and output contracts remain aligned. RKNN model
conversion and calibration requirements will be specified when that backend
enters scope.

## Architecture

OpenCV owns camera capture, image display, drawing, and the current inference
path. No RKNN SDK or cross-compilation dependency is required.

The application consists of these focused units:

- `main.cpp`: arguments, camera lifecycle, frame loop, FPS, and exit handling.
- `preprocess.cpp`: PaddleDetection-compatible resize to 320x320,
  normalization, and backend-ready RGB input.
- `opencv_backend.cpp`: load and execute the ONNX model with OpenCV DNN.
- `picodet_postprocess.cpp`: decode PicoDet outputs, keep `person`, apply
  confidence filtering and NMS, and map boxes back to source coordinates.

The small `InferenceBackend` interface and `--backend` argument reserve an
extension point for future implementations. OpenCV is the only implemented and
supported backend. Selecting `rknn` fails immediately with a clear
unavailable-backend message.

## Data Flow

1. Parse arguments and select the `opencv` backend.
2. Load the ONNX model and validate its input/output shapes.
3. Open the selected camera and request 1280x720 capture by default.
4. Resize each frame to 320x320 while recording independent horizontal and
   vertical scales.
5. Run inference through OpenCV DNN.
6. Decode the fixed PicoDet feature outputs, select class `person`, filter by
   confidence, and apply class-independent NMS.
7. Undo horizontal and vertical scaling, clamp boxes to the source frame, and
   draw confidence labels and FPS.
8. Display the frame until `q`, Escape, window closure, camera failure, or
   inference failure.

## Runtime Interface

The executable is named `human_detection` and supports:

```text
--backend opencv|rknn  backend extension point; only opencv is supported
--model PATH           ONNX model path
--camera INDEX         camera index; defaults to 0
--confidence VALUE     person threshold; defaults to 0.40
--nms VALUE            IoU threshold; defaults to 0.50
--width N              requested capture width; defaults to 1280
--height N             requested capture height; defaults to 720
--help                  print usage
```

The model path defaults to the ONNX file under `models/`. `rknn` remains
visible in the CLI as a reserved extension value but is not implemented in the
current scope.

## Model Tooling

The current model tooling downloads the official ONNX model used by OpenCV
DNN:

1. Create the local `models/` directory.
2. Fetch the fixed 320x320 PicoDet-S ONNX artifact.

RKNN conversion, calibration, and cross-build tooling are deferred and are not
documented as working project workflows.

## Error Handling

Configuration fails when OpenCV is unavailable. Startup fails with actionable
messages for missing models, unsupported backend selection, invalid thresholds,
incompatible tensor shapes, or a camera that cannot be opened. Runtime failures
include a disconnected camera, empty frames, or backend inference errors and
return a nonzero exit code.

The README includes macOS camera-permission instructions, camera selection,
threshold tuning, model download, build, test, and run commands.

## Testing

One lightweight assertion-based test suite covers:

- 320x320 resize normalization and inverse coordinate mapping.
- Confidence and `person` class filtering.
- NMS behavior and frame-bound clipping.
- Rejection of unexpected output shapes.
- ONNX loading and expected output tensors when the model is present.
- CLI help output.

Automated verification includes CMake configure/build, unit tests, CLI help,
and ONNX loading/inference when the model is present. Camera permission, live
preview, box alignment, and clean interactive exit are manual macOS checks.
Intel Linux, ARM Linux, RKNN conversion, NPU accuracy and latency, and RK3568
runtime/driver compatibility are outside the current acceptance scope.

## Acceptance Criteria

- macOS displays a responsive live USB-camera preview with correctly aligned
  person boxes and FPS.
- A standard macOS build requires only CMake, OpenCV, and the downloaded ONNX
  model.
- `opencv` is the only supported backend; `InferenceBackend` and `--backend`
  remain available as future extension points.
- Default inference input is fixed at 320x320.
- Missing models, invalid camera indices, unsupported backends, and
  tensor-contract mismatches produce actionable errors.
- `q` and Escape close the application cleanly.

## Deferred Scope

Intel Linux and ARM Linux CPU deployment, RKNN model conversion, RK3568
cross-build support, and NPU runtime integration are deferred. Their eventual
acceptance criteria must include target-specific build evidence and hardware
measurements; the macOS milestone makes no claims about NPU latency, INT8
accuracy, or RKNN driver compatibility.
