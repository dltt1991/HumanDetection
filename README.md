# Human Detection

`human_detection` is a C++17 macOS camera application that runs PicoDet person
detection through OpenCV DNN. It displays the selected camera, draws person
bounding boxes and confidence scores, and reports a rolling frame rate.

The current supported inference backend is `opencv`. `InferenceBackend` and the
`--backend` option are extension points for future CPU and NPU backends; RKNN is
not implemented in this project scope.

## macOS prerequisites

Install CMake and OpenCV with Homebrew:

```sh
brew install cmake opencv
```

The application needs a camera visible to macOS. A built-in camera, USB webcam,
or USB capture device can be used.

## Download the model

From the repository root, download the PicoDet-S 320x320 ONNX model:

```sh
./scripts/download_model.sh
```

The script writes `models/picodet_s_320_person.onnx`. Model files are excluded
from Git.

## Build and test

```sh
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The model smoke test exits successfully without exercising inference when the
model has not been downloaded. Download it first to verify ONNX loading and the
eight expected PicoDet output tensors.

## Camera permission

The first camera launch may prompt for permission. Allow camera access for the
terminal application used to start `human_detection`. To review or change the
setting, open **System Settings > Privacy & Security > Camera**, enable that
terminal application, and restart it before trying again.

Permission prompts and live preview behavior require an interactive macOS
session and must be verified manually.

## Run

Run with camera index `0`:

```sh
./build/human_detection --camera 0
```

Tune bounding-box smoothing with:

```sh
./build/human_detection --camera 0 --box-smoothing 0.35
```

Lower values produce steadier but slower boxes, while higher values follow
movement faster. A value of `1.0` disables coordinate smoothing. The tracker
never buffers video frames.

Press `q` or Escape, or close the preview window, to exit.

If the wrong camera opens, try another zero-based index:

```sh
./build/human_detection --camera 1
```

Use `--width` and `--height` to request a capture size. Camera hardware may
choose the nearest supported size:

```sh
./build/human_detection --camera 1 --width 1920 --height 1080
```

Override the model path and detection thresholds as needed:

```sh
./build/human_detection \
  --model /path/to/picodet_s_320_person.onnx \
  --confidence 0.50 \
  --nms 0.45
```

`--confidence` controls the minimum person score and defaults to `0.60`;
increasing it reduces weak detections. `--nms` controls overlap suppression;
both values must be between `0` and `1`. Run `./build/human_detection --help`
for the complete CLI.

## Backend scope

`opencv` is the only backend currently available:

```sh
./build/human_detection --backend opencv --camera 0
```

The abstract `InferenceBackend` interface and `--backend` selector reserve a
stable integration point for later backends. Selecting `--backend rknn`
currently exits with an explicit unsupported-backend error. Intel Linux, ARM
Linux CPU deployment, RKNN conversion, cross-compilation, and RK3568 NPU
runtime support are deferred future work; no working RKNN build or conversion
commands are provided yet.
