# Default Confidence 0.60 Design

## Goal

Raise the default person-detection confidence threshold from `0.40` to `0.60`
to reduce weak detections, while preserving the existing `--confidence`
override.

## Changes

- Set `Options::confidence` to `0.60f`.
- Update `--help` to report `default: 0.60`.
- Strengthen the existing help test so a future default-value regression fails.
- Update the README threshold description to state the new default.

The PicoDet model, score comparison, NMS threshold, tracking behavior, camera
pipeline, and backend interfaces remain unchanged. Historical design and plan
documents retain the values that described their original implementation.

## Verification

Build the Release target, run the full CTest suite with the real ONNX model,
verify help output contains the new default, and confirm a clean Git diff before
pushing `main` to `git@github.com:dltt1991/HumanDetection.git`.
