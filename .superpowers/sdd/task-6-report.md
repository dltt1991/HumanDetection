# Task 6 Report: macOS Documentation and Scope Update

## Status

Complete. Documentation now treats macOS with the OpenCV backend as the
current acceptance scope. Intel Linux, ARM Linux CPU, RKNN conversion,
cross-build support, and RK3568 NPU execution are explicitly deferred.

No production code was changed.

Commit: `fb267e6 docs: add macOS build and deployment guide`

## Documentation

- Added `README.md` with macOS prerequisites, model download, configure/build,
  CTest, camera permission, camera selection, thresholds, model override, and
  run instructions.
- Documented `InferenceBackend` and `--backend` as extension points while
  identifying `opencv` as the only supported backend.
- Updated the design specification to make macOS/OpenCV the current acceptance
  scope and to defer Intel/ARM/RKNN implementation and validation.
- Added a concise scope-update note to the implementation plan while retaining
  the original Task 5 details as planning history.

## Automated Verification

Run from the `picodet-camera` worktree on macOS:

```text
cmake -S . -B build                         PASS
cmake --build build --parallel             PASS
ctest --test-dir build --output-on-failure PASS (4/4)
./build/human_detection --help             PASS (exit 0)
rg -n 'TBD|TODO|FIXME' ...                 PASS (no matches)
git diff --check                           PASS
```

The downloaded 4.6 MB `models/picodet_s_320_person.onnx` file was present, so
`model_smoke_test` exercised ONNX model loading and inference instead of its
missing-model skip path.

CTest results:

- `preprocess_test`: passed
- `postprocess_test`: passed
- `model_smoke_test`: passed
- `help_test`: passed

## Manual Verification Boundary

Camera permission prompts, camera selection, live preview, bounding-box
alignment, FPS display, and clean interactive exit require a user-controlled
macOS desktop and camera. They were documented but not claimed as automated
verification.

Intel Linux and ARM Linux builds, RKNN conversion, RK3568 board execution, NPU
latency, INT8 accuracy, and RKNN runtime/driver compatibility were not tested
and remain deferred future work.

## Concerns

No documentation or automated-test blocker was found. Final acceptance still
requires the documented interactive macOS camera check on the intended camera.

## Documentation Consistency Review Fix

The follow-up review found that the implementation plan's active goal, global
constraints, Task 5, and Task 6 still described Intel/ARM/RKNN work as current
deliverables despite the scope note. The plan now defines macOS/OpenCV as the
only implemented and accepted scope, limits the active CLI requirement to the
`opencv` backend, labels all original Task 5 material as historical/deferred
and non-acceptance content, and removes Linux/RKNN deliverables from active
Task 6 instructions. The historical implementation detail remains available
for future respecification.

Exact verification run after this review fix:

```text
rg -n -i 'current|active|acceptance|implemented|supported|historical|deferred|rknn|rk3568|intel|arm|linux' docs/superpowers/plans/2026-09-18-picodet-camera-detection.md .superpowers/sdd/task-6-report.md
PASS (all Intel/ARM/Linux/RKNN current-scope references identify that work as historical, deferred, unsupported, or outside acceptance)

rg -n 'TBD|TODO|FIXME' README.md src tests scripts CMakeLists.txt
PASS (no matches; rg exit 1)

git diff --check
PASS (exit 0, no output)

ctest --test-dir build --output-on-failure
PASS (4/4 tests; 0 failures)
```

No production code or README content was changed by this review fix. The
interactive macOS camera check remains the only current manual acceptance
boundary; deferred Intel/ARM/RKNN work was not tested.
