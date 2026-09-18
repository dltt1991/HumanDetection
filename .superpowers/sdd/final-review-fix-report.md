# Final Review Fix Report

Date: 2026-09-18

## Status

All requested final-review findings are fixed for the macOS OpenCV scope. No
RKNN implementation was added.

## Changes

- `OpenCvBackend` now runs a zero-filled `1x3x320x320` inference during
  construction. The existing strict count and shape checks therefore validate
  all eight official PicoDet outputs before `run()` opens `VideoCapture`.
- `scripts/download_model.sh` now downloads to a same-directory `mktemp` file,
  removes it on exit or signals, verifies exactly `4781217` bytes, verifies
  SHA-256 `f9c671308fe7c20e618c7e2b5f88cf7460443cfcb1ef4b628d00d95f0f79758e`
  using `sha256sum` on Linux or `shasum -a 256` on macOS, and atomically
  renames the verified file into place.
- Preprocessing tests now verify the complete `1x3x320x320` shape and exact
  normalized RGB plane values from a known BGR red input.
- Coordinate restoration now has an out-of-bounds clipping regression test.
- All tests use explicit failures and nonzero returns instead of `assert`, so
  checks remain effective in Release builds with `NDEBUG`.

## TDD Evidence

### RED

After adding the constructor-probe regression and before changing production
code:

```text
cmake --build build --target preprocess_test postprocess_test model_smoke_test
ctest --test-dir build -R 'preprocess_test|postprocess_test|model_smoke_test' --output-on-failure

preprocess_test: passed
postprocess_test: passed
model_smoke_test: failed
runtime_error: backend construction must probe the output contract
67% tests passed, 1 test failed out of 3
```

The focused test creates a temporary copy of the real model with an incompatible
requested output name. The prior constructor loaded it without inference, so
the test failed for the intended missing startup validation.

### GREEN

After adding the constructor probe:

```text
cmake --build build --target preprocess_test postprocess_test model_smoke_test
ctest --test-dir build -R 'preprocess_test|postprocess_test|model_smoke_test' --output-on-failure

100% tests passed, 0 tests failed out of 3
```

## Download Verification

Real download:

```text
scripts/download_model.sh
stat -f '%z' models/picodet_s_320_person.onnx
shasum -a 256 models/picodet_s_320_person.onnx

verified model: .../models/picodet_s_320_person.onnx
4781217
f9c671308fe7c20e618c7e2b5f88cf7460443cfcb1ef4b628d00d95f0f79758e
```

An isolated fake `curl` wrote a 7-byte partial artifact. The script exited 1
with the expected size mismatch, created no final model, and left no temporary
file. A second fake download copied the exact-size model and corrupted one byte.
The script exited 1 with the expected SHA-256 mismatch, created no final model,
and again left no temporary file.

## Final Verification

Fresh Release configuration and build with AppleClang 21.0.0 and OpenCV 4.14.0:

```text
cmake -S . -B build-final-review -DCMAKE_BUILD_TYPE=Release
cmake --build build-final-review --parallel
ctest --test-dir build-final-review --output-on-failure

preprocess_test: passed
postprocess_test: passed
model_smoke_test: passed
help_test: passed
100% tests passed, 0 tests failed out of 4
```

Additional checks:

```text
./build-final-review/human_detection --help  # exit 0; all expected options shown
rg -n '#include <cassert>|\bassert\s*\(' tests  # no matches
sh -n scripts/download_model.sh  # exit 0
git diff --check  # exit 0
```

## Files Changed

- `src/opencv_backend.cpp`
- `scripts/download_model.sh`
- `tests/model_smoke_test.cpp`
- `tests/preprocess_test.cpp`
- `tests/postprocess_test.cpp`
- `.superpowers/sdd/final-review-fix-report.md`

## Concerns And Deferred Work

- Backend construction now performs one CPU inference, so startup takes the
  cost of one frame. This is intentional to reject incompatible models before
  camera access.
- The model smoke test requires the real model to exercise its contract; it
  retains the existing skip behavior when the ignored model file is absent.
- Per review direction, no binary known-image fixture was added. Real model
  loading/inference plus synthetic decoder and preprocessing tests are the
  current acceptance coverage.
- No interactive camera run was part of this fix verification; camera hardware
  and GUI behavior are unchanged.
