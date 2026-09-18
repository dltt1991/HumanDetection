# Task 2 Report: Shared PicoDet Decoder And NMS

## Status

Implemented shared PicoDet person decoding, strict output shape validation,
distribution focal loss distance decoding, OpenCV NMS, tests, and CMake wiring.

## RED

Command:

```sh
cmake --build build --target postprocess_test
```

Output (exit 2):

```text
[100%] Linking CXX executable postprocess_test
Undefined symbols for architecture arm64:
  "decode_picodet(std::__1::vector<cv::Mat, std::__1::allocator<cv::Mat>> const&, float, float)"
clang++: error: linker command failed with exit code 1
```

Strict-shape regression command:

```sh
cmake --build build --target postprocess_test && ./build/postprocess_test
```

Output (exit 134 before the rank check was implemented):

```text
Assertion failed: (threw), function test_rejects_malformed_shapes,
file postprocess_test.cpp, line 83.
```

## GREEN

Focused command:

```sh
cmake --build build --target postprocess_test && ./build/postprocess_test
```

Output (exit 0):

```text
[ 60%] Built target human_detection_core
[100%] Built target postprocess_test
```

Final verification commands:

```sh
cmake --build build
ctest --test-dir build --output-on-failure
```

Output (exit 0):

```text
[ 42%] Built target human_detection_core
[ 71%] Built target preprocess_test
[100%] Built target postprocess_test
100% tests passed, 0 tests failed out of 2
Total Test time (real) = 0.53 sec
```

## Files Changed

- `src/picodet_postprocess.hpp`: declares the shared decoder interface.
- `src/picodet_postprocess.cpp`: validates and decodes all four output levels,
  then applies NMS.
- `tests/postprocess_test.cpp`: covers synthetic decoding, threshold filtering,
  overlap suppression, and malformed shapes.
- `CMakeLists.txt`: builds the decoder and registers `postprocess_test`.
- `.superpowers/sdd/task-2-report.md`: records implementation evidence.

## Self-Review

- Confirmed output ordering and fixed shapes for strides 8, 16, 32, and 64.
- Confirmed stable softmax uses per-distribution maximum subtraction.
- Confirmed NMS returns `Detection` values with `cv::Rect2f` boxes.
- Added a same-element-count wrong-rank test so shape validation is strict.
- Confirmed `src/detection.hpp`, preprocessing code, and design/plan documents
  were not modified.
- `git diff --check` reported no whitespace errors.

## Concerns

None within the Task 2 contract. The decoder intentionally assumes the fixed
320x320 PicoDet output grids and person class index 0 specified by the brief.

## Review Fix

Rejected shape-valid non-contiguous output tensors before the decoder uses flat
pointer indexing. The regression test constructs a padded 3D ROI and verifies
that the runtime error clearly identifies the continuity requirement.

RED command:

```sh
cmake --build build --target postprocess_test && ./build/postprocess_test
```

Result (exit 134):

```text
Assertion failed: (threw), function test_rejects_non_contiguous_tensors,
file postprocess_test.cpp, line 102.
```

GREEN focused command:

```sh
cmake --build build --target postprocess_test && ./build/postprocess_test
```

Result: exit 0; `postprocess_test` built and completed without output.

Full suite command:

```sh
ctest --test-dir build --output-on-failure
```

Result (exit 0):

```text
100% tests passed, 0 tests failed out of 2
Total Test time (real) = 0.25 sec
```

The reviewer's Minor coverage suggestion remains out of scope as requested.
