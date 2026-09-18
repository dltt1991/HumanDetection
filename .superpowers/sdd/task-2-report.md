# Task 2 Report: Camera Loop And CLI Integration

## Status

`DONE`. Task 2 is implemented, verified in Debug and Release, self-reviewed,
and committed.

## Changed Files

- `src/main.cpp`: adds the `--box-smoothing` option with default `0.35`,
  validates the range `(0, 1]`, constructs one `BoxTracker` before the
  capture loop, restores current-frame detections before tracking, and draws
  tracker output on that same frame.
- `CMakeLists.txt`: extends the help test and registers the invalid smoothing
  executable-level test through a portable CMake wrapper.
- `tests/expect_failure.cmake`: verifies that the child command exits nonzero
  and that combined stdout/stderr contains the exact expected literal text.
- `README.md`: documents the smoothing example, tuning tradeoff, `1.0`
  behavior, and the no-frame-buffering guarantee.
- `.superpowers/sdd/task-2-report.md`: contains this task evidence and is
  intentionally updated after the feature commit so it can record its SHA.

Task 1 implementation, the plan, and the specification were not modified.

## Baseline

Command:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Result: exit 0; 5/5 pre-Task-2 tests passed.

## RED Evidence

After adding the brief's original CLI tests:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target human_detection
ctest --test-dir build --output-on-failure -R "help_test|invalid_box_smoothing_test"
```

Result: exit 8. `help_test` failed because `--box-smoothing` was absent, and
the invalid invocation emitted `error: unknown option: --box-smoothing`
instead of the required validation message.

After implementing the application behavior, the full Debug suite remained
RED at 5/6 because the brief combined `WILL_FAIL TRUE` with
`PASS_REGULAR_EXPRESSION`. CTest 4.3.0 treats a regex match as success while
ignoring the process exit code, then `WILL_FAIL` inverts that success. The
correct exact parser message therefore made the test fail.

## User-Approved Test Deviation

The user approved replacing the semantically incorrect property combination
with the smallest portable CMake wrapper test. The test now invokes
`tests/expect_failure.cmake`, which:

1. Runs `human_detection --box-smoothing 0`.
2. Requires a numeric, nonzero child exit result.
3. Concatenates stdout and stderr.
4. Uses `string(FIND)` to require the exact literal:
   `box smoothing must be greater than 0 and at most 1`.

No shell-specific wrapper or platform-specific executable is used.

## GREEN Evidence

Focused corrected test:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target human_detection
ctest --test-dir build --output-on-failure -R '^invalid_box_smoothing_test$'
```

Result: exit 0; 1/1 test passed.

The wrapper guards were also exercised directly:

- Matching-output child with exit 0: helper exited 1 with
  `command unexpectedly succeeded`.
- Nonzero child without the expected text: helper exited 1 with
  `expected output not found`.

These checks confirm that neither condition alone can pass the wrapper.

## Full Debug Verification

Command:

```sh
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/human_detection --help
```

Result: exit 0; all 6/6 tests passed in 1.05 seconds. Help output included:

```text
  --box-smoothing VALUE box EMA factor (default: 0.35)
```

## Release Verification

Command:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
```

Result: exit 0. Release configure and build succeeded; all 6/6 tests passed in
2.56 seconds.

## Final Checks

`git diff --check` completed silently with exit 0 before the commit.

Self-review confirmed:

- The tracker is constructed once before the camera loop.
- Each iteration processes and displays only the current `cv::Mat`.
- No frame queue, deferred frame copy, sleep, or asynchronous display logic was
  added.
- Detection boxes are restored to current-frame coordinates before tracking.
- Tracker output, including its score, drives the rectangle and label.
- `--box-smoothing` uses the required default, parser, validation range,
  exact error text, and help ordering.
- Lower/higher/`1.0` tuning behavior and no buffering are documented.
- Task 1 and plan/spec files have no diff.

## Commit

`d8cffb6faad5b44fcbea458466b6786f6352331c`

Commit message:

```text
feat: integrate stable boxes into camera preview
```

## Concerns

None. The original CTest property conflict was corrected with explicit user
approval and the wrapper independently verifies both required conditions.
