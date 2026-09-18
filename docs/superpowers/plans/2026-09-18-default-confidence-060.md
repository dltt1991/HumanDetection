# Default Confidence 0.60 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the default person confidence threshold from `0.40` to `0.60` without changing explicit `--confidence` overrides.

**Architecture:** Keep threshold ownership in the existing `Options` structure. Lock the user-visible default through the existing executable help test and document it in the README.

**Tech Stack:** C++17, CMake/CTest, OpenCV

## Global Constraints

- Default confidence is exactly `0.60`.
- Explicit `--confidence VALUE` behavior and accepted range `[0,1]` remain unchanged.
- Do not change the model, NMS, tracking, camera, or backend behavior.
- Historical design and implementation documents remain unchanged.

---

### Task 1: Update And Verify The Default Threshold

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `src/main.cpp`
- Modify: `README.md`

**Interfaces:**
- Consumes: existing `Options`, `--confidence`, and `help_test` behavior.
- Produces: default `Options::confidence == 0.60f` and matching help/documentation.

- [ ] **Step 1: Strengthen the help test first**

Change the `help_test` regular expression in `CMakeLists.txt` so the confidence
line must contain `default: 0.60`:

```cmake
set_tests_properties(help_test PROPERTIES
  PASS_REGULAR_EXPRESSION "--backend[^\n]*\n  --camera[^\n]*\n  --confidence[^\n]*default: 0\\.60[^\n]*\n  --nms[^\n]*\n  --box-smoothing")
```

- [ ] **Step 2: Run the focused test and verify RED**

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target human_detection
ctest --test-dir build --output-on-failure -R '^help_test$'
```

Expected: `help_test` fails because help still reports `default: 0.40`.

- [ ] **Step 3: Update the runtime default and help text**

In `src/main.cpp`, make these exact replacements:

```cpp
float confidence = 0.60f;
```

```cpp
<< "  --confidence VALUE     person threshold (default: 0.60)\n"
```

- [ ] **Step 4: Document the default**

Update the README confidence paragraph to state:

```markdown
`--confidence` controls the minimum person score and defaults to `0.60`;
increasing it reduces weak detections.
```

- [ ] **Step 5: Verify GREEN and the complete Release build**

```sh
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
./build-release/human_detection --help
git diff --check
```

Expected: Debug and Release suites pass, help reports `default: 0.60`, and
`git diff --check` is silent.

- [ ] **Step 6: Commit**

```sh
git add CMakeLists.txt src/main.cpp README.md
git commit -m "feat: raise default detection confidence"
```
