# Detection Box Tracking Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize person boxes with lightweight temporal tracking while always displaying the current camera frame.

**Architecture:** Add a dependency-free `BoxTracker` after frame-space coordinate restoration. It greedily associates detections by IoU, applies an exponential moving average to matched boxes, and preserves unmatched tracks for at most three frames; the capture and display loop remains unbuffered.

**Tech Stack:** C++17, OpenCV core/DNN/video I/O, CMake/CTest

## Global Constraints

- Do not buffer or delay camera frames.
- Default box smoothing factor is exactly `0.35`; accepted CLI range is `(0, 1]`.
- Track association uses a fixed minimum IoU of `0.3`.
- Unmatched tracks remain visible for at most three frames and expire on the fourth miss.
- Do not add external tracking dependencies, appearance embeddings, optical flow, or RKNN code.
- Keep the tracker portable across macOS, Intel Linux, and ARM Linux CPU targets.

---

### Task 1: Lightweight Box Tracker

**Files:**
- Create: `src/box_tracker.hpp`
- Create: `src/box_tracker.cpp`
- Create: `tests/box_tracker_test.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `Detection { cv::Rect2f box; float score; }` from `src/detection.hpp`.
- Produces: `BoxTracker(float smoothing = 0.35f)` and `std::vector<Detection> BoxTracker::update(const std::vector<Detection>& detections)`.

- [ ] **Step 1: Write focused failing tracker tests**

Create `tests/box_tracker_test.cpp` with direct assertions for EMA movement,
identity preservation, dropout lifetime, and invalid-box rejection:

```cpp
#include "box_tracker.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

bool near(float actual, float expected) {
  return std::abs(actual - expected) < 0.001f;
}

Detection detection(float x, float y, float width, float height,
                    float score = 0.9f) {
  return {{x, y, width, height}, score};
}

}  // namespace

int main() {
  {
    BoxTracker tracker(0.35f);
    tracker.update({detection(100, 100, 80, 160)});
    const auto result = tracker.update({detection(104, 96, 84, 156, 0.8f)});
    assert(result.size() == 1);
    assert(near(result[0].box.x, 101.4f));
    assert(near(result[0].box.y, 98.6f));
    assert(near(result[0].box.width, 81.4f));
    assert(near(result[0].box.height, 158.6f));
    assert(near(result[0].score, 0.8f));
  }

  {
    BoxTracker tracker(0.35f);
    tracker.update({detection(100, 100, 80, 160)});
    const float first =
        tracker.update({detection(104, 100, 80, 160)})[0].box.x;
    const float second =
        tracker.update({detection(96, 100, 80, 160)})[0].box.x;
    assert(std::abs(second - first) < 4.0f);
  }

  {
    BoxTracker tracker;
    tracker.update({detection(10, 10, 40, 80),
                    detection(200, 20, 50, 90)});
    const auto result = tracker.update({detection(204, 20, 50, 90),
                                        detection(14, 10, 40, 80)});
    assert(result.size() == 2);
    assert(result[0].box.x < 30);
    assert(result[1].box.x > 190);
  }

  {
    BoxTracker tracker;
    tracker.update({detection(20, 20, 40, 80)});
    assert(tracker.update({}).size() == 1);
    assert(tracker.update({}).size() == 1);
    assert(tracker.update({}).size() == 1);
    assert(tracker.update({}).empty());
  }

  {
    BoxTracker tracker;
    const auto result = tracker.update({
        detection(0, 0, 0, 10),
        detection(0, 0, 10, -1),
        detection(30, 30, 20, 40),
    });
    assert(result.size() == 1);
    assert(near(result[0].box.x, 30));
  }
}
```

Update `CMakeLists.txt` so `human_detection_core` lists
`src/box_tracker.cpp`, and add:

```cmake
add_executable(box_tracker_test tests/box_tracker_test.cpp)
target_link_libraries(box_tracker_test PRIVATE human_detection_core)
add_test(NAME box_tracker_test COMMAND box_tracker_test)
```

- [ ] **Step 2: Run the tracker test and verify RED**

Run:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target box_tracker_test
```

Expected: build fails because `box_tracker.hpp` and `BoxTracker` do not exist.

- [ ] **Step 3: Implement the minimal tracker interface**

Create `src/box_tracker.hpp`:

```cpp
#pragma once

#include "detection.hpp"

#include <vector>

class BoxTracker {
 public:
  explicit BoxTracker(float smoothing = 0.35f);
  std::vector<Detection> update(const std::vector<Detection>& detections);

 private:
  struct Track {
    Detection detection;
    int missed = 0;
  };

  float smoothing_;
  std::vector<Track> tracks_;
};
```

Create `src/box_tracker.cpp`. Implement these exact behaviors:

```cpp
#include "box_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <tuple>

namespace {

constexpr float kMinIou = 0.3f;
constexpr int kMaxMissed = 3;

bool valid(const Detection& detection) {
  const auto& box = detection.box;
  return std::isfinite(box.x) && std::isfinite(box.y) &&
         std::isfinite(box.width) && std::isfinite(box.height) &&
         box.width > 0 && box.height > 0;
}

float iou(const cv::Rect2f& left, const cv::Rect2f& right) {
  const float intersection = (left & right).area();
  const float combined = left.area() + right.area() - intersection;
  return combined > 0 ? intersection / combined : 0;
}

cv::Rect2f smooth(const cv::Rect2f& previous, const cv::Rect2f& current,
                  float alpha) {
  const float retained = 1.0f - alpha;
  return {retained * previous.x + alpha * current.x,
          retained * previous.y + alpha * current.y,
          retained * previous.width + alpha * current.width,
          retained * previous.height + alpha * current.height};
}

}  // namespace
```

Add the constructor and update implementation below. It discards invalid
detections, sorts all eligible matches from highest to lowest IoU, updates each
track and detection at most once, and preserves stable track order:

```cpp
BoxTracker::BoxTracker(float smoothing) : smoothing_(smoothing) {
  if (!std::isfinite(smoothing_) || smoothing_ <= 0 || smoothing_ > 1) {
    throw std::invalid_argument("box smoothing must be in (0, 1]");
  }
}

std::vector<Detection> BoxTracker::update(
    const std::vector<Detection>& detections) {
  std::vector<Detection> current;
  std::copy_if(detections.begin(), detections.end(),
               std::back_inserter(current), valid);

  std::vector<std::tuple<float, std::size_t, std::size_t>> candidates;
  for (std::size_t track = 0; track < tracks_.size(); ++track) {
    for (std::size_t detection = 0; detection < current.size(); ++detection) {
      const float overlap = iou(tracks_[track].detection.box,
                                current[detection].box);
      if (overlap >= kMinIou)
        candidates.emplace_back(overlap, track, detection);
    }
  }
  std::sort(candidates.begin(), candidates.end(), std::greater<>());

  std::vector<bool> track_matched(tracks_.size());
  std::vector<bool> detection_matched(current.size());
  for (const auto& [overlap, track, detection] : candidates) {
    (void)overlap;
    if (track_matched[track] || detection_matched[detection]) continue;
    tracks_[track].detection.box = smooth(
        tracks_[track].detection.box, current[detection].box, smoothing_);
    tracks_[track].detection.score = current[detection].score;
    tracks_[track].missed = 0;
    track_matched[track] = true;
    detection_matched[detection] = true;
  }

  for (std::size_t track = 0; track < tracks_.size(); ++track) {
    if (!track_matched[track]) ++tracks_[track].missed;
  }
  tracks_.erase(
      std::remove_if(tracks_.begin(), tracks_.end(),
                     [](const Track& track) {
                       return track.missed > kMaxMissed;
                     }),
      tracks_.end());

  for (std::size_t detection = 0; detection < current.size(); ++detection) {
    if (!detection_matched[detection]) tracks_.push_back({current[detection]});
  }

  std::vector<Detection> result;
  result.reserve(tracks_.size());
  for (const auto& track : tracks_) result.push_back(track.detection);
  return result;
}
```

- [ ] **Step 4: Build and verify GREEN**

Run:

```sh
cmake --build build --target box_tracker_test
ctest --test-dir build --output-on-failure -R box_tracker_test
```

Expected: `box_tracker_test` passes.

- [ ] **Step 5: Run all existing tests**

Run:

```sh
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Expected: all tests pass, including `box_tracker_test`.

- [ ] **Step 6: Commit the tracker**

```sh
git add CMakeLists.txt src/box_tracker.hpp src/box_tracker.cpp tests/box_tracker_test.cpp
git commit -m "feat: stabilize person boxes with temporal tracking"
```

---

### Task 2: Camera Loop And CLI Integration

**Files:**
- Modify: `src/main.cpp`
- Modify: `CMakeLists.txt`
- Modify: `README.md`

**Interfaces:**
- Consumes: `BoxTracker(float smoothing)` and `BoxTracker::update(...)` from Task 1.
- Produces: `--box-smoothing VALUE`, default `0.35`, plus stabilized boxes drawn over the current unbuffered camera frame.

- [ ] **Step 1: Add failing executable-level CLI tests**

Extend the `help_test` regular expression in `CMakeLists.txt` to require
`--box-smoothing`, and add an invalid-value test that checks the exact parser
error:

```cmake
set_tests_properties(help_test PROPERTIES
  PASS_REGULAR_EXPRESSION "--backend[^\n]*\n  --camera[^\n]*\n  --confidence[^\n]*\n  --nms[^\n]*\n  --box-smoothing")

add_test(NAME invalid_box_smoothing_test
  COMMAND human_detection --box-smoothing 0)
set_tests_properties(invalid_box_smoothing_test PROPERTIES
  WILL_FAIL TRUE
  PASS_REGULAR_EXPRESSION "box smoothing must be greater than 0 and at most 1")
```

- [ ] **Step 2: Run the CLI tests and verify RED**

Run:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target human_detection
ctest --test-dir build --output-on-failure -R "help_test|invalid_box_smoothing_test"
```

Expected: `help_test` fails because the option is absent, and the invalid test
does not contain the expected validation message.

- [ ] **Step 3: Add CLI parsing and tracker construction**

In `src/main.cpp`:

1. Include `box_tracker.hpp`.
2. Add `float box_smoothing = 0.35f;` to `Options`.
3. Add this help line after `--nms`:

```cpp
<< "  --box-smoothing VALUE box EMA factor (default: 0.35)\n"
```

4. Parse it with the existing float parser:

```cpp
else if (option == "--box-smoothing")
  options.box_smoothing = parse_float(value, option);
```

5. Validate it before returning options:

```cpp
if (!std::isfinite(options.box_smoothing) ||
    options.box_smoothing <= 0 || options.box_smoothing > 1) {
  throw std::invalid_argument(
      "box smoothing must be greater than 0 and at most 1");
}
```

6. Construct `BoxTracker tracker(options.box_smoothing);` once in `run`, before
the capture loop.

- [ ] **Step 4: Feed restored detections through the tracker**

Replace the direct decode-and-draw loop with:

```cpp
std::vector<Detection> frame_detections;
for (const auto& detection :
     decode_picodet(backend.infer(prep.blob), options.confidence,
                    options.nms)) {
  frame_detections.push_back(
      {restore_box(detection.box, prep, frame.size()), detection.score});
}

for (const auto& detection : tracker.update(frame_detections)) {
  const auto& box = detection.box;
  cv::rectangle(frame, box, {0, 255, 0}, 2);
  char label[32];
  std::snprintf(label, sizeof(label), "person %.2f", detection.score);
  cv::putText(frame, label,
              {static_cast<int>(box.x),
               std::max(18, static_cast<int>(box.y) - 5)},
              cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 255, 0}, 2);
}
```

Do not add a frame queue, frame copies for later display, sleeps, or asynchronous
display logic.

- [ ] **Step 5: Verify CLI and full test suite GREEN**

Run:

```sh
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/human_detection --help
```

Expected: every test passes and help includes `--box-smoothing`.

- [ ] **Step 6: Document tuning behavior**

Add a short README example:

```sh
./build/human_detection --camera 0 --box-smoothing 0.35
```

Document that lower values produce steadier but slower boxes, higher values
follow movement faster, and `1.0` disables coordinate smoothing. State that
video frames are never buffered by the tracker.

- [ ] **Step 7: Perform final verification and commit**

Run:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
git diff --check
```

Expected: Release build succeeds, all tests pass, and `git diff --check` is
silent.

Commit:

```sh
git add CMakeLists.txt src/main.cpp README.md
git commit -m "feat: integrate stable boxes into camera preview"
```
