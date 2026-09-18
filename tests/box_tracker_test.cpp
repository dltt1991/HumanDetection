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
