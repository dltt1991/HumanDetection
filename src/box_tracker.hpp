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
