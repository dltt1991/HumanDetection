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
