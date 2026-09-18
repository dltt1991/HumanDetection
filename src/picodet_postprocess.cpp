#include "picodet_postprocess.hpp"

#include <opencv2/dnn.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace {

float decode_distance(const float* logits, int stride) {
  const float maximum = *std::max_element(logits, logits + 8);
  float denominator = 0.0f;
  float weighted_sum = 0.0f;
  for (int bin = 0; bin < 8; ++bin) {
    const float probability = std::exp(logits[bin] - maximum);
    denominator += probability;
    weighted_sum += probability * static_cast<float>(bin);
  }
  return weighted_sum / denominator * static_cast<float>(stride);
}

bool has_shape(const cv::Mat& tensor, int rows, int columns) {
  return tensor.type() == CV_32F && tensor.dims == 3 &&
         tensor.size[0] == 1 && tensor.size[1] == rows &&
         tensor.size[2] == columns;
}

}  // namespace

std::vector<Detection> decode_picodet(const std::vector<cv::Mat>& outputs,
                                      float confidence, float nms_iou) {
  constexpr std::array<int, 4> strides{8, 16, 32, 64};
  constexpr std::array<int, 4> rows{1600, 400, 100, 25};
  if (outputs.size() != strides.size() * 2) {
    throw std::runtime_error("PicoDet requires eight output tensors");
  }

  std::vector<cv::Rect2d> boxes;
  std::vector<float> scores;
  for (std::size_t level = 0; level < strides.size(); ++level) {
    const cv::Mat& score_tensor = outputs[level * 2];
    const cv::Mat& box_tensor = outputs[level * 2 + 1];
    if (!has_shape(score_tensor, rows[level], 80) ||
        !has_shape(box_tensor, rows[level], 32)) {
      throw std::runtime_error("invalid PicoDet output tensor");
    }
    if (!score_tensor.isContinuous() || !box_tensor.isContinuous()) {
      throw std::runtime_error("PicoDet output tensors must be continuous");
    }

    const int stride = strides[level];
    const int grid_width = 320 / stride;
    const float* score_data = score_tensor.ptr<float>();
    const float* box_data = box_tensor.ptr<float>();
    for (int candidate = 0; candidate < rows[level]; ++candidate) {
      const float score = score_data[candidate * 80];
      if (score < confidence) {
        continue;
      }

      const float center_x = (candidate % grid_width + 0.5f) * stride;
      const float center_y = (candidate / grid_width + 0.5f) * stride;
      const float* logits = box_data + candidate * 32;
      const float left = decode_distance(logits, stride);
      const float top = decode_distance(logits + 8, stride);
      const float right = decode_distance(logits + 16, stride);
      const float bottom = decode_distance(logits + 24, stride);
      boxes.emplace_back(center_x - left, center_y - top, left + right,
                         top + bottom);
      scores.push_back(score);
    }
  }

  std::vector<int> kept;
  cv::dnn::NMSBoxes(boxes, scores, confidence, nms_iou, kept);
  std::vector<Detection> detections;
  detections.reserve(kept.size());
  for (const int index : kept) {
    const cv::Rect2d& box = boxes[index];
    detections.push_back(
        {{static_cast<float>(box.x), static_cast<float>(box.y),
          static_cast<float>(box.width), static_cast<float>(box.height)},
         scores[index]});
  }
  return detections;
}
