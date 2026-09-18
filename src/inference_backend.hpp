#pragma once
#include <opencv2/core.hpp>
#include <vector>

class InferenceBackend {
 public:
  virtual ~InferenceBackend() = default;
  virtual std::vector<cv::Mat> infer(const cv::Mat& blob) = 0;
};
