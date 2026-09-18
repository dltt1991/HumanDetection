#pragma once
#include <opencv2/core.hpp>
#include <vector>

class InferenceBackend {
 public:
  virtual ~InferenceBackend() = default;
  // Input is backend-ready: FP32 NCHW for OpenCV or uint8 RGB NHWC for RKNN.
  virtual std::vector<cv::Mat> infer(const cv::Mat& input) = 0;
};
