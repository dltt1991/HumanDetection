#pragma once

#include "inference_backend.hpp"
#include <filesystem>
#include <opencv2/dnn.hpp>

class OpenCvBackend final : public InferenceBackend {
 public:
  explicit OpenCvBackend(std::filesystem::path model_path);
  std::vector<cv::Mat> infer(const cv::Mat& blob) override;

 private:
  cv::dnn::Net net_;
};
