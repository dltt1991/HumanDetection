#pragma once

#include "inference_backend.hpp"

#include <filesystem>
#include <rknn_api.h>

class RknnBackend final : public InferenceBackend {
 public:
  explicit RknnBackend(std::filesystem::path model_path);
  ~RknnBackend() override;

  RknnBackend(const RknnBackend&) = delete;
  RknnBackend& operator=(const RknnBackend&) = delete;
  RknnBackend(RknnBackend&& other) noexcept;
  RknnBackend& operator=(RknnBackend&& other) noexcept;

  std::vector<cv::Mat> infer(const cv::Mat& input) override;

 private:
  rknn_context context_ = 0;
};
