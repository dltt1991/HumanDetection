#pragma once
#include <opencv2/core.hpp>

struct PreprocessResult {
  cv::Mat blob;
  float scale_x;
  float scale_y;
};

PreprocessResult preprocess(const cv::Mat& bgr);
PreprocessResult preprocess_rknn(const cv::Mat& bgr);
cv::Rect2f restore_box(const cv::Rect2f& model_box,
                       const PreprocessResult& prep,
                       cv::Size source_size);
