#include "preprocess.hpp"
#include <algorithm>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

PreprocessResult preprocess(const cv::Mat& bgr) {
  if (bgr.empty()) throw std::invalid_argument("input frame is empty");
  cv::Mat resized, rgb, float_image;
  cv::resize(bgr, resized, {320, 320}, 0, 0, cv::INTER_CUBIC);
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(float_image, CV_32F, 1.0 / 255.0);
  const cv::Scalar mean(0.485, 0.456, 0.406);
  const cv::Scalar inv_std(1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225);
  cv::subtract(float_image, mean, float_image);
  cv::multiply(float_image, inv_std, float_image);
  return {cv::dnn::blobFromImage(float_image),
          320.0f / static_cast<float>(bgr.cols),
          320.0f / static_cast<float>(bgr.rows)};
}

PreprocessResult preprocess_rknn(const cv::Mat& bgr) {
  if (bgr.empty()) throw std::invalid_argument("input frame is empty");
  cv::Mat resized, rgb;
  cv::resize(bgr, resized, {320, 320}, 0, 0, cv::INTER_CUBIC);
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  return {rgb, 320.0f / static_cast<float>(bgr.cols),
          320.0f / static_cast<float>(bgr.rows)};
}

cv::Rect2f restore_box(const cv::Rect2f& box, const PreprocessResult& prep,
                       cv::Size source) {
  const float x1 = std::clamp(box.x / prep.scale_x, 0.0f, float(source.width));
  const float y1 = std::clamp(box.y / prep.scale_y, 0.0f, float(source.height));
  const float x2 = std::clamp((box.x + box.width) / prep.scale_x, 0.0f, float(source.width));
  const float y2 = std::clamp((box.y + box.height) / prep.scale_y, 0.0f, float(source.height));
  return {x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1)};
}
