#include "opencv_backend.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

const std::vector<std::string> kOutputNames = {
    "transpose_0.tmp_0", "transpose_1.tmp_0", "transpose_2.tmp_0",
    "transpose_3.tmp_0", "transpose_4.tmp_0", "transpose_5.tmp_0",
    "transpose_6.tmp_0", "transpose_7.tmp_0"};

const std::array<std::pair<int, int>, 8> kOutputShapes = {
    std::pair{1600, 80}, std::pair{1600, 32}, std::pair{400, 80},
    std::pair{400, 32},  std::pair{100, 80},  std::pair{100, 32},
    std::pair{25, 80},   std::pair{25, 32}};

}  // namespace

OpenCvBackend::OpenCvBackend(std::filesystem::path model_path) {
  try {
    net_ = cv::dnn::readNetFromONNX(model_path.string());
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  } catch (const cv::Exception& error) {
    throw std::runtime_error("failed to load model '" + model_path.string() +
                             "': " + error.what());
  }
}

std::vector<cv::Mat> OpenCvBackend::infer(const cv::Mat& blob) {
  std::vector<cv::Mat> outputs;
  try {
    net_.setInput(blob);
    net_.forward(outputs, kOutputNames);
  } catch (const cv::Exception& error) {
    throw std::runtime_error("OpenCV inference failed: " +
                             std::string(error.what()));
  }

  if (outputs.size() != kOutputShapes.size()) {
    throw std::runtime_error("unexpected output count: expected 8, got " +
                             std::to_string(outputs.size()));
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto [rows, columns] = kOutputShapes[i];
    if (outputs[i].dims != 3 || outputs[i].size[0] != 1 ||
        outputs[i].size[1] != rows || outputs[i].size[2] != columns) {
      throw std::runtime_error("unexpected output shape at index " +
                               std::to_string(i) + ": expected 1x" +
                               std::to_string(rows) + "x" +
                               std::to_string(columns));
    }
  }
  return outputs;
}
