#include "preprocess.hpp"
#include <cassert>
#include <cmath>

int main() {
  cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(0, 0, 255));
  const auto result = preprocess(image);
  assert(result.blob.dims == 4);
  assert(result.blob.size[2] == 320 && result.blob.size[3] == 320);
  assert(std::abs(result.scale_x - 0.25f) < 1e-6f);
  assert(std::abs(result.scale_y - (320.0f / 720.0f)) < 1e-6f);
  const auto box = restore_box({32, 40, 160, 200}, result, image.size());
  assert(std::abs(box.x - 128.0f) < 1e-4f);
  assert(std::abs(box.y - 90.0f) < 1e-4f);
  assert(std::abs(box.width - 640.0f) < 1e-4f);
  assert(std::abs(box.height - 450.0f) < 1e-4f);
}
