#include "preprocess.hpp"
#include <cmath>
#include <iostream>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

}  // namespace

int main() {
  cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(0, 0, 255));
  const auto result = preprocess(image);
  check(result.blob.dims == 4, "blob has four dimensions");
  check(result.blob.size[0] == 1 && result.blob.size[1] == 3 &&
            result.blob.size[2] == 320 && result.blob.size[3] == 320,
        "blob shape is 1x3x320x320");
  check(std::abs(result.scale_x - 0.25f) < 1e-6f, "horizontal scale");
  check(std::abs(result.scale_y - (320.0f / 720.0f)) < 1e-6f,
        "vertical scale");
  const float* blob = result.blob.ptr<float>();
  const int plane = 320 * 320;
  check(std::abs(blob[0] - ((1.0f - 0.485f) / 0.229f)) < 1e-5f,
        "red input becomes normalized RGB channel 0");
  check(std::abs(blob[plane] - ((0.0f - 0.456f) / 0.224f)) < 1e-5f,
        "green input becomes normalized RGB channel 1");
  check(std::abs(blob[2 * plane] - ((0.0f - 0.406f) / 0.225f)) < 1e-5f,
        "blue input becomes normalized RGB channel 2");

  const auto box = restore_box({32, 40, 160, 200}, result, image.size());
  check(std::abs(box.x - 128.0f) < 1e-4f, "restored x");
  check(std::abs(box.y - 90.0f) < 1e-4f, "restored y");
  check(std::abs(box.width - 640.0f) < 1e-4f, "restored width");
  check(std::abs(box.height - 450.0f) < 1e-4f, "restored height");

  const auto clipped =
      restore_box({-32, -40, 400, 400}, result, image.size());
  check(clipped == cv::Rect2f(0, 0, 1280, 720),
        "out-of-bounds box clips to source image");
  return failures == 0 ? 0 : 1;
}
