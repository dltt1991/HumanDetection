#include "opencv_backend.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <utility>
#include <vector>

int main() {
  const std::filesystem::path model =
      std::filesystem::path(PROJECT_SOURCE_DIR) / "models" /
      "picodet_s_320_person.onnx";
  if (!std::filesystem::exists(model)) {
    std::cout << "SKIP: model not found at " << model << '\n';
    return 0;
  }

  OpenCvBackend backend(model);
  const int blob_shape[] = {1, 3, 320, 320};
  const cv::Mat blob(4, blob_shape, CV_32F, cv::Scalar(0));
  const auto outputs = backend.infer(blob);
  const std::vector<std::pair<int, int>> expected = {
      {1600, 80}, {1600, 32}, {400, 80}, {400, 32},
      {100, 80},  {100, 32},  {25, 80},  {25, 32}};

  assert(outputs.size() == expected.size());
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    assert(outputs[i].dims == 3);
    assert(outputs[i].size[0] == 1);
    assert(outputs[i].size[1] == expected[i].first);
    assert(outputs[i].size[2] == expected[i].second);
  }
  std::cout << "OpenCV PicoDet smoke test: 8 valid outputs\n";
}
