#include "opencv_backend.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void test_constructor_probes_output_contract(
    const std::filesystem::path& model) {
  std::ifstream input(model, std::ios::binary);
  std::string bytes(std::istreambuf_iterator<char>(input), {});
  const std::string expected_name = "transpose_0.tmp_0";
  const std::string incompatible_name = "transpose_X.tmp_0";
  std::size_t replaced = 0;
  for (std::size_t at = bytes.find(expected_name); at != std::string::npos;
       at = bytes.find(expected_name, at + incompatible_name.size())) {
    bytes.replace(at, expected_name.size(), incompatible_name);
    ++replaced;
  }
  check(replaced > 0, "test model output name was not found");

  const auto incompatible =
      std::filesystem::temp_directory_path() /
      ("human_detection_incompatible_output_" +
       std::to_string(std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) +
       ".onnx");
  struct RemoveFile {
    std::filesystem::path path;
    ~RemoveFile() { std::filesystem::remove(path); }
  } cleanup{incompatible};
  std::ofstream output(incompatible, std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  output.close();

  bool threw = false;
  try {
    OpenCvBackend backend(incompatible);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "backend construction must probe the output contract");
}

}  // namespace

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

  check(outputs.size() == expected.size(), "expected eight model outputs");
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    check(outputs[i].dims == 3, "output must have three dimensions");
    check(outputs[i].size[0] == 1, "output batch must be one");
    check(outputs[i].size[1] == expected[i].first, "unexpected output rows");
    check(outputs[i].size[2] == expected[i].second,
          "unexpected output columns");
  }
  test_constructor_probes_output_contract(model);
  std::cout << "OpenCV PicoDet smoke test: 8 valid outputs\n";
}
