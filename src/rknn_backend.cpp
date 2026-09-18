#include "rknn_backend.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

const std::array<std::pair<int, int>, 8> kOutputShapes = {
    std::pair{1600, 80}, std::pair{1600, 32}, std::pair{400, 80},
    std::pair{400, 32},  std::pair{100, 80},  std::pair{100, 32},
    std::pair{25, 80},   std::pair{25, 32}};

void check(int status, const char* operation) {
  if (status < 0)
    throw std::runtime_error(std::string(operation) + " failed: " +
                             std::to_string(status));
}

std::vector<unsigned char> read_model(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("failed to open model '" + path.string() + "'");
  const auto size = file.tellg();
  if (size <= 0) throw std::runtime_error("model is empty: '" + path.string() + "'");
  std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
  file.seekg(0);
  if (!file.read(reinterpret_cast<char*>(bytes.data()), size))
    throw std::runtime_error("failed to read model '" + path.string() + "'");
  return bytes;
}

}  // namespace

RknnBackend::RknnBackend(std::filesystem::path model_path) {
  const auto model = read_model(model_path);
  check(rknn_init(&context_, const_cast<unsigned char*>(model.data()),
                  static_cast<uint32_t>(model.size()), 0, nullptr),
        "rknn_init");
  try {
    rknn_input_output_num counts{};
    check(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &counts, sizeof(counts)),
          "rknn_query input/output count");
    if (counts.n_input != 1 || counts.n_output != kOutputShapes.size())
      throw std::runtime_error("RKNN model must have one input and eight outputs");

    rknn_tensor_attr input{};
    input.index = 0;
    check(rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input, sizeof(input)),
          "rknn_query input attributes");
    if (input.n_dims != 4 || input.fmt != RKNN_TENSOR_NHWC ||
        input.type != RKNN_TENSOR_UINT8 || input.dims[0] != 1 ||
        input.dims[1] != 320 || input.dims[2] != 320 || input.dims[3] != 3)
      throw std::runtime_error("RKNN input must be uint8 NHWC 1x320x320x3");

    for (uint32_t i = 0; i < counts.n_output; ++i) {
      rknn_tensor_attr output{};
      output.index = i;
      check(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &output, sizeof(output)),
            "rknn_query output attributes");
      const auto [rows, columns] = kOutputShapes[i];
      if (output.n_elems != static_cast<uint32_t>(rows * columns))
        throw std::runtime_error("unexpected RKNN output size at index " +
                                 std::to_string(i));
    }
  } catch (...) {
    rknn_destroy(context_);
    context_ = 0;
    throw;
  }
}

RknnBackend::~RknnBackend() {
  if (context_) rknn_destroy(context_);
}

RknnBackend::RknnBackend(RknnBackend&& other) noexcept
    : context_(std::exchange(other.context_, 0)) {}

RknnBackend& RknnBackend::operator=(RknnBackend&& other) noexcept {
  if (this != &other) {
    if (context_) rknn_destroy(context_);
    context_ = std::exchange(other.context_, 0);
  }
  return *this;
}

std::vector<cv::Mat> RknnBackend::infer(const cv::Mat& input) {
  if (input.rows != 320 || input.cols != 320 || input.type() != CV_8UC3 ||
      !input.isContinuous())
    throw std::invalid_argument("RKNN input must be contiguous uint8 RGB 320x320");

  rknn_input rknn_input_tensor{};
  rknn_input_tensor.index = 0;
  rknn_input_tensor.buf = input.data;
  rknn_input_tensor.size = static_cast<uint32_t>(input.total() * input.elemSize());
  rknn_input_tensor.type = RKNN_TENSOR_UINT8;
  rknn_input_tensor.fmt = RKNN_TENSOR_NHWC;
  rknn_input_tensor.pass_through = 0;
  check(rknn_inputs_set(context_, 1, &rknn_input_tensor), "rknn_inputs_set");
  check(rknn_run(context_, nullptr), "rknn_run");

  std::array<rknn_output, 8> rknn_outputs{};
  for (uint32_t i = 0; i < rknn_outputs.size(); ++i) {
    rknn_outputs[i].index = i;
    rknn_outputs[i].want_float = 1;
  }
  check(rknn_outputs_get(context_, rknn_outputs.size(), rknn_outputs.data(),
                         nullptr),
        "rknn_outputs_get");

  std::vector<cv::Mat> outputs;
  outputs.reserve(rknn_outputs.size());
  try {
    for (std::size_t i = 0; i < rknn_outputs.size(); ++i) {
      const auto [rows, columns] = kOutputShapes[i];
      const int shape[] = {1, rows, columns};
      cv::Mat view(3, shape, CV_32F, rknn_outputs[i].buf);
      outputs.push_back(view.clone());
    }
  } catch (...) {
    rknn_outputs_release(context_, rknn_outputs.size(), rknn_outputs.data());
    throw;
  }
  check(rknn_outputs_release(context_, rknn_outputs.size(), rknn_outputs.data()),
        "rknn_outputs_release");
  return outputs;
}
