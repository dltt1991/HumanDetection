#include "picodet_postprocess.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

cv::Mat tensor(int rows, int columns, float value) {
  const int sizes[] = {1, rows, columns};
  return cv::Mat(3, sizes, CV_32F, cv::Scalar(value));
}

std::vector<cv::Mat> empty_outputs() {
  std::vector<cv::Mat> outputs;
  for (const int rows : {1600, 400, 100, 25}) {
    outputs.push_back(tensor(rows, 80, 0.0f));
    outputs.push_back(tensor(rows, 32, -10.0f));
  }
  return outputs;
}

void set_candidate(std::vector<cv::Mat>& outputs, int candidate, float score,
                   const int (&bins)[4]) {
  outputs[0].ptr<float>()[candidate * 80] = score;
  for (int side = 0; side < 4; ++side) {
    outputs[1].ptr<float>()[candidate * 32 + side * 8 + bins[side]] = 10.0f;
  }
}

void test_decodes_distribution_distances() {
  auto outputs = empty_outputs();
  const int bins[] = {2, 2, 2, 2};
  set_candidate(outputs, 10 * 40 + 12, 0.9f, bins);

  const auto detections = decode_picodet(outputs, 0.5f, 0.5f);
  check(detections.size() == 1, "one distribution detection");
  check(std::abs(detections[0].score - 0.9f) < 1e-6f, "detection score");
  check(std::abs(detections[0].box.x - 84.0f) < 0.1f, "detection x");
  check(std::abs(detections[0].box.y - 68.0f) < 0.1f, "detection y");
  check(std::abs(detections[0].box.width - 32.0f) < 0.1f,
        "detection width");
  check(std::abs(detections[0].box.height - 32.0f) < 0.1f,
        "detection height");
}

void test_filters_low_scores() {
  auto outputs = empty_outputs();
  const int bins[] = {2, 2, 2, 2};
  set_candidate(outputs, 10 * 40 + 12, 0.4f, bins);
  check(decode_picodet(outputs, 0.5f, 0.5f).empty(), "low score filtered");
}

void test_suppresses_lower_scored_overlap() {
  auto outputs = empty_outputs();
  const int first_bins[] = {2, 2, 2, 2};
  const int second_bins[] = {3, 2, 1, 2};
  set_candidate(outputs, 10 * 40 + 12, 0.9f, first_bins);
  set_candidate(outputs, 10 * 40 + 13, 0.8f, second_bins);

  const auto detections = decode_picodet(outputs, 0.5f, 0.5f);
  check(detections.size() == 1, "overlap suppressed");
  check(std::abs(detections[0].score - 0.9f) < 1e-6f,
        "higher overlap score retained");
}

void test_rejects_malformed_shapes() {
  auto outputs = empty_outputs();
  outputs[0] = tensor(10, 80, 0.0f);
  bool threw = false;
  try {
    decode_picodet(outputs, 0.5f, 0.5f);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "rejects wrong tensor rows");

  outputs = empty_outputs();
  outputs[0] = cv::Mat(1600, 80, CV_32F, cv::Scalar(0.0f));
  threw = false;
  try {
    decode_picodet(outputs, 0.5f, 0.5f);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "rejects wrong tensor dimensions");
}

void test_rejects_non_contiguous_tensors() {
  auto outputs = empty_outputs();
  const int sizes[] = {1, 1600, 81};
  cv::Mat padded(3, sizes, CV_32F, cv::Scalar(0.0f));
  const cv::Range ranges[] = {cv::Range::all(), cv::Range::all(),
                              cv::Range(0, 80)};
  outputs[0] = padded(ranges);
  check(!outputs[0].isContinuous(), "test tensor is non-contiguous");

  bool threw = false;
  try {
    decode_picodet(outputs, 0.5f, 0.5f);
  } catch (const std::runtime_error& error) {
    threw = std::string(error.what()).find("continuous") != std::string::npos;
  }
  check(threw, "rejects non-contiguous tensor");
}

}  // namespace

int main() {
  test_decodes_distribution_distances();
  test_filters_low_scores();
  test_suppresses_lower_scored_overlap();
  test_rejects_malformed_shapes();
  test_rejects_non_contiguous_tensors();
  return failures == 0 ? 0 : 1;
}
