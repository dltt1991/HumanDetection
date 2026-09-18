#include "opencv_backend.hpp"
#include "picodet_postprocess.hpp"
#include "preprocess.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

struct Options {
  std::string backend = "opencv";
  int camera = 0;
  float confidence = 0.40f;
  float nms = 0.50f;
  int width = 1280;
  int height = 720;
  std::filesystem::path model = "models/picodet_s_320_person.onnx";
};

void print_help() {
  std::cout
      << "Usage: human_detection [options]\n"
      << "  --backend opencv|rknn  inference backend (default: opencv)\n"
      << "  --camera INDEX         camera index (default: 0)\n"
      << "  --confidence VALUE     person threshold (default: 0.40)\n"
      << "  --nms VALUE            IoU threshold (default: 0.50)\n"
      << "  --width PIXELS         requested capture width (default: 1280)\n"
      << "  --height PIXELS        requested capture height (default: 720)\n"
      << "  --model PATH           ONNX model path\n"
      << "  --help                 show this help\n";
}

int parse_int(const std::string& value, const std::string& option) {
  int result = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size())
    throw std::invalid_argument(option + " requires an integer");
  return result;
}

float parse_float(const std::string& value, const std::string& option) {
  std::size_t end = 0;
  float result = 0;
  try {
    result = std::stof(value, &end);
  } catch (const std::exception&) {
    throw std::invalid_argument(option + " requires a number");
  }
  if (end != value.size())
    throw std::invalid_argument(option + " requires a number");
  return result;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--help") {
      print_help();
      std::exit(0);
    }
    if (i + 1 == argc) throw std::invalid_argument(option + " requires a value");
    const std::string value = argv[++i];
    if (option == "--backend") options.backend = value;
    else if (option == "--camera") options.camera = parse_int(value, option);
    else if (option == "--confidence") options.confidence = parse_float(value, option);
    else if (option == "--nms") options.nms = parse_float(value, option);
    else if (option == "--width") options.width = parse_int(value, option);
    else if (option == "--height") options.height = parse_int(value, option);
    else if (option == "--model") options.model = value;
    else throw std::invalid_argument("unknown option: " + option);
  }

  if (options.backend != "opencv" && options.backend != "rknn")
    throw std::invalid_argument("backend must be 'opencv' or 'rknn'");
  if (options.camera < 0)
    throw std::invalid_argument("camera index must be non-negative");
  if (!std::isfinite(options.confidence) || options.confidence < 0 ||
      options.confidence > 1)
    throw std::invalid_argument("confidence must be between 0 and 1");
  if (!std::isfinite(options.nms) || options.nms < 0 || options.nms > 1)
    throw std::invalid_argument("nms must be between 0 and 1");
  if (options.width <= 0 || options.height <= 0)
    throw std::invalid_argument("capture width and height must be positive");
  return options;
}

int run(const Options& options) {
  if (options.backend == "rknn")
    throw std::runtime_error("rknn backend is not available in this build");

  OpenCvBackend backend(options.model);
  cv::VideoCapture camera(options.camera, cv::CAP_ANY);
  if (!camera.isOpened())
    throw std::runtime_error("failed to open camera " +
                             std::to_string(options.camera));
  camera.set(cv::CAP_PROP_FRAME_WIDTH, options.width);
  camera.set(cv::CAP_PROP_FRAME_HEIGHT, options.height);

  constexpr char kWindow[] = "Human Detection";
  cv::namedWindow(kWindow, cv::WINDOW_AUTOSIZE);
  std::deque<std::chrono::steady_clock::time_point> frame_times;
  for (;;) {
    cv::Mat frame;
    if (!camera.read(frame) || frame.empty())
      throw std::runtime_error("camera returned an empty frame");

    const auto prep = preprocess(frame);
    for (const auto& detection :
         decode_picodet(backend.infer(prep.blob), options.confidence,
                        options.nms)) {
      const auto box = restore_box(detection.box, prep, frame.size());
      cv::rectangle(frame, box, {0, 255, 0}, 2);
      char label[32];
      std::snprintf(label, sizeof(label), "person %.2f", detection.score);
      cv::putText(frame, label,
                  {static_cast<int>(box.x),
                   std::max(18, static_cast<int>(box.y) - 5)},
                  cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 255, 0}, 2);
    }

    const auto now = std::chrono::steady_clock::now();
    frame_times.push_back(now);
    if (frame_times.size() > 30) frame_times.pop_front();
    double fps = 0;
    if (frame_times.size() > 1) {
      const auto seconds =
          std::chrono::duration<double>(now - frame_times.front()).count();
      fps = (frame_times.size() - 1) / seconds;
    }
    char fps_label[32];
    std::snprintf(fps_label, sizeof(fps_label), "FPS %.1f", fps);
    cv::putText(frame, fps_label, {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                {0, 255, 0}, 2);
    cv::imshow(kWindow, frame);

    const int key = cv::waitKey(1) & 0xff;
    if (key == 'q' || key == 27 ||
        cv::getWindowProperty(kWindow, cv::WND_PROP_VISIBLE) < 1)
      break;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
