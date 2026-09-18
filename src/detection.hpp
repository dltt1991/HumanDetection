#pragma once
#include <opencv2/core.hpp>

struct Detection {
  cv::Rect2f box;
  float score;
};
