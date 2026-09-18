#pragma once

#include "detection.hpp"
#include <opencv2/core.hpp>
#include <vector>

std::vector<Detection> decode_picodet(const std::vector<cv::Mat>& outputs,
                                      float confidence, float nms_iou);
