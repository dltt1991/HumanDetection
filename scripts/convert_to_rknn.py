#!/usr/bin/env python3
import argparse

from rknn.api import RKNN


p = argparse.ArgumentParser()
p.add_argument("--onnx", default="models/picodet_s_320_person.onnx")
p.add_argument("--dataset", required=True)
p.add_argument("--output", default="models/picodet_s_320_person_int8.rknn")
a = p.parse_args()

rknn = RKNN(verbose=True)
assert rknn.config(
    target_platform="rk3568",
    mean_values=[[123.675, 116.28, 103.53]],
    std_values=[[58.395, 57.12, 57.375]],
) == 0
assert rknn.load_onnx(model=a.onnx) == 0
assert rknn.build(do_quantization=True, dataset=a.dataset) == 0
assert rknn.export_rknn(a.output) == 0
rknn.release()
