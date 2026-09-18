# 人体检测

`human_detection` 是一个使用 C++17 编写的 macOS 摄像头应用。程序通过
OpenCV DNN 运行 PicoDet 人体检测模型，可显示指定摄像头的实时画面、绘制人体
检测框和置信度，并显示滚动计算的帧率。

当前支持的推理后端为 `opencv`。`InferenceBackend` 接口和 `--backend` 参数为
后续接入 CPU 和 NPU 后端预留了扩展点；本项目当前尚未实现 RKNN。

## macOS 环境要求

使用 Homebrew 安装 CMake 和 OpenCV：

```sh
brew install cmake opencv
```

程序需要使用 macOS 能够识别的摄像头，包括内置摄像头、USB 摄像头或 USB
视频采集设备。

## 下载模型

在仓库根目录运行以下命令，下载 PicoDet-S 320x320 ONNX 模型：

```sh
./scripts/download_model.sh
```

脚本会将模型保存为 `models/picodet_s_320_person.onnx`。模型文件不会提交到
Git 仓库。

## 构建与测试

```sh
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

如果尚未下载模型，模型冒烟测试会跳过实际推理并正常退出。请先下载模型，以
验证 ONNX 文件能否正常加载，以及模型是否输出 PicoDet 预期的 8 个张量。

## 摄像头权限

首次启动摄像头时，macOS 可能会请求访问权限。请允许启动
`human_detection` 的终端应用访问摄像头。如需查看或修改权限，请打开
**系统设置 > 隐私与安全性 > 相机**，启用对应的终端应用，然后重启该应用再试。

权限弹窗和实时预览需要在可交互的 macOS 会话中手动验证。

## 运行

使用编号为 `0` 的摄像头：

```sh
./build/human_detection --camera 0
```

调整检测框平滑系数：

```sh
./build/human_detection --camera 0 --box-smoothing 0.35
```

较小的数值会让检测框更稳定，但跟随速度更慢；较大的数值会让检测框更快地
跟随目标。设置为 `1.0` 可关闭坐标平滑。跟踪器不会缓存视频帧，因此不会给
预览画面增加延迟。

按 `q`、Escape，或者关闭预览窗口即可退出。

如果打开了错误的摄像头，请尝试其他从 `0` 开始的设备编号：

```sh
./build/human_detection --camera 1
```

使用 `--width` 和 `--height` 可以指定期望的采集分辨率。摄像头硬件可能会选择
最接近的受支持分辨率：

```sh
./build/human_detection --camera 1 --width 1920 --height 1080
```

也可以按需指定模型路径和检测阈值：

```sh
./build/human_detection \
  --model /path/to/picodet_s_320_person.onnx \
  --confidence 0.50 \
  --nms 0.45
```

`--confidence` 控制人体检测的最低置信度，默认值为 `0.60`；提高该值可以减少
低置信度检测。`--nms` 控制重叠检测框的抑制程度。两个参数的取值范围均为
`0` 到 `1`。运行 `./build/human_detection --help` 可查看完整的命令行参数。

## 后端支持范围

当前仅支持 `opencv` 后端：

```sh
./build/human_detection --backend opencv --camera 0
```

抽象接口 `InferenceBackend` 和 `--backend` 参数为后续后端提供了稳定的接入点。
当前选择 `--backend rknn` 会明确提示该后端不可用。Intel Linux、ARM Linux
CPU 部署、RKNN 模型转换、交叉编译和 RK3568 NPU 运行时支持均属于后续工作；
本项目目前不提供可用的 RKNN 构建或转换命令。
