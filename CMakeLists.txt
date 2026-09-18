cmake_minimum_required(VERSION 3.16)
project(human_detection LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui videoio dnn)
option(ENABLE_RKNN "Build RK3568 RKNN backend" OFF)
if(ENABLE_RKNN)
  if(NOT DEFINED RKNN_SDK_ROOT)
    message(FATAL_ERROR "ENABLE_RKNN requires RKNN_SDK_ROOT")
  endif()
  find_path(RKNN_INCLUDE_DIR rknn_api.h PATHS "${RKNN_SDK_ROOT}/include"
            NO_DEFAULT_PATH REQUIRED)
  find_library(RKNN_RUNTIME rknnrt
               PATHS "${RKNN_SDK_ROOT}/lib" "${RKNN_SDK_ROOT}/lib64"
                     "${RKNN_SDK_ROOT}/aarch64"
               NO_DEFAULT_PATH REQUIRED)
endif()

add_library(human_detection_core
  src/opencv_backend.cpp
  src/preprocess.cpp
  src/picodet_postprocess.cpp)
target_include_directories(human_detection_core PUBLIC src)
target_link_libraries(human_detection_core PUBLIC ${OpenCV_LIBS})
if(ENABLE_RKNN)
  target_sources(human_detection_core PRIVATE src/rknn_backend.cpp)
  target_include_directories(human_detection_core PUBLIC ${RKNN_INCLUDE_DIR})
  target_link_libraries(human_detection_core PUBLIC ${RKNN_RUNTIME})
  target_compile_definitions(human_detection_core PUBLIC ENABLE_RKNN=1)
endif()

add_executable(human_detection src/main.cpp)
target_link_libraries(human_detection PRIVATE human_detection_core)

enable_testing()
add_executable(preprocess_test tests/preprocess_test.cpp)
target_link_libraries(preprocess_test PRIVATE human_detection_core)
add_test(NAME preprocess_test COMMAND preprocess_test)

add_executable(postprocess_test tests/postprocess_test.cpp)
target_link_libraries(postprocess_test PRIVATE human_detection_core)
add_test(NAME postprocess_test COMMAND postprocess_test)

add_executable(model_smoke_test tests/model_smoke_test.cpp)
target_link_libraries(model_smoke_test PRIVATE human_detection_core)
target_compile_definitions(
  model_smoke_test PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME model_smoke_test COMMAND model_smoke_test)

add_test(NAME help_test COMMAND human_detection --help)
set_tests_properties(help_test PROPERTIES
  PASS_REGULAR_EXPRESSION "--backend[^\n]*\n  --camera[^\n]*\n  --confidence[^\n]*\n  --nms")
