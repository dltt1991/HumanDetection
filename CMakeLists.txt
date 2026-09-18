cmake_minimum_required(VERSION 3.16)
project(human_detection LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui videoio dnn)

add_library(human_detection_core src/preprocess.cpp src/picodet_postprocess.cpp)
target_include_directories(human_detection_core PUBLIC src)
target_link_libraries(human_detection_core PUBLIC ${OpenCV_LIBS})

enable_testing()
add_executable(preprocess_test tests/preprocess_test.cpp)
target_link_libraries(preprocess_test PRIVATE human_detection_core)
add_test(NAME preprocess_test COMMAND preprocess_test)

add_executable(postprocess_test tests/postprocess_test.cpp)
target_link_libraries(postprocess_test PRIVATE human_detection_core)
add_test(NAME postprocess_test COMMAND postprocess_test)
