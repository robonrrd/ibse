SET(PROJECT_NAME IBSE)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
#set(CMAKE_VERBOSE_MAKEFILE ON)

#set(BUILD_TYPE Release)
set(BUILD_TYPE RelWithDebInfo)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# required libraries
#find_package(Eigen3 REQUIRED)
set(EIGEN3_INCLUDE_DIR /usr/include/eigen3/)

find_package(OpenCV 2.0 QUIET)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV > 2.0 not found.")
endif()

set(CMAKE_CXX_FLAGS
   "-O3 -g -std=c++11 -march=native -Wall -Werror"
)

# Sources files
set(ibse_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/ibse.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_impl.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_internal.cpp
)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
) 

# compile main library.
add_library(ibse SHARED
  ${ibse_SOURCE_FILES}
)

target_link_libraries(ibse
  ${OpenCV_LIBS}
)

# Enable and compile tests
#
enable_testing()

set(TEST_LIBS  nlopt ${OpenCV_LIBS})

# Internal component tests
add_executable(
  linear_interpolation_test
  ${PROJECT_SOURCE_DIR}/src/ibse_internal.cpp
  ${PROJECT_SOURCE_DIR}/src/linear_interpolation_test.cpp )
target_link_libraries(linear_interpolation_test
  ${TEST_LIBS}
)
add_test(linear_interpolation_test bin/linear_interpolation_test)

add_executable(
  ltsi_test
  ${PROJECT_SOURCE_DIR}/src/ibse_internal.cpp
  ${PROJECT_SOURCE_DIR}/src/ltsi_test.cpp )
target_link_libraries(ltsi_test
  ${TEST_LIBS}
)
add_test(ltsi_test bin/ltsi_test)

add_executable(golden_section_search_test
  ${PROJECT_SOURCE_DIR}/src/golden_section_search_test.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_testing.cpp 
  ${PROJECT_SOURCE_DIR}/src/ibse_internal.cpp)
target_link_libraries(golden_section_search_test
  ${TEST_LIBS}
)
add_test(golden_search_test bin/golden_section_search_test)

add_executable(moving_average_test
  ${PROJECT_SOURCE_DIR}/src/moving_average_test.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_testing.cpp 
  ${PROJECT_SOURCE_DIR}/src/ibse_internal.cpp)
target_link_libraries(moving_average_test
  ${TEST_LIBS}
)
add_test(moving_average_test bin/moving_average_test)

add_executable(objective_function_test
  ${PROJECT_SOURCE_DIR}/src/objective_function_test.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_testing.cpp 
  ${PROJECT_SOURCE_DIR}/src/ibse_internal.cpp)
target_link_libraries(objective_function_test
  ${TEST_LIBS}
)
add_test(moving_average_test bin/moving_average_test)

add_executable(rotation_test
  ${PROJECT_SOURCE_DIR}/src/rotation_test.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_testing.cpp 
  ${PROJECT_SOURCE_DIR}/src/ibse_internal.cpp)
target_link_libraries(rotation_test
  ${TEST_LIBS}
)
add_test(rotation_test bin/rotation_test)

add_executable(kalman_test
  ${PROJECT_SOURCE_DIR}/src/kalman_test.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_testing.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_internal.cpp)
target_link_libraries(kalman_test 
  ${TEST_LIBS}
)
add_test(kalman_test bin/kalman_test)

add_executable(estimate_alignment_test
  ${PROJECT_SOURCE_DIR}/src/estimate_alignment_test.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_testing.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_internal.cpp)
target_link_libraries(estimate_alignment_test
  ${TEST_LIBS}
)
add_test(estimate_alignment_test bin/estimate_alignment_test)

add_executable(align_camera_imu_test
  ${PROJECT_SOURCE_DIR}/src/align_camera_imu_test.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_testing.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_internal.cpp)
target_link_libraries(align_camera_imu_test
  ${TEST_LIBS}
)
add_test(align_camera_imu_test bin/align_camera_imu_test)

add_executable(initialize_estimates_test
  ${PROJECT_SOURCE_DIR}/src/initialize_estimates_test.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_testing.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_internal.cpp)
target_link_libraries(initialize_estimates_test
  ${TEST_LIBS}
)
add_test(initialize_estimates_test bin/initialize_estimates_test)

add_executable(estimate_scale_test
  ${PROJECT_SOURCE_DIR}/src/estimate_scale_test.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_testing.cpp
  ${PROJECT_SOURCE_DIR}/src/ibse_internal.cpp)
target_link_libraries(estimate_scale_test
  ${TEST_LIBS}
)
add_test(estimate_scale_test bin/estimate_scale_test)



# Full library tests
add_executable(ibse_test
  ${PROJECT_SOURCE_DIR}/src/ibse_test.cpp)
target_link_libraries(ibse_test
  ibse
  ${TEST_LIBS}
)
add_test(ibse_test bin/ibse_test)
