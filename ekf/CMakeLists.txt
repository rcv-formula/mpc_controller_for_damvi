cmake_minimum_required(VERSION 3.8)
project(ekf)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)  # Required for Eigen3
find_package(Eigen3 REQUIRED)

# Executable
add_executable(imu_carto_ekf src/imu_carto_ekf.cpp)

ament_target_dependencies(imu_carto_ekf
  rclcpp
  sensor_msgs
  nav_msgs
  Eigen3
)

# Include directories
target_include_directories(imu_carto_ekf PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Set required C++ standard
target_compile_features(imu_carto_ekf PUBLIC cxx_std_17)

# Install target
install(TARGETS 
  imu_carto_ekf
  DESTINATION lib/${PROJECT_NAME}
)

# Linting & Tests
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
