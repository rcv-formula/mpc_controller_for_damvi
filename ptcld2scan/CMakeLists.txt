cmake_minimum_required(VERSION 3.8)
project(ptcld2scan)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

# add executable
add_executable(projector src/projector.cpp)
add_executable(fuser src/fuser.cpp)

# link dependencies
ament_target_dependencies(projector
  rclcpp
  tf2_ros
  sensor_msgs
  tf2_geometry_msgs
  pcl_conversions
  PCL
)
ament_target_dependencies(fuser
  rclcpp
  tf2_ros
  sensor_msgs
  tf2_geometry_msgs
)

# install targets
install(TARGETS
  projector
  fuser
  DESTINATION lib/${PROJECT_NAME}
)

# install additional directories
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
