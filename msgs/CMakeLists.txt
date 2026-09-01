cmake_minimum_required(VERSION 3.8)
project(divas_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Track.msg"
  "msg/TrackArray.msg"
  "msg/TrajectoryMode.msg"
  "msg/PredictedTrajectory.msg"
  "msg/TrajectorySet.msg"
  "msg/PathWithCurvature.msg"
  DEPENDENCIES std_msgs geometry_msgs nav_msgs
)

ament_package()
