cmake_minimum_required(VERSION 2.8.3)
project(hsrb_meshes)
find_package(catkin REQUIRED)
catkin_package(
  )

install(DIRECTORY meshes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/
  )
