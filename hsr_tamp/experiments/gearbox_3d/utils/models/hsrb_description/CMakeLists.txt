# Copyright (C) 2016 Toyota Motor Corporation
# vim: fileencoding=utf-8 :
cmake_minimum_required(VERSION 2.8.3)
project(hsrb_description)

find_package(catkin REQUIRED COMPONENTS xacro rosbash)

# define robots to be installed
set(robots hsrb4s)

# Generate robot URDFs
file(GLOB_RECURSE _component_xacros "${PROJECT_SOURCE_DIR}/urdf/*.xacro")
foreach(_robot ${robots})
    set(_urdf ${PROJECT_SOURCE_DIR}/robots/${_robot}.urdf)
    set(_xacro ${_urdf}.xacro)

    # add custom target
    add_custom_command(
        OUTPUT ${_urdf}
        DEPENDS ${_xacro} ${_component_xacros}
        COMMAND ${CATKIN_DEVEL_PREFIX}/env.sh rosrun xacro xacro --inorder ${_xacro} > ${_urdf})
    add_custom_target(${PROJECT_NAME}_${_robot}_gen_urdf ALL DEPENDS ${_urdf})
endforeach()

catkin_package(CATKIN_DEPENDS xacro)

install(DIRECTORY config launch robots urdf
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/
)

# make a symbolic link
add_custom_target(link_target ALL
     COMMAND ${CMAKE_COMMAND} -E create_symlink hsrb4s.urdf ${PROJECT_SOURCE_DIR}/robots/hsrb.urdf
)

if(CATKIN_ENABLE_TESTING)
    find_package(roslaunch REQUIRED)
    roslaunch_add_file_check(launch USE_TEST_DEPENDENCIES)
    catkin_add_nosetests(test)
endif()
