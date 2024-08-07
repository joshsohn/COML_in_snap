cmake_minimum_required(VERSION 2.8.3)
project(snap)

## Compile as C++11, supported in ROS Indigo and newer
set(CMAKE_CXX_STANDARD 11)
add_definitions(-std=c++11) # snapdragon wants it like this :(
if (NOT CMAKE_BUILD_TYPE)
    # Options: Debug, Release, MinSizeRel, RelWithDebInfo
    message(STATUS "No build type selected, default to Release")
#    set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_BUILD_TYPE "Debug")  # See issue #1

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  std_srvs
  geometry_msgs
  snapstack_msgs
  nav_msgs
)

find_package(snap_sim QUIET)
if(snap_sim_FOUND)
  message("Building snap for software-in-the-loop simulation")

  set(ROS_PKG_DEPS ${ROS_PKG_DEPS} snap_sim)

  # very gross hack --- IMPORTED_LOCATION only allows one lib. Just use
  # the first lib in the list. If it is wrong, no matter---we add the
  # rest of the libs via INTERFACE_LINK_LIBRARIES
  list(GET snap_sim_LIBRARIES 0 FIRST)

  add_library(esc_interface SHARED IMPORTED)
  set_property(TARGET esc_interface PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${snap_sim_INCLUDE_DIRS})
  set_property(TARGET esc_interface PROPERTY IMPORTED_LOCATION ${FIRST})
  set_property(TARGET esc_interface PROPERTY INTERFACE_LINK_LIBRARIES ${snap_sim_LIBRARIES})

  add_library(sensor_imu SHARED IMPORTED)
  set_property(TARGET sensor_imu PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${snap_sim_INCLUDE_DIRS})
  set_property(TARGET sensor_imu PROPERTY IMPORTED_LOCATION ${FIRST})
  set_property(TARGET sensor_imu PROPERTY INTERFACE_LINK_LIBRARIES ${snap_sim_LIBRARIES})

  # add macro definitions so that targets that link to these targets know we are in SIL
  set_property(TARGET esc_interface APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS SNAP_SIM)
  set_property(TARGET sensor_imu APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS SNAP_SIM)
endif()

## System dependencies are found with CMake's conventions
find_package(Curses REQUIRED)

# Setup cmake paths
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Download adaptnotch library
set(ADAPTNOTCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/adaptnotch-download)
set(BUILD_EXAPPS OFF CACHE INTERNAL "") # don't build adaptnotch examples
configure_file("${CMAKE_MODULE_PATH}/adaptnotch.cmake.in" "${ADAPTNOTCH_DIR}/CMakeLists.txt" IMMEDIATE @ONLY)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" . WORKING_DIRECTORY ${ADAPTNOTCH_DIR} )
execute_process(COMMAND ${CMAKE_COMMAND} --build . WORKING_DIRECTORY ${ADAPTNOTCH_DIR})
add_subdirectory(${ADAPTNOTCH_DIR}/src ${ADAPTNOTCH_DIR}/build)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   DIRECTORY msg
#   FILES
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES snap
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
#  CATKIN_DEPENDS snapstack_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include include/nodes ${catkin_INCLUDE_DIRS} ${CURSES_INCLUDE_DIR})

## Declare a C++ library
add_library(snap
  src/common/SnapdragonDebugPrint.cpp
  src/utils/SnapdragonUtils.cpp
  src/imu/SnapdragonImuManager.cpp
  src/observer/SnapdragonObserverManager.cpp
  src/control/SnapdragonControllerManager.cpp
  src/esc/SnapdragonEscManager.cpp
)

target_include_directories(snap PUBLIC
  src/imu
  src/common
  src/observer
  src/control
  src/esc
  src/utils
 )

# snap_apm only if snap_apm_FOUND
find_package(snap_apm QUIET)
if( snap_apm_FOUND )
  message("snap_apm found")
  add_definitions(-DSNAP_APM)
  target_link_libraries(snap snap_apm)
endif()
target_link_libraries(snap sensor_imu esc_interface pthread adaptnotch)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(snap ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(snap_node src/nodes/SnapdragonRosNode.cpp src/nodes/SnapdragonRosNode_main.cpp)
add_executable(esc_interface_node src/nodes/esc_interface_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
set_target_properties(snap_node PROPERTIES OUTPUT_NAME snap PREFIX "")
set_target_properties(esc_interface_node PROPERTIES OUTPUT_NAME esc_interface PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(snap_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(esc_interface_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(snap_node ${catkin_LIBRARIES} snap)
target_link_libraries(esc_interface_node ${catkin_LIBRARIES} ${CURSES_LIBRARIES})

## Snapdragon specific directives
# set(CMAKE_EXE_LINKER_FLAGS "-Wl,-gc-sections -Wl,-rpath-link,${CMAKE_SYSROOT}/linaro-rootfs/lib/arm-linux-gnueabihf -Wl,-rpath-link,${CMAKE_SYSROOT}/linaro-rootfs/usr/lib/arm-linux-gnueabihf")

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
install(TARGETS snap snap_node esc_interface_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY cfg/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/cfg/
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_snap.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
