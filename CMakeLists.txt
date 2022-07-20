cmake_minimum_required(VERSION 2.8.3)
project(rgbd2pointcloud)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
find_package(Eigen3)
include_directories(  ${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS} )

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    )

include_directories(${catkin_INCLUDE_DIRS} )
catkin_package()

add_executable(rgbd2pointcloud_from_topic src/rgbd2pointcloud_from_topic.cpp)
add_dependencies(rgbd2pointcloud_from_topic ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ) 
target_link_libraries(rgbd2pointcloud_from_topic ${catkin_LIBRARIES}  ${OpenCV_LIBS} ${PCL_LIBRARIES}) 


add_executable(rgbd2pointcloud_from_data src/rgbd2pointcloud_from_data.cpp)
#add_dependencies(rgbd2pointcloud_from_data ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ) 
target_link_libraries(rgbd2pointcloud_from_data ${catkin_LIBRARIES}  ${OpenCV_LIBS} ${PCL_LIBRARIES}) 
