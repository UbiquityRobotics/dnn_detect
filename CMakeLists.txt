
cmake_minimum_required(VERSION 2.8.3)
project(dnn_detect)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf2_geometry_msgs
  tf2_ros
  tf2
  visualization_msgs
  image_transport
  cv_bridge
  std_msgs
)

find_package(OpenCV REQUIRED)


add_message_files(
   FILES
   DetectedObject.msg
   DetectedObjectArray.msg
)


add_service_files(
  FILES
  Detect.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)
catkin_package(INCLUDE_DIRS DEPENDS OpenCV)

###########
## Build ##
###########


add_definitions(-std=c++11)

include_directories(${catkin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(dnn_detect src/dnn_detect.cpp)

add_dependencies(dnn_detect ${${PROJECT_NAME}_EXPORTED_TARGETS}
                 ${catkin_EXPORTED_TARGETS})

target_link_libraries(dnn_detect ${catkin_LIBRARIES} ${OpenCV_LIBS})

#############
## Install ##
#############

## Mark executables and/or libraries for installation
install(TARGETS dnn_detect
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
) 

install(DIRECTORY model/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/model
) 

###########
## Tests ##
###########

if(CATKIN_ENABLE_TESTING)
        find_package(rostest REQUIRED)

        # Tests need c++11
        add_definitions(-std=c++11)
        
        add_rostest_gtest(dnn_images_test 
          test/dnn_images.test 
          test/dnn_images_test.cpp)
        add_dependencies(dnn_images_test ${PROJECT_NAME}_generate_messages)
        target_link_libraries(dnn_images_test ${catkin_LIBRARIES} ${OpenCV_LIBS})
endif()
