cmake_minimum_required(VERSION 2.6)
project(Classifier)

      
find_package(OpenCV REQUIRED PATHS /usr/local/ NO_DEFAULT_PATH ) 
set (CMAKE_CXX_STANDARD 11)
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
set(sources
  Distance.h
  FeatureExtractor.h
  FeatureExtractor.cpp
  KMeans.h
  KMeans.cpp
  ImageClassifier.h
  ImageClassifier.cpp
  Driver.cpp
)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(Classifier  ${sources})
TARGET_LINK_LIBRARIES (Classifier ${OpenCV_LIBS})
