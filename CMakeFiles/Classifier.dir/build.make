# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /export/home/softwares/anaconda3/lib/python3.7/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /export/home/softwares/anaconda3/lib/python3.7/site-packages/cmake/data/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /export/home/users/media/media_shared/Himmat/Classifier

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /export/home/users/media/media_shared/Himmat/Classifier

# Include any dependencies generated for this target.
include CMakeFiles/Classifier.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Classifier.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Classifier.dir/flags.make

CMakeFiles/Classifier.dir/FeatureExtractor.cpp.o: CMakeFiles/Classifier.dir/flags.make
CMakeFiles/Classifier.dir/FeatureExtractor.cpp.o: FeatureExtractor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/export/home/users/media/media_shared/Himmat/Classifier/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Classifier.dir/FeatureExtractor.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Classifier.dir/FeatureExtractor.cpp.o -c /export/home/users/media/media_shared/Himmat/Classifier/FeatureExtractor.cpp

CMakeFiles/Classifier.dir/FeatureExtractor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Classifier.dir/FeatureExtractor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/home/users/media/media_shared/Himmat/Classifier/FeatureExtractor.cpp > CMakeFiles/Classifier.dir/FeatureExtractor.cpp.i

CMakeFiles/Classifier.dir/FeatureExtractor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Classifier.dir/FeatureExtractor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/home/users/media/media_shared/Himmat/Classifier/FeatureExtractor.cpp -o CMakeFiles/Classifier.dir/FeatureExtractor.cpp.s

CMakeFiles/Classifier.dir/KMeans.cpp.o: CMakeFiles/Classifier.dir/flags.make
CMakeFiles/Classifier.dir/KMeans.cpp.o: KMeans.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/export/home/users/media/media_shared/Himmat/Classifier/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Classifier.dir/KMeans.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Classifier.dir/KMeans.cpp.o -c /export/home/users/media/media_shared/Himmat/Classifier/KMeans.cpp

CMakeFiles/Classifier.dir/KMeans.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Classifier.dir/KMeans.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/home/users/media/media_shared/Himmat/Classifier/KMeans.cpp > CMakeFiles/Classifier.dir/KMeans.cpp.i

CMakeFiles/Classifier.dir/KMeans.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Classifier.dir/KMeans.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/home/users/media/media_shared/Himmat/Classifier/KMeans.cpp -o CMakeFiles/Classifier.dir/KMeans.cpp.s

CMakeFiles/Classifier.dir/ImageClassifier.cpp.o: CMakeFiles/Classifier.dir/flags.make
CMakeFiles/Classifier.dir/ImageClassifier.cpp.o: ImageClassifier.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/export/home/users/media/media_shared/Himmat/Classifier/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/Classifier.dir/ImageClassifier.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Classifier.dir/ImageClassifier.cpp.o -c /export/home/users/media/media_shared/Himmat/Classifier/ImageClassifier.cpp

CMakeFiles/Classifier.dir/ImageClassifier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Classifier.dir/ImageClassifier.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/home/users/media/media_shared/Himmat/Classifier/ImageClassifier.cpp > CMakeFiles/Classifier.dir/ImageClassifier.cpp.i

CMakeFiles/Classifier.dir/ImageClassifier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Classifier.dir/ImageClassifier.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/home/users/media/media_shared/Himmat/Classifier/ImageClassifier.cpp -o CMakeFiles/Classifier.dir/ImageClassifier.cpp.s

CMakeFiles/Classifier.dir/Driver.cpp.o: CMakeFiles/Classifier.dir/flags.make
CMakeFiles/Classifier.dir/Driver.cpp.o: Driver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/export/home/users/media/media_shared/Himmat/Classifier/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/Classifier.dir/Driver.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Classifier.dir/Driver.cpp.o -c /export/home/users/media/media_shared/Himmat/Classifier/Driver.cpp

CMakeFiles/Classifier.dir/Driver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Classifier.dir/Driver.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/home/users/media/media_shared/Himmat/Classifier/Driver.cpp > CMakeFiles/Classifier.dir/Driver.cpp.i

CMakeFiles/Classifier.dir/Driver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Classifier.dir/Driver.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/home/users/media/media_shared/Himmat/Classifier/Driver.cpp -o CMakeFiles/Classifier.dir/Driver.cpp.s

# Object files for target Classifier
Classifier_OBJECTS = \
"CMakeFiles/Classifier.dir/FeatureExtractor.cpp.o" \
"CMakeFiles/Classifier.dir/KMeans.cpp.o" \
"CMakeFiles/Classifier.dir/ImageClassifier.cpp.o" \
"CMakeFiles/Classifier.dir/Driver.cpp.o"

# External object files for target Classifier
Classifier_EXTERNAL_OBJECTS =

Classifier: CMakeFiles/Classifier.dir/FeatureExtractor.cpp.o
Classifier: CMakeFiles/Classifier.dir/KMeans.cpp.o
Classifier: CMakeFiles/Classifier.dir/ImageClassifier.cpp.o
Classifier: CMakeFiles/Classifier.dir/Driver.cpp.o
Classifier: CMakeFiles/Classifier.dir/build.make
Classifier: /usr/local/lib/libopencv_gapi.so.4.2.0
Classifier: /usr/local/lib/libopencv_stitching.so.4.2.0
Classifier: /usr/local/lib/libopencv_bgsegm.so.4.2.0
Classifier: /usr/local/lib/libopencv_xphoto.so.4.2.0
Classifier: /usr/local/lib/libopencv_superres.so.4.2.0
Classifier: /usr/local/lib/libopencv_hdf.so.4.2.0
Classifier: /usr/local/lib/libopencv_freetype.so.4.2.0
Classifier: /usr/local/lib/libopencv_cvv.so.4.2.0
Classifier: /usr/local/lib/libopencv_line_descriptor.so.4.2.0
Classifier: /usr/local/lib/libopencv_img_hash.so.4.2.0
Classifier: /usr/local/lib/libopencv_dpm.so.4.2.0
Classifier: /usr/local/lib/libopencv_surface_matching.so.4.2.0
Classifier: /usr/local/lib/libopencv_reg.so.4.2.0
Classifier: /usr/local/lib/libopencv_xobjdetect.so.4.2.0
Classifier: /usr/local/lib/libopencv_rgbd.so.4.2.0
Classifier: /usr/local/lib/libopencv_structured_light.so.4.2.0
Classifier: /usr/local/lib/libopencv_quality.so.4.2.0
Classifier: /usr/local/lib/libopencv_bioinspired.so.4.2.0
Classifier: /usr/local/lib/libopencv_xfeatures2d.so.4.2.0
Classifier: /usr/local/lib/libopencv_videostab.so.4.2.0
Classifier: /usr/local/lib/libopencv_stereo.so.4.2.0
Classifier: /usr/local/lib/libopencv_aruco.so.4.2.0
Classifier: /usr/local/lib/libopencv_fuzzy.so.4.2.0
Classifier: /usr/local/lib/libopencv_ccalib.so.4.2.0
Classifier: /usr/local/lib/libopencv_optflow.so.4.2.0
Classifier: /usr/local/lib/libopencv_shape.so.4.2.0
Classifier: /usr/local/lib/libopencv_tracking.so.4.2.0
Classifier: /usr/local/lib/libopencv_face.so.4.2.0
Classifier: /usr/local/lib/libopencv_saliency.so.4.2.0
Classifier: /usr/local/lib/libopencv_hfs.so.4.2.0
Classifier: /usr/local/lib/libopencv_ximgproc.so.4.2.0
Classifier: /usr/local/lib/libopencv_phase_unwrapping.so.4.2.0
Classifier: /usr/local/lib/libopencv_video.so.4.2.0
Classifier: /usr/local/lib/libopencv_datasets.so.4.2.0
Classifier: /usr/local/lib/libopencv_ml.so.4.2.0
Classifier: /usr/local/lib/libopencv_plot.so.4.2.0
Classifier: /usr/local/lib/libopencv_highgui.so.4.2.0
Classifier: /usr/local/lib/libopencv_videoio.so.4.2.0
Classifier: /usr/local/lib/libopencv_imgcodecs.so.4.2.0
Classifier: /usr/local/lib/libopencv_photo.so.4.2.0
Classifier: /usr/local/lib/libopencv_objdetect.so.4.2.0
Classifier: /usr/local/lib/libopencv_calib3d.so.4.2.0
Classifier: /usr/local/lib/libopencv_features2d.so.4.2.0
Classifier: /usr/local/lib/libopencv_flann.so.4.2.0
Classifier: /usr/local/lib/libopencv_imgproc.so.4.2.0
Classifier: /usr/local/lib/libopencv_core.so.4.2.0
Classifier: CMakeFiles/Classifier.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/export/home/users/media/media_shared/Himmat/Classifier/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable Classifier"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Classifier.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Classifier.dir/build: Classifier

.PHONY : CMakeFiles/Classifier.dir/build

CMakeFiles/Classifier.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Classifier.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Classifier.dir/clean

CMakeFiles/Classifier.dir/depend:
	cd /export/home/users/media/media_shared/Himmat/Classifier && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /export/home/users/media/media_shared/Himmat/Classifier /export/home/users/media/media_shared/Himmat/Classifier /export/home/users/media/media_shared/Himmat/Classifier /export/home/users/media/media_shared/Himmat/Classifier /export/home/users/media/media_shared/Himmat/Classifier/CMakeFiles/Classifier.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Classifier.dir/depend
