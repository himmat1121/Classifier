# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/export/home/softwares/anaconda3/lib/python3.7/site-packages/cmake/data/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/export/home/softwares/anaconda3/lib/python3.7/site-packages/cmake/data/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /export/home/users/media/media_shared/Himmat/Classifier/CMakeFiles /export/home/users/media/media_shared/Himmat/Classifier/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /export/home/users/media/media_shared/Himmat/Classifier/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named Classifier

# Build rule for target.
Classifier: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 Classifier
.PHONY : Classifier

# fast build rule for target.
Classifier/fast:
	$(MAKE) -f CMakeFiles/Classifier.dir/build.make CMakeFiles/Classifier.dir/build
.PHONY : Classifier/fast

Driver.o: Driver.cpp.o

.PHONY : Driver.o

# target to build an object file
Driver.cpp.o:
	$(MAKE) -f CMakeFiles/Classifier.dir/build.make CMakeFiles/Classifier.dir/Driver.cpp.o
.PHONY : Driver.cpp.o

Driver.i: Driver.cpp.i

.PHONY : Driver.i

# target to preprocess a source file
Driver.cpp.i:
	$(MAKE) -f CMakeFiles/Classifier.dir/build.make CMakeFiles/Classifier.dir/Driver.cpp.i
.PHONY : Driver.cpp.i

Driver.s: Driver.cpp.s

.PHONY : Driver.s

# target to generate assembly for a file
Driver.cpp.s:
	$(MAKE) -f CMakeFiles/Classifier.dir/build.make CMakeFiles/Classifier.dir/Driver.cpp.s
.PHONY : Driver.cpp.s

FeatureExtractor.o: FeatureExtractor.cpp.o

.PHONY : FeatureExtractor.o

# target to build an object file
FeatureExtractor.cpp.o:
	$(MAKE) -f CMakeFiles/Classifier.dir/build.make CMakeFiles/Classifier.dir/FeatureExtractor.cpp.o
.PHONY : FeatureExtractor.cpp.o

FeatureExtractor.i: FeatureExtractor.cpp.i

.PHONY : FeatureExtractor.i

# target to preprocess a source file
FeatureExtractor.cpp.i:
	$(MAKE) -f CMakeFiles/Classifier.dir/build.make CMakeFiles/Classifier.dir/FeatureExtractor.cpp.i
.PHONY : FeatureExtractor.cpp.i

FeatureExtractor.s: FeatureExtractor.cpp.s

.PHONY : FeatureExtractor.s

# target to generate assembly for a file
FeatureExtractor.cpp.s:
	$(MAKE) -f CMakeFiles/Classifier.dir/build.make CMakeFiles/Classifier.dir/FeatureExtractor.cpp.s
.PHONY : FeatureExtractor.cpp.s

ImageClassifier.o: ImageClassifier.cpp.o

.PHONY : ImageClassifier.o

# target to build an object file
ImageClassifier.cpp.o:
	$(MAKE) -f CMakeFiles/Classifier.dir/build.make CMakeFiles/Classifier.dir/ImageClassifier.cpp.o
.PHONY : ImageClassifier.cpp.o

ImageClassifier.i: ImageClassifier.cpp.i

.PHONY : ImageClassifier.i

# target to preprocess a source file
ImageClassifier.cpp.i:
	$(MAKE) -f CMakeFiles/Classifier.dir/build.make CMakeFiles/Classifier.dir/ImageClassifier.cpp.i
.PHONY : ImageClassifier.cpp.i

ImageClassifier.s: ImageClassifier.cpp.s

.PHONY : ImageClassifier.s

# target to generate assembly for a file
ImageClassifier.cpp.s:
	$(MAKE) -f CMakeFiles/Classifier.dir/build.make CMakeFiles/Classifier.dir/ImageClassifier.cpp.s
.PHONY : ImageClassifier.cpp.s

KMeans.o: KMeans.cpp.o

.PHONY : KMeans.o

# target to build an object file
KMeans.cpp.o:
	$(MAKE) -f CMakeFiles/Classifier.dir/build.make CMakeFiles/Classifier.dir/KMeans.cpp.o
.PHONY : KMeans.cpp.o

KMeans.i: KMeans.cpp.i

.PHONY : KMeans.i

# target to preprocess a source file
KMeans.cpp.i:
	$(MAKE) -f CMakeFiles/Classifier.dir/build.make CMakeFiles/Classifier.dir/KMeans.cpp.i
.PHONY : KMeans.cpp.i

KMeans.s: KMeans.cpp.s

.PHONY : KMeans.s

# target to generate assembly for a file
KMeans.cpp.s:
	$(MAKE) -f CMakeFiles/Classifier.dir/build.make CMakeFiles/Classifier.dir/KMeans.cpp.s
.PHONY : KMeans.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... Classifier"
	@echo "... Driver.o"
	@echo "... Driver.i"
	@echo "... Driver.s"
	@echo "... FeatureExtractor.o"
	@echo "... FeatureExtractor.i"
	@echo "... FeatureExtractor.s"
	@echo "... ImageClassifier.o"
	@echo "... ImageClassifier.i"
	@echo "... ImageClassifier.s"
	@echo "... KMeans.o"
	@echo "... KMeans.i"
	@echo "... KMeans.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

