# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/ipc1/.local/lib/python3.7/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/ipc1/.local/lib/python3.7/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ipc1/SLAM/VSLAM/mono-vo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ipc1/SLAM/VSLAM/mono-vo/build

# Include any dependencies generated for this target.
include CMakeFiles/visual_odometry.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/visual_odometry.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/visual_odometry.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/visual_odometry.dir/flags.make

CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.o: CMakeFiles/visual_odometry.dir/flags.make
CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.o: /home/ipc1/SLAM/VSLAM/mono-vo/src/visual_odometry.cpp
CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.o: CMakeFiles/visual_odometry.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ipc1/SLAM/VSLAM/mono-vo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.o -MF CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.o.d -o CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.o -c /home/ipc1/SLAM/VSLAM/mono-vo/src/visual_odometry.cpp

CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ipc1/SLAM/VSLAM/mono-vo/src/visual_odometry.cpp > CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.i

CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ipc1/SLAM/VSLAM/mono-vo/src/visual_odometry.cpp -o CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.s

# Object files for target visual_odometry
visual_odometry_OBJECTS = \
"CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.o"

# External object files for target visual_odometry
visual_odometry_EXTERNAL_OBJECTS =

visual_odometry: CMakeFiles/visual_odometry.dir/src/visual_odometry.cpp.o
visual_odometry: CMakeFiles/visual_odometry.dir/build.make
visual_odometry: /usr/local/lib/libopencv_stitching.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_superres.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_videostab.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_aruco.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_bgsegm.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_bioinspired.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_ccalib.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_dpm.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_face.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_freetype.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_fuzzy.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_hdf.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_hfs.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_img_hash.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_line_descriptor.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_optflow.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_reg.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_rgbd.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_saliency.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_stereo.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_structured_light.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_surface_matching.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_tracking.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_xfeatures2d.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_ximgproc.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_xobjdetect.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_xphoto.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_shape.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_highgui.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_videoio.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_viz.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_video.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_datasets.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_plot.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_text.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_dnn.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_ml.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_imgcodecs.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_objdetect.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_calib3d.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_features2d.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_flann.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_photo.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_imgproc.so.3.4.19
visual_odometry: /usr/local/lib/libopencv_core.so.3.4.19
visual_odometry: CMakeFiles/visual_odometry.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ipc1/SLAM/VSLAM/mono-vo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable visual_odometry"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/visual_odometry.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/visual_odometry.dir/build: visual_odometry
.PHONY : CMakeFiles/visual_odometry.dir/build

CMakeFiles/visual_odometry.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/visual_odometry.dir/cmake_clean.cmake
.PHONY : CMakeFiles/visual_odometry.dir/clean

CMakeFiles/visual_odometry.dir/depend:
	cd /home/ipc1/SLAM/VSLAM/mono-vo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ipc1/SLAM/VSLAM/mono-vo /home/ipc1/SLAM/VSLAM/mono-vo /home/ipc1/SLAM/VSLAM/mono-vo/build /home/ipc1/SLAM/VSLAM/mono-vo/build /home/ipc1/SLAM/VSLAM/mono-vo/build/CMakeFiles/visual_odometry.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/visual_odometry.dir/depend

