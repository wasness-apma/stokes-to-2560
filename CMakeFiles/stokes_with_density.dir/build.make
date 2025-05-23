# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

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
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/stokes-to-2560

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/stokes-to-2560

# Include any dependencies generated for this target.
include CMakeFiles/stokes_with_density.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/stokes_with_density.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/stokes_with_density.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/stokes_with_density.dir/flags.make

CMakeFiles/stokes_with_density.dir/codegen:
.PHONY : CMakeFiles/stokes_with_density.dir/codegen

CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.o: CMakeFiles/stokes_with_density.dir/flags.make
CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.o: stokes_with_density.cpp
CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.o: CMakeFiles/stokes_with_density.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/stokes-to-2560/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.o"
	/opt/homebrew/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.o -MF CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.o.d -o CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.o -c /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/stokes-to-2560/stokes_with_density.cpp

CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.i"
	/opt/homebrew/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/stokes-to-2560/stokes_with_density.cpp > CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.i

CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.s"
	/opt/homebrew/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/stokes-to-2560/stokes_with_density.cpp -o CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.s

# Object files for target stokes_with_density
stokes_with_density_OBJECTS = \
"CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.o"

# External object files for target stokes_with_density
stokes_with_density_EXTERNAL_OBJECTS =

stokes_with_density: CMakeFiles/stokes_with_density.dir/stokes_with_density.cpp.o
stokes_with_density: CMakeFiles/stokes_with_density.dir/build.make
stokes_with_density: /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/mfem/build/libmfem.a
stokes_with_density: /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/hypre/src/hypre/lib/libHYPRE.a
stokes_with_density: /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/SuiteSparse/lib/libumfpack.dylib
stokes_with_density: /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/SuiteSparse/lib/libklu.dylib
stokes_with_density: /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/SuiteSparse/lib/libamd.dylib
stokes_with_density: /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/SuiteSparse/lib/libbtf.dylib
stokes_with_density: /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/SuiteSparse/lib/libcholmod.dylib
stokes_with_density: /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/SuiteSparse/lib/libcolamd.dylib
stokes_with_density: /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/SuiteSparse/lib/libcamd.dylib
stokes_with_density: /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/SuiteSparse/lib/libccolamd.dylib
stokes_with_density: /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/SuiteSparse/lib/libsuitesparseconfig.dylib
stokes_with_density: /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/metis-5.1.0/lib/libmetis.a
stokes_with_density: /opt/homebrew/Cellar/open-mpi/5.0.6/lib/libmpi.dylib
stokes_with_density: CMakeFiles/stokes_with_density.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/stokes-to-2560/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable stokes_with_density"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stokes_with_density.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/stokes_with_density.dir/build: stokes_with_density
.PHONY : CMakeFiles/stokes_with_density.dir/build

CMakeFiles/stokes_with_density.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/stokes_with_density.dir/cmake_clean.cmake
.PHONY : CMakeFiles/stokes_with_density.dir/clean

CMakeFiles/stokes_with_density.dir/depend:
	cd /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/stokes-to-2560 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/stokes-to-2560 /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/stokes-to-2560 /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/stokes-to-2560 /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/stokes-to-2560 /Users/asness_will/Desktop/Everything_Academic/research/numerical_pdes/stokes-to-2560/CMakeFiles/stokes_with_density.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/stokes_with_density.dir/depend

