cmake_minimum_required(VERSION 3.8)
project(SiMPL)

# Ensure C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Use strict C++ standard
# Ensure MPI is found before including MFEM
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})  # Set MPI compiler if needed

# Use the MFEM build directory directly
include("${CMAKE_SOURCE_DIR}/config.cmake")

# Include MFEM
include("${MFEM_DIR}/MFEMConfig.cmake")

# Create executable
add_executable(stokes stokes.cpp)
add_executable(stokes_with_density stokes_with_density.cpp)
add_executable(stokes_to stokes_to.cpp)

# Link with MFEM
target_link_libraries(stokes PRIVATE mfem)
target_link_libraries(stokes_with_density PRIVATE helper mfem)
target_link_libraries(stokes_to PRIVATE mfem)

add_custom_target(dataclean
  COMMAND ${CMAKE_COMMAND} -E echo "Cleaning up mesh and result files..."
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/*.mesh
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/*.gf
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/ParaView
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/results
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/data
)
