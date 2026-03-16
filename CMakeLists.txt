cmake_minimum_required(VERSION 3.24)
project(gpu-trace-analyzer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# sm_89 for RTX 4090 (Ada Lovelace). RTX 5090 uses PTX JIT from sm_89.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# MPI (needed by OTF2/otf2xx at link time, and for parallel reading)
find_package(MPI REQUIRED)

# Enable otf2xx MPI support for parallel trace reading
set(OTF2XX_WITH_MPI ON CACHE BOOL "Enable MPI support in otf2xx" FORCE)

# otf2xx (uses its own FindOTF2.cmake to locate OTF2 via otf2-config)
add_subdirectory(third_party/otf2xx)

# Main library
add_library(gpu_analyzer_lib STATIC
    src/reader/OTF2SoAReader.cpp
    src/matching/P2PMatching.cu
    src/matching/CollectiveGrouping.cpp
    src/analysis/AnalysisKernels.cu
    src/analysis/Statistics.cpp
)
target_include_directories(gpu_analyzer_lib PUBLIC include/)
target_link_libraries(gpu_analyzer_lib PUBLIC
    otf2xx::otf2xx
    MPI::MPI_CXX
)
set_target_properties(gpu_analyzer_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(gpu_analyzer_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)

# Main executable
add_executable(gpu_analyzer src/main.cu)
target_link_libraries(gpu_analyzer PRIVATE gpu_analyzer_lib)
set_target_properties(gpu_analyzer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Tests
enable_testing()
add_subdirectory(test)
