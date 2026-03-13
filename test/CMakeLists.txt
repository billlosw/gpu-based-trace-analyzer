cmake_minimum_required(VERSION 3.24)

# Helper function for adding tests
function(gpu_analyzer_add_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} PRIVATE gpu_analyzer_lib)
    set_target_properties(${test_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Test: P2P Matching
gpu_analyzer_add_test(test_p2p_matching test_p2p_matching.cu)

# Test: Analysis Kernels (all 8 analyses with synthetic data)
gpu_analyzer_add_test(test_analysis_kernels test_analysis_kernels.cu)

# Test: Statistics computation
gpu_analyzer_add_test(test_statistics test_statistics.cpp)
