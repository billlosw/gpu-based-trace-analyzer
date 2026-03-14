#include "analysis/AnalysisKernels.h"
#include "analysis/Statistics.h"
#include "common/cuda_check.h"
#include "data/AnalysisResults.h"
#include "matching/CollectiveGrouping.h"
#include "matching/P2PMatching.h"
#include "reader/OTF2SoAReader.h"

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <mpi.h>

static void printGpuInfo() {
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  std::cout << "=== GPU Device Info ===" << std::endl;
  std::cout << "Device: " << prop.name << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor
            << std::endl;
  std::cout << "VRAM: " << prop.totalGlobalMem / (1024 * 1024) << " MB"
            << std::endl;
  std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
  std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bit"
            << std::endl;
  std::cout << "========================" << std::endl;
}

static void printResult(const char *name, const AnalysisResult &r) {
  std::cout << "------------------ " << name << " -------------------"
            << std::endl;
  if (r.count == 0) {
    std::cout << "There is no " << name << std::endl;
    return;
  }
  std::cout << "Count: " << std::fixed << std::setprecision(10) << r.count
            << std::endl;
  std::cout << "Mean: " << std::fixed << std::setprecision(10) << r.mean
            << std::endl;
  std::cout << "Median: " << std::fixed << std::setprecision(10) << r.median
            << std::endl;
  std::cout << "Minimum: " << std::fixed << std::setprecision(10) << r.min_val
            << std::endl;
  std::cout << "Maximum: " << std::fixed << std::setprecision(10) << r.max_val
            << std::endl;
  std::cout << "Sum: " << std::fixed << std::setprecision(10) << r.sum
            << std::endl;
  std::cout << "Variance: " << std::fixed << std::setprecision(10)
            << r.variance << std::endl;
  std::cout << "Quartile 25: " << std::fixed << std::setprecision(10) << r.q25
            << std::endl;
  std::cout << "Quartile 75: " << std::fixed << std::setprecision(10) << r.q75
            << std::endl;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (argc < 2) {
    if (mpi_rank == 0)
      std::cerr << "Usage: " << argv[0] << " <path/to/traces.otf2>" << std::endl;
    MPI_Finalize();
    return 1;
  }

  std::string trace_path = argv[1];

  if (mpi_rank == 0) {
    printGpuInfo();
    std::cout << std::endl;
    std::cout << "Trace file: " << trace_path << std::endl;
    if (mpi_size > 1)
      std::cout << "MPI ranks for reading: " << mpi_size << std::endl;
    std::cout << std::endl;
  }

  auto t_total_start = std::chrono::high_resolution_clock::now();

  // Step 1: Read OTF2 trace into SoA (MPI-parallel)
  if (mpi_rank == 0)
    std::cout << "=== Step 1: Reading OTF2 trace ===" << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();
  ReaderOutput reader_output = readOTF2Trace(trace_path);
  auto t2 = std::chrono::high_resolution_clock::now();
  double read_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();

  if (mpi_rank == 0) {
    std::cout << "[Timer] OTF2 read: " << read_ms << " ms" << std::endl;
    std::cout << "Events: " << reader_output.data.count << std::endl;
    std::cout << "Data size: "
              << reader_output.data.sizeInBytes() / (1024.0 * 1024.0) << " MB"
              << std::endl;
    std::cout << std::endl;
  }

  // Only rank 0 has the full data — other ranks can finalize
  if (mpi_rank != 0) {
    MPI_Finalize();
    return 0;
  }

  if (reader_output.data.count == 0) {
    std::cerr << "[ERROR] No events read from trace. Check trace file path."
              << std::endl;
    MPI_Finalize();
    return 1;
  }

  // Step 2: P2P Matching (CPU, timestamp-sorted)
  std::cout << "=== Step 2: P2P Matching ===" << std::endl;
  t1 = std::chrono::high_resolution_clock::now();
  runP2PMatching(reader_output.data);
  t2 = std::chrono::high_resolution_clock::now();
  double match_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  std::cout << "[Timer] P2P matching: " << match_ms << " ms" << std::endl;
  std::cout << std::endl;

  // Step 3: Collective Grouping on CPU
  std::cout << "=== Step 3: Collective Grouping (CPU) ===" << std::endl;
  t1 = std::chrono::high_resolution_clock::now();
  CollectiveGroupCSR csr;
  buildCollectiveGroups(reader_output.data, reader_output.comm_sets, csr);
  t2 = std::chrono::high_resolution_clock::now();
  double group_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  std::cout << "[Timer] Collective grouping: " << group_ms << " ms"
            << std::endl;
  std::cout << std::endl;

  // Step 4: Run 8 Analysis Kernels on GPU
  std::cout << "=== Step 4: Analysis Kernels (GPU) ===" << std::endl;
  t1 = std::chrono::high_resolution_clock::now();
  RawAnalysisOutput raw = runAnalysisKernels(reader_output.data, csr);
  t2 = std::chrono::high_resolution_clock::now();
  double analysis_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  std::cout << "[Timer] Analysis kernels: " << analysis_ms << " ms"
            << std::endl;

  // Print raw counts for diagnostics
  std::cout << "[Analysis] Raw counts: "
            << "LS=" << raw.late_sender.size()
            << " LR=" << raw.late_receiver.size()
            << " BW=" << raw.barrier_wait.size()
            << " BC=" << raw.barrier_completion.size()
            << " ER=" << raw.early_reduce.size()
            << " LB=" << raw.late_broadcast.size()
            << " WN=" << raw.wait_nxn.size()
            << " NC=" << raw.nxn_completion.size()
            << std::endl;
  std::cout << std::endl;

  // Step 5: Compute Statistics
  std::cout << "=== Step 5: Computing Statistics ===" << std::endl;
  t1 = std::chrono::high_resolution_clock::now();
  AllAnalysisResults results;
  results.late_sender = computeStatistics(raw.late_sender);
  results.late_receiver = computeStatistics(raw.late_receiver);
  results.barrier_wait = computeStatistics(raw.barrier_wait);
  results.barrier_completion = computeStatistics(raw.barrier_completion);
  results.early_reduce = computeStatistics(raw.early_reduce);
  results.late_broadcast = computeStatistics(raw.late_broadcast);
  results.wait_nxn = computeStatistics(raw.wait_nxn);
  results.nxn_completion = computeStatistics(raw.nxn_completion);
  t2 = std::chrono::high_resolution_clock::now();
  double stats_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  std::cout << "[Timer] Statistics: " << stats_ms << " ms" << std::endl;
  std::cout << std::endl;

  auto t_total_end = std::chrono::high_resolution_clock::now();
  double total_ms =
      std::chrono::duration<double, std::milli>(t_total_end - t_total_start)
          .count();

  // Print results in same format as TileTrace's integration test
  std::cout << "=== Analysis Results ===" << std::endl;
  printResult("late_sender", results.late_sender);
  printResult("late_receiver", results.late_receiver);
  printResult("barrier_wait", results.barrier_wait);
  printResult("barrier_completion", results.barrier_completion);
  printResult("earlyreduce", results.early_reduce);
  printResult("latebroadcast", results.late_broadcast);
  printResult("wait_nxn", results.wait_nxn);
  printResult("nxn_completion", results.nxn_completion);

  std::cout << std::endl;
  std::cout << "=== Timing Summary ===" << std::endl;
  std::cout << "OTF2 Read:            " << std::fixed << std::setprecision(2)
            << read_ms << " ms" << std::endl;
  std::cout << "P2P Matching:         " << match_ms << " ms" << std::endl;
  std::cout << "Coll. Grouping (CPU): " << group_ms << " ms" << std::endl;
  std::cout << "Analysis (GPU):       " << analysis_ms << " ms" << std::endl;
  std::cout << "Statistics (CPU):     " << stats_ms << " ms" << std::endl;
  std::cout << "Total:                " << total_ms << " ms" << std::endl;

  MPI_Finalize();
  return 0;
}
