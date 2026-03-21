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

// Gather raw analysis duration vectors from all ranks to rank 0
static void gatherRawResults(const RawAnalysisOutput &local,
                             RawAnalysisOutput &global, int rank, int nprocs) {
  auto gatherVec = [&](const std::vector<double> &loc,
                       std::vector<double> &glob) {
    int local_sz = (int)loc.size();
    std::vector<int> sizes(nprocs), displs(nprocs);
    MPI_Gather(&local_sz, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);

    int total = 0;
    if (rank == 0) {
      for (int i = 0; i < nprocs; i++) {
        displs[i] = total;
        total += sizes[i];
      }
      glob.resize(total);
    }

    MPI_Gatherv(loc.data(), local_sz, MPI_DOUBLE,
                rank == 0 ? glob.data() : nullptr, sizes.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
  };

  gatherVec(local.late_sender, global.late_sender);
  gatherVec(local.late_receiver, global.late_receiver);
  gatherVec(local.barrier_wait, global.barrier_wait);
  gatherVec(local.barrier_completion, global.barrier_completion);
  gatherVec(local.early_reduce, global.early_reduce);
  gatherVec(local.late_broadcast, global.late_broadcast);
  gatherVec(local.wait_nxn, global.wait_nxn);
  gatherVec(local.nxn_completion, global.nxn_completion);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (argc < 2) {
    if (mpi_rank == 0)
      std::cerr << "Usage: " << argv[0] << " <path/to/traces.otf2>"
                << std::endl;
    MPI_Finalize();
    return 1;
  }

  std::string trace_path = argv[1];

  if (mpi_rank == 0) {
    printGpuInfo();
    std::cout << std::endl;
    std::cout << "Trace file: " << trace_path << std::endl;
#ifdef USE_SCALASCA_TIMESTAMPS
    std::cout << "Timestamp mode: SCALASCA (Enter-region timestamps)"
              << std::endl;
#else
    std::cout << "Timestamp mode: TILETRACE (point-event timestamps)"
              << std::endl;
#endif
    std::cout << "MPI ranks: " << mpi_size << " (distributed reading)"
              << std::endl;
    std::cout << std::endl;
  }

  auto t_total_start = std::chrono::high_resolution_clock::now();

  // Step 1: Read OTF2 trace into SoA (distributed two-pass)
  if (mpi_rank == 0)
    std::cout << "=== Step 1: Reading OTF2 trace (distributed) ==="
              << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();
  ReaderOutput reader_output = readOTF2Trace(trace_path);
  auto t2 = std::chrono::high_resolution_clock::now();
  double read_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();

  if (mpi_rank == 0) {
    std::cout << "[Timer] OTF2 read: " << read_ms << " ms" << std::endl;
    std::cout << std::endl;
  }

  // Step 2: P2P Matching (local, each rank independently)
  if (mpi_rank == 0)
    std::cout << "=== Step 2: P2P Matching (local) ===" << std::endl;
  t1 = std::chrono::high_resolution_clock::now();
  runP2PMatching(reader_output.data);
  t2 = std::chrono::high_resolution_clock::now();
  double match_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  if (mpi_rank == 0) {
    std::cout << "[Timer] P2P matching: " << match_ms << " ms" << std::endl;
    std::cout << std::endl;
  }

  // Step 3: Collective Grouping (local, each rank independently)
  if (mpi_rank == 0)
    std::cout << "=== Step 3: Collective Grouping (local) ===" << std::endl;
  t1 = std::chrono::high_resolution_clock::now();
  CollectiveGroupCSR csr;
  buildCollectiveGroups(reader_output.data, reader_output.comm_sets, csr);
  t2 = std::chrono::high_resolution_clock::now();
  double group_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  if (mpi_rank == 0) {
    std::cout << "[Timer] Collective grouping: " << group_ms << " ms"
              << std::endl;
    std::cout << std::endl;
  }

  // Step 4: Run 8 Analysis Kernels on GPU (local, each rank independently)
  if (mpi_rank == 0)
    std::cout << "=== Step 4: Analysis Kernels (GPU, local) ===" << std::endl;
  t1 = std::chrono::high_resolution_clock::now();
  RawAnalysisOutput raw = runAnalysisKernels(reader_output.data, csr);
  t2 = std::chrono::high_resolution_clock::now();
  double analysis_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  if (mpi_rank == 0) {
    std::cout << "[Timer] Analysis kernels: " << analysis_ms << " ms"
              << std::endl;
    std::cout << "[Analysis] Sub-phases: H2D=" << std::fixed
              << std::setprecision(2) << raw.h2d_ms
              << " ms, P2P kernel=" << raw.p2p_kernel_ms
              << " ms, Coll kernels=" << raw.coll_kernel_ms << " ms"
              << std::endl;
    std::cout << std::endl;
  }

  // Step 5: Gather results from all ranks to rank 0
  if (mpi_rank == 0)
    std::cout << "=== Step 5: Gathering Results ===" << std::endl;
  t1 = std::chrono::high_resolution_clock::now();

  RawAnalysisOutput global_raw;
  if (mpi_size > 1) {
    gatherRawResults(raw, global_raw, mpi_rank, mpi_size);
  } else {
    global_raw = std::move(raw);
  }

  t2 = std::chrono::high_resolution_clock::now();
  double gather_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  if (mpi_rank == 0) {
    std::cout << "[Timer] Result gathering: " << gather_ms << " ms"
              << std::endl;
    std::cout << std::endl;
  }

  // Step 6: Compute Statistics (rank 0 only)
  if (mpi_rank == 0) {
    std::cout << "=== Step 6: Computing Statistics ===" << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    AllAnalysisResults results;
    results.late_sender = computeStatistics(global_raw.late_sender);
    results.late_receiver = computeStatistics(global_raw.late_receiver);
    results.barrier_wait = computeStatistics(global_raw.barrier_wait);
    results.barrier_completion =
        computeStatistics(global_raw.barrier_completion);
    results.early_reduce = computeStatistics(global_raw.early_reduce);
    results.late_broadcast = computeStatistics(global_raw.late_broadcast);
    results.wait_nxn = computeStatistics(global_raw.wait_nxn);
    results.nxn_completion = computeStatistics(global_raw.nxn_completion);
    t2 = std::chrono::high_resolution_clock::now();
    double stats_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "[Timer] Statistics: " << stats_ms << " ms" << std::endl;
    std::cout << std::endl;

    auto t_total_end = std::chrono::high_resolution_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(t_total_end - t_total_start)
            .count();

    // Print results
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
    std::cout << "Coll. Grouping:       " << group_ms << " ms" << std::endl;
    std::cout << "Analysis (GPU):       " << analysis_ms << " ms" << std::endl;
    std::cout << "Result Gathering:     " << gather_ms << " ms" << std::endl;
    std::cout << "Statistics:           " << stats_ms << " ms" << std::endl;
    std::cout << "Total:                " << total_ms << " ms" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
