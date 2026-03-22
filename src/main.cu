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
#include <numeric>
#include <string>
#include <vector>

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

// Gather SoA trace data from all ranks to rank 0, remapping indices.
// On rank 0, merged_data and merged_csr are filled with the combined data.
static void gatherTraceData(TraceDataSoA &local_data,
                            CollectiveGroupCSR &local_csr,
                            TraceDataSoA &merged_data,
                            CollectiveGroupCSR &merged_csr, int rank,
                            int nprocs) {
  // 1. Gather event counts from all ranks
  int local_count = (int)local_data.count;
  std::vector<int> counts(nprocs), soa_displs(nprocs);
  MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  // Compute prefix sums (SoA index offsets per rank)
  std::vector<int> soa_prefix(nprocs, 0);
  int total_events = 0;
  if (rank == 0) {
    for (int i = 0; i < nprocs; i++) {
      soa_displs[i] = total_events;
      soa_prefix[i] = total_events;
      total_events += counts[i];
    }
  }

  // Broadcast prefix sums so each rank knows its offset for remapping
  MPI_Bcast(soa_prefix.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);
  int my_offset = soa_prefix[rank];

  // 2. Remap match_partner indices: add this rank's offset
  // Unmatched entries (-1) stay as -1
  for (size_t i = 0; i < local_data.count; i++) {
    if (local_data.match_partner[i] >= 0) {
      local_data.match_partner[i] += my_offset;
    }
  }

  // 3. Gather SoA arrays to rank 0
  if (rank == 0)
    merged_data.allocate(total_events);

  // Helper: gatherv for a typed array
  auto gatherArray_u32 = [&](id_t *local, id_t *global) {
    MPI_Gatherv(local, local_count, MPI_UINT32_T,
                rank == 0 ? global : nullptr, counts.data(), soa_displs.data(),
                MPI_UINT32_T, 0, MPI_COMM_WORLD);
  };
  auto gatherArray_u64 = [&](timestamp_t *local, timestamp_t *global) {
    MPI_Gatherv(local, local_count, MPI_UINT64_T,
                rank == 0 ? global : nullptr, counts.data(), soa_displs.data(),
                MPI_UINT64_T, 0, MPI_COMM_WORLD);
  };
  auto gatherArray_i32 = [&](int32_t *local, int32_t *global) {
    MPI_Gatherv(local, local_count, MPI_INT32_T,
                rank == 0 ? global : nullptr, counts.data(), soa_displs.data(),
                MPI_INT32_T, 0, MPI_COMM_WORLD);
  };

  // event_t and event_type_t are small enums; transfer as int
  {
    std::vector<int> local_events(local_count), local_types(local_count);
    for (int i = 0; i < local_count; i++) {
      local_events[i] = (int)local_data.events[i];
      local_types[i] = (int)local_data.types[i];
    }
    std::vector<int> g_events, g_types;
    if (rank == 0) {
      g_events.resize(total_events);
      g_types.resize(total_events);
    }
    MPI_Gatherv(local_events.data(), local_count, MPI_INT,
                rank == 0 ? g_events.data() : nullptr, counts.data(),
                soa_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_types.data(), local_count, MPI_INT,
                rank == 0 ? g_types.data() : nullptr, counts.data(),
                soa_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      for (int i = 0; i < total_events; i++) {
        merged_data.events[i] = (event_t)g_events[i];
        merged_data.types[i] = (event_type_t)g_types[i];
      }
    }
  }

  gatherArray_u64(local_data.timestamps, merged_data.timestamps);
  gatherArray_u64(local_data.end_timestamps, merged_data.end_timestamps);
  gatherArray_u32(local_data.pids, merged_data.pids);
  gatherArray_u32(local_data.tids, merged_data.tids);
  gatherArray_u32(local_data.replay_pids, merged_data.replay_pids);
  gatherArray_u32(local_data.srcs, merged_data.srcs);
  gatherArray_u32(local_data.dsts, merged_data.dsts);
  gatherArray_u32(local_data.tags, merged_data.tags);
  gatherArray_u32(local_data.roots, merged_data.roots);
  gatherArray_u32(local_data.indices, merged_data.indices);
  gatherArray_i32(local_data.match_partner, merged_data.match_partner);
  gatherArray_i32(local_data.coll_group_id, merged_data.coll_group_id);

  if (rank == 0)
    merged_data.count = total_events;

  // 4. Gather CSR data to rank 0
  int local_num_groups = (int)local_csr.num_groups;
  int local_total_members = (int)local_csr.total_members;
  std::vector<int> group_counts(nprocs), member_counts(nprocs);
  std::vector<int> group_displs(nprocs), member_displs(nprocs);

  MPI_Gather(&local_num_groups, 1, MPI_INT, group_counts.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);
  MPI_Gather(&local_total_members, 1, MPI_INT, member_counts.data(), 1,
             MPI_INT, 0, MPI_COMM_WORLD);

  int total_groups = 0, total_members = 0;
  if (rank == 0) {
    for (int i = 0; i < nprocs; i++) {
      group_displs[i] = total_groups;
      total_groups += group_counts[i];
      member_displs[i] = total_members;
      total_members += member_counts[i];
    }
  }

  // Remap CSR members: add this rank's SoA offset
  for (size_t i = 0; i < local_csr.total_members; i++) {
    local_csr.members[i] += my_offset;
  }

  // Gather members
  if (rank == 0) {
    merged_csr.total_members = total_members;
    merged_csr.num_groups = total_groups;
    merged_csr.members = (int32_t *)malloc(total_members * sizeof(int32_t));
    merged_csr.offsets =
        (int32_t *)malloc((total_groups + 1) * sizeof(int32_t));
    merged_csr.group_types = (event_t *)malloc(total_groups * sizeof(event_t));
    merged_csr.group_roots = (id_t *)malloc(total_groups * sizeof(id_t));
  }

  MPI_Gatherv(local_csr.members, local_total_members, MPI_INT32_T,
              rank == 0 ? merged_csr.members : nullptr, member_counts.data(),
              member_displs.data(), MPI_INT32_T, 0, MPI_COMM_WORLD);

  // Gather group_types and group_roots
  {
    std::vector<int> local_gtypes(local_num_groups);
    for (int i = 0; i < local_num_groups; i++)
      local_gtypes[i] = (int)local_csr.group_types[i];
    std::vector<int> g_gtypes;
    if (rank == 0)
      g_gtypes.resize(total_groups);
    MPI_Gatherv(local_gtypes.data(), local_num_groups, MPI_INT,
                rank == 0 ? g_gtypes.data() : nullptr, group_counts.data(),
                group_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      for (int i = 0; i < total_groups; i++)
        merged_csr.group_types[i] = (event_t)g_gtypes[i];
    }
  }
  MPI_Gatherv(local_csr.group_roots, local_num_groups, MPI_UINT32_T,
              rank == 0 ? merged_csr.group_roots : nullptr,
              group_counts.data(), group_displs.data(), MPI_UINT32_T, 0,
              MPI_COMM_WORLD);

  // Gather and reconstruct offsets:
  // Each rank's offsets are local to its members. On rank 0, shift by
  // accumulated member offset. Only gather the first num_groups entries
  // (not the trailing sentinel), then reconstruct the sentinel.
  {
    // Shift local offsets by member displacement before sending
    std::vector<int32_t> shifted_offsets(local_num_groups);
    int member_base = 0;
    // Each rank needs to know its member displacement
    MPI_Scatter(member_displs.data(), 1, MPI_INT, &member_base, 1, MPI_INT, 0,
                MPI_COMM_WORLD);
    for (int i = 0; i < local_num_groups; i++)
      shifted_offsets[i] = local_csr.offsets[i] + member_base;

    MPI_Gatherv(shifted_offsets.data(), local_num_groups, MPI_INT32_T,
                rank == 0 ? merged_csr.offsets : nullptr, group_counts.data(),
                group_displs.data(), MPI_INT32_T, 0, MPI_COMM_WORLD);

    if (rank == 0)
      merged_csr.offsets[total_groups] = total_members;
  }
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
    std::cout << "MPI ranks: " << mpi_size
              << " (distributed reading, single-GPU analysis)" << std::endl;
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

  // Step 4: Gather all data to rank 0 (SoA + CSR with index remapping)
  if (mpi_rank == 0)
    std::cout << "=== Step 4: Gathering Trace Data to Rank 0 ===" << std::endl;
  t1 = std::chrono::high_resolution_clock::now();

  TraceDataSoA merged_data;
  CollectiveGroupCSR merged_csr;

  if (mpi_size > 1) {
    gatherTraceData(reader_output.data, csr, merged_data, merged_csr, mpi_rank,
                    mpi_size);
  }

  // Use merged data on rank 0, or local data if single rank
  TraceDataSoA &analysis_data =
      (mpi_size > 1 && mpi_rank == 0) ? merged_data : reader_output.data;
  CollectiveGroupCSR &analysis_csr =
      (mpi_size > 1 && mpi_rank == 0) ? merged_csr : csr;

  t2 = std::chrono::high_resolution_clock::now();
  double gather_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  if (mpi_rank == 0) {
    std::cout << "[Timer] Data gathering: " << gather_ms << " ms" << std::endl;
    std::cout << "[Gather] Merged events: " << analysis_data.count
              << ", groups: " << analysis_csr.num_groups
              << ", total members: " << analysis_csr.total_members << std::endl;
    std::cout << std::endl;
  }

  // Step 5: Run 8 Analysis Kernels on GPU (rank 0 only, single GPU)
  RawAnalysisOutput raw;
  double analysis_ms = 0;
  if (mpi_rank == 0) {
    std::cout << "=== Step 5: Analysis Kernels (GPU, rank 0 only) ==="
              << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    raw = runAnalysisKernels(analysis_data, analysis_csr);
    t2 = std::chrono::high_resolution_clock::now();
    analysis_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "[Timer] Analysis kernels: " << analysis_ms << " ms"
              << std::endl;
    std::cout << "[Analysis] Sub-phases: H2D=" << std::fixed
              << std::setprecision(2) << raw.h2d_ms
              << " ms, P2P kernel=" << raw.p2p_kernel_ms
              << " ms, Coll kernels=" << raw.coll_kernel_ms << " ms"
              << std::endl;
    std::cout << std::endl;
  }

  // Step 6: Compute Statistics (rank 0 only)
  if (mpi_rank == 0) {
    std::cout << "=== Step 6: Computing Statistics ===" << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    AllAnalysisResults results;
    results.late_sender = computeStatistics(raw.late_sender);
    results.late_receiver = computeStatistics(raw.late_receiver);
    results.barrier_wait = computeStatistics(raw.barrier_wait);
    results.barrier_completion =
        computeStatistics(raw.barrier_completion);
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
    std::cout << "Data Gathering:       " << gather_ms << " ms" << std::endl;
    std::cout << "Analysis (GPU):       " << analysis_ms << " ms" << std::endl;
    std::cout << "Statistics:           " << stats_ms << " ms" << std::endl;
    std::cout << "Total:                " << total_ms << " ms" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
