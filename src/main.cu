#include "analysis/AnalysisKernels.h"
#include "analysis/Statistics.h"
#include "common/cuda_check.h"
#include "data/AnalysisResults.h"
#include "matching/CollectiveGrouping.h"
#include "matching/P2PMatching.h"
#include "reader/OTF2SoAReader.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
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

// ============================================================
// Architecture C: Adaptive Batch Streaming
// ============================================================
// Instead of gathering all data to rank 0 (O(N) memory), process
// K ranks' data per GPU batch. K is adaptively computed based on
// available VRAM. For small traces K=P (all fit), for large traces
// K=1 (pure streaming). Memory bounded by O(K*N/P), not O(N).

// Send local SoA + CSR data to rank 0 via MPI point-to-point.
static void sendLocalData(const TraceDataSoA &data,
                          const CollectiveGroupCSR &csr) {
  int count = (int)data.count;
  MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  if (count == 0) {
    // Still send CSR header (may have 0 groups)
    int csr_header[2] = {0, 0};
    MPI_Send(csr_header, 2, MPI_INT, 0, 15, MPI_COMM_WORLD);
    return;
  }

  // Send enum arrays as int (portable across platforms)
  {
    std::vector<int> ev_buf(count), type_buf(count);
    for (int i = 0; i < count; i++) {
      ev_buf[i] = (int)data.events[i];
      type_buf[i] = (int)data.types[i];
    }
    MPI_Send(ev_buf.data(), count, MPI_INT, 0, 1, MPI_COMM_WORLD);
    MPI_Send(type_buf.data(), count, MPI_INT, 0, 2, MPI_COMM_WORLD);
  }

  MPI_Send(data.timestamps, count, MPI_UINT64_T, 0, 3, MPI_COMM_WORLD);
  MPI_Send(data.end_timestamps, count, MPI_UINT64_T, 0, 4, MPI_COMM_WORLD);
  MPI_Send(data.pids, count, MPI_UINT32_T, 0, 5, MPI_COMM_WORLD);
  MPI_Send(data.tids, count, MPI_UINT32_T, 0, 6, MPI_COMM_WORLD);
  MPI_Send(data.replay_pids, count, MPI_UINT32_T, 0, 7, MPI_COMM_WORLD);
  MPI_Send(data.srcs, count, MPI_UINT32_T, 0, 8, MPI_COMM_WORLD);
  MPI_Send(data.dsts, count, MPI_UINT32_T, 0, 9, MPI_COMM_WORLD);
  MPI_Send(data.tags, count, MPI_UINT32_T, 0, 10, MPI_COMM_WORLD);
  MPI_Send(data.roots, count, MPI_UINT32_T, 0, 11, MPI_COMM_WORLD);
  MPI_Send(data.indices, count, MPI_UINT32_T, 0, 12, MPI_COMM_WORLD);
  MPI_Send(data.match_partner, count, MPI_INT32_T, 0, 13, MPI_COMM_WORLD);
  MPI_Send(data.coll_group_id, count, MPI_INT32_T, 0, 14, MPI_COMM_WORLD);

  // Send CSR
  int csr_header[2] = {(int)csr.num_groups, (int)csr.total_members};
  MPI_Send(csr_header, 2, MPI_INT, 0, 15, MPI_COMM_WORLD);
  if (csr.num_groups > 0) {
    MPI_Send(csr.offsets, (int)csr.num_groups + 1, MPI_INT32_T, 0, 16,
             MPI_COMM_WORLD);
    MPI_Send(csr.members, (int)csr.total_members, MPI_INT32_T, 0, 17,
             MPI_COMM_WORLD);
    std::vector<int> gt_buf(csr.num_groups);
    for (size_t i = 0; i < csr.num_groups; i++)
      gt_buf[i] = (int)csr.group_types[i];
    MPI_Send(gt_buf.data(), (int)csr.num_groups, MPI_INT, 0, 18,
             MPI_COMM_WORLD);
    MPI_Send(csr.group_roots, (int)csr.num_groups, MPI_UINT32_T, 0, 19,
             MPI_COMM_WORLD);
  }
}

// Receive SoA + CSR data from a specific rank via MPI point-to-point.
static void recvRankData(TraceDataSoA &data, CollectiveGroupCSR &csr,
                         int src) {
  int count;
  MPI_Recv(&count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (count == 0) {
    data.count = 0;
    int csr_header[2];
    MPI_Recv(csr_header, 2, MPI_INT, src, 15, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    return;
  }

  data.allocate(count);
  data.count = count;

  {
    std::vector<int> ev_buf(count), type_buf(count);
    MPI_Recv(ev_buf.data(), count, MPI_INT, src, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Recv(type_buf.data(), count, MPI_INT, src, 2, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    for (int i = 0; i < count; i++) {
      data.events[i] = (event_t)ev_buf[i];
      data.types[i] = (event_type_t)type_buf[i];
    }
  }

  MPI_Recv(data.timestamps, count, MPI_UINT64_T, src, 3, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(data.end_timestamps, count, MPI_UINT64_T, src, 4, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(data.pids, count, MPI_UINT32_T, src, 5, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(data.tids, count, MPI_UINT32_T, src, 6, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(data.replay_pids, count, MPI_UINT32_T, src, 7, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(data.srcs, count, MPI_UINT32_T, src, 8, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(data.dsts, count, MPI_UINT32_T, src, 9, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(data.tags, count, MPI_UINT32_T, src, 10, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(data.roots, count, MPI_UINT32_T, src, 11, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(data.indices, count, MPI_UINT32_T, src, 12, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(data.match_partner, count, MPI_INT32_T, src, 13, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(data.coll_group_id, count, MPI_INT32_T, src, 14, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  // Recv CSR
  int csr_header[2];
  MPI_Recv(csr_header, 2, MPI_INT, src, 15, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  csr.num_groups = csr_header[0];
  csr.total_members = csr_header[1];
  if (csr.num_groups > 0) {
    csr.offsets = (int32_t *)malloc((csr.num_groups + 1) * sizeof(int32_t));
    csr.members = (int32_t *)malloc(csr.total_members * sizeof(int32_t));
    csr.group_types = (event_t *)malloc(csr.num_groups * sizeof(event_t));
    csr.group_roots = (id_t *)malloc(csr.num_groups * sizeof(id_t));

    MPI_Recv(csr.offsets, (int)csr.num_groups + 1, MPI_INT32_T, src, 16,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(csr.members, (int)csr.total_members, MPI_INT32_T, src, 17,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::vector<int> gt_buf(csr.num_groups);
    MPI_Recv(gt_buf.data(), (int)csr.num_groups, MPI_INT, src, 18,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (size_t i = 0; i < csr.num_groups; i++)
      csr.group_types[i] = (event_t)gt_buf[i];
    MPI_Recv(csr.group_roots, (int)csr.num_groups, MPI_UINT32_T, src, 19,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

// Merge K ranks' data into a single batch buffer with index remapping.
// Each rank's match_partner indices and CSR member indices are remapped
// to batch-global offsets.
static void mergeBatchData(std::vector<TraceDataSoA> &rank_data,
                           std::vector<CollectiveGroupCSR> &rank_csr, int k,
                           TraceDataSoA &batch_soa,
                           CollectiveGroupCSR &batch_csr) {
  std::vector<size_t> soa_offsets(k);
  size_t total_events = 0;
  for (int i = 0; i < k; i++) {
    soa_offsets[i] = total_events;
    total_events += rank_data[i].count;
  }

  if (total_events == 0)
    return;

  batch_soa.allocate(total_events);
  batch_soa.count = total_events;

  for (int r = 0; r < k; r++) {
    size_t off = soa_offsets[r];
    size_t cnt = rank_data[r].count;
    if (cnt == 0)
      continue;

    memcpy(batch_soa.events + off, rank_data[r].events,
           cnt * sizeof(event_t));
    memcpy(batch_soa.types + off, rank_data[r].types,
           cnt * sizeof(event_type_t));
    memcpy(batch_soa.timestamps + off, rank_data[r].timestamps,
           cnt * sizeof(timestamp_t));
    memcpy(batch_soa.end_timestamps + off, rank_data[r].end_timestamps,
           cnt * sizeof(timestamp_t));
    memcpy(batch_soa.pids + off, rank_data[r].pids, cnt * sizeof(id_t));
    memcpy(batch_soa.tids + off, rank_data[r].tids, cnt * sizeof(id_t));
    memcpy(batch_soa.replay_pids + off, rank_data[r].replay_pids,
           cnt * sizeof(id_t));
    memcpy(batch_soa.srcs + off, rank_data[r].srcs, cnt * sizeof(id_t));
    memcpy(batch_soa.dsts + off, rank_data[r].dsts, cnt * sizeof(id_t));
    memcpy(batch_soa.tags + off, rank_data[r].tags, cnt * sizeof(id_t));
    memcpy(batch_soa.roots + off, rank_data[r].roots, cnt * sizeof(id_t));
    memcpy(batch_soa.indices + off, rank_data[r].indices,
           cnt * sizeof(id_t));

    // Remap match_partner: add batch-local offset
    for (size_t i = 0; i < cnt; i++) {
      int32_t mp = rank_data[r].match_partner[i];
      batch_soa.match_partner[off + i] =
          (mp >= 0) ? (int32_t)(mp + (int32_t)off) : -1;
    }

    memcpy(batch_soa.coll_group_id + off, rank_data[r].coll_group_id,
           cnt * sizeof(int32_t));
  }

  // Merge CSR
  size_t total_groups = 0, total_members = 0;
  std::vector<size_t> group_offsets(k), member_offsets(k);
  for (int i = 0; i < k; i++) {
    group_offsets[i] = total_groups;
    member_offsets[i] = total_members;
    total_groups += rank_csr[i].num_groups;
    total_members += rank_csr[i].total_members;
  }

  batch_csr.num_groups = total_groups;
  batch_csr.total_members = total_members;

  if (total_groups > 0) {
    batch_csr.offsets =
        (int32_t *)malloc((total_groups + 1) * sizeof(int32_t));
    batch_csr.members = (int32_t *)malloc(total_members * sizeof(int32_t));
    batch_csr.group_types = (event_t *)malloc(total_groups * sizeof(event_t));
    batch_csr.group_roots = (id_t *)malloc(total_groups * sizeof(id_t));

    for (int r = 0; r < k; r++) {
      size_t goff = group_offsets[r];
      size_t moff = member_offsets[r];
      size_t soff = soa_offsets[r];
      size_t ng = rank_csr[r].num_groups;
      size_t nm = rank_csr[r].total_members;

      if (ng == 0)
        continue;

      // Shift CSR offsets by accumulated member offset
      for (size_t g = 0; g < ng; g++) {
        batch_csr.offsets[goff + g] =
            (int32_t)(rank_csr[r].offsets[g] + (int32_t)moff);
      }

      // Shift CSR member indices by SoA event offset
      for (size_t m = 0; m < nm; m++) {
        batch_csr.members[moff + m] =
            (int32_t)(rank_csr[r].members[m] + (int32_t)soff);
      }

      memcpy(batch_csr.group_types + goff, rank_csr[r].group_types,
             ng * sizeof(event_t));
      memcpy(batch_csr.group_roots + goff, rank_csr[r].group_roots,
             ng * sizeof(id_t));
    }
    batch_csr.offsets[total_groups] = (int32_t)total_members;
  }
}

// Accumulate per-batch raw analysis results into the global output.
static void accumulateResults(RawAnalysisOutput &total,
                              const RawAnalysisOutput &batch) {
  auto append = [](std::vector<double> &dst, const std::vector<double> &src) {
    dst.insert(dst.end(), src.begin(), src.end());
  };
  append(total.late_sender, batch.late_sender);
  append(total.late_receiver, batch.late_receiver);
  append(total.barrier_wait, batch.barrier_wait);
  append(total.barrier_completion, batch.barrier_completion);
  append(total.early_reduce, batch.early_reduce);
  append(total.late_broadcast, batch.late_broadcast);
  append(total.wait_nxn, batch.wait_nxn);
  append(total.nxn_completion, batch.nxn_completion);
  total.h2d_ms += batch.h2d_ms;
  total.p2p_kernel_ms += batch.p2p_kernel_ms;
  total.coll_kernel_ms += batch.coll_kernel_ms;
}

// Compute adaptive batch size K based on available VRAM and per-rank data.
// Returns the max number of ranks whose data fits in one GPU launch.
static int computeBatchSize(const std::vector<int> &all_counts, int nprocs) {
  size_t vram_free = 0, vram_total = 0;
  cudaMemGetInfo(&vram_free, &vram_total);

  // Use 80% of free VRAM as budget (leave room for CUDA runtime overhead)
  size_t vram_budget = (size_t)(vram_free * 0.8);

  // Per-event VRAM usage:
  //   Input: events(4) + timestamps(8) + end_timestamps(8) + match_partner(4)
  //          + pids(4) + roots(4) = 32 bytes
  //   Output: 2 P2P output arrays (8 bytes each) = 16 bytes worst case
  //   Total: ~48 bytes/event
  const size_t BYTES_PER_EVENT = 48;

  int max_rank_events = 0;
  for (int i = 0; i < nprocs; i++) {
    if (all_counts[i] > max_rank_events)
      max_rank_events = all_counts[i];
  }

  if (max_rank_events == 0)
    return nprocs;

  size_t max_events_per_batch = vram_budget / BYTES_PER_EVENT;
  int K = (int)std::min((size_t)nprocs,
                        max_events_per_batch / (size_t)max_rank_events);
  return std::max(K, 1);
}

// Run adaptive batch streaming analysis. All ranks participate:
// - Non-zero ranks send their local data to rank 0
// - Rank 0 receives data in batches of K ranks, processes on GPU,
//   and accumulates results across batches
// Returns the accumulated RawAnalysisOutput (meaningful only on rank 0).
static RawAnalysisOutput
streamBatchAnalysis(TraceDataSoA &local_data, CollectiveGroupCSR &local_csr,
                    int rank, int nprocs) {
  RawAnalysisOutput output;

  // All ranks participate: gather event counts to rank 0
  int local_count = (int)local_data.count;
  std::vector<int> all_counts(nprocs);
  MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  if (rank != 0) {
    // Non-zero ranks: send data to rank 0, then done
    sendLocalData(local_data, local_csr);
    return output;
  }

  // ---- Rank 0 only below ----
  int K = computeBatchSize(all_counts, nprocs);
  int total_events = 0;
  for (int i = 0; i < nprocs; i++)
    total_events += all_counts[i];
  int num_batches = (nprocs + K - 1) / K;

  std::cout << "[Batch] VRAM-based batch size K=" << K << " ranks, "
            << num_batches << " batch(es) for " << nprocs << " ranks ("
            << total_events << " total events)" << std::endl;

  // Process each batch
  for (int batch_start = 0; batch_start < nprocs; batch_start += K) {
    int batch_end = std::min(batch_start + K, nprocs);
    int batch_k = batch_end - batch_start;

    // Collect data for ranks in this batch
    std::vector<TraceDataSoA> batch_rank_data(batch_k);
    std::vector<CollectiveGroupCSR> batch_rank_csr(batch_k);

    for (int bi = 0; bi < batch_k; bi++) {
      int src_rank = batch_start + bi;
      if (src_rank == 0) {
        // Use rank 0's local data directly (move to avoid copy)
        batch_rank_data[bi] = std::move(local_data);
        batch_rank_csr[bi] = std::move(local_csr);
      } else {
        recvRankData(batch_rank_data[bi], batch_rank_csr[bi], src_rank);
      }
    }

    // Run GPU analysis on this batch
    RawAnalysisOutput batch_raw;
    if (batch_k == 1) {
      // Single rank in batch: no merge needed, analyze directly
      batch_raw =
          runAnalysisKernels(batch_rank_data[0], batch_rank_csr[0]);
    } else {
      // Multiple ranks: merge with index remapping, then analyze
      TraceDataSoA batch_soa;
      CollectiveGroupCSR batch_csr;
      mergeBatchData(batch_rank_data, batch_rank_csr, batch_k, batch_soa,
                     batch_csr);
      batch_raw = runAnalysisKernels(batch_soa, batch_csr);
    }

    accumulateResults(output, batch_raw);

    if (num_batches > 1) {
      int batch_events = 0;
      for (int bi = 0; bi < batch_k; bi++)
        batch_events += all_counts[batch_start + bi];
      std::cout << "[Batch] Batch " << (batch_start / K + 1) << "/"
                << num_batches << ": ranks " << batch_start << "-"
                << (batch_end - 1) << " (" << batch_events
                << " events), H2D=" << std::fixed << std::setprecision(2)
                << batch_raw.h2d_ms << " ms, P2P=" << batch_raw.p2p_kernel_ms
                << " ms, Coll=" << batch_raw.coll_kernel_ms << " ms"
                << std::endl;
    }
  }

  return output;
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
              << " (distributed reading, adaptive batch streaming GPU analysis)"
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

  // Step 4+5: Adaptive Batch Streaming Analysis (Architecture C)
  // Replaces separate gather + GPU analysis steps. Processes K ranks per
  // GPU batch, where K is adaptively computed from available VRAM.
  // - Single rank: analyze local data directly on GPU
  // - Multi rank: non-zero ranks send data; rank 0 receives in batches,
  //   merges with index remapping, and runs GPU kernels per batch.
  RawAnalysisOutput raw;
  double analysis_ms = 0;
  if (mpi_rank == 0)
    std::cout << "=== Step 4: Adaptive Batch Streaming Analysis (GPU) ==="
              << std::endl;
  t1 = std::chrono::high_resolution_clock::now();

  if (mpi_size == 1) {
    // Single rank: analyze directly, no MPI communication needed
    raw = runAnalysisKernels(reader_output.data, csr);
  } else {
    // Multi-rank: batch streaming
    raw = streamBatchAnalysis(reader_output.data, csr, mpi_rank, mpi_size);
  }

  t2 = std::chrono::high_resolution_clock::now();
  analysis_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  if (mpi_rank == 0) {
    std::cout << "[Timer] Batch streaming analysis: " << analysis_ms << " ms"
              << std::endl;
    std::cout << "[Analysis] Sub-phases: H2D=" << std::fixed
              << std::setprecision(2) << raw.h2d_ms
              << " ms, P2P kernel=" << raw.p2p_kernel_ms
              << " ms, Coll kernels=" << raw.coll_kernel_ms << " ms"
              << std::endl;
    std::cout << std::endl;
  }

  // Step 5: Compute Statistics (rank 0 only)
  if (mpi_rank == 0) {
    std::cout << "=== Step 5: Computing Statistics ===" << std::endl;
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
    std::cout << "Batch Analysis (GPU): " << analysis_ms << " ms" << std::endl;
    std::cout << "Statistics:           " << stats_ms << " ms" << std::endl;
    std::cout << "Total:                " << total_ms << " ms" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
