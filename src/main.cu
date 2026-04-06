#include "analysis/AnalysisKernels.h"
#include "analysis/Statistics.h"
#include "analysis/TimestampCorrection.h"
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
#include <sys/mman.h>

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
    MPI_Send(csr.member_bytes_sent, (int)csr.total_members, MPI_UINT64_T, 0, 20,
             MPI_COMM_WORLD);
    MPI_Send(csr.member_bytes_received, (int)csr.total_members, MPI_UINT64_T, 0, 21,
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
    csr.member_bytes_sent = (uint64_t *)malloc(csr.total_members * sizeof(uint64_t));
    csr.member_bytes_received = (uint64_t *)malloc(csr.total_members * sizeof(uint64_t));
    MPI_Recv(csr.member_bytes_sent, (int)csr.total_members, MPI_UINT64_T, src, 20,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(csr.member_bytes_received, (int)csr.total_members, MPI_UINT64_T, src, 21,
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
    batch_csr.member_bytes_sent = (uint64_t *)malloc(total_members * sizeof(uint64_t));
    batch_csr.member_bytes_received = (uint64_t *)malloc(total_members * sizeof(uint64_t));

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
      if (rank_csr[r].member_bytes_sent)
        memcpy(batch_csr.member_bytes_sent + moff, rank_csr[r].member_bytes_sent,
               nm * sizeof(uint64_t));
      if (rank_csr[r].member_bytes_received)
        memcpy(batch_csr.member_bytes_received + moff, rank_csr[r].member_bytes_received,
               nm * sizeof(uint64_t));
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
  total.d2h_ms += batch.d2h_ms;
  total.gpu_alloc_ms += batch.gpu_alloc_ms;
  total.gpu_free_ms += batch.gpu_free_ms;
  total.pin_ms += batch.pin_ms;
  total.unpin_ms += batch.unpin_ms;
  total.batch_prep_ms += batch.batch_prep_ms;
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
  int64_t local_count_64 = (int64_t)local_data.count;
  std::vector<int64_t> all_counts_64(nprocs);
  MPI_Gather(&local_count_64, 1, MPI_INT64_T, all_counts_64.data(), 1,
             MPI_INT64_T, 0, MPI_COMM_WORLD);

  // Convert to int for downstream (per-rank counts fit in int)
  std::vector<int> all_counts(nprocs);
  if (rank == 0) {
    for (int i = 0; i < nprocs; i++)
      all_counts[i] = (int)all_counts_64[i];
  }

  if (rank != 0) {
    // Non-zero ranks: send data to rank 0, then done
    sendLocalData(local_data, local_csr);
    return output;
  }

  // ---- Rank 0 only below ----
  int K = computeBatchSize(all_counts, nprocs);
  int64_t total_events = 0;
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

// ============================================================
// Architecture D: Shared Memory — Bypass MPI Gather
// ============================================================
// When all ranks share physical memory (same node), use MPI-3
// shared memory windows. Each rank writes its SoA data directly
// into a common buffer. Only a fence is needed for synchronization.
// Falls back to Architecture C if ranks span multiple nodes.

struct ShmSoALayout {
  size_t total_bytes;
  size_t events_off, types_off, timestamps_off, end_timestamps_off;
  size_t pids_off, tids_off, replay_pids_off;
  size_t srcs_off, dsts_off, tags_off, roots_off, indices_off;
  size_t match_partner_off, coll_group_id_off;
  size_t leave_recv_ts_off;

  // Page-align offsets so each sub-array can be independently pinned
  // with cudaHostRegister (requires page-aligned ptr and size).
  static size_t alignPage(size_t x) {
    const size_t PAGE = 4096;
    return (x + PAGE - 1) & ~(PAGE - 1);
  }

  void compute(size_t n) {
    size_t off = 0;
    events_off = off;         off += alignPage(n * sizeof(event_t));
    types_off = off;          off += alignPage(n * sizeof(event_type_t));
    timestamps_off = off;     off += alignPage(n * sizeof(timestamp_t));
    end_timestamps_off = off; off += alignPage(n * sizeof(timestamp_t));
    pids_off = off;           off += alignPage(n * sizeof(id_t));
    tids_off = off;           off += alignPage(n * sizeof(id_t));
    replay_pids_off = off;    off += alignPage(n * sizeof(id_t));
    srcs_off = off;           off += alignPage(n * sizeof(id_t));
    dsts_off = off;           off += alignPage(n * sizeof(id_t));
    tags_off = off;           off += alignPage(n * sizeof(id_t));
    roots_off = off;          off += alignPage(n * sizeof(id_t));
    indices_off = off;        off += alignPage(n * sizeof(id_t));
    match_partner_off = off;  off += alignPage(n * sizeof(int32_t));
    coll_group_id_off = off;  off += alignPage(n * sizeof(int32_t));
    leave_recv_ts_off = off;  off += alignPage(n * sizeof(timestamp_t));
    total_bytes = off;
  }
};

// Timing info returned from shared memory analysis, separating
// preprocessing (CPU-side data prep) from GPU analysis.
struct ShmAnalysisResult {
  RawAnalysisOutput raw;
  double preprocess_ms = 0; // fill_soa + p2p + coll + ts_correct + remap + csr_gather + pinning
  double gpu_analysis_ms = 0; // GPU batches only (H2D + kernels + D2H)
  double cleanup_ms = 0;
};

// Set up a non-owning TraceDataSoA pointing at a rank's slice of the shared window.
static void shmSetupLocalSoA(TraceDataSoA &soa, char *base,
                              const ShmSoALayout &layout, size_t offset,
                              size_t count) {
  soa.owns_memory = false;
  soa.capacity = count;
  soa.events = (event_t *)(base + layout.events_off) + offset;
  soa.types = (event_type_t *)(base + layout.types_off) + offset;
  soa.timestamps = (timestamp_t *)(base + layout.timestamps_off) + offset;
  soa.end_timestamps =
      (timestamp_t *)(base + layout.end_timestamps_off) + offset;
  soa.pids = (id_t *)(base + layout.pids_off) + offset;
  soa.tids = (id_t *)(base + layout.tids_off) + offset;
  soa.replay_pids = (id_t *)(base + layout.replay_pids_off) + offset;
  soa.srcs = (id_t *)(base + layout.srcs_off) + offset;
  soa.dsts = (id_t *)(base + layout.dsts_off) + offset;
  soa.tags = (id_t *)(base + layout.tags_off) + offset;
  soa.roots = (id_t *)(base + layout.roots_off) + offset;
  soa.indices = (id_t *)(base + layout.indices_off) + offset;
  soa.match_partner = (int32_t *)(base + layout.match_partner_off) + offset;
  soa.coll_group_id = (int32_t *)(base + layout.coll_group_id_off) + offset;
}

// Pin the 7 GPU-accessed SoA arrays for DMA H2D transfers.
// Returns the number of pinned regions (0 if skipped or failed).
struct PinRegion {
  void *ptr;
  size_t len;
};
static int shmPinHostMemory(char *base, const ShmSoALayout &layout,
                            size_t total_events, PinRegion *regions,
                            int max_regions) {
  const size_t PIN_THRESHOLD = (size_t)10 * 1024 * 1024 * 1024; // 10 GB
  if (total_events == 0 || max_regions < 7)
    return 0;

  regions[0] = {base + layout.events_off,
                ShmSoALayout::alignPage(total_events * sizeof(event_t))};
  regions[1] = {base + layout.timestamps_off,
                ShmSoALayout::alignPage(total_events * sizeof(timestamp_t))};
  regions[2] = {base + layout.end_timestamps_off,
                ShmSoALayout::alignPage(total_events * sizeof(timestamp_t))};
  regions[3] = {base + layout.match_partner_off,
                ShmSoALayout::alignPage(total_events * sizeof(int32_t))};
  regions[4] = {base + layout.pids_off,
                ShmSoALayout::alignPage(total_events * sizeof(id_t))};
  regions[5] = {base + layout.roots_off,
                ShmSoALayout::alignPage(total_events * sizeof(id_t))};
  regions[6] = {base + layout.leave_recv_ts_off,
                ShmSoALayout::alignPage(total_events * sizeof(timestamp_t))};
  int n_pin = 7;

  size_t total_pin_bytes = 0;
  for (int i = 0; i < n_pin; i++)
    total_pin_bytes += regions[i].len;

  if (total_pin_bytes > PIN_THRESHOLD) {
    std::cout << "[SHM] Skipping cudaHostRegister: pin size "
              << (total_pin_bytes / (1024 * 1024))
              << " MB exceeds 10 GB threshold" << std::endl;
    return 0;
  }

  for (int i = 0; i < n_pin; i++) {
    cudaError_t err =
        cudaHostRegister(regions[i].ptr, regions[i].len, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
      std::cerr << "[SHM] Warning: cudaHostRegister failed for region " << i
                << " (" << cudaGetErrorString(err) << ")" << std::endl;
      cudaGetLastError();
      for (int j = 0; j < i; j++)
        cudaHostUnregister(regions[j].ptr);
      return 0;
    }
  }

  std::cout << "[SHM] Pinned " << n_pin << " regions ("
            << (total_pin_bytes / (1024 * 1024)) << " MB) for DMA H2D"
            << std::endl;
  return n_pin;
}

static void shmUnpinHostMemory(PinRegion *regions, int n_pin) {
  for (int i = 0; i < n_pin; i++)
    cudaHostUnregister(regions[i].ptr);
}

// Gather all ranks' CSR data to rank 0 via MPI_Gatherv.
// Each rank adjusts its local CSR indices to global offsets before gathering.
struct MergedCSR {
  std::vector<int32_t> offsets, members;
  std::vector<int> group_types;
  std::vector<id_t> group_roots;
  std::vector<uint64_t> member_bytes_sent, member_bytes_received;
  size_t total_groups = 0, total_members = 0;
  std::vector<int> group_counts, group_displs;
  std::vector<int> member_counts, member_displs;
};

static MergedCSR shmGatherCSR(const CollectiveGroupCSR &local_csr,
                               const std::vector<size_t> &soa_offsets,
                               int rank, int nprocs, MPI_Comm shm_comm) {
  MergedCSR merged;

  int csr_header[2] = {(int)local_csr.num_groups,
                       (int)local_csr.total_members};
  std::vector<int> all_csr_headers(nprocs * 2);
  MPI_Allgather(csr_header, 2, MPI_INT, all_csr_headers.data(), 2, MPI_INT,
                shm_comm);

  merged.group_counts.resize(nprocs);
  merged.group_displs.resize(nprocs + 1);
  merged.member_counts.resize(nprocs);
  merged.member_displs.resize(nprocs + 1);

  for (int i = 0; i < nprocs; i++) {
    merged.group_counts[i] = all_csr_headers[2 * i];
    merged.member_counts[i] = all_csr_headers[2 * i + 1];
    merged.group_displs[i] = (int)merged.total_groups;
    merged.member_displs[i] = (int)merged.total_members;
    merged.total_groups += merged.group_counts[i];
    merged.total_members += merged.member_counts[i];
  }
  merged.group_displs[nprocs] = (int)merged.total_groups;
  merged.member_displs[nprocs] = (int)merged.total_members;

  // Adjust local CSR indices to global
  std::vector<int32_t> adj_members(local_csr.total_members);
  for (size_t i = 0; i < local_csr.total_members; i++)
    adj_members[i] = local_csr.members[i] + (int32_t)soa_offsets[rank];

  std::vector<int32_t> adj_offsets(local_csr.num_groups);
  for (size_t i = 0; i < local_csr.num_groups; i++)
    adj_offsets[i] =
        local_csr.offsets[i] + (int32_t)merged.member_displs[rank];

  std::vector<int> local_gt(local_csr.num_groups);
  for (size_t i = 0; i < local_csr.num_groups; i++)
    local_gt[i] = (int)local_csr.group_types[i];

  if (rank == 0 && merged.total_groups > 0) {
    merged.offsets.resize(merged.total_groups + 1);
    merged.members.resize(merged.total_members);
    merged.group_types.resize(merged.total_groups);
    merged.group_roots.resize(merged.total_groups);
    merged.member_bytes_sent.resize(merged.total_members);
    merged.member_bytes_received.resize(merged.total_members);
  }

  MPI_Gatherv(adj_offsets.data(), (int)local_csr.num_groups, MPI_INT32_T,
              (merged.total_groups > 0 && rank == 0) ? merged.offsets.data()
                                                     : nullptr,
              merged.group_counts.data(), merged.group_displs.data(),
              MPI_INT32_T, 0, shm_comm);

  MPI_Gatherv(
      adj_members.data(), (int)local_csr.total_members, MPI_INT32_T,
      (merged.total_members > 0 && rank == 0) ? merged.members.data() : nullptr,
      merged.member_counts.data(), merged.member_displs.data(), MPI_INT32_T, 0,
      shm_comm);

  MPI_Gatherv(local_gt.data(), (int)local_csr.num_groups, MPI_INT,
              (merged.total_groups > 0 && rank == 0) ? merged.group_types.data()
                                                     : nullptr,
              merged.group_counts.data(), merged.group_displs.data(), MPI_INT,
              0, shm_comm);

  MPI_Gatherv(
      local_csr.group_roots, (int)local_csr.num_groups, MPI_UINT32_T,
      (merged.total_groups > 0 && rank == 0) ? merged.group_roots.data()
                                             : nullptr,
      merged.group_counts.data(), merged.group_displs.data(), MPI_UINT32_T, 0,
      shm_comm);

  // Gather member bytes (indexed by member, not group)
  MPI_Gatherv(
      local_csr.member_bytes_sent, (int)local_csr.total_members, MPI_UINT64_T,
      (merged.total_members > 0 && rank == 0) ? merged.member_bytes_sent.data()
                                              : nullptr,
      merged.member_counts.data(), merged.member_displs.data(), MPI_UINT64_T, 0,
      shm_comm);

  MPI_Gatherv(
      local_csr.member_bytes_received, (int)local_csr.total_members, MPI_UINT64_T,
      (merged.total_members > 0 && rank == 0) ? merged.member_bytes_received.data()
                                              : nullptr,
      merged.member_counts.data(), merged.member_displs.data(), MPI_UINT64_T, 0,
      shm_comm);

  if (rank == 0 && merged.total_groups > 0)
    merged.offsets[merged.total_groups] = (int32_t)merged.total_members;

  return merged;
}

// Run batched GPU analysis on shared memory data. Only rank 0 calls this.
// Creates batch-relative TraceDataSoA views and CSR slices, runs kernels.
// Uses pre-allocated GPU memory pool to avoid per-batch cudaMalloc/cudaFree.
static RawAnalysisOutput
shmRunBatchedGPU(char *base, const ShmSoALayout &layout,
                 const std::vector<size_t> &soa_offsets,
                 const std::vector<int> &all_counts, const MergedCSR &merged,
                 int nprocs, bool global_pinned) {
  RawAnalysisOutput output;

  int K = computeBatchSize(all_counts, nprocs);
  int num_batches = (nprocs + K - 1) / K;

  std::cout << "[SHM] VRAM batch size K=" << K << ", " << num_batches
            << " batch(es)" << std::endl;

  // Compute max batch sizes for pool pre-allocation
  size_t max_batch_events = 0;
  size_t max_batch_members = 0;
  size_t max_batch_groups = 0;
  for (int batch_start = 0; batch_start < nprocs; batch_start += K) {
    int batch_end = std::min(batch_start + K, nprocs);
    size_t batch_events = soa_offsets[batch_end] - soa_offsets[batch_start];
    int batch_group_off = merged.group_displs[batch_start];
    int batch_num_groups = merged.group_displs[batch_end] - batch_group_off;
    int batch_member_off = merged.member_displs[batch_start];
    int batch_total_members = merged.member_displs[batch_end] - batch_member_off;
    if (batch_events > max_batch_events) max_batch_events = batch_events;
    if ((size_t)batch_total_members > max_batch_members) max_batch_members = batch_total_members;
    if ((size_t)batch_num_groups > max_batch_groups) max_batch_groups = batch_num_groups;
  }

  // Pre-allocate GPU memory pool once
  auto tp_alloc0 = std::chrono::high_resolution_clock::now();
  GPUMemoryPool pool;
  pool.allocate(max_batch_events, max_batch_members, max_batch_groups);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  auto tp_alloc1 = std::chrono::high_resolution_clock::now();
  float pool_alloc_ms = (float)std::chrono::duration<double, std::milli>(tp_alloc1 - tp_alloc0).count();
  std::cout << "[SHM] GPU pool allocated: " << (max_batch_events * 48 / (1024 * 1024))
            << " MB device memory, took " << std::fixed << std::setprecision(1)
            << pool_alloc_ms << " ms" << std::endl;

  for (int batch_start = 0; batch_start < nprocs; batch_start += K) {
    int batch_end = std::min(batch_start + K, nprocs);

    size_t batch_soa_off = soa_offsets[batch_start];
    size_t batch_events = soa_offsets[batch_end] - batch_soa_off;
    int batch_group_off = merged.group_displs[batch_start];
    int batch_num_groups = merged.group_displs[batch_end] - batch_group_off;
    int batch_member_off = merged.member_displs[batch_start];
    int batch_total_members = merged.member_displs[batch_end] - batch_member_off;

    if (batch_events == 0)
      continue;

    // --- Batch prep: match_partner remap + CSR construction ---
    auto tp_prep0 = std::chrono::high_resolution_clock::now();

    // Non-owning TraceDataSoA pointing into shared buffer
    TraceDataSoA batch_soa;
    shmSetupLocalSoA(batch_soa, base, layout, batch_soa_off, batch_events);
    batch_soa.count = batch_events;

    // match_partner: adjust global indices to batch-relative
    int32_t *batch_mp = (int32_t *)malloc(batch_events * sizeof(int32_t));
    int32_t *shm_mp =
        (int32_t *)(base + layout.match_partner_off) + batch_soa_off;
    for (size_t i = 0; i < batch_events; i++) {
      int32_t mp = shm_mp[i];
      batch_mp[i] = (mp >= 0) ? (int32_t)(mp - (int32_t)batch_soa_off) : -1;
    }
    batch_soa.match_partner = batch_mp;

    // Build batch CSR with batch-relative indices
    CollectiveGroupCSR batch_csr;
    if (batch_num_groups > 0) {
      batch_csr.num_groups = batch_num_groups;
      batch_csr.total_members = batch_total_members;
      batch_csr.offsets =
          (int32_t *)malloc((batch_num_groups + 1) * sizeof(int32_t));
      batch_csr.members =
          (int32_t *)malloc(batch_total_members * sizeof(int32_t));
      batch_csr.group_types =
          (event_t *)malloc(batch_num_groups * sizeof(event_t));
      batch_csr.group_roots =
          (id_t *)malloc(batch_num_groups * sizeof(id_t));
      batch_csr.member_bytes_sent =
          (uint64_t *)malloc(batch_total_members * sizeof(uint64_t));
      batch_csr.member_bytes_received =
          (uint64_t *)malloc(batch_total_members * sizeof(uint64_t));

      for (int g = 0; g < batch_num_groups; g++)
        batch_csr.offsets[g] =
            merged.offsets[batch_group_off + g] - batch_member_off;
      batch_csr.offsets[batch_num_groups] = batch_total_members;

      for (int m = 0; m < batch_total_members; m++)
        batch_csr.members[m] =
            merged.members[batch_member_off + m] - (int32_t)batch_soa_off;

      for (int g = 0; g < batch_num_groups; g++)
        batch_csr.group_types[g] =
            (event_t)merged.group_types[batch_group_off + g];

      memcpy(batch_csr.group_roots,
             merged.group_roots.data() + batch_group_off,
             batch_num_groups * sizeof(id_t));
      memcpy(batch_csr.member_bytes_sent,
             merged.member_bytes_sent.data() + batch_member_off,
             batch_total_members * sizeof(uint64_t));
      memcpy(batch_csr.member_bytes_received,
             merged.member_bytes_received.data() + batch_member_off,
             batch_total_members * sizeof(uint64_t));
    }

    auto tp_prep1 = std::chrono::high_resolution_clock::now();
    float batch_prep_ms = (float)std::chrono::duration<double, std::milli>(tp_prep1 - tp_prep0).count();

    // --- Per-batch pinning (only if global pinning was skipped) ---
    auto tp_pin0 = std::chrono::high_resolution_clock::now();
    int batch_n_pin = 0;
    PinRegion batch_pin[7];
    if (!global_pinned && batch_events > 0) {
      size_t ev_bytes = batch_events * sizeof(event_t);
      size_t ts_bytes = batch_events * sizeof(timestamp_t);
      size_t id_bytes = batch_events * sizeof(id_t);
      size_t mp_bytes = batch_events * sizeof(int32_t);

      batch_pin[0] = {batch_soa.events, ev_bytes};
      batch_pin[1] = {batch_soa.timestamps, ts_bytes};
      batch_pin[2] = {batch_soa.end_timestamps, ts_bytes};
      batch_pin[3] = {batch_mp, mp_bytes};
      batch_pin[4] = {batch_soa.pids, id_bytes};
      batch_pin[5] = {batch_soa.roots, id_bytes};
      batch_pin[6] = {(void *)((timestamp_t *)(base + layout.leave_recv_ts_off) + batch_soa_off), ts_bytes};
      batch_n_pin = 7;

      size_t pin_total = ev_bytes + 3 * ts_bytes + mp_bytes + 2 * id_bytes;
      bool pin_ok = true;
      for (int i = 0; i < batch_n_pin; i++) {
        cudaError_t err = cudaHostRegister(batch_pin[i].ptr, batch_pin[i].len,
                                           cudaHostRegisterDefault);
        if (err != cudaSuccess) {
          cudaGetLastError();
          for (int j = 0; j < i; j++)
            cudaHostUnregister(batch_pin[j].ptr);
          batch_n_pin = 0;
          pin_ok = false;
          break;
        }
      }
      if (pin_ok && num_batches > 1) {
        std::cout << "[SHM] Batch " << (batch_start / K + 1)
                  << ": pinned " << (pin_total / (1024 * 1024))
                  << " MB for DMA" << std::endl;
      }
    }
    auto tp_pin1 = std::chrono::high_resolution_clock::now();
    float batch_pin_ms = (float)std::chrono::duration<double, std::milli>(tp_pin1 - tp_pin0).count();

    // --- Run GPU kernels (using pool + stream) ---
#ifdef USE_SCALASCA_TIMESTAMPS
    const timestamp_t *batch_leave_recv =
        (const timestamp_t *)(base + layout.leave_recv_ts_off) + batch_soa_off;
    RawAnalysisOutput batch_raw = runAnalysisKernelsAsync(batch_soa, batch_csr, pool, stream, batch_leave_recv);
#else
    RawAnalysisOutput batch_raw = runAnalysisKernelsAsync(batch_soa, batch_csr, pool, stream);
#endif

    // --- Per-batch unpinning ---
    auto tp_unpin0 = std::chrono::high_resolution_clock::now();
    if (batch_n_pin > 0) {
      for (int i = 0; i < batch_n_pin; i++)
        cudaHostUnregister(batch_pin[i].ptr);
    }
    auto tp_unpin1 = std::chrono::high_resolution_clock::now();
    float batch_unpin_ms = (float)std::chrono::duration<double, std::milli>(tp_unpin1 - tp_unpin0).count();

    // Store host-side timing in batch_raw before accumulation
    batch_raw.pin_ms = batch_pin_ms;
    batch_raw.unpin_ms = batch_unpin_ms;
    batch_raw.batch_prep_ms = batch_prep_ms;

    accumulateResults(output, batch_raw);

    if (num_batches > 1) {
      std::cout << "[SHM] Batch " << (batch_start / K + 1) << "/"
                << num_batches << ": ranks " << batch_start << "-"
                << (batch_end - 1) << " (" << batch_events
                << " events), H2D=" << std::fixed << std::setprecision(2)
                << batch_raw.h2d_ms << " ms, P2P=" << batch_raw.p2p_kernel_ms
                << " ms, Coll=" << batch_raw.coll_kernel_ms << " ms"
                << std::endl;
    }

    free(batch_mp);
    batch_soa.match_partner = nullptr;
  }

  // Cleanup pool + stream
  CUDA_CHECK(cudaStreamDestroy(stream));
  pool.deallocate();

  return output;
}

// Direct-to-shared-memory analysis: reads OTF2 vectors directly into shared
// window, then runs P2P matching, collective grouping, and timestamp correction
// on the shared data before GPU analysis. Eliminates double memory allocation.
static ShmAnalysisResult
sharedMemoryDirectAnalysis(ReaderPhase1Output &phase1, int rank, int nprocs,
                           bool time_correct) {
  ShmAnalysisResult result;
  auto tp0 = std::chrono::high_resolution_clock::now();
  auto tp1 = tp0;
  double t_shm_split = 0, t_allgather = 0, t_win_alloc = 0, t_prefault = 0,
         t_fill_soa = 0, t_p2p_match = 0, t_coll_group = 0,
         t_ts_correct = 0, t_remap_fence = 0, t_csr_gather = 0,
         t_gpu_batches = 0, t_cleanup = 0;

  // --- Create shared-memory communicator ---
  tp0 = std::chrono::high_resolution_clock::now();
  MPI_Comm shm_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                      MPI_INFO_NULL, &shm_comm);
  int shm_size;
  MPI_Comm_size(shm_comm, &shm_size);
  tp1 = std::chrono::high_resolution_clock::now();
  t_shm_split = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

  // Fallback if not all ranks share memory (multi-node)
  if (shm_size != nprocs) {
    if (rank == 0)
      std::cerr << "[SHM-Direct] Only " << shm_size << "/" << nprocs
                << " ranks share memory. Falling back to MPI Send/Recv."
                << std::endl;
    MPI_Comm_free(&shm_comm);
    TraceDataSoA local_data;
    local_data.allocate(phase1.event_count);
    readerFillSoA(phase1.handle, local_data);
    readerRelease(phase1.handle);
    phase1.handle = nullptr;
    runP2PMatching(local_data);
    CollectiveGroupCSR local_csr;
    buildCollectiveGroups(local_data, phase1.comm_sets, phase1.coll_bytes_sent, phase1.coll_bytes_received, local_csr);
#ifdef USE_SCALASCA_TIMESTAMPS
    if (time_correct)
      applyTimestampCorrection(local_data);
#endif
    result.raw = streamBatchAnalysis(local_data, local_csr, rank, nprocs);
    return result;
  }

  // --- Exchange event counts, compute offsets ---
  tp0 = std::chrono::high_resolution_clock::now();
  int64_t local_count_64 = (int64_t)phase1.event_count;
  std::vector<int64_t> all_counts_64(nprocs);
  MPI_Allgather(&local_count_64, 1, MPI_INT64_T, all_counts_64.data(), 1,
                MPI_INT64_T, shm_comm);

  // Convert to int for downstream APIs that need int (per-rank counts safely fit)
  std::vector<int> all_counts(nprocs);
  for (int i = 0; i < nprocs; i++)
    all_counts[i] = (int)all_counts_64[i];

  std::vector<size_t> soa_offsets(nprocs + 1);
  size_t total_events = 0;
  for (int i = 0; i < nprocs; i++) {
    soa_offsets[i] = total_events;
    total_events += all_counts[i];
  }
  soa_offsets[nprocs] = total_events;
  tp1 = std::chrono::high_resolution_clock::now();
  t_allgather = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

  if (total_events == 0) {
    readerRelease(phase1.handle);
    phase1.handle = nullptr;
    MPI_Comm_free(&shm_comm);
    return result;
  }

  // --- Compute layout & allocate shared window ---
  tp0 = std::chrono::high_resolution_clock::now();
  ShmSoALayout layout;
  layout.compute(total_events);

  if (rank == 0) {
    std::cout << "[SHM-Direct] Shared memory mode: " << total_events
              << " total events, " << (layout.total_bytes / (1024 * 1024))
              << " MB shared buffer" << std::endl;
  }

  MPI_Win win;
  void *base_ptr = nullptr;
  MPI_Aint win_size = (rank == 0) ? (MPI_Aint)layout.total_bytes : 0;
  MPI_Win_allocate_shared(win_size, 1, MPI_INFO_NULL, shm_comm, &base_ptr,
                          &win);

  if (rank != 0) {
    MPI_Aint sz;
    int disp;
    MPI_Win_shared_query(win, 0, &sz, &disp, &base_ptr);
  }
  char *base = (char *)base_ptr;
  tp1 = std::chrono::high_resolution_clock::now();
  t_win_alloc = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

  // --- Prefault shared memory + pin for GPU DMA ---
  // SHM pages are lazily allocated by the OS. Writing to them during fill_soa
  // causes ~5s of scattered page faults. Each rank prefaults its own slice in
  // parallel, distributing the cost across all cores (~16GB/16 ranks = 1GB each).
  // Also request huge pages (2MB THP) to reduce TLB miss rate.
  // Pinning is done by rank 0 after all ranks finish prefaulting.
  tp0 = std::chrono::high_resolution_clock::now();
  PinRegion pin_regions[7];
  int n_pin = 0;
  {
    // Rank 0 requests huge pages for the entire buffer
    if (rank == 0)
      madvise(base, layout.total_bytes, MADV_HUGEPAGE);
    MPI_Barrier(shm_comm);

    // Each rank prefaults its own slice of each array in parallel
    size_t my_off = soa_offsets[rank];
    size_t my_cnt = (size_t)all_counts[rank];
    if (my_cnt > 0) {
      memset((event_t *)(base + layout.events_off) + my_off, 0, my_cnt * sizeof(event_t));
      memset((event_type_t *)(base + layout.types_off) + my_off, 0, my_cnt * sizeof(event_type_t));
      memset((timestamp_t *)(base + layout.timestamps_off) + my_off, 0, my_cnt * sizeof(timestamp_t));
      memset((timestamp_t *)(base + layout.end_timestamps_off) + my_off, 0, my_cnt * sizeof(timestamp_t));
      memset((id_t *)(base + layout.pids_off) + my_off, 0, my_cnt * sizeof(id_t));
      memset((id_t *)(base + layout.tids_off) + my_off, 0, my_cnt * sizeof(id_t));
      memset((id_t *)(base + layout.replay_pids_off) + my_off, 0, my_cnt * sizeof(id_t));
      memset((id_t *)(base + layout.srcs_off) + my_off, 0, my_cnt * sizeof(id_t));
      memset((id_t *)(base + layout.dsts_off) + my_off, 0, my_cnt * sizeof(id_t));
      memset((id_t *)(base + layout.tags_off) + my_off, 0, my_cnt * sizeof(id_t));
      memset((id_t *)(base + layout.roots_off) + my_off, 0, my_cnt * sizeof(id_t));
      memset((id_t *)(base + layout.indices_off) + my_off, 0, my_cnt * sizeof(id_t));
      memset((int32_t *)(base + layout.match_partner_off) + my_off, 0, my_cnt * sizeof(int32_t));
      memset((int32_t *)(base + layout.coll_group_id_off) + my_off, 0, my_cnt * sizeof(int32_t));
      memset((timestamp_t *)(base + layout.leave_recv_ts_off) + my_off, 0, my_cnt * sizeof(timestamp_t));
    }

    MPI_Barrier(shm_comm);

    // Rank 0 pins after all pages are faulted (hot pages → fast pin)
    if (rank == 0)
      n_pin = shmPinHostMemory(base, layout, total_events, pin_regions, 7);
    MPI_Barrier(shm_comm);
  }
  tp1 = std::chrono::high_resolution_clock::now();
  t_prefault = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

  // --- Fill SoA directly into shared window ---
  tp0 = std::chrono::high_resolution_clock::now();
  MPI_Win_fence(0, win);

  size_t my_off = soa_offsets[rank];
  size_t my_cnt = phase1.event_count;

  TraceDataSoA local_data;
  shmSetupLocalSoA(local_data, base, layout, my_off, my_cnt);

  readerFillSoA(phase1.handle, local_data);
#ifdef USE_SCALASCA_TIMESTAMPS
  // Copy leave_recv_ts into shared window before releasing reader
  {
    std::vector<timestamp_t> lrt = readerGetLeaveRecvTs(phase1.handle);
    timestamp_t *shm_lrt = (timestamp_t *)(base + layout.leave_recv_ts_off) + my_off;
    if (my_cnt > 0 && lrt.size() == my_cnt)
      memcpy(shm_lrt, lrt.data(), my_cnt * sizeof(timestamp_t));
  }
#endif
  readerRelease(phase1.handle);
  phase1.handle = nullptr;
  tp1 = std::chrono::high_resolution_clock::now();
  t_fill_soa = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

  // --- P2P Matching (local, on shared window) ---
  tp0 = std::chrono::high_resolution_clock::now();
  runP2PMatching(local_data);
  tp1 = std::chrono::high_resolution_clock::now();
  t_p2p_match = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

  // --- Collective Grouping (local, on shared window) ---
  tp0 = std::chrono::high_resolution_clock::now();
  CollectiveGroupCSR local_csr;
  buildCollectiveGroups(local_data, phase1.comm_sets, phase1.coll_bytes_sent, phase1.coll_bytes_received, local_csr);
  tp1 = std::chrono::high_resolution_clock::now();
  t_coll_group = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

  // --- Timestamp Correction (conditional) ---
#ifdef USE_SCALASCA_TIMESTAMPS
  if (time_correct) {
    tp0 = std::chrono::high_resolution_clock::now();
    size_t local_violations = applyTimestampCorrection(local_data);
    tp1 = std::chrono::high_resolution_clock::now();
    t_ts_correct =
        std::chrono::duration<double, std::milli>(tp1 - tp0).count();
    size_t total_violations = 0;
    MPI_Reduce(&local_violations, &total_violations, 1, MPI_UNSIGNED_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << "[CLC] Total violations across all ranks: "
                << total_violations << std::endl;
    }
  }
#endif

  // --- Remap match_partner to global + fence ---
  tp0 = std::chrono::high_resolution_clock::now();
  if (my_cnt > 0 && my_off > 0) {
    for (size_t i = 0; i < my_cnt; i++) {
      int32_t mp = local_data.match_partner[i];
      if (mp >= 0)
        local_data.match_partner[i] = mp + (int32_t)my_off;
    }
  }
  MPI_Win_fence(0, win);
  tp1 = std::chrono::high_resolution_clock::now();
  t_remap_fence = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

  // (Pinning done during prefault step above — no separate pin needed)

  // --- Gather CSR data via MPI_Gatherv ---
  tp0 = std::chrono::high_resolution_clock::now();
  MergedCSR merged =
      shmGatherCSR(local_csr, soa_offsets, rank, nprocs, shm_comm);
  tp1 = std::chrono::high_resolution_clock::now();
  t_csr_gather = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

  // --- Rank 0 runs batched GPU analysis ---
  tp0 = std::chrono::high_resolution_clock::now();
  if (rank == 0) {
    result.raw = shmRunBatchedGPU(base, layout, soa_offsets, all_counts,
                                  merged, nprocs, n_pin > 0);
  }
  tp1 = std::chrono::high_resolution_clock::now();
  t_gpu_batches = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

  // --- Cleanup ---
  tp0 = std::chrono::high_resolution_clock::now();
  if (rank == 0 && n_pin > 0)
    shmUnpinHostMemory(pin_regions, n_pin);

  // Nullify non-owning pointers before win_free
  local_data.deallocate();

  MPI_Win_free(&win);
  MPI_Comm_free(&shm_comm);
  tp1 = std::chrono::high_resolution_clock::now();
  t_cleanup = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

  // Compute aggregate timing categories
  result.preprocess_ms = t_shm_split + t_allgather + t_win_alloc + t_prefault +
                         t_fill_soa + t_p2p_match + t_coll_group +
                         t_ts_correct + t_remap_fence + t_csr_gather;
  result.gpu_analysis_ms = t_gpu_batches;
  result.cleanup_ms = t_cleanup;

  if (rank == 0) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[SHM-Direct Preprocess] shm_comm_split: " << t_shm_split
              << " ms" << std::endl;
    std::cout << "[SHM-Direct Preprocess] allgather:      " << t_allgather
              << " ms" << std::endl;
    std::cout << "[SHM-Direct Preprocess] win_allocate:   " << t_win_alloc
              << " ms" << std::endl;
    std::cout << "[SHM-Direct Preprocess] prefault+pin:   " << t_prefault
              << " ms" << std::endl;
    std::cout << "[SHM-Direct Preprocess] fill_soa:       " << t_fill_soa
              << " ms" << std::endl;
    std::cout << "[SHM-Direct Preprocess] p2p_matching:   " << t_p2p_match
              << " ms" << std::endl;
    std::cout << "[SHM-Direct Preprocess] coll_grouping:  " << t_coll_group
              << " ms" << std::endl;
    std::cout << "[SHM-Direct Preprocess] ts_correction:  " << t_ts_correct
              << " ms" << std::endl;
    std::cout << "[SHM-Direct Preprocess] remap+fence:    " << t_remap_fence
              << " ms" << std::endl;
    std::cout << "[SHM-Direct Preprocess] csr_gather:     " << t_csr_gather
              << " ms" << std::endl;
    std::cout << "[SHM-Direct GPU]        batches:        " << t_gpu_batches
              << " ms" << std::endl;
    std::cout << "[SHM-Direct Cleanup]    cleanup:        " << t_cleanup
              << " ms" << std::endl;
  }

  return result;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (argc < 2) {
    if (mpi_rank == 0)
      std::cerr << "Usage: " << argv[0]
                << " <path/to/traces.otf2> [--time-correct]" << std::endl;
    MPI_Finalize();
    return 1;
  }

  std::string trace_path = argv[1];
  bool time_correct = false;
  for (int i = 2; i < argc; i++) {
    if (std::string(argv[i]) == "--time-correct") {
      time_correct = true;
    }
  }

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
              << " (distributed reading, direct-to-shared-memory GPU analysis)"
              << std::endl;
    if (time_correct)
      std::cout << "Timestamp correction: ENABLED (--time-correct)" << std::endl;
    std::cout << std::endl;
  }

  auto t_total_start = std::chrono::high_resolution_clock::now();

  // Step 1: Read OTF2 trace phase 1 (pass1 + pass2 + redistribution)
  if (mpi_rank == 0)
    std::cout << "=== Step 1: Reading OTF2 trace (distributed) ==="
              << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();
  ReaderPhase1Output phase1 = readOTF2TracePhase1(trace_path);
  auto t2 = std::chrono::high_resolution_clock::now();
  double read_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();

  if (mpi_rank == 0) {
    std::cout << "[Timer] OTF2 read: " << read_ms << " ms" << std::endl;
    std::cout << std::endl;
  }

  // Steps 2-4: Shared memory direct analysis (single function for multi-rank)
  // or local analysis for single-rank. P2P matching, collective grouping,
  // and timestamp correction are done inside sharedMemoryDirectAnalysis
  // to operate directly on the shared window.
  RawAnalysisOutput raw;
  double preprocess_ms = 0, gpu_analysis_ms = 0, cleanup_ms = 0;
  if (mpi_rank == 0)
    std::cout << "=== Step 2-4: Direct-to-SHM Analysis (GPU) ==="
              << std::endl;
  t1 = std::chrono::high_resolution_clock::now();

  if (mpi_size == 1) {
    // Single rank: local allocation, no shared memory
    TraceDataSoA local_data;
    local_data.allocate(phase1.event_count);
    readerFillSoA(phase1.handle, local_data);
#ifdef USE_SCALASCA_TIMESTAMPS
    std::vector<timestamp_t> leave_recv_ts = readerGetLeaveRecvTs(phase1.handle);
#endif
    readerRelease(phase1.handle);
    phase1.handle = nullptr;

    runP2PMatching(local_data);
    CollectiveGroupCSR csr;
    buildCollectiveGroups(local_data, phase1.comm_sets, phase1.coll_bytes_sent, phase1.coll_bytes_received, csr);
#ifdef USE_SCALASCA_TIMESTAMPS
    if (time_correct)
      applyTimestampCorrection(local_data);
#endif

#ifdef USE_SCALASCA_TIMESTAMPS
    raw = runAnalysisKernels(local_data, csr, leave_recv_ts.data());
#else
    raw = runAnalysisKernels(local_data, csr);
#endif
  } else {
    // Multi-rank: direct-to-shared-memory analysis
    ShmAnalysisResult shm_result =
        sharedMemoryDirectAnalysis(phase1, mpi_rank, mpi_size, time_correct);
    raw = std::move(shm_result.raw);
    preprocess_ms = shm_result.preprocess_ms;
    gpu_analysis_ms = shm_result.gpu_analysis_ms;
    cleanup_ms = shm_result.cleanup_ms;
  }

  t2 = std::chrono::high_resolution_clock::now();
  double total_analysis_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  if (mpi_rank == 0) {
    std::cout << std::fixed << std::setprecision(2);
    if (mpi_size > 1) {
      std::cout << "[Timer] Preprocess (CPU):  " << preprocess_ms << " ms"
                << std::endl;
      std::cout << "[Timer] GPU analysis:      " << gpu_analysis_ms << " ms"
                << std::endl;
      std::cout << "[Timer] Cleanup:           " << cleanup_ms << " ms"
                << std::endl;
    }
    std::cout << "[Timer] Analysis (total):  " << total_analysis_ms << " ms"
              << std::endl;
    float gpu_kernel_ms = raw.p2p_kernel_ms + raw.coll_kernel_ms;
    float gpu_accounted = raw.batch_prep_ms + raw.pin_ms + raw.gpu_alloc_ms +
                          raw.h2d_ms + gpu_kernel_ms + raw.d2h_ms +
                          raw.unpin_ms;
    float gpu_other_ms = (float)gpu_analysis_ms - gpu_accounted;
    std::cout << "[Analysis] GPU breakdown: batch_prep=" << raw.batch_prep_ms
              << " ms, pin=" << raw.pin_ms
              << " ms, alloc=" << raw.gpu_alloc_ms
              << " ms, H2D=" << raw.h2d_ms
              << " ms, kernels=" << gpu_kernel_ms
              << " ms (P2P=" << raw.p2p_kernel_ms
              << " + Coll=" << raw.coll_kernel_ms
              << "), D2H=" << raw.d2h_ms
              << " ms, unpin=" << raw.unpin_ms
              << " ms, other=" << gpu_other_ms << " ms"
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
    if (mpi_size > 1) {
      std::cout << "Preprocess (CPU):     " << preprocess_ms << " ms" << std::endl;
      std::cout << "GPU Analysis:         " << gpu_analysis_ms << " ms" << std::endl;
      std::cout << "Cleanup:              " << cleanup_ms << " ms" << std::endl;
    }
    std::cout << "Analysis (total):     " << total_analysis_ms << " ms" << std::endl;
    std::cout << "Statistics:           " << stats_ms << " ms" << std::endl;
    std::cout << "Total:                " << total_ms << " ms" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
