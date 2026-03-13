#include "analysis/AnalysisKernels.h"
#include "common/cuda_check.h"

#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>

// ============================================================
// Kernel 1: Late Sender + Late Receiver (P2P pairs)
// ============================================================
__global__ void
kernelLateSenderReceiver(const event_t *__restrict__ events,
                         const timestamp_t *__restrict__ timestamps,
                         const int32_t *__restrict__ match_partner, size_t n,
                         double *__restrict__ late_sender_out,
                         unsigned int *__restrict__ late_sender_cnt,
                         double *__restrict__ late_receiver_out,
                         unsigned int *__restrict__ late_receiver_cnt) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;

  for (size_t i = idx; i < n; i += stride) {
    event_t ev = events[i];
    // Only process Recv/Irecv events to avoid double-counting
    if ((ev != TT_MPI_Recv && ev != TT_MPI_Irecv) || match_partner[i] < 0)
      continue;

    int32_t send_idx = match_partner[i];
    timestamp_t recv_ts = timestamps[i];
    timestamp_t send_ts = timestamps[send_idx];

    if (send_ts > recv_ts) {
      // Late sender: sender arrived after receiver
      unsigned int pos = atomicAdd(late_sender_cnt, 1u);
      late_sender_out[pos] = (double)(send_ts - recv_ts);
    } else if (recv_ts > send_ts) {
      // Late receiver: receiver arrived after sender
      unsigned int pos = atomicAdd(late_receiver_cnt, 1u);
      late_receiver_out[pos] = (double)(recv_ts - send_ts);
    }
  }
}

// ============================================================
// Kernel 2: Barrier Wait + Barrier Completion
// One block per barrier group. Thread 0 does sequential scan
// (groups are typically small: tens to hundreds of members).
// ============================================================
__global__ void kernelBarrierWaitCompletion(
    const timestamp_t *__restrict__ timestamps,
    const timestamp_t *__restrict__ end_timestamps,
    const int32_t *__restrict__ coll_offsets,
    const int32_t *__restrict__ coll_members,
    const event_t *__restrict__ group_types, size_t num_groups,
    double *__restrict__ barrier_wait_out,
    unsigned int *__restrict__ barrier_wait_cnt,
    double *__restrict__ barrier_completion_out,
    unsigned int *__restrict__ barrier_completion_cnt) {
  int gid = blockIdx.x;
  if ((size_t)gid >= num_groups)
    return;
  if (group_types[gid] != TT_MPI_Barrier)
    return;

  int start = coll_offsets[gid];
  int end = coll_offsets[gid + 1];
  int group_size = end - start;
  if (group_size < 2)
    return;

  // Thread 0 does all work for this group (groups are typically small)
  if (threadIdx.x == 0) {
    // Find max enter timestamp and min end timestamp
    timestamp_t max_enter = 0;
    timestamp_t min_end = UINT64_MAX;
    for (int j = start; j < end; j++) {
      int ev_idx = coll_members[j];
      timestamp_t ts = timestamps[ev_idx];
      timestamp_t ets = end_timestamps[ev_idx];
      if (ts > max_enter)
        max_enter = ts;
      if (ets < min_end)
        min_end = ets;
    }

    // Each member contributes its wait and completion time
    for (int j = start; j < end; j++) {
      int ev_idx = coll_members[j];
      timestamp_t ts = timestamps[ev_idx];
      timestamp_t ets = end_timestamps[ev_idx];
      if (max_enter > ts) {
        unsigned int pos = atomicAdd(barrier_wait_cnt, 1u);
        barrier_wait_out[pos] = (double)(max_enter - ts);
      }
      if (ets > min_end) {
        unsigned int pos = atomicAdd(barrier_completion_cnt, 1u);
        barrier_completion_out[pos] = (double)(ets - min_end);
      }
    }
  }
}

// ============================================================
// Kernel 3: Early Reduce (Reduce, Gather, Gatherv)
// max(member_ts) - root_ts when root arrives early
// ============================================================
__global__ void kernelEarlyReduce(
    const timestamp_t *__restrict__ timestamps,
    const id_t *__restrict__ pids, const id_t *__restrict__ roots,
    const int32_t *__restrict__ coll_offsets,
    const int32_t *__restrict__ coll_members,
    const event_t *__restrict__ group_types, const id_t *__restrict__ group_roots,
    size_t num_groups, double *__restrict__ early_reduce_out,
    unsigned int *__restrict__ early_reduce_cnt) {
  int gid = blockIdx.x;
  if ((size_t)gid >= num_groups)
    return;

  event_t gtype = group_types[gid];
  if (gtype != TT_MPI_Reduce && gtype != TT_MPI_Gather &&
      gtype != TT_MPI_Gatherv)
    return;

  if (threadIdx.x != 0)
    return;

  int start = coll_offsets[gid];
  int end = coll_offsets[gid + 1];
  id_t root_pid = group_roots[gid];

  timestamp_t root_ts = 0;
  timestamp_t max_ts = 0;

  for (int j = start; j < end; j++) {
    int ev_idx = coll_members[j];
    timestamp_t ts = timestamps[ev_idx];
    if (pids[ev_idx] == root_pid)
      root_ts = ts;
    if (ts > max_ts)
      max_ts = ts;
  }

  if (max_ts > root_ts) {
    unsigned int pos = atomicAdd(early_reduce_cnt, 1u);
    early_reduce_out[pos] = (double)(max_ts - root_ts);
  }
}

// ============================================================
// Kernel 4: Late Broadcast (Bcast, Scatter, Scatterv)
// root_ts - member_ts for members that arrived before root
// ============================================================
__global__ void kernelLateBroadcast(
    const timestamp_t *__restrict__ timestamps,
    const id_t *__restrict__ pids, const id_t *__restrict__ roots,
    const int32_t *__restrict__ coll_offsets,
    const int32_t *__restrict__ coll_members,
    const event_t *__restrict__ group_types, const id_t *__restrict__ group_roots,
    size_t num_groups, double *__restrict__ late_bcast_out,
    unsigned int *__restrict__ late_bcast_cnt) {
  int gid = blockIdx.x;
  if ((size_t)gid >= num_groups)
    return;

  event_t gtype = group_types[gid];
  if (gtype != TT_MPI_Bcast && gtype != TT_MPI_Scatter &&
      gtype != TT_MPI_Scatterv)
    return;

  if (threadIdx.x != 0)
    return;

  int start = coll_offsets[gid];
  int end = coll_offsets[gid + 1];
  id_t root_pid = group_roots[gid];

  // Find root timestamp
  timestamp_t root_ts = 0;
  for (int j = start; j < end; j++) {
    int ev_idx = coll_members[j];
    if (pids[ev_idx] == root_pid) {
      root_ts = timestamps[ev_idx];
      break;
    }
  }

  // Check each member
  for (int j = start; j < end; j++) {
    int ev_idx = coll_members[j];
    timestamp_t member_ts = timestamps[ev_idx];
    if (member_ts < root_ts) {
      unsigned int pos = atomicAdd(late_bcast_cnt, 1u);
      late_bcast_out[pos] = (double)(root_ts - member_ts);
    }
  }
}

// ============================================================
// Kernel 5: Wait NxN + NxN Completion
// Same as barrier wait/completion but for AlltoAll-type collectives
// ============================================================
__global__ void kernelNxNWaitCompletion(
    const timestamp_t *__restrict__ timestamps,
    const timestamp_t *__restrict__ end_timestamps,
    const int32_t *__restrict__ coll_offsets,
    const int32_t *__restrict__ coll_members,
    const event_t *__restrict__ group_types, size_t num_groups,
    double *__restrict__ wait_nxn_out, unsigned int *__restrict__ wait_nxn_cnt,
    double *__restrict__ nxn_completion_out,
    unsigned int *__restrict__ nxn_completion_cnt) {
  int gid = blockIdx.x;
  if ((size_t)gid >= num_groups)
    return;

  event_t gtype = group_types[gid];
  bool is_nxn = (gtype == TT_MPI_Reduce_Scatter ||
                 gtype == TT_MPI_Reduce_Scatter_Block ||
                 gtype == TT_MPI_All_Gather || gtype == TT_MPI_All_Gatherv ||
                 gtype == TT_MPI_All_Reduce || gtype == TT_MPI_AlltoAll);
  if (!is_nxn)
    return;

  if (threadIdx.x != 0)
    return;

  int start = coll_offsets[gid];
  int end = coll_offsets[gid + 1];
  int group_size = end - start;
  if (group_size < 2)
    return;

  timestamp_t max_enter = 0;
  timestamp_t min_end = UINT64_MAX;
  for (int j = start; j < end; j++) {
    int ev_idx = coll_members[j];
    timestamp_t ts = timestamps[ev_idx];
    timestamp_t ets = end_timestamps[ev_idx];
    if (ts > max_enter)
      max_enter = ts;
    if (ets < min_end)
      min_end = ets;
  }

  for (int j = start; j < end; j++) {
    int ev_idx = coll_members[j];
    timestamp_t ts = timestamps[ev_idx];
    timestamp_t ets = end_timestamps[ev_idx];
    if (max_enter > ts) {
      unsigned int pos = atomicAdd(wait_nxn_cnt, 1u);
      wait_nxn_out[pos] = (double)(max_enter - ts);
    }
    if (ets > min_end) {
      unsigned int pos = atomicAdd(nxn_completion_cnt, 1u);
      nxn_completion_out[pos] = (double)(ets - min_end);
    }
  }
}

// ============================================================
// Host function: Run all 8 analyses on GPU
// ============================================================
RawAnalysisOutput runAnalysisKernels(const TraceDataSoA &data,
                                     const CollectiveGroupCSR &csr) {
  RawAnalysisOutput output;
  size_t n = data.count;
  if (n == 0)
    return output;

  // ---- Allocate device arrays for trace data ----
  event_t *d_events;
  timestamp_t *d_timestamps, *d_end_timestamps;
  int32_t *d_match;
  id_t *d_pids, *d_roots;

  CUDA_CHECK(cudaMalloc(&d_events, n * sizeof(event_t)));
  CUDA_CHECK(cudaMalloc(&d_timestamps, n * sizeof(timestamp_t)));
  CUDA_CHECK(cudaMalloc(&d_end_timestamps, n * sizeof(timestamp_t)));
  CUDA_CHECK(cudaMalloc(&d_match, n * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_pids, n * sizeof(id_t)));
  CUDA_CHECK(cudaMalloc(&d_roots, n * sizeof(id_t)));

  CUDA_CHECK(cudaMemcpy(d_events, data.events, n * sizeof(event_t),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_timestamps, data.timestamps,
                         n * sizeof(timestamp_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_end_timestamps, data.end_timestamps,
                         n * sizeof(timestamp_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_match, data.match_partner, n * sizeof(int32_t),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pids, data.pids, n * sizeof(id_t),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_roots, data.roots, n * sizeof(id_t),
                         cudaMemcpyHostToDevice));

  // ---- Allocate output arrays on device ----
  // Max possible output size = n (every event produces a result)
  double *d_ls_out, *d_lr_out;
  unsigned int *d_ls_cnt, *d_lr_cnt;
  CUDA_CHECK(cudaMalloc(&d_ls_out, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_lr_out, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_ls_cnt, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_lr_cnt, sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(d_ls_cnt, 0, sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(d_lr_cnt, 0, sizeof(unsigned int)));

  // ---- Run Late Sender/Receiver kernel ----
  int blockSize = 256;
  int gridSize = (int)std::min((n + 255) / 256, (size_t)1024);
  kernelLateSenderReceiver<<<gridSize, blockSize>>>(
      d_events, d_timestamps, d_match, n, d_ls_out, d_ls_cnt, d_lr_out,
      d_lr_cnt);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Read back P2P results
  unsigned int h_ls_cnt = 0, h_lr_cnt = 0;
  CUDA_CHECK(cudaMemcpy(&h_ls_cnt, d_ls_cnt, sizeof(unsigned int),
                         cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_lr_cnt, d_lr_cnt, sizeof(unsigned int),
                         cudaMemcpyDeviceToHost));

  if (h_ls_cnt > 0) {
    output.late_sender.resize(h_ls_cnt);
    CUDA_CHECK(cudaMemcpy(output.late_sender.data(), d_ls_out,
                           h_ls_cnt * sizeof(double), cudaMemcpyDeviceToHost));
  }
  if (h_lr_cnt > 0) {
    output.late_receiver.resize(h_lr_cnt);
    CUDA_CHECK(cudaMemcpy(output.late_receiver.data(), d_lr_out,
                           h_lr_cnt * sizeof(double), cudaMemcpyDeviceToHost));
  }

  CUDA_CHECK(cudaFree(d_ls_out));
  CUDA_CHECK(cudaFree(d_lr_out));
  CUDA_CHECK(cudaFree(d_ls_cnt));
  CUDA_CHECK(cudaFree(d_lr_cnt));

  // ---- Collective analysis kernels ----
  if (csr.num_groups > 0) {
    // Transfer CSR to device
    int32_t *d_coll_offsets, *d_coll_members;
    event_t *d_group_types;
    id_t *d_group_roots;

    CUDA_CHECK(cudaMalloc(&d_coll_offsets,
                           (csr.num_groups + 1) * sizeof(int32_t)));
    CUDA_CHECK(
        cudaMalloc(&d_coll_members, csr.total_members * sizeof(int32_t)));
    CUDA_CHECK(
        cudaMalloc(&d_group_types, csr.num_groups * sizeof(event_t)));
    CUDA_CHECK(
        cudaMalloc(&d_group_roots, csr.num_groups * sizeof(id_t)));

    CUDA_CHECK(cudaMemcpy(d_coll_offsets, csr.offsets,
                           (csr.num_groups + 1) * sizeof(int32_t),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coll_members, csr.members,
                           csr.total_members * sizeof(int32_t),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_group_types, csr.group_types,
                           csr.num_groups * sizeof(event_t),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_group_roots, csr.group_roots,
                           csr.num_groups * sizeof(id_t),
                           cudaMemcpyHostToDevice));

    // Allocate output arrays for all collective analyses
    // Max possible: total_members entries per analysis
    size_t max_coll_out = csr.total_members;

    double *d_bw_out, *d_bc_out, *d_er_out, *d_lb_out, *d_wn_out, *d_nc_out;
    unsigned int *d_bw_cnt, *d_bc_cnt, *d_er_cnt, *d_lb_cnt, *d_wn_cnt,
        *d_nc_cnt;

    CUDA_CHECK(cudaMalloc(&d_bw_out, max_coll_out * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bc_out, max_coll_out * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_er_out, max_coll_out * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lb_out, max_coll_out * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_wn_out, max_coll_out * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nc_out, max_coll_out * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_bw_cnt, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_bc_cnt, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_er_cnt, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_lb_cnt, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_wn_cnt, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_nc_cnt, sizeof(unsigned int)));

    CUDA_CHECK(cudaMemset(d_bw_cnt, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_bc_cnt, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_er_cnt, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_lb_cnt, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_wn_cnt, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_nc_cnt, 0, sizeof(unsigned int)));

    int coll_grid = (int)csr.num_groups;

    // Barrier Wait + Completion
    kernelBarrierWaitCompletion<<<coll_grid, 1>>>(
        d_timestamps, d_end_timestamps, d_coll_offsets, d_coll_members,
        d_group_types, csr.num_groups, d_bw_out, d_bw_cnt, d_bc_out,
        d_bc_cnt);

    // Early Reduce
    kernelEarlyReduce<<<coll_grid, 1>>>(
        d_timestamps, d_pids, d_roots, d_coll_offsets, d_coll_members,
        d_group_types, d_group_roots, csr.num_groups, d_er_out, d_er_cnt);

    // Late Broadcast
    kernelLateBroadcast<<<coll_grid, 1>>>(
        d_timestamps, d_pids, d_roots, d_coll_offsets, d_coll_members,
        d_group_types, d_group_roots, csr.num_groups, d_lb_out, d_lb_cnt);

    // NxN Wait + Completion
    kernelNxNWaitCompletion<<<coll_grid, 1>>>(
        d_timestamps, d_end_timestamps, d_coll_offsets, d_coll_members,
        d_group_types, csr.num_groups, d_wn_out, d_wn_cnt, d_nc_out,
        d_nc_cnt);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back collective results
    unsigned int h_bw_cnt = 0, h_bc_cnt = 0, h_er_cnt = 0, h_lb_cnt = 0,
                h_wn_cnt = 0, h_nc_cnt = 0;
    CUDA_CHECK(cudaMemcpy(&h_bw_cnt, d_bw_cnt, sizeof(unsigned int),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_bc_cnt, d_bc_cnt, sizeof(unsigned int),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_er_cnt, d_er_cnt, sizeof(unsigned int),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_lb_cnt, d_lb_cnt, sizeof(unsigned int),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_wn_cnt, d_wn_cnt, sizeof(unsigned int),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_nc_cnt, d_nc_cnt, sizeof(unsigned int),
                           cudaMemcpyDeviceToHost));

    auto copyBack = [](std::vector<double> &out, double *d_ptr,
                       unsigned int cnt) {
      if (cnt > 0) {
        out.resize(cnt);
        CUDA_CHECK(cudaMemcpy(out.data(), d_ptr, cnt * sizeof(double),
                               cudaMemcpyDeviceToHost));
      }
    };

    copyBack(output.barrier_wait, d_bw_out, h_bw_cnt);
    copyBack(output.barrier_completion, d_bc_out, h_bc_cnt);
    copyBack(output.early_reduce, d_er_out, h_er_cnt);
    copyBack(output.late_broadcast, d_lb_out, h_lb_cnt);
    copyBack(output.wait_nxn, d_wn_out, h_wn_cnt);
    copyBack(output.nxn_completion, d_nc_out, h_nc_cnt);

    // Cleanup collective device memory
    CUDA_CHECK(cudaFree(d_coll_offsets));
    CUDA_CHECK(cudaFree(d_coll_members));
    CUDA_CHECK(cudaFree(d_group_types));
    CUDA_CHECK(cudaFree(d_group_roots));
    CUDA_CHECK(cudaFree(d_bw_out));
    CUDA_CHECK(cudaFree(d_bc_out));
    CUDA_CHECK(cudaFree(d_er_out));
    CUDA_CHECK(cudaFree(d_lb_out));
    CUDA_CHECK(cudaFree(d_wn_out));
    CUDA_CHECK(cudaFree(d_nc_out));
    CUDA_CHECK(cudaFree(d_bw_cnt));
    CUDA_CHECK(cudaFree(d_bc_cnt));
    CUDA_CHECK(cudaFree(d_er_cnt));
    CUDA_CHECK(cudaFree(d_lb_cnt));
    CUDA_CHECK(cudaFree(d_wn_cnt));
    CUDA_CHECK(cudaFree(d_nc_cnt));
  }

  // Cleanup trace device memory
  CUDA_CHECK(cudaFree(d_events));
  CUDA_CHECK(cudaFree(d_timestamps));
  CUDA_CHECK(cudaFree(d_end_timestamps));
  CUDA_CHECK(cudaFree(d_match));
  CUDA_CHECK(cudaFree(d_pids));
  CUDA_CHECK(cudaFree(d_roots));

  return output;
}
