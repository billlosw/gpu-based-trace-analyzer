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
                         const timestamp_t *__restrict__ end_timestamps,
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
    // recv_enter: Enter(MPI_Recv) or Enter(MPI_Wait) for non-blocking
    // send_enter: point event timestamp ≈ Enter(MPI_Send/MPI_Isend)
    timestamp_t recv_enter = timestamps[i];
    timestamp_t send_enter = timestamps[send_idx];

    // Late sender: sender arrived after receiver started waiting.
    // Scalasca: idle = min(Enter(MPI_Send), Leave(MPI_Recv/Wait)) - Enter(MPI_Recv/Wait)
    // Simplified: idle = Enter(MPI_Send) - Enter(MPI_Recv/Wait) when > 0
    if (send_enter > recv_enter) {
      unsigned int pos = atomicAdd(late_sender_cnt, 1u);
      late_sender_out[pos] = (double)(send_enter - recv_enter);
    }

#ifdef USE_SCALASCA_TIMESTAMPS
    // Late receiver (independent check, NOT mutually exclusive with late sender).
    // Scalasca algorithm (SCOUT Patterns_gen.cpp lines 2819-2843):
    //   enter_sendcmp = Enter(send completion) = Enter(MPI_Send) for blocking
    //   leave_sendcmp = Leave(send completion) = Leave(MPI_Send) for blocking
    //   enter_recvreq = Enter(recv request)    = Enter(MPI_Irecv) for non-blocking
    //                                          = Enter(MPI_Recv) for blocking
    // Condition: leave_sendcmp > enter_recvreq (sender still blocked when recv posted)
    // Duration:  enter_recvreq - enter_sendcmp (only counted when > 0)
    //
    // Note: Cross-node clock violations are handled in TimestampCorrection.cpp
    // which adjusts end_timestamps for affected recv events before GPU analysis.
    {
      timestamp_t send_leave = end_timestamps[send_idx];   // Leave(MPI_Send)
      timestamp_t recv_req_enter = end_timestamps[i];      // Enter(MPI_Irecv) or Enter(MPI_Recv)
      if (send_leave > recv_req_enter && recv_req_enter > send_enter) {
        unsigned int pos = atomicAdd(late_receiver_cnt, 1u);
        late_receiver_out[pos] = (double)(recv_req_enter - send_enter);
      }
    }
#else
    // Non-Scalasca mode: simple late receiver check
    if (recv_enter > send_enter) {
      unsigned int pos = atomicAdd(late_receiver_cnt, 1u);
      late_receiver_out[pos] = (double)(recv_enter - send_enter);
    }
#endif
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
// Scalasca: only counts when root has getBytesReceived() != 0
// ============================================================
__global__ void kernelEarlyReduce(
    const timestamp_t *__restrict__ timestamps,
    const id_t *__restrict__ pids, const id_t *__restrict__ roots,
    const int32_t *__restrict__ coll_offsets,
    const int32_t *__restrict__ coll_members,
    const event_t *__restrict__ group_types, const id_t *__restrict__ group_roots,
    const uint64_t *__restrict__ member_bytes_sent,
    const uint64_t *__restrict__ member_bytes_received,
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
  bool root_receives = false;

  for (int j = start; j < end; j++) {
    int ev_idx = coll_members[j];
    timestamp_t ts = timestamps[ev_idx];
    if (pids[ev_idx] == root_pid) {
      root_ts = ts;
      root_receives = (member_bytes_received[j] != 0);
    }
    // Scalasca: exclude zero-byte-sent members from max
    if (member_bytes_sent[j] != 0 && ts > max_ts)
      max_ts = ts;
  }

  // Scalasca: early_reduce only when root has getBytesReceived() != 0
  if (root_receives && max_ts > root_ts) {
    unsigned int pos = atomicAdd(early_reduce_cnt, 1u);
    early_reduce_out[pos] = (double)(max_ts - root_ts);
  }
}

// ============================================================
// Kernel 4: Late Broadcast (Bcast, Scatter, Scatterv)
// root_ts - member_ts for members that arrived before root
// Scalasca: excludes root and zero-byte-received members
// ============================================================
__global__ void kernelLateBroadcast(
    const timestamp_t *__restrict__ timestamps,
    const id_t *__restrict__ pids, const id_t *__restrict__ roots,
    const int32_t *__restrict__ coll_offsets,
    const int32_t *__restrict__ coll_members,
    const event_t *__restrict__ group_types, const id_t *__restrict__ group_roots,
    const uint64_t *__restrict__ member_bytes_received,
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

  // Check each member (Scalasca: skip root and zero-byte-received)
  for (int j = start; j < end; j++) {
    int ev_idx = coll_members[j];
    if (pids[ev_idx] == root_pid)
      continue;
    if (member_bytes_received[j] == 0)
      continue;
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
// Scalasca: filters by bytes_sent/bytes_received for zero-byte members
// ============================================================
__global__ void kernelNxNWaitCompletion(
    const timestamp_t *__restrict__ timestamps,
    const timestamp_t *__restrict__ end_timestamps,
    const int32_t *__restrict__ coll_offsets,
    const int32_t *__restrict__ coll_members,
    const event_t *__restrict__ group_types,
    const uint64_t *__restrict__ member_bytes_sent,
    const uint64_t *__restrict__ member_bytes_received,
    size_t num_groups,
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
    // Scalasca: exclude zero-byte-sent from max_enter (latest)
    if (member_bytes_sent[j] != 0 && ts > max_enter)
      max_enter = ts;
    // Scalasca: exclude zero-byte-received from min_end (earliest_end)
    if (member_bytes_received[j] != 0 && ets < min_end)
      min_end = ets;
  }

  for (int j = start; j < end; j++) {
    int ev_idx = coll_members[j];
    timestamp_t ts = timestamps[ev_idx];
    timestamp_t ets = end_timestamps[ev_idx];
    // Scalasca wait_nxn: non-receivers don't have to wait
    if (member_bytes_received[j] != 0 && max_enter > ts) {
      unsigned int pos = atomicAdd(wait_nxn_cnt, 1u);
      wait_nxn_out[pos] = (double)(max_enter - ts);
    }
    // Scalasca nxn_completion: only for members with both sent and received
    if (member_bytes_sent[j] != 0 && member_bytes_received[j] != 0 && ets > min_end) {
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

  // ---- CUDA event timing ----
  cudaEvent_t ev_start, ev_alloc_done, ev_h2d_done, ev_p2p_done, ev_coll_done,
              ev_d2h_done;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_alloc_done));
  CUDA_CHECK(cudaEventCreate(&ev_h2d_done));
  CUDA_CHECK(cudaEventCreate(&ev_p2p_done));
  CUDA_CHECK(cudaEventCreate(&ev_coll_done));
  CUDA_CHECK(cudaEventCreate(&ev_d2h_done));

  CUDA_CHECK(cudaEventRecord(ev_start));

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

  CUDA_CHECK(cudaEventRecord(ev_alloc_done));

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

  CUDA_CHECK(cudaEventRecord(ev_h2d_done));

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
      d_events, d_timestamps, d_end_timestamps, d_match, n, d_ls_out, d_ls_cnt, d_lr_out,
      d_lr_cnt);
  CUDA_CHECK(cudaEventRecord(ev_p2p_done));
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
    uint64_t *d_member_bytes_sent, *d_member_bytes_received;

    CUDA_CHECK(cudaMalloc(&d_coll_offsets,
                           (csr.num_groups + 1) * sizeof(int32_t)));
    CUDA_CHECK(
        cudaMalloc(&d_coll_members, csr.total_members * sizeof(int32_t)));
    CUDA_CHECK(
        cudaMalloc(&d_group_types, csr.num_groups * sizeof(event_t)));
    CUDA_CHECK(
        cudaMalloc(&d_group_roots, csr.num_groups * sizeof(id_t)));
    CUDA_CHECK(
        cudaMalloc(&d_member_bytes_sent, csr.total_members * sizeof(uint64_t)));
    CUDA_CHECK(
        cudaMalloc(&d_member_bytes_received, csr.total_members * sizeof(uint64_t)));

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
    CUDA_CHECK(cudaMemcpy(d_member_bytes_sent, csr.member_bytes_sent,
                           csr.total_members * sizeof(uint64_t),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_member_bytes_received, csr.member_bytes_received,
                           csr.total_members * sizeof(uint64_t),
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
        d_group_types, d_group_roots, d_member_bytes_sent,
        d_member_bytes_received, csr.num_groups, d_er_out, d_er_cnt);

    // Late Broadcast
    kernelLateBroadcast<<<coll_grid, 1>>>(
        d_timestamps, d_pids, d_roots, d_coll_offsets, d_coll_members,
        d_group_types, d_group_roots, d_member_bytes_received,
        csr.num_groups, d_lb_out, d_lb_cnt);

    // NxN Wait + Completion
    kernelNxNWaitCompletion<<<coll_grid, 1>>>(
        d_timestamps, d_end_timestamps, d_coll_offsets, d_coll_members,
        d_group_types, d_member_bytes_sent, d_member_bytes_received,
        csr.num_groups, d_wn_out, d_wn_cnt, d_nc_out,
        d_nc_cnt);

    CUDA_CHECK(cudaEventRecord(ev_coll_done));
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

    CUDA_CHECK(cudaEventRecord(ev_d2h_done));
    CUDA_CHECK(cudaEventSynchronize(ev_d2h_done));

    // Cleanup collective device memory
    CUDA_CHECK(cudaFree(d_coll_offsets));
    CUDA_CHECK(cudaFree(d_coll_members));
    CUDA_CHECK(cudaFree(d_group_types));
    CUDA_CHECK(cudaFree(d_group_roots));
    CUDA_CHECK(cudaFree(d_member_bytes_sent));
    CUDA_CHECK(cudaFree(d_member_bytes_received));
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
  } else {
    // No collectives: D2H already done above for P2P
    CUDA_CHECK(cudaEventRecord(ev_d2h_done));
    CUDA_CHECK(cudaEventSynchronize(ev_d2h_done));
  }

  // Cleanup trace device memory
  CUDA_CHECK(cudaFree(d_events));
  CUDA_CHECK(cudaFree(d_timestamps));
  CUDA_CHECK(cudaFree(d_end_timestamps));
  CUDA_CHECK(cudaFree(d_match));
  CUDA_CHECK(cudaFree(d_pids));
  CUDA_CHECK(cudaFree(d_roots));

  // ---- Compute sub-phase timings ----
  CUDA_CHECK(cudaEventElapsedTime(&output.gpu_alloc_ms, ev_start, ev_alloc_done));
  CUDA_CHECK(cudaEventElapsedTime(&output.h2d_ms, ev_alloc_done, ev_h2d_done));
  CUDA_CHECK(cudaEventElapsedTime(&output.p2p_kernel_ms, ev_h2d_done, ev_p2p_done));
  if (csr.num_groups > 0) {
    CUDA_CHECK(cudaEventElapsedTime(&output.coll_kernel_ms, ev_p2p_done, ev_coll_done));
    CUDA_CHECK(cudaEventElapsedTime(&output.d2h_ms, ev_coll_done, ev_d2h_done));
  } else {
    CUDA_CHECK(cudaEventElapsedTime(&output.d2h_ms, ev_p2p_done, ev_d2h_done));
  }

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_alloc_done));
  CUDA_CHECK(cudaEventDestroy(ev_h2d_done));
  CUDA_CHECK(cudaEventDestroy(ev_p2p_done));
  CUDA_CHECK(cudaEventDestroy(ev_coll_done));
  CUDA_CHECK(cudaEventDestroy(ev_d2h_done));

  return output;
}

// ============================================================
// GPU Memory Pool — pre-allocate once, reuse across batches
// ============================================================
void GPUMemoryPool::allocate(size_t max_n, size_t max_members, size_t max_groups) {
  deallocate();
  max_events = max_n;
  max_coll_members = max_members;
  max_coll_groups = max_groups;

  if (max_n == 0) return;

  // Trace input
  CUDA_CHECK(cudaMalloc(&d_events, max_n * sizeof(event_t)));
  CUDA_CHECK(cudaMalloc(&d_timestamps, max_n * sizeof(timestamp_t)));
  CUDA_CHECK(cudaMalloc(&d_end_timestamps, max_n * sizeof(timestamp_t)));
  CUDA_CHECK(cudaMalloc(&d_match, max_n * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_pids, max_n * sizeof(id_t)));
  CUDA_CHECK(cudaMalloc(&d_roots, max_n * sizeof(id_t)));

  // P2P output (max_n each)
  CUDA_CHECK(cudaMalloc(&d_ls_out, max_n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_lr_out, max_n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_ls_cnt, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_lr_cnt, sizeof(unsigned int)));

  // Collective input
  if (max_groups > 0) {
    CUDA_CHECK(cudaMalloc(&d_coll_offsets, (max_groups + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_coll_members, max_members * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_group_types, max_groups * sizeof(event_t)));
    CUDA_CHECK(cudaMalloc(&d_group_roots, max_groups * sizeof(id_t)));
    CUDA_CHECK(cudaMalloc(&d_member_bytes_sent, max_members * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_member_bytes_received, max_members * sizeof(uint64_t)));

    // Collective output (max_members each)
    CUDA_CHECK(cudaMalloc(&d_bw_out, max_members * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bc_out, max_members * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_er_out, max_members * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lb_out, max_members * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_wn_out, max_members * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nc_out, max_members * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bw_cnt, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_bc_cnt, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_er_cnt, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_lb_cnt, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_wn_cnt, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_nc_cnt, sizeof(unsigned int)));
  }
}

void GPUMemoryPool::deallocate() {
  if (max_events == 0) return;

  cudaFree(d_events); cudaFree(d_timestamps); cudaFree(d_end_timestamps);
  cudaFree(d_match); cudaFree(d_pids); cudaFree(d_roots);
  cudaFree(d_ls_out); cudaFree(d_lr_out);
  cudaFree(d_ls_cnt); cudaFree(d_lr_cnt);

  if (max_coll_groups > 0) {
    cudaFree(d_coll_offsets); cudaFree(d_coll_members);
    cudaFree(d_group_types); cudaFree(d_group_roots);
    cudaFree(d_member_bytes_sent); cudaFree(d_member_bytes_received);
    cudaFree(d_bw_out); cudaFree(d_bc_out);
    cudaFree(d_er_out); cudaFree(d_lb_out);
    cudaFree(d_wn_out); cudaFree(d_nc_out);
    cudaFree(d_bw_cnt); cudaFree(d_bc_cnt);
    cudaFree(d_er_cnt); cudaFree(d_lb_cnt);
    cudaFree(d_wn_cnt); cudaFree(d_nc_cnt);
  }

  d_events = nullptr; d_timestamps = nullptr; d_end_timestamps = nullptr;
  d_match = nullptr; d_pids = nullptr; d_roots = nullptr;
  d_ls_out = nullptr; d_lr_out = nullptr;
  d_ls_cnt = nullptr; d_lr_cnt = nullptr;
  d_coll_offsets = nullptr; d_coll_members = nullptr;
  d_group_types = nullptr; d_group_roots = nullptr;
  d_member_bytes_sent = nullptr; d_member_bytes_received = nullptr;
  d_bw_out = nullptr; d_bc_out = nullptr;
  d_er_out = nullptr; d_lb_out = nullptr;
  d_wn_out = nullptr; d_nc_out = nullptr;
  d_bw_cnt = nullptr; d_bc_cnt = nullptr;
  d_er_cnt = nullptr; d_lb_cnt = nullptr;
  d_wn_cnt = nullptr; d_nc_cnt = nullptr;
  max_events = 0; max_coll_members = 0; max_coll_groups = 0;
}

// ============================================================
// Async variant: uses pre-allocated pool + CUDA stream
// ============================================================
RawAnalysisOutput runAnalysisKernelsAsync(const TraceDataSoA &data,
                                          const CollectiveGroupCSR &csr,
                                          GPUMemoryPool &pool,
                                          cudaStream_t stream) {
  RawAnalysisOutput output;
  size_t n = data.count;
  if (n == 0) return output;

  // CUDA event timing
  cudaEvent_t ev_start, ev_h2d_done, ev_p2p_done, ev_coll_done, ev_d2h_done;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_h2d_done));
  CUDA_CHECK(cudaEventCreate(&ev_p2p_done));
  CUDA_CHECK(cudaEventCreate(&ev_coll_done));
  CUDA_CHECK(cudaEventCreate(&ev_d2h_done));

  CUDA_CHECK(cudaEventRecord(ev_start, stream));

  // H2D using async memcpy on the given stream
  CUDA_CHECK(cudaMemcpyAsync(pool.d_events, data.events, n * sizeof(event_t),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(pool.d_timestamps, data.timestamps,
                              n * sizeof(timestamp_t), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(pool.d_end_timestamps, data.end_timestamps,
                              n * sizeof(timestamp_t), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(pool.d_match, data.match_partner, n * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(pool.d_pids, data.pids, n * sizeof(id_t),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(pool.d_roots, data.roots, n * sizeof(id_t),
                              cudaMemcpyHostToDevice, stream));

  CUDA_CHECK(cudaEventRecord(ev_h2d_done, stream));

  // Reset P2P counters
  CUDA_CHECK(cudaMemsetAsync(pool.d_ls_cnt, 0, sizeof(unsigned int), stream));
  CUDA_CHECK(cudaMemsetAsync(pool.d_lr_cnt, 0, sizeof(unsigned int), stream));

  // P2P kernel
  int blockSize = 256;
  int gridSize = (int)std::min((n + 255) / 256, (size_t)1024);
  kernelLateSenderReceiver<<<gridSize, blockSize, 0, stream>>>(
      pool.d_events, pool.d_timestamps, pool.d_end_timestamps, pool.d_match,
      n, pool.d_ls_out, pool.d_ls_cnt, pool.d_lr_out, pool.d_lr_cnt);
  CUDA_CHECK(cudaEventRecord(ev_p2p_done, stream));

  // Sync stream to read P2P counts for D2H sizing
  CUDA_CHECK(cudaStreamSynchronize(stream));

  unsigned int h_ls_cnt = 0, h_lr_cnt = 0;
  CUDA_CHECK(cudaMemcpy(&h_ls_cnt, pool.d_ls_cnt, sizeof(unsigned int),
                         cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_lr_cnt, pool.d_lr_cnt, sizeof(unsigned int),
                         cudaMemcpyDeviceToHost));

  if (h_ls_cnt > 0) {
    output.late_sender.resize(h_ls_cnt);
    CUDA_CHECK(cudaMemcpyAsync(output.late_sender.data(), pool.d_ls_out,
                                h_ls_cnt * sizeof(double), cudaMemcpyDeviceToHost, stream));
  }
  if (h_lr_cnt > 0) {
    output.late_receiver.resize(h_lr_cnt);
    CUDA_CHECK(cudaMemcpyAsync(output.late_receiver.data(), pool.d_lr_out,
                                h_lr_cnt * sizeof(double), cudaMemcpyDeviceToHost, stream));
  }

  // Collective analysis
  if (csr.num_groups > 0) {
    CUDA_CHECK(cudaMemcpyAsync(pool.d_coll_offsets, csr.offsets,
                                (csr.num_groups + 1) * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(pool.d_coll_members, csr.members,
                                csr.total_members * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(pool.d_group_types, csr.group_types,
                                csr.num_groups * sizeof(event_t),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(pool.d_group_roots, csr.group_roots,
                                csr.num_groups * sizeof(id_t),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(pool.d_member_bytes_sent, csr.member_bytes_sent,
                                csr.total_members * sizeof(uint64_t),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(pool.d_member_bytes_received, csr.member_bytes_received,
                                csr.total_members * sizeof(uint64_t),
                                cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemsetAsync(pool.d_bw_cnt, 0, sizeof(unsigned int), stream));
    CUDA_CHECK(cudaMemsetAsync(pool.d_bc_cnt, 0, sizeof(unsigned int), stream));
    CUDA_CHECK(cudaMemsetAsync(pool.d_er_cnt, 0, sizeof(unsigned int), stream));
    CUDA_CHECK(cudaMemsetAsync(pool.d_lb_cnt, 0, sizeof(unsigned int), stream));
    CUDA_CHECK(cudaMemsetAsync(pool.d_wn_cnt, 0, sizeof(unsigned int), stream));
    CUDA_CHECK(cudaMemsetAsync(pool.d_nc_cnt, 0, sizeof(unsigned int), stream));

    int coll_grid = (int)csr.num_groups;

    kernelBarrierWaitCompletion<<<coll_grid, 1, 0, stream>>>(
        pool.d_timestamps, pool.d_end_timestamps, pool.d_coll_offsets,
        pool.d_coll_members, pool.d_group_types, csr.num_groups,
        pool.d_bw_out, pool.d_bw_cnt, pool.d_bc_out, pool.d_bc_cnt);

    kernelEarlyReduce<<<coll_grid, 1, 0, stream>>>(
        pool.d_timestamps, pool.d_pids, pool.d_roots, pool.d_coll_offsets,
        pool.d_coll_members, pool.d_group_types, pool.d_group_roots,
        pool.d_member_bytes_sent, pool.d_member_bytes_received,
        csr.num_groups, pool.d_er_out, pool.d_er_cnt);

    kernelLateBroadcast<<<coll_grid, 1, 0, stream>>>(
        pool.d_timestamps, pool.d_pids, pool.d_roots, pool.d_coll_offsets,
        pool.d_coll_members, pool.d_group_types, pool.d_group_roots,
        pool.d_member_bytes_received,
        csr.num_groups, pool.d_lb_out, pool.d_lb_cnt);

    kernelNxNWaitCompletion<<<coll_grid, 1, 0, stream>>>(
        pool.d_timestamps, pool.d_end_timestamps, pool.d_coll_offsets,
        pool.d_coll_members, pool.d_group_types,
        pool.d_member_bytes_sent, pool.d_member_bytes_received,
        csr.num_groups,
        pool.d_wn_out, pool.d_wn_cnt, pool.d_nc_out, pool.d_nc_cnt);

    CUDA_CHECK(cudaEventRecord(ev_coll_done, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    unsigned int h_bw_cnt = 0, h_bc_cnt = 0, h_er_cnt = 0, h_lb_cnt = 0,
                h_wn_cnt = 0, h_nc_cnt = 0;
    CUDA_CHECK(cudaMemcpy(&h_bw_cnt, pool.d_bw_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_bc_cnt, pool.d_bc_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_er_cnt, pool.d_er_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_lb_cnt, pool.d_lb_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_wn_cnt, pool.d_wn_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_nc_cnt, pool.d_nc_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    auto copyBack = [&stream](std::vector<double> &out, double *d_ptr, unsigned int cnt) {
      if (cnt > 0) {
        out.resize(cnt);
        CUDA_CHECK(cudaMemcpyAsync(out.data(), d_ptr, cnt * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream));
      }
    };
    copyBack(output.barrier_wait, pool.d_bw_out, h_bw_cnt);
    copyBack(output.barrier_completion, pool.d_bc_out, h_bc_cnt);
    copyBack(output.early_reduce, pool.d_er_out, h_er_cnt);
    copyBack(output.late_broadcast, pool.d_lb_out, h_lb_cnt);
    copyBack(output.wait_nxn, pool.d_wn_out, h_wn_cnt);
    copyBack(output.nxn_completion, pool.d_nc_out, h_nc_cnt);

    CUDA_CHECK(cudaEventRecord(ev_d2h_done, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } else {
    CUDA_CHECK(cudaEventRecord(ev_d2h_done, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Timing
  output.gpu_alloc_ms = 0; // pool already allocated
  CUDA_CHECK(cudaEventElapsedTime(&output.h2d_ms, ev_start, ev_h2d_done));
  CUDA_CHECK(cudaEventElapsedTime(&output.p2p_kernel_ms, ev_h2d_done, ev_p2p_done));
  if (csr.num_groups > 0) {
    CUDA_CHECK(cudaEventElapsedTime(&output.coll_kernel_ms, ev_p2p_done, ev_coll_done));
    CUDA_CHECK(cudaEventElapsedTime(&output.d2h_ms, ev_coll_done, ev_d2h_done));
  } else {
    CUDA_CHECK(cudaEventElapsedTime(&output.d2h_ms, ev_p2p_done, ev_d2h_done));
  }

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_h2d_done));
  CUDA_CHECK(cudaEventDestroy(ev_p2p_done));
  CUDA_CHECK(cudaEventDestroy(ev_coll_done));
  CUDA_CHECK(cudaEventDestroy(ev_d2h_done));

  return output;
}
