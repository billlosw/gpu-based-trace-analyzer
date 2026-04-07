#include "matching/GPUP2PMatching.h"
#include "matching/P2PMatching.h"
#include "common/cuda_check.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

// ============================================================================
// GPU P2P Matching: Sort-and-Rank Algorithm
//
// The key insight: for a fixed matching key (sender, receiver, tag),
// MPI's FIFO ordering guarantees that the i-th send matches the i-th recv
// (both ordered by timestamp). This reduces matching to:
//   1. Partition into sends and recvs
//   2. Sort each by (key, timestamp)
//   3. Assign within-group ordinals
//   4. Match sends and recvs with the same (key, ordinal)
// ============================================================================

// Pack (sender, receiver, tag) into a 64-bit sort key.
// Same encoding as CPU P2PMatching: 22 + 21 + 21 bits.
__host__ __device__ static inline uint64_t packMatchKey(uint32_t sender,
                                                         uint32_t receiver,
                                                         uint32_t tag) {
  return ((uint64_t)sender << 42) | ((uint64_t)receiver << 21) | (uint64_t)tag;
}

// Kernel: classify events, compute match keys, copy timestamps
__global__ void kernelClassifyP2P(
    const event_t *__restrict__ events,
    const uint32_t *__restrict__ pids,
    const uint32_t *__restrict__ srcs,
    const uint32_t *__restrict__ dsts,
    const uint32_t *__restrict__ tags,
    const uint64_t *__restrict__ timestamps,
    size_t n,
    int *__restrict__ is_send,
    int *__restrict__ is_recv,
    uint64_t *__restrict__ match_keys,
    uint64_t *__restrict__ ts_copy
) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (idx >= n) return;

  event_t ev = events[idx];
  int s = (ev == TT_MPI_Send || ev == TT_MPI_Isend) ? 1 : 0;
  int r = (ev == TT_MPI_Recv || ev == TT_MPI_Irecv) ? 1 : 0;

  is_send[idx] = s;
  is_recv[idx] = r;
  ts_copy[idx] = timestamps[idx];

  if (s) {
    match_keys[idx] = packMatchKey(pids[idx], dsts[idx], tags[idx]);
  } else if (r) {
    match_keys[idx] = packMatchKey(srcs[idx], pids[idx], tags[idx]);
  } else {
    match_keys[idx] = UINT64_MAX;
  }
}

// Functor for copy_if predicate
struct IsNonZero {
  __host__ __device__ bool operator()(int x) const { return x != 0; }
};

// Kernel: match sends to recvs via binary search on sorted recv keys
__global__ void kernelWriteMatches(
    const uint32_t *__restrict__ send_orig_indices,
    const uint64_t *__restrict__ send_keys,
    const uint32_t *__restrict__ send_ordinals,
    size_t num_sends,
    const uint32_t *__restrict__ recv_orig_indices,
    const uint64_t *__restrict__ recv_keys,
    const uint32_t *__restrict__ recv_ordinals,
    size_t num_recvs,
    int32_t *__restrict__ match_partner
) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (idx >= num_sends) return;

  uint64_t key = send_keys[idx];
  uint32_t ordinal = send_ordinals[idx];
  uint32_t send_orig = send_orig_indices[idx];

  // Binary search for lower_bound of key in recv_keys
  size_t lo = 0, hi = num_recvs;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (recv_keys[mid] < key)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo >= num_recvs || recv_keys[lo] != key) return;

  size_t target = lo + ordinal;
  if (target >= num_recvs || recv_keys[target] != key) return;
  if (recv_ordinals[target] != ordinal) return;

  uint32_t recv_orig = recv_orig_indices[target];
  match_partner[send_orig] = (int32_t)recv_orig;
  match_partner[recv_orig] = (int32_t)send_orig;
}

void runGPUP2PMatching(TraceDataSoA &data) {
  size_t n = data.count;
  if (n == 0) return;

  auto t_start = std::chrono::high_resolution_clock::now();

  // Check GPU availability
  int device;
  cudaError_t dev_err = cudaGetDevice(&device);
  if (dev_err != cudaSuccess) {
    std::cerr << "[GPU P2P] No GPU, falling back to CPU" << std::endl;
    runP2PMatching(data);
    return;
  }

  // Clear any stale CUDA errors
  cudaGetLastError();

  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  std::cout << "[GPU P2P] GPU memory: " << (free_mem / (1024*1024))
            << " MB free / " << (total_mem / (1024*1024)) << " MB total, "
            << n << " events" << std::endl;

  // Estimate: ~80 bytes per event peak (multiple arrays simultaneously live)
  size_t estimated_mem = n * 80;
  if (estimated_mem > free_mem * 0.8) {
    std::cout << "[GPU P2P] Insufficient GPU memory, falling back to CPU" << std::endl;
    runP2PMatching(data);
    return;
  }

  try {
    auto t_upload = std::chrono::high_resolution_clock::now();

    // Step 0: Upload input arrays to GPU via cudaMalloc + cudaMemcpy
    event_t *d_events = nullptr;
    uint32_t *d_pids = nullptr, *d_srcs = nullptr, *d_dsts = nullptr, *d_tags = nullptr;
    uint64_t *d_timestamps = nullptr;
    int *d_is_send = nullptr, *d_is_recv = nullptr;
    uint64_t *d_match_keys = nullptr, *d_ts_copy = nullptr;

    CUDA_CHECK(cudaMalloc(&d_events, n * sizeof(event_t)));
    CUDA_CHECK(cudaMalloc(&d_pids, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_srcs, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dsts, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_tags, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_timestamps, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_is_send, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_recv, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_match_keys, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ts_copy, n * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(d_events, data.events, n * sizeof(event_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pids, data.pids, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_srcs, data.srcs, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dsts, data.dsts, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tags, data.tags, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_timestamps, data.timestamps, n * sizeof(uint64_t), cudaMemcpyHostToDevice));

    auto t_classify = std::chrono::high_resolution_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(t_classify - t_upload).count();

    // Step 1: Classify events
    int block_size = 256;
    int num_blocks = (int)((n + block_size - 1) / block_size);

    kernelClassifyP2P<<<num_blocks, block_size>>>(
        d_events, d_pids, d_srcs, d_dsts, d_tags, d_timestamps, n,
        d_is_send, d_is_recv, d_match_keys, d_ts_copy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free input arrays no longer needed
    CUDA_CHECK(cudaFree(d_events));
    CUDA_CHECK(cudaFree(d_pids));
    CUDA_CHECK(cudaFree(d_srcs));
    CUDA_CHECK(cudaFree(d_dsts));
    CUDA_CHECK(cudaFree(d_tags));
    CUDA_CHECK(cudaFree(d_timestamps));

    // Count sends and recvs on CPU (download is_send/is_recv and sum).
    // thrust::reduce was found unreliable in multi-TU separable compilation
    // builds — returns garbage for the second reduce call while cudaMemcpy
    // returns correct data. CPU counting is negligible overhead for ~500K ints.
    std::vector<int> h_is_send(n), h_is_recv(n);
    CUDA_CHECK(cudaMemcpy(h_is_send.data(), d_is_send, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_is_recv.data(), d_is_recv, n * sizeof(int), cudaMemcpyDeviceToHost));
    size_t num_sends = 0, num_recvs = 0;
    for (size_t i = 0; i < n; i++) {
      num_sends += h_is_send[i];
      num_recvs += h_is_recv[i];
    }

    auto tp_is_send = thrust::device_pointer_cast(d_is_send);
    auto tp_is_recv = thrust::device_pointer_cast(d_is_recv);

    auto t_compact = std::chrono::high_resolution_clock::now();
    double classify_ms = std::chrono::duration<double, std::milli>(t_compact - t_classify).count();

    if (num_sends == 0 && num_recvs == 0) {
      CUDA_CHECK(cudaFree(d_is_send));
      CUDA_CHECK(cudaFree(d_is_recv));
      CUDA_CHECK(cudaFree(d_match_keys));
      CUDA_CHECK(cudaFree(d_ts_copy));
      auto t_end = std::chrono::high_resolution_clock::now();
      double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
      std::cout << "[GPU P2P] No P2P events (" << total_ms << " ms)" << std::endl;
      return;
    }

    // Step 2: Compact sends and recvs
    thrust::device_vector<uint32_t> d_orig_indices(n);
    thrust::sequence(d_orig_indices.begin(), d_orig_indices.end());

    auto tp_match_keys = thrust::device_pointer_cast(d_match_keys);
    auto tp_ts_copy = thrust::device_pointer_cast(d_ts_copy);

    // Compact sends
    thrust::device_vector<uint32_t> d_send_orig(num_sends);
    thrust::device_vector<uint64_t> d_send_keys(num_sends);
    thrust::device_vector<uint64_t> d_send_ts(num_sends);

    auto src_begin = thrust::make_zip_iterator(thrust::make_tuple(
        d_orig_indices.begin(), tp_match_keys, tp_ts_copy));
    auto src_end = thrust::make_zip_iterator(thrust::make_tuple(
        d_orig_indices.end(), tp_match_keys + n, tp_ts_copy + n));
    auto send_dst = thrust::make_zip_iterator(thrust::make_tuple(
        d_send_orig.begin(), d_send_keys.begin(), d_send_ts.begin()));

    thrust::copy_if(src_begin, src_end, tp_is_send, send_dst, IsNonZero());

    // Compact recvs
    thrust::device_vector<uint32_t> d_recv_orig(num_recvs);
    thrust::device_vector<uint64_t> d_recv_keys(num_recvs);
    thrust::device_vector<uint64_t> d_recv_ts(num_recvs);

    auto recv_dst = thrust::make_zip_iterator(thrust::make_tuple(
        d_recv_orig.begin(), d_recv_keys.begin(), d_recv_ts.begin()));

    thrust::copy_if(src_begin, src_end, tp_is_recv, recv_dst, IsNonZero());

    // Free intermediates
    d_orig_indices.clear(); d_orig_indices.shrink_to_fit();
    CUDA_CHECK(cudaFree(d_is_send));
    CUDA_CHECK(cudaFree(d_is_recv));
    CUDA_CHECK(cudaFree(d_match_keys));
    CUDA_CHECK(cudaFree(d_ts_copy));

    auto t_sort = std::chrono::high_resolution_clock::now();
    double compact_ms = std::chrono::duration<double, std::milli>(t_sort - t_compact).count();

    // Step 3: Sort by (key, timestamp) using two-pass stable sort
    thrust::stable_sort_by_key(d_send_ts.begin(), d_send_ts.end(),
                                thrust::make_zip_iterator(thrust::make_tuple(
                                    d_send_orig.begin(), d_send_keys.begin())));
    thrust::stable_sort_by_key(d_send_keys.begin(), d_send_keys.end(),
                                thrust::make_zip_iterator(thrust::make_tuple(
                                    d_send_orig.begin(), d_send_ts.begin())));

    thrust::stable_sort_by_key(d_recv_ts.begin(), d_recv_ts.end(),
                                thrust::make_zip_iterator(thrust::make_tuple(
                                    d_recv_orig.begin(), d_recv_keys.begin())));
    thrust::stable_sort_by_key(d_recv_keys.begin(), d_recv_keys.end(),
                                thrust::make_zip_iterator(thrust::make_tuple(
                                    d_recv_orig.begin(), d_recv_ts.begin())));

    d_send_ts.clear(); d_send_ts.shrink_to_fit();
    d_recv_ts.clear(); d_recv_ts.shrink_to_fit();

    auto t_ordinal = std::chrono::high_resolution_clock::now();
    double sort_ms = std::chrono::duration<double, std::milli>(t_ordinal - t_sort).count();

    // Step 4: Assign within-group ordinals via exclusive_scan_by_key
    thrust::device_vector<uint32_t> d_send_ordinals(num_sends);
    thrust::device_vector<uint32_t> d_recv_ordinals(num_recvs);

    {
      thrust::device_vector<uint32_t> ones(num_sends, 1);
      thrust::exclusive_scan_by_key(d_send_keys.begin(), d_send_keys.end(),
                                     ones.begin(), d_send_ordinals.begin(),
                                     (uint32_t)0, thrust::equal_to<uint64_t>(),
                                     thrust::plus<uint32_t>());
    }
    {
      thrust::device_vector<uint32_t> ones(num_recvs, 1);
      thrust::exclusive_scan_by_key(d_recv_keys.begin(), d_recv_keys.end(),
                                     ones.begin(), d_recv_ordinals.begin(),
                                     (uint32_t)0, thrust::equal_to<uint64_t>(),
                                     thrust::plus<uint32_t>());
    }

    auto t_match = std::chrono::high_resolution_clock::now();
    double ordinal_ms = std::chrono::duration<double, std::milli>(t_match - t_ordinal).count();

    // Step 5: Match sends to recvs via binary search
    thrust::device_vector<int32_t> d_match_partner(n, -1);

    int match_blocks = (int)((num_sends + block_size - 1) / block_size);
    if (match_blocks > 0) {
      kernelWriteMatches<<<match_blocks, block_size>>>(
          thrust::raw_pointer_cast(d_send_orig.data()),
          thrust::raw_pointer_cast(d_send_keys.data()),
          thrust::raw_pointer_cast(d_send_ordinals.data()),
          num_sends,
          thrust::raw_pointer_cast(d_recv_orig.data()),
          thrust::raw_pointer_cast(d_recv_keys.data()),
          thrust::raw_pointer_cast(d_recv_ordinals.data()),
          num_recvs,
          thrust::raw_pointer_cast(d_match_partner.data()));
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(data.match_partner,
                           thrust::raw_pointer_cast(d_match_partner.data()),
                           n * sizeof(int32_t), cudaMemcpyDeviceToHost));

    auto t_end = std::chrono::high_resolution_clock::now();
    double match_ms = std::chrono::duration<double, std::milli>(t_end - t_match).count();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Count matches and unmatched on host
    size_t match_count = 0, unmatched_send = 0, unmatched_recv = 0;
    for (size_t i = 0; i < n; i++) {
      event_t ev = data.events[i];
      if (ev == TT_MPI_Send || ev == TT_MPI_Isend) {
        if (data.match_partner[i] >= 0) match_count++;
        else unmatched_send++;
      } else if (ev == TT_MPI_Recv || ev == TT_MPI_Irecv) {
        if (data.match_partner[i] < 0) unmatched_recv++;
      }
    }

    std::cout << "[GPU P2P] " << match_count << " pairs from "
              << num_sends << " sends + " << num_recvs << " recvs" << std::endl;
    if (unmatched_send > 0 || unmatched_recv > 0)
      std::cout << "[GPU P2P] Unmatched: " << unmatched_send << " sends, "
                << unmatched_recv << " recvs" << std::endl;
    std::cout << "[GPU P2P] " << std::fixed << std::setprecision(1)
              << total_ms << " ms (upload=" << upload_ms
              << " classify=" << classify_ms << " compact=" << compact_ms
              << " sort=" << sort_ms << " ordinal=" << ordinal_ms
              << " match=" << match_ms << ")" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "[GPU P2P] Failed (" << e.what()
              << "), falling back to CPU" << std::endl;
    cudaGetLastError();
    memset(data.match_partner, 0xFF, n * sizeof(int32_t));
    runP2PMatching(data);
  }
}
