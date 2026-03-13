#include "matching/P2PMatching.h"
#include "common/cuda_check.h"

#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>

// Hash table entry for P2P matching
struct MatchHashEntry {
  unsigned long long key; // composite key, 0 = empty sentinel
  int32_t event_index;    // index of the recv event, -1 = consumed
};

// Composite key: encode (src, dst, tag) into a 64-bit value.
// +1 offsets avoid zero (the empty sentinel).
__device__ __host__ unsigned long long makeMatchKey(id_t a, id_t b, id_t c) {
  return ((unsigned long long)(a + 1) << 42) |
         ((unsigned long long)(b + 1) << 21) | (unsigned long long)(c + 1);
}

// Phase 1: Insert Recv/Irecv events into hash table.
// For a Recv event at index i with src=srcs[i], pid=pids[i] (=dst), tag=tags[i],
// key = (srcs[i], pids[i], tags[i]) so matching Send can find it.
__global__ void kernelBuildRecvHash(const event_t *__restrict__ events,
                                    const id_t *__restrict__ srcs,
                                    const id_t *__restrict__ pids,
                                    const id_t *__restrict__ tags,
                                    size_t n,
                                    MatchHashEntry *__restrict__ table,
                                    size_t table_size) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;

  for (size_t i = idx; i < n; i += stride) {
    event_t ev = events[i];
    if (ev != TT_MPI_Recv && ev != TT_MPI_Irecv)
      continue;

    // Key: (src_of_recv, pid_of_recv, tag)
    unsigned long long k = makeMatchKey(srcs[i], pids[i], tags[i]);

    // Open-addressing insert with linear probing
    size_t slot = k % table_size;
    for (size_t probe = 0; probe < table_size; probe++) {
      size_t s = (slot + probe) % table_size;
      unsigned long long old = atomicCAS(&table[s].key, 0ULL, k);
      if (old == 0ULL) {
        // Claimed empty slot
        table[s].event_index = (int32_t)i;
        break;
      }
      // Slot taken - continue probing to find another empty slot
      // (handles multiple recvs with same key)
    }
  }
}

// Phase 2: Match Send/Isend events against the recv hash table.
// For a Send at index i with pid=pids[i] (=src), dst=dsts[i], tag=tags[i],
// key = (pids[i], dsts[i], tags[i]) which matches Recv's (srcs[recv], pids[recv], tags[recv]).
__global__ void kernelMatchSendRecv(const event_t *__restrict__ events,
                                    const id_t *__restrict__ pids,
                                    const id_t *__restrict__ dsts,
                                    const id_t *__restrict__ tags,
                                    size_t n,
                                    MatchHashEntry *__restrict__ table,
                                    size_t table_size,
                                    int32_t *__restrict__ match_partner) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;

  for (size_t i = idx; i < n; i += stride) {
    event_t ev = events[i];
    if (ev != TT_MPI_Send && ev != TT_MPI_Isend)
      continue;

    // Key: (pid_of_send, dst_of_send, tag)
    unsigned long long k = makeMatchKey(pids[i], dsts[i], tags[i]);

    size_t slot = k % table_size;
    for (size_t probe = 0; probe < table_size; probe++) {
      size_t s = (slot + probe) % table_size;
      if (table[s].key == 0ULL)
        break; // empty slot => no match found

      if (table[s].key == k) {
        // Try to claim this recv
        int32_t recv_idx = atomicExch(&table[s].event_index, -1);
        if (recv_idx >= 0) {
          // Matched: send[i] <-> recv[recv_idx]
          match_partner[i] = recv_idx;
          match_partner[recv_idx] = (int32_t)i;
          break;
        }
        // Already consumed, continue probing
      }
    }
  }
}

void runP2PMatching(TraceDataSoA &data) {
  size_t n = data.count;
  if (n == 0)
    return;

  // Count recv events for hash table sizing
  size_t recv_count = 0;
  for (size_t i = 0; i < n; i++) {
    if (data.events[i] == TT_MPI_Recv || data.events[i] == TT_MPI_Irecv)
      recv_count++;
  }
  if (recv_count == 0)
    return;

  size_t table_size = std::max(recv_count * 2, (size_t)1024);

  // Allocate device memory
  event_t *d_events;
  id_t *d_pids, *d_srcs, *d_dsts, *d_tags;
  int32_t *d_match;
  MatchHashEntry *d_table;

  CUDA_CHECK(cudaMalloc(&d_events, n * sizeof(event_t)));
  CUDA_CHECK(cudaMalloc(&d_pids, n * sizeof(id_t)));
  CUDA_CHECK(cudaMalloc(&d_srcs, n * sizeof(id_t)));
  CUDA_CHECK(cudaMalloc(&d_dsts, n * sizeof(id_t)));
  CUDA_CHECK(cudaMalloc(&d_tags, n * sizeof(id_t)));
  CUDA_CHECK(cudaMalloc(&d_match, n * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_table, table_size * sizeof(MatchHashEntry)));

  // Transfer H2D
  CUDA_CHECK(
      cudaMemcpy(d_events, data.events, n * sizeof(event_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_pids, data.pids, n * sizeof(id_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_srcs, data.srcs, n * sizeof(id_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_dsts, data.dsts, n * sizeof(id_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_tags, data.tags, n * sizeof(id_t), cudaMemcpyHostToDevice));

  // Initialize match_partner to -1 and hash table to 0
  CUDA_CHECK(cudaMemset(d_match, 0xFF, n * sizeof(int32_t)));
  CUDA_CHECK(cudaMemset(d_table, 0, table_size * sizeof(MatchHashEntry)));

  int blockSize = 256;
  int gridSize = (int)std::min((n + 255) / 256, (size_t)1024);

  // Phase 1: Build recv hash table
  kernelBuildRecvHash<<<gridSize, blockSize>>>(d_events, d_srcs, d_pids,
                                               d_tags, n, d_table, table_size);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Phase 2: Match sends
  kernelMatchSendRecv<<<gridSize, blockSize>>>(d_events, d_pids, d_dsts,
                                               d_tags, n, d_table, table_size,
                                               d_match);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy results back
  CUDA_CHECK(cudaMemcpy(data.match_partner, d_match, n * sizeof(int32_t),
                         cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_events));
  CUDA_CHECK(cudaFree(d_pids));
  CUDA_CHECK(cudaFree(d_srcs));
  CUDA_CHECK(cudaFree(d_dsts));
  CUDA_CHECK(cudaFree(d_tags));
  CUDA_CHECK(cudaFree(d_match));
  CUDA_CHECK(cudaFree(d_table));
}
