#include "matching/P2PMatching.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

// CPU-based P2P matching using timestamp-sorted events and FIFO queues.
// This mirrors TileTrace's InteractionPattern::analysis() logic:
// events are processed in timestamp order; for each send, we look for
// a matching recv (same src, dst, tag) in the unmatched recv queue and
// vice versa. This ensures correct temporal ordering of matches.
//
// Note on GPU sort: thrust::sort_by_key was tested but fails when 64 MPI
// ranks compete for the same GPU (cudaErrorInvalidValue / OOM). The
// matching loop itself is inherently sequential (FIFO queue state). Since
// per-rank data is ~4M events and std::sort takes ~40ms without contention,
// GPU sort is not worthwhile here.
void runP2PMatching(TraceDataSoA &data) {
  size_t n = data.count;
  if (n == 0)
    return;

  // Build sorted index by timestamp
  std::vector<size_t> sorted_idx(n);
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::sort(sorted_idx.begin(), sorted_idx.end(),
            [&](size_t a, size_t b) {
              return data.timestamps[a] < data.timestamps[b];
            });

  // Queue-based matching with composite key = (sender, receiver, tag)
  // For Send: sender=pids[i], receiver=dsts[i], tag=tags[i]
  //   -> key = (pids[i], dsts[i], tags[i])
  // For Recv: sender=srcs[i], receiver=pids[i], tag=tags[i]
  //   -> key = (srcs[i], pids[i], tags[i])
  // Both produce key = (sender_rank, receiver_rank, tag)

  auto makeKey = [](id_t a, id_t b, id_t c) -> uint64_t {
    return ((uint64_t)a << 42) | ((uint64_t)b << 21) | (uint64_t)c;
  };

  // Unmatched send/recv queues indexed by key
  std::unordered_map<uint64_t, std::queue<size_t>> unmatched_sends;
  std::unordered_map<uint64_t, std::queue<size_t>> unmatched_recvs;

  size_t match_count = 0;

  for (size_t si = 0; si < n; si++) {
    size_t i = sorted_idx[si];
    event_t ev = data.events[i];

    if (ev == TT_MPI_Send || ev == TT_MPI_Isend) {
      // Send: key = (sender=pid, receiver=dst, tag)
      uint64_t k = makeKey(data.pids[i], data.dsts[i], data.tags[i]);

      // Try to match with an existing unmatched recv
      auto it = unmatched_recvs.find(k);
      if (it != unmatched_recvs.end() && !it->second.empty()) {
        size_t recv_idx = it->second.front();
        it->second.pop();
        data.match_partner[i] = (int32_t)recv_idx;
        data.match_partner[recv_idx] = (int32_t)i;
        match_count++;
      } else {
        // No matching recv yet, queue this send
        unmatched_sends[k].push(i);
      }
    } else if (ev == TT_MPI_Recv || ev == TT_MPI_Irecv) {
      // Recv: key = (sender=src, receiver=pid, tag)
      uint64_t k = makeKey(data.srcs[i], data.pids[i], data.tags[i]);

      // Try to match with an existing unmatched send
      auto it = unmatched_sends.find(k);
      if (it != unmatched_sends.end() && !it->second.empty()) {
        size_t send_idx = it->second.front();
        it->second.pop();
        data.match_partner[i] = (int32_t)send_idx;
        data.match_partner[send_idx] = (int32_t)i;
        match_count++;
      } else {
        // No matching send yet, queue this recv
        unmatched_recvs[k].push(i);
      }
    }
  }

  // Report unmatched
  size_t unmatched_send = 0, unmatched_recv = 0;
  for (auto &[k, q] : unmatched_sends)
    unmatched_send += q.size();
  for (auto &[k, q] : unmatched_recvs)
    unmatched_recv += q.size();
  if (unmatched_send > 0 || unmatched_recv > 0) {
    std::cout << "[P2P Matching] Unmatched: " << unmatched_send << " sends, "
              << unmatched_recv << " recvs" << std::endl;
  }
}
