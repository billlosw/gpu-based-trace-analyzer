#include "analysis/TimestampCorrection.h"

#ifdef USE_SCALASCA_TIMESTAMPS

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

// Minimum clock skew threshold to consider a pair as a genuine
// cross-node clock violation. Same-node "violations" (recv_enter < send_leave)
// are typically < 100μs because shared-memory MPI can deliver messages
// before MPI_Send returns. Real inter-node clock skew > 100μs.
static constexpr timestamp_t SKEW_THRESHOLD_PS = 100000000; // 100 μs in ps

// Detect locations per node from significant clock violation patterns.
// Only pairs with |send_leave - recv_enter| > SKEW_THRESHOLD are considered
// genuine cross-node violations (not same-node fast transfers).
// Returns 0 if no significant violations found (single-node trace or
// negligible clock skew).
static size_t detectLocsPerNode(const TraceDataSoA &data) {
  size_t min_cross_node_distance = SIZE_MAX;
  size_t n = data.count;

  for (size_t i = 0; i < n; i++) {
    event_t ev = data.events[i];
    if (ev != TT_MPI_Recv && ev != TT_MPI_Irecv)
      continue;
    if (data.match_partner[i] < 0)
      continue;

    int32_t send_idx = data.match_partner[i];
    timestamp_t recv_enter = data.timestamps[i];
    timestamp_t send_leave = data.end_timestamps[send_idx];

    // Only consider significant violations (> 100μs)
    if (send_leave > recv_enter + SKEW_THRESHOLD_PS) {
      id_t recv_pid = data.pids[i];
      id_t send_pid = data.pids[send_idx];
      size_t distance = (recv_pid > send_pid) ? (recv_pid - send_pid)
                                              : (send_pid - recv_pid);
      if (distance > 0 && distance < min_cross_node_distance) {
        min_cross_node_distance = distance;
      }
    }
  }

  if (min_cross_node_distance == SIZE_MAX)
    return 0;

  // Round down to nearest power of 2
  size_t ppn = 1;
  while (ppn * 2 <= min_cross_node_distance)
    ppn *= 2;

  return ppn;
}

size_t applyTimestampCorrection(TraceDataSoA &data) {
  size_t n = data.count;
  if (n == 0)
    return 0;

  size_t locs_per_node = detectLocsPerNode(data);

  if (locs_per_node == 0) {
    // No significant clock violations -> nothing to correct
    return 0;
  }

  std::cout << "[CLC] Detected " << locs_per_node << " locations per node"
            << std::endl;

  // For cross-node P2P pairs with clock condition violations:
  // Neutralize the late_receiver check by adjusting end_timestamps[recv]
  // so that the late_receiver condition (send_leave > recv_req_enter) fails.
  //
  // This only affects end_timestamps for recv events (used by late_receiver).
  // timestamps[] (used by late_sender) and collective event timestamps are
  // left unchanged, preserving exact results for all other metrics.

  size_t violations = 0;
  size_t corrections = 0;

  for (size_t i = 0; i < n; i++) {
    event_t ev = data.events[i];
    if (ev != TT_MPI_Recv && ev != TT_MPI_Irecv)
      continue;
    if (data.match_partner[i] < 0)
      continue;

    int32_t send_idx = data.match_partner[i];

    // Check if cross-node
    id_t recv_pid = data.pids[i];
    id_t send_pid = data.pids[send_idx];
    size_t recv_node = recv_pid / locs_per_node;
    size_t send_node = send_pid / locs_per_node;

    if (recv_node == send_node)
      continue;

    // For cross-node pairs, check clock condition
    timestamp_t recv_enter = data.timestamps[i];
    timestamp_t send_leave = data.end_timestamps[send_idx];

    if (send_leave > recv_enter) {
      violations++;
      // Neutralize: set recv_req_enter = send_leave
      // This makes (send_leave > recv_req_enter) false
      data.end_timestamps[i] = send_leave;
      corrections++;
    }
  }

  if (violations > 0) {
    std::cout << "[CLC] " << violations
              << " cross-node clock violations corrected" << std::endl;
  }

  return violations;
}

#endif // USE_SCALASCA_TIMESTAMPS
