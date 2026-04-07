#ifndef GPU_ANALYZER_GPU_P2P_MATCHING_H
#define GPU_ANALYZER_GPU_P2P_MATCHING_H

#include "data/TraceDataSoA.h"

// GPU sort-based P2P matching.
// Fills data.match_partner[i] with the index of the matched partner event.
// The algorithm uses the sort-and-rank equivalence:
//   For a fixed matching key (sender, receiver, tag), the i-th send
//   (ordered by timestamp) matches the i-th recv (ordered by timestamp).
// This is equivalent to the CPU FIFO queue approach but fully parallelizable.
//
// Requires: data arrays accessible from host (mmap or heap).
// GPU scratch memory is allocated internally.
// Falls back to CPU matching if GPU allocation fails.
void runGPUP2PMatching(TraceDataSoA &data);

#endif // GPU_ANALYZER_GPU_P2P_MATCHING_H
