#ifndef GPU_ANALYZER_GPU_COLLECTIVE_GROUPING_H
#define GPU_ANALYZER_GPU_COLLECTIVE_GROUPING_H

#include "data/TraceDataSoA.h"
#include <cstdint>
#include <vector>

// GPU sort-based collective grouping.
// Uses the segment-scan algorithm:
//   1. Filter collective events
//   2. Sort by (event_type, comm_set_hash, timestamp)
//   3. Segment by (event_type, comm_set_hash)
//   4. group_id = position_within_segment / communicator_size
//   5. Build CSR from sorted results
//
// Falls back to CPU grouping if GPU allocation fails.
void buildGPUCollectiveGroups(
    const TraceDataSoA &data,
    const std::vector<std::vector<uint64_t>> &comm_sets,
    const std::vector<uint64_t> &coll_bytes_sent,
    const std::vector<uint64_t> &coll_bytes_received,
    CollectiveGroupCSR &out_csr);

#endif // GPU_ANALYZER_GPU_COLLECTIVE_GROUPING_H
