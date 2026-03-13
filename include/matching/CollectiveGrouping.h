#ifndef GPU_ANALYZER_COLLECTIVE_GROUPING_H
#define GPU_ANALYZER_COLLECTIVE_GROUPING_H

#include "data/TraceDataSoA.h"
#include <cstdint>
#include <vector>

// Build collective group CSR from SoA data on the CPU.
// Events must already be sorted by timestamp (as read from OTF2).
// comm_sets: per-event communicator member lists (from OTF2 reader).
void buildCollectiveGroups(const TraceDataSoA &data,
                           const std::vector<std::vector<uint64_t>> &comm_sets,
                           CollectiveGroupCSR &out_csr);

#endif // GPU_ANALYZER_COLLECTIVE_GROUPING_H
