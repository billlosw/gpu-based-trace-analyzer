#ifndef GPU_ANALYZER_OTF2_SOA_READER_H
#define GPU_ANALYZER_OTF2_SOA_READER_H

#include "data/TraceDataSoA.h"
#include <string>
#include <vector>

struct ReaderOutput {
  TraceDataSoA data;
  // Per-event comm_set for collective events (indexed by position in SoA).
  // Empty vector for non-collective events.
  std::vector<std::vector<uint64_t>> comm_sets;
};

// Read an OTF2 trace file into SoA format using distributed two-pass reading.
// Each MPI rank reads a contiguous block of locations plus related locations
// (communication partners discovered in pass 1). Collective events are
// redistributed by root location. Each rank returns only its local portion.
// trace_path: path to the .otf2 anchor file
ReaderOutput readOTF2Trace(const std::string &trace_path);

#endif // GPU_ANALYZER_OTF2_SOA_READER_H
