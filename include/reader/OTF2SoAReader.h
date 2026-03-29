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

// Split-phase reader API: allows caller to allocate SoA memory between
// pass 2 and fillSoA (e.g., for shared memory windows).

struct ReaderPhase1Output {
  size_t event_count;
  std::vector<std::vector<uint64_t>> comm_sets;
  void *handle; // opaque; must pass to readerFillSoA then readerRelease
};

// Phase 1: pass1 + pass2 + redistribution. Returns event count, comm_sets,
// and an opaque handle. Does NOT allocate or fill TraceDataSoA.
ReaderPhase1Output readOTF2TracePhase1(const std::string &trace_path);

// Phase 2: copy parsed vectors into pre-set SoA pointers.
// Caller must have set all 14 pointer fields and data.capacity.
// Initializes: match_partner=-1, coll_group_id=-1, tids=0, indices[i]=i,
// replay_pids=pids. Does NOT call data.allocate().
void readerFillSoA(void *handle, TraceDataSoA &data);

// Free the opaque handle. Must be called exactly once after readerFillSoA.
void readerRelease(void *handle);

#endif // GPU_ANALYZER_OTF2_SOA_READER_H
