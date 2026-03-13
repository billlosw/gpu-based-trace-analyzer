#ifndef GPU_ANALYZER_TRACE_DATA_SOA_H
#define GPU_ANALYZER_TRACE_DATA_SOA_H

#include "common/types.h"
#include <cstdlib>
#include <cstring>
#include <vector>

struct TraceDataSoA {
  size_t count = 0;
  size_t capacity = 0;

  // 12 base arrays
  event_t *events = nullptr;
  event_type_t *types = nullptr;
  timestamp_t *timestamps = nullptr;
  timestamp_t *end_timestamps = nullptr;
  id_t *pids = nullptr;
  id_t *tids = nullptr;
  id_t *replay_pids = nullptr;
  id_t *srcs = nullptr;
  id_t *dsts = nullptr;
  id_t *tags = nullptr;
  id_t *roots = nullptr;
  id_t *indices = nullptr;

  // P2P matching result: index of matched partner, -1 if unmatched
  int32_t *match_partner = nullptr;

  // Collective group ID for each event, -1 for non-collective
  int32_t *coll_group_id = nullptr;

  TraceDataSoA() = default;
  TraceDataSoA(const TraceDataSoA &) = delete;
  TraceDataSoA &operator=(const TraceDataSoA &) = delete;
  TraceDataSoA(TraceDataSoA &&o) noexcept { moveFrom(o); }
  TraceDataSoA &operator=(TraceDataSoA &&o) noexcept {
    if (this != &o) {
      deallocate();
      moveFrom(o);
    }
    return *this;
  }
  ~TraceDataSoA() { deallocate(); }

  void allocate(size_t n) {
    capacity = n;
    count = 0;
#define ALLOC_FIELD(field, type) field = (type *)calloc(n, sizeof(type))
    ALLOC_FIELD(events, event_t);
    ALLOC_FIELD(types, event_type_t);
    ALLOC_FIELD(timestamps, timestamp_t);
    ALLOC_FIELD(end_timestamps, timestamp_t);
    ALLOC_FIELD(pids, id_t);
    ALLOC_FIELD(tids, id_t);
    ALLOC_FIELD(replay_pids, id_t);
    ALLOC_FIELD(srcs, id_t);
    ALLOC_FIELD(dsts, id_t);
    ALLOC_FIELD(tags, id_t);
    ALLOC_FIELD(roots, id_t);
    ALLOC_FIELD(indices, id_t);
    ALLOC_FIELD(match_partner, int32_t);
    ALLOC_FIELD(coll_group_id, int32_t);
#undef ALLOC_FIELD
    // Initialize match_partner and coll_group_id to -1
    memset(match_partner, 0xFF, n * sizeof(int32_t));
    memset(coll_group_id, 0xFF, n * sizeof(int32_t));
  }

  void deallocate() {
#define FREE_FIELD(field) \
  do {                    \
    free(field);          \
    field = nullptr;      \
  } while (0)
    FREE_FIELD(events);
    FREE_FIELD(types);
    FREE_FIELD(timestamps);
    FREE_FIELD(end_timestamps);
    FREE_FIELD(pids);
    FREE_FIELD(tids);
    FREE_FIELD(replay_pids);
    FREE_FIELD(srcs);
    FREE_FIELD(dsts);
    FREE_FIELD(tags);
    FREE_FIELD(roots);
    FREE_FIELD(indices);
    FREE_FIELD(match_partner);
    FREE_FIELD(coll_group_id);
#undef FREE_FIELD
    count = 0;
    capacity = 0;
  }

  size_t sizeInBytes() const {
    return capacity * (sizeof(event_t) + sizeof(event_type_t) +
                       2 * sizeof(timestamp_t) + 8 * sizeof(id_t) +
                       2 * sizeof(int32_t));
  }

private:
  void moveFrom(TraceDataSoA &o) {
    count = o.count;
    capacity = o.capacity;
    events = o.events;
    types = o.types;
    timestamps = o.timestamps;
    end_timestamps = o.end_timestamps;
    pids = o.pids;
    tids = o.tids;
    replay_pids = o.replay_pids;
    srcs = o.srcs;
    dsts = o.dsts;
    tags = o.tags;
    roots = o.roots;
    indices = o.indices;
    match_partner = o.match_partner;
    coll_group_id = o.coll_group_id;
    o.count = 0;
    o.capacity = 0;
    o.events = nullptr;
    o.types = nullptr;
    o.timestamps = nullptr;
    o.end_timestamps = nullptr;
    o.pids = nullptr;
    o.tids = nullptr;
    o.replay_pids = nullptr;
    o.srcs = nullptr;
    o.dsts = nullptr;
    o.tags = nullptr;
    o.roots = nullptr;
    o.indices = nullptr;
    o.match_partner = nullptr;
    o.coll_group_id = nullptr;
  }
};

// Collective group CSR for GPU transfer
struct CollectiveGroupCSR {
  size_t num_groups = 0;
  int32_t *offsets = nullptr;   // size: num_groups + 1
  int32_t *members = nullptr;   // size: total members
  event_t *group_types = nullptr; // size: num_groups
  id_t *group_roots = nullptr;     // size: num_groups (root pid for each group)
  size_t total_members = 0;

  CollectiveGroupCSR() = default;
  ~CollectiveGroupCSR() { deallocate(); }
  CollectiveGroupCSR(const CollectiveGroupCSR &) = delete;
  CollectiveGroupCSR &operator=(const CollectiveGroupCSR &) = delete;

  void deallocate() {
    free(offsets);
    free(members);
    free(group_types);
    free(group_roots);
    offsets = nullptr;
    members = nullptr;
    group_types = nullptr;
    group_roots = nullptr;
    num_groups = 0;
    total_members = 0;
  }
};

#endif // GPU_ANALYZER_TRACE_DATA_SOA_H
