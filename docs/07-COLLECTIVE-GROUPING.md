# Collective Event Grouping

**Source**: `src/matching/CollectiveGrouping.cpp`
**Header**: `include/matching/CollectiveGrouping.h`

## Overview

Collective MPI operations (Barrier, Reduce, Bcast, AllReduce, etc.) involve multiple processes participating in a single operation. Before analysis, we must group individual per-process events into complete "groups" where all communicator members have participated.

The output is a **CSR (Compressed Sparse Row)** structure that maps each group to its member event indices, suitable for efficient GPU kernel processing.

## Algorithm: Comm-Set-Based Temporal Grouping

### Input

- `TraceDataSoA data`: events with type, timestamp, pid, root
- `comm_sets`: per-collective-event communicator member lists (from OTF2 reader)

### Key Insight

A collective group is uniquely identified by `(event_type, communicator_member_set)`. Two events belong to the same group instance if:

1. They have the same MPI operation type (e.g., both are `MPI_Barrier`)
2. They use the same communicator (same set of member ranks)
3. The pid hasn't already contributed to this group (no double-counting)
4. They occur in temporal order (processed in timestamp order to match iterations)

### Algorithm Steps

```
1. Collect all collective event indices from the SoA data
2. Sort by timestamp
3. For each collective event (in timestamp order):
   a. Compute its key: (event_type, sorted_comm_set)
   b. Search pending groups of the same type for one with:
      - Matching comm_set
      - This pid still needed (hasn't contributed yet)
   c. If found:
      - Add this event to the group
      - Remove this pid from the group's needed_pids set
      - If needed_pids is empty → group is complete → move to completed list
   d. If not found:
      - Create a new pending group with this event as first member
      - Set needed_pids = all comm_set members except this pid
4. After processing all events, include incomplete groups with ≥2 members
5. Convert completed groups to CSR format
```

### Data Structures

```cpp
struct PendingGroup {
    event_t type;                          // MPI operation type
    id_t root;                             // Root PID (for rooted collectives)
    std::vector<uint64_t> comm_set_key;    // Sorted communicator members (for comparison)
    std::set<id_t> needed_pids;            // PIDs that haven't contributed yet
    std::vector<size_t> member_indices;    // SoA event indices of participating events
    size_t expected_size;                  // Total communicator size
};

// Groups organized by event type for faster lookup
std::unordered_map<int, std::vector<PendingGroup>> pending_by_type;
```

### Comm-Set Normalization

Communicator member sets are sorted before comparison, so `{0, 3, 1, 2}` and `{1, 0, 2, 3}` are treated as the same communicator:

```cpp
auto normalizeCommSet = [](const std::vector<uint64_t> &cs) {
    std::vector<uint64_t> sorted_cs(cs);
    std::sort(sorted_cs.begin(), sorted_cs.end());
    return sorted_cs;
};
```

### Handling Multiple Concurrent Collectives

The algorithm correctly handles multiple concurrent collectives of the same type with the same communicator. For example:

```
Iteration 1: All 64 processes call MPI_Barrier
Iteration 2: All 64 processes call MPI_Barrier
```

Because events are processed in timestamp order and groups are completed when all members arrive, iteration 1's barrier is completed before iteration 2's events start a new group.

## Output: CSR Format

The CSR output contains:

```
num_groups = 63 (for CG trace: 63 broadcast groups)
offsets    = [0, 64, 128, ..., 4032]  (each group has 64 members)
members    = [idx0, idx1, ..., idx63, idx64, ...]  (SoA event indices)
group_types = [TT_MPI_Bcast, TT_MPI_Bcast, ...]
group_roots = [0, 0, ...]
```

See [04-DATA-STRUCTURES.md](./04-DATA-STRUCTURES.md) for CSR layout details.

## Why CPU, Not GPU

Collective grouping runs on CPU because:

1. **Small data volume**: Collective events are typically <15% of total events. For CG-B with 2.5M events, only ~4K are collective events, forming ~63 groups.
2. **Variable-length structures**: Groups have variable sizes, and the pending group management requires dynamic allocation (sets, vectors)
3. **Negligible time**: Grouping takes ~1.8 ms — 0.01% of total execution time

## Comm-Set to Event Index Mapping

The reader stores comm_sets in a separate flat list (not embedded in the SoA). The mapping between SoA event indices and comm_set indices is maintained by processing events in order:

```cpp
size_t cs_idx = 0;
for (size_t i = 0; i < n; i++) {
    event_t ev = data.events[i];
    if (ev >= TT_MPI_Bcast && ev <= TT_MPI_AlltoAll) {
        soa_to_commset[i] = cs_idx;
        cs_idx++;
    }
}
```

This assumes:
1. Collective events in the SoA appear in the same order as comm_sets
2. The number of collective events equals the number of comm_sets

This invariant is maintained because both are populated by the same reader callback in the same order.

## Incomplete Groups

After processing all events, pending groups with ≥2 members are included in the completed list:

```cpp
for (auto &[type_key, plist] : pending_by_type) {
    for (auto &pg : plist) {
        if (pg.member_indices.size() >= 2) {
            completed_groups.push_back(std::move(pg));
        }
    }
}
```

This handles cases where some processes' events were lost or filtered. In practice, with correct OTF2 traces, all groups should be complete.

## Diagnostic Output

The module prints group counts per collective type:

```
[CollectiveGrouping] Built 63 groups with 4032 total members
  MPI_Bcast: 63 groups
```
