# OTF2 Reader — MPI-Parallel Trace Reading

**Source**: `src/reader/OTF2SoAReader.cpp`
**Header**: `include/reader/OTF2SoAReader.h`

## Overview

The OTF2 reader is the most complex module (~480 lines). It reads OTF2 trace files produced by Score-P and converts them into the SoA format used by the rest of the pipeline. It supports MPI-parallel reading for performance.

## otf2xx Callback System

The reader uses the **otf2xx** library (C++ wrapper around the OTF2 C API). otf2xx uses a callback pattern:

1. Create a `reader` object with the trace file path
2. Register a callback class (inheriting `otf2::reader::callback`)
3. Call `reader.read_definitions()` — triggers `definition()` callbacks
4. Call `reader.read_events()` — triggers `event()` callbacks
5. The library calls the appropriate overloaded `event()` method for each event type

```cpp
class SoAReaderCallback : public otf2::reader::callback {
    // Definition callbacks
    void definition(const otf2::definition::location &loc) override;
    void definitions_done(const otf2::reader::reader &) override;

    // Event callbacks
    void event(const otf2::definition::location &, const otf2::event::enter &) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_send &) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_receive &) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_isend_request &) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_ireceive_complete &) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_collective_begin &) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_collective_end &) override;
};
```

## MPI-Parallel Reading

### Location Distribution

OTF2 traces contain one "location" per MPI process in the traced application. Locations are distributed across reader MPI ranks using round-robin:

```cpp
void definition(const otf2::definition::location &loc) override {
    uint64_t loc_id = loc.ref().get();
    if ((int)(loc_id % m_size) == m_rank) {
        m_rdr.register_location(loc);  // Only register locations for this rank
    }
}
```

For a 64-location trace with 8 reading ranks: rank 0 reads locations {0,8,16,...56}, rank 1 reads {1,9,17,...57}, etc.

### MPI Reader Constructor

The key to parallel reading is using otf2xx's MPI-aware reader constructor:

```cpp
otf2::reader::reader rdr(trace_path, MPI_COMM_WORLD);
```

This requires `OTF2XX_WITH_MPI ON` set before `add_subdirectory(third_party/otf2xx)`. Without it, the `OTF2XX_HAS_MPI` define is absent and the MPI constructor is not available, causing a compile error.

### Gathering to Rank 0

After reading, all data is gathered to rank 0 via `MPI_Gatherv`:

```
Step 1: MPI_Allgather event counts (each rank reports how many events it read)
Step 2: Compute displacements (prefix sum of counts)
Step 3: MPI_Gatherv for each SoA array (events[], timestamps[], pids[], etc.)
Step 4: MPI_Gatherv for comm_sets (flattened format)
```

The comm_sets gathering uses a flattened format because `MPI_Gatherv` requires contiguous buffers:

```
Flat format: [size0, member0_0, member0_1, ..., size1, member1_0, ...]
```

Each comm_set is prefixed with its size, then serialized members follow. Rank 0 reconstructs the nested vectors after gathering.

### Single-Process Path

When `comm_sz == 1`, the `MPI_Gatherv` calls are bypassed entirely:

```cpp
if (comm_sz == 1) {
    cb.fillSoA(output.data);
    output.comm_sets = std::move(cb.getCommSets());
} else {
    gatherEventsToRank0(cb, rank, comm_sz, output, ...);
}
```

## Event Processing Details

### Single-Pass Reading

The reader uses a **single-pass** approach with `std::vector` dynamic arrays:

```cpp
std::vector<event_t> m_v_events;
std::vector<timestamp_t> m_v_timestamps;
// ... 9 vectors total

void pushEvent(event_t ev, event_type_t type, timestamp_t ts, ...) {
    m_v_events.push_back(ev);
    m_v_timestamps.push_back(ts);
    // ...
}
```

After reading, the vectors are copied into the fixed-size `TraceDataSoA` via `memcpy`:

```cpp
void fillSoA(TraceDataSoA &data) const {
    size_t n = m_v_events.size();
    data.allocate(n);
    data.count = n;
    std::memcpy(data.events, m_v_events.data(), n * sizeof(event_t));
    // ...
}
```

**History**: The original implementation used a two-pass approach (pass 1: count events, pass 2: fill arrays). This doubled I/O time unnecessarily. The single-pass approach with vectors halves the I/O time at the cost of dynamic memory allocation (which is negligible compared to I/O).

### Enter Timestamp Tracking

The `m_last_enter_ts` map tracks the most recent `Enter` region timestamp for each location:

```cpp
std::unordered_map<id_t, timestamp_t> m_last_enter_ts;

void event(const otf2::definition::location &loc,
           const otf2::event::enter &event) override {
    id_t pid = loc.ref().get();
    m_last_enter_ts[pid] = extractTimestamp(event.timestamp());
}
```

This is used by send/recv handlers to get the Enter timestamp instead of the point-event timestamp. The `enter` callback fires for every region entry (MPI_Send, MPI_Recv, MPI_Wait, etc.), so `m_last_enter_ts` always holds the most recent one when a point event fires.

### P2P Event Field Mapping

For send events:
```
events[i]     = TT_MPI_Send or TT_MPI_Isend
timestamps[i] = Enter(MPI_Send) or Enter(MPI_Isend)
pids[i]       = sender rank
srcs[i]       = sender rank (= pids[i])
dsts[i]       = receiver rank
tags[i]       = message tag
```

For recv events:
```
events[i]         = TT_MPI_Recv or TT_MPI_Irecv
timestamps[i]     = Enter(MPI_Recv) or Enter(MPI_Wait)
end_timestamps[i] = point-event timestamp (completion time)
pids[i]           = receiver rank
srcs[i]           = sender rank
dsts[i]           = receiver rank (= pids[i])
tags[i]           = message tag
```

**Key**: For both sends and receives, the matching key is `(srcs[i], dsts[i], tags[i])` = `(sender_rank, receiver_rank, tag)`.

### Collective Event Processing

Collectives use a begin/end pair:

```cpp
void event(..., const otf2::event::mpi_collective_begin &event) override {
    m_coll_begin_ts[pid] = timestamp;
    m_coll_begin_valid[pid] = true;
}

void event(..., const otf2::event::mpi_collective_end &event) override {
    if (!m_coll_begin_valid[pid]) return;  // Orphan end event
    m_coll_begin_valid[pid] = false;

    // Map OTF2 op type to event_t enum
    // Extract communicator members from event.comm()
    // Push event with begin_ts as timestamp, end_ts as end_timestamp
}
```

The communicator member list is extracted via:
```cpp
auto comm_set = std::get<otf2::definition::comm_group>(
    std::get<otf2::definition::comm>(event.comm()).group()
).members();
```

This navigates: `event.comm()` → `definition::comm` → `group()` → `definition::comm_group` → `members()`.

### Sentinel Root Handling

OTF2 uses `4294967295` (UINT32_MAX) as a sentinel for "no root" in non-rooted collectives (Barrier, AllReduce, etc.). The reader maps this to the minimum rank in the communicator:

```cpp
if (root == 4294967295u) {
    root = pid;
    for (auto id : comm_set)
        root = std::min(root, (uint32_t)id);
}
```

This matches TileTrace's behavior.

## Performance

| MPI Ranks | Read Time (CG trace, 2.5M events) | Speedup |
|-----------|-------------------------------------|---------|
| 1 | 107,000 ms | 1.0x |
| 8 | 14,000 ms | 7.5x |
| 32 | 2,400 ms | 44x |

The read phase dominates total execution time (98%+ for 8-rank reading). As shown, parallel reading provides significant speedup.
