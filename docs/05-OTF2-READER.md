# OTF2 Reader — Distributed Two-Pass Trace Reading

**Source**: `src/reader/OTF2SoAReader.cpp`
**Header**: `include/reader/OTF2SoAReader.h`

## Overview

The OTF2 reader uses a **distributed two-pass** strategy inspired by TileTrace. Each MPI rank reads a contiguous block of locations plus their communication partners, performs local P2P matching and collective grouping, and runs GPU analysis independently. Only final result vectors are gathered to rank 0.

## otf2xx Callback System

The reader uses the **otf2xx** library (C++ wrapper around the OTF2 C API). otf2xx uses a callback pattern:

1. Create a `reader` object with the trace file path
2. Register a callback class (inheriting `otf2::reader::callback`)
3. Call `reader.read_definitions()` — triggers `definition()` callbacks
4. Call `reader.read_events()` — triggers `event()` callbacks
5. The library calls the appropriate overloaded `event()` method for each event type

Two separate callback classes are used for the two passes:

```cpp
// Pass 1: Discovery — lightweight, only finds communication partners
class Pass1DiscoveryCallback : public otf2::reader::callback {
    void definition(const otf2::definition::location &loc) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_send &) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_isend_request &) override;
};

// Pass 2: Data loading — full event processing with collective routing
class Pass2DataCallback : public otf2::reader::callback {
    void definition(const otf2::definition::location &loc) override;
    void event(const otf2::definition::location &, const otf2::event::enter &) override;
    void event(const otf2::definition::location &, const otf2::event::leave &) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_send &) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_receive &) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_isend_request &) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_ireceive_request &) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_ireceive_complete &) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_collective_begin &) override;
    void event(const otf2::definition::location &, const otf2::event::mpi_collective_end &) override;
};
```

## Distributed Two-Pass Reading

### Location Assignment

Locations are assigned as contiguous blocks using `traceRange(nlocs, nprocs, rank)`:

```
For 16 locations with 3 ranks:
  Rank 0: [0, 6)   — 6 locations (remainder distributed to early ranks)
  Rank 1: [6, 11)  — 5 locations
  Rank 2: [11, 16) — 5 locations
```

The inverse function `owningRank(nlocs, nprocs, loc_id)` maps a location to its owning rank, used for routing collective events.

### Pass 1: Discovery

Each rank opens the OTF2 trace, registers only its own locations, and reads events. Only `mpi_send` and `mpi_isend_request` callbacks are active — they record receiver locations in a `std::unordered_set<id_t> related_locs`. After pass 1, `related_locs` contains all locations this rank needs for pass 2 (own + receivers of local sends).

### Pass 2: Data Loading

Each rank opens a second reader, registers all locations in `related_locs`, and reads events with full processing. Filtering rules:

- **Sends** (`mpi_send`, `mpi_isend_request`): Only store events from own locations `[start, end)`
- **Recvs** (`mpi_receive`, `mpi_ireceive_complete`): Only store events where `sender in [start, end)`
- **Collective begin**: Only track for own locations
- **Collective end**: If root is local → store directly; if root is remote → buffer for redistribution

This ensures all send-recv pairs where the sender is local are co-located on the same rank.

### Collective Redistribution

After pass 2, each rank has buffered collective events whose root belongs to other ranks. These are redistributed using MPI:

```
For each target rank t in [0, nprocs):
  1. MPI_Gather: each rank sends count of events destined for t
  2. MPI_Gatherv × 5: send op_type, begin_ts, end_ts, root, pid arrays
  3. MPI_Gatherv × 1: send flattened comm_sets [size, member0, member1, ...]
  4. Target rank reconstructs and appends events via appendCollectiveEvents()
```

### MPI Reader Constructor

The key to parallel reading is using otf2xx's MPI-aware reader constructor:

```cpp
otf2::reader::reader rdr(trace_path, MPI_COMM_WORLD);
```

This requires `OTF2XX_WITH_MPI ON` set before `add_subdirectory(third_party/otf2xx)`. Without it, the `OTF2XX_HAS_MPI` define is absent and the MPI constructor is not available, causing a compile error.

## Event Processing Details

### Vector-Based Event Accumulation

The reader uses `std::vector` dynamic arrays in Pass 2:

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

### Leave Timestamp and Send SoA Index Tracking (Scalasca Mode)

Under `#ifdef USE_SCALASCA_TIMESTAMPS`, the reader also captures additional state needed for the late_receiver analysis:

```cpp
// Leave handler — captures Leave(MPI_Send) for late_receiver condition
void event(const otf2::definition::location &loc,
           const otf2::event::leave &event) override {
    id_t pid = loc.ref().get();
    m_last_leave_ts[pid] = extractTimestamp(event.timestamp());
    // If this Leave follows a Send, update the send's end_timestamps
    auto it = m_last_send_soa_idx.find(pid);
    if (it != m_last_send_soa_idx.end() && it->second >= 0) {
        m_v_end_timestamps[it->second] = m_last_leave_ts[pid];
        it->second = -1;  // Reset
    }
}

// mpi_ireceive_request handler — saves Enter(MPI_Irecv) timestamp
void event(const otf2::definition::location &loc,
           const otf2::event::mpi_ireceive_request &event) override {
    id_t pid = loc.ref().get();
    m_irecv_enter_ts[pid] = m_last_enter_ts[pid];
}
```

The send handlers (`mpi_send`, `mpi_isend_request`) record their SoA index in `m_last_send_soa_idx[pid]` so the subsequent Leave callback can write `Leave(MPI_Send)` into the correct `end_timestamps` entry.

The `mpi_ireceive_complete` handler stores `m_irecv_enter_ts[pid]` (i.e., `Enter(MPI_Irecv)`) into `end_timestamps[i]` for the recv event, repurposing this field for the late_receiver check.

New member variables (under `#ifdef USE_SCALASCA_TIMESTAMPS`):
- `m_last_leave_ts` — Last Leave timestamp per location
- `m_irecv_enter_ts` — Enter(MPI_Irecv) timestamp per location
- `m_last_send_soa_idx` — SoA index of last send event per location

### P2P Event Field Mapping

For send events:
```
events[i]         = TT_MPI_Send or TT_MPI_Isend
timestamps[i]     = Enter(MPI_Send) or Enter(MPI_Isend)
end_timestamps[i] = Leave(MPI_Send) or Leave(MPI_Isend)  [Scalasca mode only]
pids[i]           = sender rank
srcs[i]           = sender rank (= pids[i])
dsts[i]           = receiver rank
tags[i]           = message tag
```

For recv events:
```
events[i]         = TT_MPI_Recv or TT_MPI_Irecv
timestamps[i]     = Enter(MPI_Recv) or Enter(MPI_Wait)
end_timestamps[i] = Enter(MPI_Irecv) [Scalasca mode] or point-event timestamp [non-Scalasca]
pids[i]           = receiver rank
srcs[i]           = sender rank
dsts[i]           = receiver rank (= pids[i])
tags[i]           = message tag
```

**Note** (Scalasca mode): For send events, `end_timestamps` holds `Leave(MPI_Send)`, used for the late_receiver blocking condition. For recv events, `end_timestamps` holds `Enter(MPI_Irecv)` (the time the receive request was posted), used as the recv timestamp in the late_receiver formula.

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
