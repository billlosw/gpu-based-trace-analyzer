# P2P Send-Recv Matching

**Source**: `src/matching/P2PMatching.cpp`
**Header**: `include/matching/P2PMatching.h`

## Overview

P2P matching pairs each `MPI_Send`/`MPI_Isend` event with its corresponding `MPI_Recv`/`MPI_Irecv` event. This is a prerequisite for the Late Sender and Late Receiver analyses.

## Algorithm: Timestamp-Sorted FIFO Queues

The algorithm mirrors TileTrace's `InteractionPattern::analysis()`:

1. **Sort** all events by timestamp (create a sorted index, don't move the data)
2. **Process** events in timestamp order
3. For each **Send**: look for a matching Recv in `unmatched_recvs[key]` FIFO queue
4. For each **Recv**: look for a matching Send in `unmatched_sends[key]` FIFO queue
5. If match found: link both events via `match_partner`; if not: queue the event

### Matching Key

The key ensures we match the correct sender-receiver pair:

```cpp
// For Send: key = (sender=pids[i], receiver=dsts[i], tag=tags[i])
// For Recv: key = (sender=srcs[i], receiver=pids[i], tag=tags[i])
// Both produce: key = (sender_rank, receiver_rank, tag)
```

Key encoding (fits in 64 bits):
```cpp
auto makeKey = [](id_t a, id_t b, id_t c) -> uint64_t {
    return ((uint64_t)a << 42) | ((uint64_t)b << 21) | (uint64_t)c;
};
```

This packs three values into a single 64-bit integer: 22 bits for `a`, 21 bits for `b`, 21 bits for `c`. This supports up to ~4M ranks and ~2M tags, which is sufficient for typical HPC applications.

### Why Timestamp Ordering Matters

The same `(sender, receiver, tag)` triplet can appear multiple times in a trace (e.g., process 0 sends to process 1 with tag 0 in every iteration). Temporal ordering ensures:

- The first send matches the first recv
- The second send matches the second recv
- Etc.

Without temporal ordering, a GPU hash table might match sends and receives from different iterations, producing incorrect timing measurements.

### Implementation

```cpp
void runP2PMatching(TraceDataSoA &data) {
    // 1. Build sorted index
    std::vector<size_t> sorted_idx(n);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&](size_t a, size_t b) {
                  return data.timestamps[a] < data.timestamps[b];
              });

    // 2. FIFO queues per key
    std::unordered_map<uint64_t, std::queue<size_t>> unmatched_sends;
    std::unordered_map<uint64_t, std::queue<size_t>> unmatched_recvs;

    // 3. Process in timestamp order
    for (size_t si = 0; si < n; si++) {
        size_t i = sorted_idx[si];
        event_t ev = data.events[i];

        if (ev == TT_MPI_Send || ev == TT_MPI_Isend) {
            uint64_t k = makeKey(data.pids[i], data.dsts[i], data.tags[i]);
            auto it = unmatched_recvs.find(k);
            if (it != unmatched_recvs.end() && !it->second.empty()) {
                // Match found
                size_t recv_idx = it->second.front();
                it->second.pop();
                data.match_partner[i] = (int32_t)recv_idx;
                data.match_partner[recv_idx] = (int32_t)i;
            } else {
                unmatched_sends[k].push(i);
            }
        } else if (ev == TT_MPI_Recv || ev == TT_MPI_Irecv) {
            // Symmetric: try unmatched_sends first
            // ...
        }
    }
}
```

### Output

- `data.match_partner[i]` = index of matched partner (-1 if unmatched)
- For a Send: `match_partner[i]` = index of the matching Recv
- For a Recv: `match_partner[i]` = index of the matching Send
- Both partners point to each other (bidirectional link)

## Why This Runs on CPU, Not GPU

P2P matching uses pure C++ (`.cpp` file, no CUDA dependency). Three attempts at GPU matching were investigated and rejected:

1. **GPU hash table** (initial, 2026-03-13): Open-addressing with `atomicCAS`/`atomicExch`. Failed due to temporal ordering requirement (same key appears multiple times across iterations), probing chain breaks, and race conditions.

2. **GPU sort-based** (`thrust::sort_by_key`, 2026-03-29): Failed because 64 MPI ranks sharing one GPU cannot all use thrust simultaneously — causes cudaErrorMemoryAllocation (OOM) and cudaErrorInvalidValue. Each rank would need ~48 MB GPU memory for sort workspace; 64 ranks × 48 MB = 3 GB concurrent + radix sort temp space exceeds GPU memory.

3. **nvcc overhead discovery** (2026-03-29): Even without any GPU code, compiling P2PMatching as `.cu` (processed by nvcc) caused CUDA runtime initialization overhead. Renaming to `.cpp` improved P2P matching from 2,799ms to 617ms — **4.5x faster** — purely by avoiding CUDA runtime init.

The CPU queue-based approach is correct by construction and runs in O(N log N) time (dominated by sorting). For 4M events per rank, matching takes ~617 ms.

## Performance Characteristics

| Trace | Events | Match Time | Matched Pairs | Unmatched |
|-------|--------|------------|---------------|-----------|
| CG-B (64 locs) | 2.5M | 255 ms | 1,279,175 | 0 sends, 0 recvs |
| CG-C (64 locs) | 2.5M | 260 ms | 1,279,232 | 0 sends, 0 recvs |

All send-recv pairs are correctly matched with 0 unmatched events.

## Potential Optimization: GPU Matching

For traces with tens of millions of events, the CPU matching could become a bottleneck. A correct GPU matching would need:

1. **Sort events on GPU** (CUB radix sort)
2. **Two-pass matching**:
   - Pass 1: For each unique key, count sends and recvs
   - Pass 2: Match in order within each key group
3. **Key-based partitioning**: Partition by key, then sequential matching within each partition

This is deferred as a future optimization since matching is currently <1% of total time.
