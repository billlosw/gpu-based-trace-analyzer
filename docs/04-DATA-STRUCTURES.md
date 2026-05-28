# Data Structures

## TraceDataSoA — Core Event Storage

**Location**: `include/data/TraceDataSoA.h`

The fundamental data structure. Stores all trace events in Structure-of-Arrays (SoA) layout for GPU-coalesced memory access.

### Why SoA Instead of AoS?

TileTrace uses AoS (Array of Structures):
```cpp
// TileTrace AoS — BAD for GPU
struct Event { event_t type; uint64_t timestamp; uint32_t pid; ... };
Event events[N];  // events[0].timestamp, events[1].timestamp are 56 bytes apart
```

This project uses SoA:
```cpp
// GPU Analyzer SoA — GOOD for GPU
event_t events[N];        // events[0], events[1] are 4 bytes apart → coalesced
uint64_t timestamps[N];   // timestamps[0], timestamps[1] are 8 bytes apart → coalesced
uint32_t pids[N];         // etc.
```

When a CUDA warp (32 threads) reads `timestamps[threadIdx]`, all 32 reads fall within a contiguous 256-byte memory region, triggering a single coalesced memory transaction. With AoS, the same operation would scatter across memory, causing 32 separate transactions.

### Fields

```cpp
struct TraceDataSoA {
    size_t count;       // Number of events currently stored
    size_t capacity;    // Allocated capacity
    bool owns_memory;   // If true, deallocate() calls free(); if false, only nullifies pointers

    // === 12 base arrays (one element per event) ===
    event_t    *events;          // MPI event type (TT_MPI_Send, TT_MPI_Barrier, etc.)
    event_type_t *types;         // ENTER or LEAVE (always ENTER in current impl)
    timestamp_t *timestamps;     // Enter timestamp in picoseconds
    timestamp_t *end_timestamps; // End/Leave timestamp (for collectives: collective_end time)
    id_t *pids;                  // Process ID (== location ID in OTF2 terms)
    id_t *tids;                  // Thread ID (always 0, reserved for future use)
    id_t *replay_pids;           // Replay PID (== pids, kept for TileTrace compatibility)
    id_t *srcs;                  // Source rank (for P2P: sender; for collective: unused/0)
    id_t *dsts;                  // Destination rank (for P2P: receiver; for collective: unused/0)
    id_t *tags;                  // Message tag (for P2P; 0 for collectives)
    id_t *roots;                 // Root rank (for rooted collectives; 0 for others)
    id_t *indices;               // Original event index (0..N-1)

    // === Matching results (filled by P2PMatching / CollectiveGrouping) ===
    int32_t *match_partner;      // P2P: index of matched partner (-1 if unmatched)
    int32_t *coll_group_id;      // Collective: group ID (-1 if not collective)
};
```

### Memory Layout Per Event

| Field | Type | Size | Purpose |
|-------|------|------|---------|
| `events` | `event_t` (int) | 4B | Event type enum |
| `types` | `event_type_t` (int) | 4B | Enter/Leave |
| `timestamps` | `uint64_t` | 8B | Enter timestamp (ps) |
| `end_timestamps` | `uint64_t` | 8B | End timestamp (ps) |
| `pids` | `uint32_t` | 4B | Process ID |
| `tids` | `uint32_t` | 4B | Thread ID |
| `replay_pids` | `uint32_t` | 4B | Replay PID |
| `srcs` | `uint32_t` | 4B | Source rank |
| `dsts` | `uint32_t` | 4B | Destination rank |
| `tags` | `uint32_t` | 4B | Message tag |
| `roots` | `uint32_t` | 4B | Root rank |
| `indices` | `uint32_t` | 4B | Original index |
| `match_partner` | `int32_t` | 4B | P2P partner index |
| `coll_group_id` | `int32_t` | 4B | Collective group ID |
| **Total** | | **60B/event** | |

### Memory Budget

With a 24GB RTX 4090:
- **~400 million events** fit in GPU VRAM (24GB / 60B)
- In practice, the analysis kernels only transfer a subset of arrays to GPU (events, timestamps, end_timestamps, match_partner, pids, roots = 36B/event), so the GPU can handle even more

### Allocation and Initialization

```cpp
void allocate(size_t n) {
    // All arrays allocated with calloc (zero-initialized)
    // match_partner and coll_group_id initialized to -1 (0xFF bytes)
    memset(match_partner, 0xFF, n * sizeof(int32_t));
    memset(coll_group_id, 0xFF, n * sizeof(int32_t));
}
```

**Implementation trick**: Uses `calloc` instead of `malloc` for zero-initialization, then `memset(0xFF)` for the -1 sentinel in match arrays. The `0xFF` pattern sets all bits to 1, which represents -1 in two's complement for `int32_t`.

### Ownership

`TraceDataSoA` has an `owns_memory` flag (default `true`):
- When `owns_memory == true`: `allocate()` uses `calloc`, `deallocate()` calls `free` on all arrays. This is the standard path for single-rank mode.
- When `owns_memory == false`: `deallocate()` only nullifies pointers without calling `free()`. This is used when the SoA points into an MPI shared memory window (`MPI_Win_allocate_shared`), where the window manages the memory lifecycle.

The struct is move-only (deleted copy constructor/assignment). The destructor calls `deallocate()` which respects the `owns_memory` flag.

## CollectiveGroupCSR — GPU-Friendly Group Format

**Location**: `include/data/TraceDataSoA.h` (same file)

Compressed Sparse Row format for representing collective groups on the GPU.

### Structure

```cpp
struct CollectiveGroupCSR {
    size_t num_groups;        // Number of collective groups
    int32_t *offsets;         // [num_groups + 1] CSR row pointers
    int32_t *members;         // [total_members] SoA indices of group members
    event_t *group_types;     // [num_groups] Event type per group
    id_t *group_roots;        // [num_groups] Root PID per group
    size_t total_members;     // Sum of all group sizes
};
```

### CSR Layout Example

For 3 groups with sizes 4, 3, 2:
```
offsets = [0, 4, 7, 9]
members = [idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8]
           |-- group 0 --|  |-- group 1 --|  |- group 2-|
```

To iterate group `g`:
```cpp
for (int j = offsets[g]; j < offsets[g+1]; j++) {
    int event_idx = members[j];
    // Access data.timestamps[event_idx], data.pids[event_idx], etc.
}
```

### Why CSR?

- **Fixed-size allocations**: No variable-length arrays on GPU
- **Contiguous memory**: Efficient GPU memory access
- **Simple indexing**: One `offsets` lookup per group, then linear scan
- **Standard format**: Well-understood in GPU computing (same as sparse matrix CSR)

## ReaderPhase1Output — Split-Phase Reader Return Type

**Location**: `include/reader/OTF2SoAReader.h`

```cpp
struct ReaderPhase1Output {
    size_t event_count;                         // Number of events this rank will store
    std::vector<std::vector<uint64_t>> comm_sets;  // Communicator member lists
    std::vector<uint64_t> coll_bytes_sent;      // Per-event collective bytes
    std::vector<uint64_t> coll_bytes_received;
    void *handle;                               // Opaque pointer to Pass2DataCallback
};
```

The split-phase API separates reading from storage: `readOTF2TracePhase1()` returns event count and comm_sets without allocating the SoA, allowing the caller to choose the target memory (local calloc or shared window). Then `readerFillSoA(handle, data)` copies internal vectors into the pre-set SoA, and `readerRelease(handle)` frees the reader's internal state.

## RawAnalysisOutput — Kernel Results

**Location**: `include/analysis/AnalysisKernels.h`

```cpp
struct RawAnalysisOutput {
    std::vector<double> late_sender;       // Duration values (picoseconds)
    std::vector<double> late_receiver;
    std::vector<double> barrier_wait;
    std::vector<double> barrier_completion;
    std::vector<double> early_reduce;
    std::vector<double> late_broadcast;
    std::vector<double> wait_nxn;
    std::vector<double> nxn_completion;

    float h2d_ms;         // Host-to-device transfer time
    float p2p_kernel_ms;  // P2P analysis kernel time
    float coll_kernel_ms; // Collective analysis kernel time
    float d2h_ms;         // Device-to-host transfer time
    float gpu_alloc_ms;   // cudaMalloc time (0 when using GPUMemoryPool)
    float gpu_free_ms;    // cudaFree time (placeholder)
    float pin_ms;         // cudaHostRegister time per batch
    float unpin_ms;       // cudaHostUnregister time per batch
    float batch_prep_ms;  // match_partner remap + CSR batch construction time
};
```

Each vector contains the raw duration values in picoseconds. These are later converted to seconds by `computeStatistics` (dividing by 1e12).

## GPUMemoryPool — Pre-Allocated Device Memory

**Location**: `include/analysis/AnalysisKernels.h`

```cpp
struct GPUMemoryPool {
    // Trace input arrays (6)
    event_t *d_events; timestamp_t *d_timestamps, *d_end_timestamps;
    int32_t *d_match; id_t *d_pids, *d_roots;

    // P2P output (4)
    double *d_ls_out, *d_lr_out;
    unsigned int *d_ls_cnt, *d_lr_cnt;

    // Collective input (4)
    int32_t *d_coll_offsets, *d_coll_members;
    event_t *d_group_types; id_t *d_group_roots;

    // Collective output (12)
    double *d_bw_out, *d_bc_out, *d_er_out, *d_lb_out, *d_wn_out, *d_nc_out;
    unsigned int *d_bw_cnt, *d_bc_cnt, *d_er_cnt, *d_lb_cnt, *d_wn_cnt, *d_nc_cnt;

    // Capacities
    size_t max_events, max_coll_members, max_coll_groups;

    void allocate(size_t max_n, size_t max_members, size_t max_groups);
    void deallocate();
};
```

The pool allocates all 26 device pointers once at the start of batched GPU analysis, reused across all K-rank batches. This eliminates per-batch `cudaMalloc`/`cudaFree` overhead (~170ms savings). For LAMMPS n1024, pool allocation takes ~2.2 ms and uses ~12 GB device memory.

## AnalysisResult — Statistics Output

**Location**: `include/data/AnalysisResults.h`

```cpp
struct AnalysisResult {
    size_t count;     // Number of events with this wait pattern
    double sum;       // Total wait time (seconds)
    double min_val;   // Minimum duration (seconds)
    double max_val;   // Maximum duration (seconds)
    double mean;      // Average duration (seconds)
    double variance;  // Population variance (seconds^2)
    double median;    // Median duration (seconds)
    double q25;       // 25th percentile (seconds)
    double q75;       // 75th percentile (seconds)
};
```

## Type Definitions

**Location**: `include/common/types.h`

```cpp
typedef uint64_t timestamp_t;  // Picoseconds (from OTF2)
typedef uint32_t id_t;         // Process/rank IDs and tags

enum event_t {
    TT_MPI_Init = 0,
    TT_MPI_Send = 3,      // Blocking send
    TT_MPI_Recv = 4,      // Blocking recv
    TT_MPI_Isend = 5,     // Non-blocking send
    TT_MPI_Irecv = 6,     // Non-blocking recv
    TT_MPI_Bcast = 7,     // Broadcast
    TT_MPI_Barrier = 8,   // Barrier
    TT_MPI_Reduce = 9,    // Reduce
    // ... (21 total, matching TileTrace Utils.h exactly)
    TT_MPI_Finalize = 20
};
```

The `TT_` prefix stands for "TileTrace" — these values match TileTrace's `Utils.h` exactly for compatibility. The enum values are used as array indices throughout the code (e.g., `event_strings[ev]` for printing).
