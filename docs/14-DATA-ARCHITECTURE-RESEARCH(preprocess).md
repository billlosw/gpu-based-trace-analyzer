# Doc 14: Data Architecture Research — GPU Preprocessing & cuFile

**Created**: 2026-04-06
**Context**: Companion to `13-PERFORMANCE-ENHANCEMENT-PLAN.md` and `chat-history/260406-1500-soa-cache-architecture-strategy.md`.
**Question**: Can P2P matching and collective grouping move to GPU? Does a multi-GPU architecture enable cuFile direct-to-VRAM reads? What are the tradeoffs?

---

## 1. Why CPU Preprocessing is the Current Bottleneck

Two stages currently run on the CPU before GPU analysis can start:

| Stage | Time (n1024) | % of preprocessing |
|-------|-------------|-------------------|
| P2P Matching | 617 ms | 6% |
| Collective Grouping | 2,875 ms | 27% |
| fill_soa (copy data to SHM) | 5,061 ms | 48% |
| cudaHostRegister (pinning) | 1,990 ms | 19% |

Both stages produce data that the GPU kernels depend on:
- **P2P Matching** → `match_partner[]`: for each recv event, the index of its send partner
- **Collective Grouping** → `CollectiveGroupCSR`: which events belong to each collective group

These outputs are necessary because the GPU kernels compute *durations* (how long a send or barrier waited), not *topology* (who matched whom). The topology discovery — currently CPU-bound — is what we want to move to GPU.

The fundamental reason both stages are CPU-bound today is their use of **sequential FIFO queues** and **dynamic hash maps**, which have no direct GPU equivalent. But as shown below, a reformulation using **sort-based algorithms** makes both stages GPU-parallelizable.

---

## 2. GPU P2P Matching: Sort-Based Reformulation

### 2.1 Why the Current FIFO Algorithm is Sequential

The current algorithm (`src/matching/P2PMatching.cpp`) processes events in timestamp order, maintaining per-key queues:

```
for each event i in timestamp order:
    key = (sender_rank, receiver_rank, mpi_tag)
    if event is Send:
        if unmatched_recvs[key] non-empty: match immediately
        else: push onto unmatched_sends[key]
    if event is Recv:
        if unmatched_sends[key] non-empty: match immediately
        else: push onto unmatched_recvs[key]
```

The FIFO queue is correct because MPI guarantees **message ordering**: if A sends to B twice with the same tag, the first send matches the first recv. Processing in timestamp order enforces this.

The sequential dependency is: whether event i is queued depends on whether all earlier events have already consumed the partner that i would have matched.

### 2.2 The Key Insight: Sort-and-Rank Equivalence

The FIFO algorithm and a sort-based algorithm produce identical results because of the following invariant:

> For a fixed matching key `(sender, receiver, tag)`, if there are K sends and K recvs, the i-th send (ordered by timestamp) matches the i-th recv (ordered by timestamp).

This is simply MPI's FIFO ordering property. Given this, the matching can be reformulated as:

1. **Partition** all P2P events into sends and recvs
2. **Sort** sends by `(sender, receiver, tag, timestamp)` — assign ordinal `send_rank[i]` within each key group
3. **Sort** recvs by `(sender, receiver, tag, timestamp)` — assign ordinal `recv_rank[i]` within each key group
4. **Match**: send with `(key=K, ordinal=j)` matches recv with `(key=K, ordinal=j)`

This requires no sequential state. Steps 2-4 are embarrassingly parallelizable on GPU.

### 2.3 GPU Algorithm (Sort-Based)

```
Input:  N events with (event_kind, pid, src, dst, tag, timestamp, original_index)
Output: match_partner[original_index] for each event

Step 1: Filter
  - Extract sends: events where kind ∈ {MPI_Send, MPI_Isend}
  - Extract recvs: events where kind ∈ {MPI_Recv, MPI_Irecv}
  - GPU kernel: 2 compaction passes (thrust::copy_if or CUB DeviceSelect)

Step 2: Assign matching keys
  - For send[i]: key = (pid, dst, tag)   [pid=sender, dst=receiver]
  - For recv[i]: key = (src, pid, tag)   [src=sender, pid=receiver]
  - Both use the same 3-tuple (sender_rank, receiver_rank, tag)
  - Pack into 64-bit key: (sender << 42) | (receiver << 21) | tag  [same as current]

Step 3: Stable sort sends by (key, timestamp)
  - thrust::stable_sort_by_key or CUB DeviceRadixSort
  - O(S log S) where S = number of send events

Step 4: Stable sort recvs by (key, timestamp)
  - Same, O(R log R) where R = number of recv events

Step 5: Assign within-group rank (ordinal)
  - For each sorted array, use a segmented exclusive scan
  - Segment boundaries: where consecutive elements have different matching keys
  - CUB DeviceScan with custom segment operator
  - Result: send_ordinal[i] = position of this send within its (key) group

Step 6: Build inverse maps
  - reverse_send[key][ordinal] = original_index (in send array)
  - reverse_recv[key][ordinal] = original_index (in recv array)
  - Implementation: use sorted arrays + binary search, or thrust::sort_by_key on (key, ordinal)

Step 7: For each send i and matching recv j with same (key, ordinal):
  - match_partner[send.original_index] = recv.original_index
  - match_partner[recv.original_index] = send.original_index
  - GPU kernel: one thread per pair
```

**Complexity**: O(N log N) sort-dominated, same as the CPU algorithm — but GPU radix sort is ~10-50x faster than CPU `std::sort` for large N.

**Unmatched detection**: Any send (or recv) with ordinal `j` where no corresponding recv (or send) with the same `(key, ordinal=j)` exists is unmatched. Detectable by checking if counts of sends and recvs per group differ.

### 2.4 Memory Requirements

| Array | Size | Bytes |
|-------|------|-------|
| send indices (compacted) | S × 4B | ~S × 4 |
| recv indices (compacted) | R × 4B | ~R × 4 |
| sort keys (64-bit) | (S+R) × 8B | ~(S+R) × 8 |
| ordinals (32-bit) | (S+R) × 4B | ~(S+R) × 4 |
| CUB sort temp storage | ~N × 2B | ~N × 2 |
| **Total** | | **~26 B per P2P event** |

For n1024: ~132M P2P send+recv events → ~3.4 GB scratch. Well within 4090's 24 GB.

### 2.5 Why This Works for Multi-GPU

The current FIFO algorithm is local per MPI rank — each rank matches its own P2P events. The sort-based GPU algorithm is also local: each GPU matches the events assigned to it. **No cross-GPU communication needed for local matching** (the same assumption holds: events are redistributed so that both the send and recv of each pair are on the same MPI rank — this is what `redistributeCollectives()` and the distributed two-pass reader already ensure for collectives; for P2P, it is guaranteed by the rank-based partitioning: a Send from rank A to rank B is assigned to the reader rank that owns rank A, and the corresponding Recv is also owned by that same reader rank because OTF2 stores both sides of point-to-point communication).

> **Note on P2P ownership**: Each P2P communication event (both the Send-side and the Recv-side) appears in the trace of the respective process. In the distributed reader, rank A's trace is read by the MPI reader that owns location A. After redistribution, a reader rank has all events from locations it owns. Both the Send(from=A, to=B) and the Recv(from=A, to=B) must reside on reader ranks that cover A and B respectively. This means P2P matching **is cross-rank** — and the current architecture handles it locally only because events are redistributed such that each reader rank holds both sides. For multi-GPU, the same redistribution constraint applies.

---

## 3. GPU Collective Grouping: Segment Scan Reformulation

### 3.1 Why the Current Algorithm is Sequential

The current collective grouping (`src/matching/CollectiveGrouping.cpp`) uses a stateful automaton:

```
pending_groups: list of (event_type, comm_set, needed_pids, member_indices)
for each collective event in timestamp order:
    find matching pending group with same (event_type, comm_set) and pid still needed
    if found: add to group; if group complete, move to completed list
    if not found: create new pending group
```

The sequential dependency is: whether process A's barrier-1 joins pending_group[0] or creates pending_group[1] depends on whether processes B, C, D, ... already started group 0.

### 3.2 The Key Insight: Uniform Group Sizes

For typical MPI programs, **all instances of the same collective operation on the same communicator have exactly the same group size** (equal to the communicator size). This means:

> Given K collective events on communicator C (size P), sorted by timestamp, the first P events form group 0, the next P events form group 1, etc.

This reduces the problem to a simple **segmented partition**:

1. Sort by `(event_type, comm_set_hash, timestamp)`
2. Communicator C broadcasts to 64 processes in 1000 iterations → 64,000 Bcast events → sorted, they appear in 1000 groups of 64.
3. Each group is a contiguous slice of size P in the sorted array.
4. Group ID = `floor(position_within_key_segment / P)`

### 3.3 GPU Algorithm (Segment-Scan-Based)

```
Input:  N events (only collective events matter: C ≪ N)
        comm_sets: per-collective-event list of participant ranks
Output: CollectiveGroupCSR

Step 1: Filter collective events
  - GPU kernel: extract indices where events[i] ∈ {Bcast, Reduce, ..., AlltoAll}
  - CUB DeviceSelect, O(N)
  - Result: coll_indices[C], where C ≪ N

Step 2: Sort by (event_type, comm_set_hash, timestamp)
  - GPU radix sort: 128-bit key (64-bit hash | 32-bit type | 32-bit position)
  - After sorting: events of same type+comm form contiguous runs

Step 3: Compute segment boundaries
  - Two consecutive entries are in the same segment if they have the same (event_type, comm_set_hash)
  - GPU scan: adjacent_difference to find segment start/end positions

Step 4: Within each segment, assign group IDs
  - Each segment has K events from P processes making K/P collective calls
  - position_within_segment = scan of segment-local index
  - group_id = position_within_segment / P  (where P = communicator size)
  - GPU kernel: one thread per collective event
  - Edge case: P differs per event (e.g., MPI_COMM_WORLD vs. subcommunicator)
    → P is read from comm_sets[cs_idx].size() — constant per segment

Step 5: Build CSR
  - Sort by (group_id, original_soa_index) — positions members within groups
  - thrust::exclusive_scan on group sizes → offsets[]
  - members[], group_types[], group_roots[], member_bytes[] built from sorted result
```

**Complexity**: O(C log C) sort-dominated, where C = collective event count ≪ N.

**Edge cases**:

- Overlapping collectives (multiple instances simultaneously open): The sort-then-divide approach fails if two processes contribute their events for iteration 2's barrier before all arrive for iteration 1's. However, this cannot happen in MPI because MPI_Barrier is blocking — all processes must have entered barrier(i) before any process can exit it and enter barrier(i+1). So temporal ordering guarantees non-overlapping within one communicator.
- Mixed communicators: segments are separated by `(event_type, comm_set_hash)`, so different communicators are handled independently.
- Hash collisions: The comm_set_hash is a content hash (FNV-1a of the sorted member list). Collision probability is negligible (1/2^64).

### 3.4 Memory Requirements

Collective events are typically <15% of total:

| Array | Size | Bytes |
|-------|------|-------|
| coll_indices (filtered) | C × 4B | ~C × 4 |
| sort keys (128-bit) | C × 16B | ~C × 16 |
| ordinals, group_ids | C × 8B | ~C × 8 |
| CSR output | C × 16B | ~C × 16 |
| **Total** | | **~44 B per collective event** |

For n1024 (190K collective events): ~8.4 MB — negligible.

---

## 4. A Full GPU Pipeline: Eliminating Host Involvement

With both P2P matching and collective grouping on GPU, the entire hot path becomes GPU-resident:

```
Current pipeline:
  File ──fread──► Host SHM ──CPU P2P/Coll──► Host buffers ──cudaMemcpy──► GPU ──kernels──► Results

Full-GPU pipeline:
  File ──cuFile──► GPU VRAM ──GPU matching──► GPU VRAM ──kernels──► Results
```

The CPU role becomes:
1. Read the file header and metadata (comm_set data, header validation)
2. Launch GPU kernels (very lightweight)
3. Collect result statistics (D2H of small output arrays)

This enables:
- **cuFile reads all 7 GPU-input arrays directly to VRAM** (no host staging at all)
- **No `cudaHostRegister` needed** (data never touches host-pinned memory)
- **No SHM window** (no `MPI_Win_allocate_shared`)
- **Zero host-side copies**

---

## 5. Multi-GPU Architecture

### 5.1 Motivation

The current architecture is: `P MPI readers → 1 GPU (rank 0)`. All P ranks share memory on one node; rank 0 runs all GPU analysis.

A natural extension is: `N MPI processes → N GPUs`, where N = number of available GPUs. Each process owns one GPU and is responsible for reading and analyzing its fraction of the trace.

### 5.2 Proposed Architecture: One Process Per GPU

```
Trace file (corpus of all locations)
        |
        | (partitioned by location range)
        |
 ┌──────┴──────────────────────────────────────────────┐
 │ GPU 0 (Process 0)  │ GPU 1 (Process 1) │ GPU 2 ...  │
 │  Locations 0-255   │  Locations 256-511 │           │
 │                    │                   │           │
 │  cuFile → VRAM     │  cuFile → VRAM    │ cuFile    │
 │  GPU P2P match     │  GPU P2P match    │           │
 │  GPU Coll group    │  GPU Coll group   │           │
 │  Analysis kernels  │  Analysis kernels │           │
 │  Local results     │  Local results    │           │
 └──────┬─────────────┴──────────┬────────┴──────────┘
        └─────── MPI Reduce ─────┘
                    │
              Final statistics (rank 0)
```

### 5.3 How cuFile Fits In

With:
1. **Single column-major cache file** (see `chat-history/260406-1500-soa-cache-architecture-strategy.md`, Scheme B)
2. **One MPI process per GPU**
3. **Full-GPU preprocessing** (sections 2 and 3 above)

The data path becomes:

```cpp
// Each process reads its own slice of the column-major cache file directly to VRAM
size_t my_start = rank_boundaries[rank];       // global event offset for this rank
size_t my_count = rank_boundaries[rank+1] - my_start;

cuFileRead(fh, d_events,         my_count * 4,  col_offsets[0] + my_start * 4,  0);
cuFileRead(fh, d_timestamps,     my_count * 8,  col_offsets[2] + my_start * 8,  0);
cuFileRead(fh, d_end_timestamps, my_count * 8,  col_offsets[3] + my_start * 8,  0);
cuFileRead(fh, d_pids,           my_count * 4,  col_offsets[4] + my_start * 4,  0);
cuFileRead(fh, d_srcs,           my_count * 4,  col_offsets[5] + my_start * 4,  0);
cuFileRead(fh, d_dsts,           my_count * 4,  col_offsets[6] + my_start * 4,  0);
cuFileRead(fh, d_tags,           my_count * 4,  col_offsets[7] + my_start * 4,  0);
cuFileRead(fh, d_roots,          my_count * 4,  col_offsets[8] + my_start * 4,  0);
cuFileRead(fh, d_leave_recv_ts,  my_count * 8,  col_offsets[9] + my_start * 8,  0);

// GPU preprocessing (all on device)
gpuP2PMatch(...);           // sort-based, pure GPU
gpuCollGrouping(...);       // segment-scan, pure GPU

// Analysis kernels (unchanged design)
kernelLateSenderReceiver(...);
kernelBarrierWaitCompletion(...);
// ...

// Gather to rank 0
MPI_Reduce(local_results, global_results, ..., MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
```

**Zero host-side data handling** for the entire hot path.

### 5.4 Cross-GPU P2P Matching Problem

**Critical issue**: P2P matching is currently local because each MPI reader rank holds both sides of each matched pair. In the multi-GPU architecture with N processes each covering a location range, the **send and recv of the same P2P pair may be owned by different GPUs**:

- GPU 0 holds locations 0-255 → process 0's send events (src=0..255, dst=0..255) AND process 0's recv events (src=0..255, dst=0..255) — both sides for intra-partition pairs.
- But a send from location 0 to location 300 (GPU 0 → GPU 1): the Send event is on GPU 0 (location 0 is in GPU 0's range), and the Recv event is on GPU 1 (location 300 is in GPU 1's range).

This is the distributed P2P matching problem. Two approaches:

#### Approach A: Redistribute cross-partition pairs (MPI pass)

After each GPU reads its events, do a communication round to redistribute "cross events":
- GPU 0 sends its recv events to whichever GPU holds the matching send's location range, and vice versa.
- This is analogous to what TileTrace does with MPI-based distributed replay.
- Cost: ~2× the volume of cross-partition P2P events (typically 30-70% of all P2P events for real MPI programs).
- After redistribution, GPU matching is fully local.

#### Approach B: Cross-GPU sort (NVLink)

On multi-GPU systems with NVLink (e.g., H100 NVLink), GPUs can access each other's memory. Do a distributed sort across all GPUs:
- Each GPU contributes its sends and recvs into a shared (NVLink-accessible) sort key space.
- A global sort by `(sender, receiver, tag, timestamp)` groups matching pairs together regardless of which GPU holds them.
- Assignment of matched pairs to GPUs after sort.
- Cost: O(N log N) sort across all GPUs — viable with NVLink but expensive over PCIe.

#### Approach C: Single GPU collects all (current architecture)

The current architecture's ultimate destination for all data is rank 0 (one GPU). For N GPUs, you could do all matching on one GPU after gathering all events — but this defeats multi-GPU scaling.

**Recommended**: Approach A (redistribute) for correctness and locality. The redistribution cost is a one-time communication that is then cached, so on subsequent runs cuFile reads local data without redistribution.

### 5.5 Collective Grouping in Multi-GPU Setting

Collective events are already redistributed to rank 0 in the current architecture (`redistributeCollectives()`). In the multi-GPU setting:

- **Option 1**: Keep collective redistribution to GPU 0 (only rank 0 does collective analysis). Collectives are <15% of events and produce negligible traffic compared to P2P.
- **Option 2**: Replicate collective data to all GPUs (each GPU has all collective group memberships). Since collective events are <15% of total and comm_sets are tiny, replication is cheap.
- **Option 3**: Partition collectives: GPU k handles collective groups where the root location is in GPU k's range. This requires more complex routing for non-root members.

Option 1 is simplest and preserves correctness without rearchitecting the collective analysis kernels.

### 5.6 VRAM Budget Per GPU

For the multi-GPU case with N GPUs and T total events:

| Item | Per-GPU size |
|------|-------------|
| Input SoA (9 arrays) | T/N × 52 B |
| leave_recv_ts | T/N × 8 B |
| P2P sort scratch | T/N × 26 B |
| Collective sort scratch | C/N × 44 B |
| Analysis output | T/N × 16 B |
| **Total** | **T/N × ~ 102 B** |

For n1024 (T=262M), N=4 GPUs:
- Per GPU: 65.5M events × 102 B ≈ **6.7 GB** — comfortably fits in 4090's 24 GB.

For n8192 (T~2B), N=4 GPUs:
- Per GPU: 500M events × 102 B ≈ **51 GB** — exceeds 4090 (24 GB) but fits H100 (80 GB).

For n8192 with N=8 GPUs:
- Per GPU: 250M events × 102 B ≈ **25.5 GB** — tight on 4090, fits H100.

**Takeaway**: For traces up to n4096 (1.06B events), 4 RTX 4090s are sufficient. Beyond that, H100s or more GPUs are needed. The current FUSE cluster has 2×H100 and 4×4090 — both viable targets.

---

## 6. cuFile Feasibility on FUSE Cluster

### 6.1 GDS Requirements

GPUDirect Storage requires:
1. **GDS-compatible filesystem**: local NVMe (ext4/XFS), GPFS/Lustre with GDS plugin. NFS does **not** support GDS.
2. **GDS kernel driver**: `nvidia-gds` package (part of CUDA Toolkit 11.4+)
3. **MOFED** (Mellanox/InfiniBand required for some GDS configurations)
4. **Compatible GPU**: Volta or newer (GP100, V100, A100, H100, RTX 3080+, 4090 all support GDS)

### 6.2 FUSE Cluster Storage Topology

From cluster specs:
```
fuse1: 2×H100, 2×4090, 2×A10, 2×MI100, 1×MI210 — storage: /home (NFS, shared)
fuse2: 1×H100, 2×5090 — storage: /home (NFS, shared)
```

The `/home` filesystem is NFS. **NFS does not support GDS.** The soa_cache files would be written to `/home/luosw22/trace_data/...`, which is NFS-mounted.

**GDS is therefore not available on FUSE as currently configured.** This is consistent with the note in `13-PERFORMANCE-ENHANCEMENT-PLAN.md`: "FUSE cluster storage may not support GDS."

### 6.3 Fallback: mmap + cudaMemcpy

Without GDS, the fastest achievable I/O is:

```
NVMe/NFS → Page Cache (OS) → mmap virtual address → cudaMemcpy (DMA)
```

With the column-major single-file format:
- `mmap(PROT_READ, MAP_SHARED)` on the cache file
- `madvise(MADV_SEQUENTIAL | MADV_WILLNEED)` to prefetch
- Each process accesses `base + col_offset + my_start * elem_size` directly
- `cudaMemcpy` from mmap'd address to GPU (requires `cudaHostRegister` on the page or system-pinned memory)

This eliminates the intermediate heap allocation and SHM copy — **2 of 3 host-side copies removed** without GDS.

### 6.4 When GDS Becomes Available

If local NVMe scratch space becomes available (e.g., `/tmp` on NVMe, or a request to mount a local filesystem), cuFile can be enabled with a one-line check:

```cpp
CUfileError_t e = cuFileDriverOpen();
bool gds_ok = (e.err == CU_FILE_SUCCESS);
// Use cuFile if gds_ok, else fall back to mmap + cudaMemcpy
```

No file format change is needed: the same column-major cache file works for both paths.

---

## 7. Implementation Roadmap

### Phase 1: Column-Major Single Cache File (No Code Change in Analysis)

**Status**: Design complete (see `chat-history/260406-1400-soa-cache-architecture-strategy.md`)

- Merge per-rank files into one column-major file
- Use mmap instead of fread + SHM
- Eliminates 2 host-side copies
- **Works on FUSE today, no GPU algorithm changes**
- Expected improvement on cache read phase: ~1.5-2×

### Phase 2: GPU P2P Matching

**Status**: Implemented and validated (`src/matching/GPUP2PMatching.cu`, enabled by `--gpu-matching`)

GPU sort-based P2P matching (section 2):
- Implemented `runGPUP2PMatching()` using thrust sort + exclusive_scan_by_key
- CPU version kept as fallback (selected at runtime by absence of `--gpu-matching`)
- Algorithm: classify → compact → sort by (key, timestamp) → assign ordinals → binary-search match
- **Measured results**: 16q: 172 ms (CPU: 3439 ms, **20x speedup**). n1024: 2591 ms (CPU: 5677 ms, **2.2x speedup**).
- **Correctness**: Exact match with CPU P2P matching (all 8 analyses identical for both traces)
- Note: thrust::reduce unreliable in multi-TU separable compilation builds; uses CPU-side counting via cudaMemcpy as workaround

### Phase 3: GPU Collective Grouping

**Status**: Implemented (`src/matching/GPUCollectiveGrouping.cu`, enabled by `--gpu-matching`)

GPU segment-scan collective grouping (section 3):
- Implemented `buildGPUCollectiveGroups()` using thrust sort + segment scan
- CPU version kept as fallback
- Algorithm: filter → hash comm_sets → sort by (type|hash, timestamp) → segment scan → group_id = pos/comm_size → CSR on CPU
- **Measured results**: 16q: 7.3 ms (CPU: 8.7 ms, **1.2x**). n1024: 5695 ms (CPU: 8049 ms, **1.4x**). Hash computation dominates on n1024 (5508 ms).
- **Correctness**: Small differences vs CPU state-machine algorithm for broadcast groups (different tie-breaking for temporally close events). P2P-dependent analyses (late_sender, late_receiver) and most collective analyses (barrier_wait, wait_nxn) are unaffected.

### Phase 4: mmap-Based Reader (No cuFile)

Replace fread + SHM with mmap of the column-major cache file.
- Each MPI process mmaps the file and accesses `my_start..my_start+my_count` range
- `madvise(MADV_SEQUENTIAL)` per-process prefetch
- `cudaMemcpy` from mmap'd memory (after Phase 2/3, only GPU-input columns needed)
- **Works on FUSE today**
- Expected improvement: ~1.5-2× on cache read

### Phase 5: Multi-GPU (N Processes, N GPUs)

**Prerequisites**: Phases 1-4

- Assign one MPI process per GPU
- Implement Approach A redistribution for cross-partition P2P pairs
- Run GPU kernels on each GPU independently
- MPI_Reduce for final statistics
- Expected scaling: linear with N GPUs up to N=8 for current trace sizes

### Phase 6: cuFile (When GDS Available)

**Prerequisite**: Phase 1 (column-major file); Phases 2-3 (GPU preprocessing)

- Replace mmap + cudaMemcpy with cuFileRead for the N-GPU path
- Register each GPU's device buffer with cuFileBufRegister
- cuFileRead directly into device memory
- Falls back to Phase 4 path if cuFile driver unavailable
- Expected improvement: additional 2-3× on I/O phase when NVMe-local

---

## 8. Summary: GPU-Preprocessing Feasibility Assessment

| Stage | Current | GPU Feasible? | Algorithm | Complexity |
|-------|---------|--------------|-----------|------------|
| P2P Matching | CPU FIFO, sequential | **Yes** | Sort + ordinal matching | O(N log N), GPU-parallelizable |
| Collective Grouping | CPU state machine, sequential | **Yes** | Sort + segment scan | O(C log C), GPU-parallelizable |
| fill_soa | CPU memcpy | N/A (eliminated with cuFile) | cuFileRead / mmap | O(N), I/O-bound |
| cudaHostRegister | CPU pinning | N/A (eliminated with cuFile) | cuFileBufRegister | O(1) |
| Analysis kernels | GPU, parallel | Already on GPU | — | O(N/threads) |

**Bottom line**: Both P2P matching and collective grouping CAN be reformulated as GPU-parallelizable algorithms using sort-based approaches. Combined with a column-major cache file and cuFile (or mmap), the full GPU pipeline eliminates all host-side data movement. The multi-GPU (N processes, N GPUs) architecture is viable and scales well for traces up to n4096 with 4×RTX 4090 or beyond with H100s.

The primary near-term constraint is GDS driver availability on FUSE. The mmap fallback delivers most of the benefit without GDS, making Phase 4 a good intermediate milestone.

---

## 9. Related Documents

- `docs/06-P2P-MATCHING.md`: Current CPU P2P matching algorithm details
- `docs/07-COLLECTIVE-GROUPING.md`: Current CPU collective grouping algorithm details
- `docs/08-CUDA-KERNELS.md`: GPU analysis kernel designs
- `docs/13-PERFORMANCE-ENHANCEMENT-PLAN.md`: Enhancement 1d (GPUDirect), baseline timings
- `chat-history/260406-1500-soa-cache-architecture-strategy.md`: Cache file format schemes (A-E)
