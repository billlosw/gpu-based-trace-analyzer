# SoA Cache Architecture Strategy Report

**Date**: 2026-04-06
**Goal**: Evaluate schemes for compacting per-rank SoA binary cache files into a single file, enabling cuFile (GPUDirect Storage) for direct-to-VRAM reads, and handling VRAM overflow.

---

## 1. Current Architecture (Baseline)

### 1.1 Cache File Layout

Each MPI rank writes its own binary file:
```
<trace_dir>/soa_cache/soa_cache_r<rank>_n<nprocs>.bin
```

**Per-file structure**:
```
[SoACacheHeader: 100 bytes]
[events[N]:       N × 4B  = int32]
[types[N]:        N × 4B  = int32]
[timestamps[N]:   N × 8B  = uint64]
[end_timestamps[N]: N × 8B = uint64]
[pids[N]:         N × 4B  = uint32]
[srcs[N]:         N × 4B  = uint32]
[dsts[N]:         N × 4B  = uint32]
[tags[N]:         N × 4B  = uint32]
[roots[N]:        N × 4B  = uint32]
[leave_recv_ts[N]: N × 8B = uint64]  (conditional flag)
[comm_sets: variable-length nested arrays]
[coll_bytes_sent[M]:     M × 8B]
[coll_bytes_received[M]: M × 8B]
```

**Per-event byte counts in cache**:
| Field | Type | Bytes |
|-------|------|-------|
| events | int32 | 4 |
| types | int32 | 4 |
| timestamps | uint64 | 8 |
| end_timestamps | uint64 | 8 |
| pids | uint32 | 4 |
| srcs | uint32 | 4 |
| dsts | uint32 | 4 |
| tags | uint32 | 4 |
| roots | uint32 | 4 |
| leave_recv_ts | uint64 | 8 |
| **Subtotal (fixed)** | | **52 B/event** |

Plus variable-length comm_sets and collective byte data (negligible for large traces — n1024 has only 186 comm_sets vs 262M events).

### 1.2 Data Pipeline: Cache → Host → GPU

```
Cache files (disk)
  ↓ fread() per rank (std::ifstream)
Per-rank SoACacheData vectors (heap)
  ↓ readerFillSoA() — memcpy into SHM window
SHM window (MPI_Win_allocate_shared, 15 arrays × page-aligned)
  ↓ cudaHostRegister (pin 7 arrays) or per-batch pinning
Pinned host memory
  ↓ cudaMemcpy H2D (7 arrays: events, timestamps, end_timestamps,
  ↓                  leave_recv_ts, match_partner, pids, roots)
GPUMemoryPool (pre-allocated device buffers)
  ↓ kernel execution
Results (D2H copy)
```

**Key bottleneck**: The current pipeline has **3 copies** for cached data:
1. Disk → heap (fread into `SoACacheData` vectors)
2. Heap → SHM window (memcpy in `readerFillSoA`)
3. SHM → GPU (cudaMemcpy H2D)

### 1.3 Scale Reference

| Trace | Locations | Events | Cache Size (total) | SHM Buffer | GPU Batches (4090) |
|-------|-----------|--------|-------------------|------------|-------------------|
| 16q | 16 | 4.1M | ~35 MB | 279 MB | 1 |
| n1024 | 1,024 | 262M | ~4.5 GB (64×~70MB) | 18 GB | 1 |
| n2048 | 2,048 | 525M | ~9 GB | 32 GB | 2 |
| n4096 | 4,096 | 1.06B | ~18 GB | 72.6 GB | 3 |
| n8192 | 8,192 | ~2B | ~36 GB | ~140 GB | 6+ |

**GPU VRAM**: RTX 4090 = 24 GB, H100 = 80 GB.
**GPU-transferred bytes per event**: 40 B (7 arrays: 3×uint64 + 3×uint32 + 1×int32 = 24+12+4 = 40B).
**Per-batch capacity (4090, ~20 GB usable)**: ~500M events input + output buffers.

### 1.4 Current Cache Performance

| Trace | Cache Read | End-to-End (hit) | Speedup vs OTF2 |
|-------|-----------|-----------------|-----------------|
| 16q | 24 ms | 453 ms | 43x |
| n1024 | 2,167 ms | 24,408 ms | 11x |
| n4096 | ~15s (est.) | 638s | — (first run, no cache) |

---

## 2. What cuFile / GPUDirect Storage Can Do

### 2.1 cuFile Overview

NVIDIA GPUDirect Storage (GDS) via the cuFile API allows **direct DMA from NVMe storage to GPU VRAM**, bypassing the CPU page cache and host memory entirely:

```
Traditional:  NVMe → Page Cache → User Buffer → Pinned Buffer → GPU
GDS:          NVMe → GPU (via DMA, bypassing CPU entirely)
```

**Key cuFile operations**:
- `cuFileRead(handle, devPtr, size, file_offset, devPtr_offset)` — read from file directly to a GPU device pointer
- `cuFileBufRegister(devPtr, size)` — register GPU buffer for GDS
- Requires: GDS-compatible filesystem (ext4, XFS, GPFS, Lustre), NVIDIA MOFED or GDS driver, CUDA 11.4+

**Alignment requirements**:
- File offsets must be aligned to **4 KB** (filesystem block size)
- GPU buffer offsets should be aligned to **at least 512 B** (recommended 4 KB)
- Transfer sizes should be multiples of **4 KB** for best performance

### 2.2 Benefits

- Eliminates 2 of 3 copies (disk → GPU directly, no host staging)
- Reduces CPU memory pressure (no page cache pollution)
- Theoretical bandwidth: NVMe line rate directly to GPU (up to 7 GB/s per NVMe, 25+ GB/s with RAID)

### 2.3 Limitations

- **FUSE cluster**: Storage may not support GDS (NFS/shared filesystem typically does NOT support GDS). Need local NVMe or GDS-compatible parallel FS.
- **cuFile reads to a device pointer**: data must already be in a format the GPU can directly consume — no on-the-fly transformation
- **Variable-length data** (comm_sets) cannot be efficiently loaded via cuFile
- **File must be opened with O_DIRECT** semantics (no page caching)

### 2.4 Implication for Cache Design

For cuFile to be useful, the cache file must have:
1. **Fixed-stride, aligned data blocks** — so cuFile can read contiguous regions at known offsets
2. **GPU-consumable layout** — data lands directly in GPU buffers matching kernel expectations
3. **4 KB alignment** for all array start offsets in the file

---

## 3. Proposed Schemes

### Scheme A: Single Concatenated File with Rank Index Table

**Concept**: Merge all per-rank cache files into one file with a global header containing an index table. Each rank's data block is stored sequentially, self-contained.

**File layout**:
```
[GlobalHeader: magic, version, nprocs, fingerprint[32], total_events]
[RankIndex[nprocs]: {rank, offset, event_count, num_comm_sets}]
[Rank 0 data: events[N0] | types[N0] | timestamps[N0] | ... | comm_sets | coll_bytes]
[Rank 1 data: events[N1] | types[N1] | timestamps[N1] | ... | comm_sets | coll_bytes]
...
[Rank P-1 data: ...]
```

Each rank's data block is identical to the current per-rank file body (no header per rank — the global index table replaces it).

**Reading**: Each rank reads the global header + index table, seeks to its offset, reads its block.

**cuFile compatibility**: Partially. Each rank can cuFile-read its own block's fixed arrays into GPU, but:
- The arrays within each rank block are **row-grouped** (all of rank k's arrays together), not column-grouped
- cuFile would need one read per array per rank (9-10 reads per rank), each at a different file offset
- comm_sets remain variable-length and must be handled on CPU

**Pros**:
- Simplest change from current architecture — just concatenation with an index
- Single file is easier to manage (no directory of N files)
- Rank 0 can write the whole file, or MPI-IO can be used for parallel write
- Cache validation is simpler (one fingerprint check)

**Cons**:
- No cuFile advantage over current per-rank files (still need per-array seeks)
- Parallel write is harder (need to know all event counts before writing to compute offsets)
- Large file I/O contention: all ranks reading from the same file simultaneously (NFS bottleneck)
- comm_sets variable-length sections prevent clean cuFile reads
- Rank count changes invalidate entire cache (same as current)

**Best for**: Simplifying file management while keeping the same read pipeline.

---

### Scheme B: Column-Major Single File (cuFile-Optimized)

**Concept**: Store all events from all ranks in a single file, organized by **column** (one contiguous block per SoA array). Each column contains all ranks' data concatenated. This is the ideal layout for cuFile bulk reads.

**File layout**:
```
[GlobalHeader: 256 bytes, 4KB-aligned]
  magic, version, nprocs, fingerprint[32], total_events
  rank_event_counts[nprocs]   — how many events each rank owns
  column_offsets[15]           — file byte offset for each column
  metadata_offset              — file offset for variable-length section

[Column 0: events]       all_events[total_N]      @ 4KB-aligned offset
[Column 1: types]        all_types[total_N]        @ 4KB-aligned offset
[Column 2: timestamps]   all_timestamps[total_N]   @ 4KB-aligned offset
[Column 3: end_ts]       all_end_timestamps[total_N] @ 4KB-aligned offset
[Column 4: pids]         all_pids[total_N]         @ 4KB-aligned offset
[Column 5: srcs]         ...
[Column 6: dsts]         ...
[Column 7: tags]         ...
[Column 8: roots]        ...
[Column 9: leave_recv_ts] (optional, flagged)      @ 4KB-aligned offset

[Metadata section: variable-length]
  Per-rank comm_sets (serialized with length prefixes)
  Per-rank coll_bytes_sent, coll_bytes_received
```

Within each column, data is ordered by rank: `[rank0_events | rank1_events | ... | rankP-1_events]`.

**cuFile usage**:
```cpp
// To load timestamps for batch [event_start, event_start + batch_size):
size_t file_off = header.column_offsets[2] + event_start * sizeof(uint64_t);
cuFileRead(fh, pool.d_timestamps, batch_size * sizeof(uint64_t), file_off, 0);
```

A single `cuFileRead()` per column directly populates the GPU buffer — **no host memory involved**.

**For batched processing**: Each batch is a contiguous range of the global event array. The file offset for column C, batch starting at event E, is simply:
```
column_offsets[C] + E * element_size[C]
```

**Pros**:
- **Optimal for cuFile/GDS**: one read per column per batch, directly to GPU device pointer
- Eliminates ALL host-side copies (no SHM, no heap vectors, no cudaMemcpy)
- Natural batch boundaries: just slide a window over the global event array
- Column-selective loading: only read the 7 columns the GPU actually needs (skip types, srcs, dsts, tags)
- 4KB alignment per column is easy to guarantee
- Massive bandwidth utilization: 7 parallel cuFileRead calls, each sequential I/O

**Cons**:
- **Writing requires two passes or coordination**: all ranks must participate (MPI-IO with collective writes, or rank 0 gathers and writes serially)
- **CPU processing still needs host data**: P2P matching and collective grouping run on CPU before GPU analysis — those CPU stages need types, srcs, dsts, tags which cuFile would skip
- Variable-length comm_sets are awkward (shoved into a metadata section)
- Ranking change invalidates entire cache
- Single large file may hit filesystem limits on NFS (but unlikely under 200 GB)
- **GDS driver support required** — FUSE cluster may not have it

**Critical issue — CPU stages need host data**:

The current pipeline does P2P matching and collective grouping on the CPU *using the SHM window*, before GPU analysis. These CPU stages need `types`, `srcs`, `dsts`, `tags` (columns that cuFile would skip for GPU). Two options:
1. **Dual read**: cuFile the 7 GPU columns to VRAM, and fread the CPU columns to host memory (complex but doable)
2. **Restructure to GPU-only**: move P2P matching + collective grouping to GPU (major rearchitecture)
3. **Read everything to host first, then cuFile for GPU transfer only** — but this defeats the purpose

Option 1 is realistic: read the full file to host (fast with mmap or buffered I/O), do CPU work, then use cuFile only for the H2D transfer step. But then the benefit is limited to replacing cudaMemcpy with cuFileRead, which only saves the pinning overhead.

**Best for**: Future architectures where the entire pipeline is GPU-resident, or where CPU preprocessing is eliminated.

---

### Scheme C: Hybrid Two-Tier (Fixed SoA File + Metadata File)

**Concept**: Split the cache into two files:
1. **Data file** (`.soa`): Fixed-stride column-major binary, cuFile-friendly
2. **Meta file** (`.meta`): Variable-length data (comm_sets, coll_bytes, header info)

**Data file layout** (`.soa`):
```
[Superblock: 4KB]
  magic, version, fingerprint[32], nprocs, total_events
  rank_boundaries[nprocs+1]  — cumulative event counts (offset array)
  column_descriptors[10]     — {offset, element_size, total_bytes} per column

[Column 0: events]       @ 4KB-aligned
[Column 1: types]        @ 4KB-aligned
...
[Column 9: leave_recv_ts] @ 4KB-aligned
```

**Meta file layout** (`.meta`):
```
[Header: nprocs, fingerprint[32]]
[Per-rank metadata]:
  rank 0: num_comm_sets, comm_sets data, coll_bytes_sent, coll_bytes_received
  rank 1: ...
  ...
```

**Pros**:
- Clean separation: data file is 100% fixed-stride, no variable-length sections
- Data file can be mmap'd, cuFile'd, or fread'd — maximum flexibility
- Meta file is small (KBs) and only read by CPU
- cuFile can selectively read individual columns
- Easier to validate (fingerprint in both files, but data file is self-describing)

**Cons**:
- Two files to manage (but contained in same `soa_cache/` directory)
- Same CPU preprocessing issue as Scheme B (need host copies for P2P/coll phases)
- Writing requires coordination across ranks (MPI-IO or gather-and-write)
- Slightly more complex cache path management

**Best for**: Incremental upgrade path — start with host reads, later enable cuFile for the data file when GDS is available.

---

### Scheme D: Streaming cuFile with Batched GPU Transfer

**Concept**: Keep the column-major single file from Scheme B, but design explicitly for **streaming** — the file is read in batches matching the GPU batch size, with double-buffering to overlap I/O and compute.

**Pipeline**:
```
Batch 0:  cuFileRead(col, batch0) → GPU buf A → kernel(A)
Batch 1:  cuFileRead(col, batch1) → GPU buf B → kernel(B)   [overlapped with kernel(A)]
Batch 2:  cuFileRead(col, batch2) → GPU buf A → kernel(A)   [overlapped with kernel(B)]
...
```

**Two GPU buffers** (A and B) in the memory pool, each sized for `max_batch_events`. While kernels execute on buffer A, cuFileRead populates buffer B.

**VRAM budget per buffer** (7 arrays × element sizes):
```
events:        4B × batch_size
timestamps:    8B × batch_size
end_timestamps: 8B × batch_size
leave_recv_ts: 8B × batch_size
match_partner: 4B × batch_size  ← NOT from file; computed on CPU
pids:          4B × batch_size
roots:         4B × batch_size
Input total:   40B × batch_size

Output arrays: ~16B × batch_size (8 analysis durations + counters)
Total per buffer: 56B × batch_size
Double buffer: 112B × batch_size
```

For RTX 4090 (24 GB, ~20 GB usable): `batch_size ≈ 20GB / 112B ≈ 178M events` per buffer (but with double buffering: `~89M` events each).

Wait — `match_partner` is computed by CPU P2P matching, not read from the file. This means cuFile cannot be the sole data path. The CPU **must** see the full data to compute match_partner, which is then transferred to GPU.

**This is the fundamental constraint**: P2P matching and collective grouping are CPU-bound preprocessing steps that require host-resident data. cuFile's direct-to-GPU path can only bypass host memory for data that does NOT need CPU processing.

**Resolution — Column-selective hybrid**:
- Columns needed **only by GPU** (and already computed by GPU): `timestamps`, `end_timestamps`, `leave_recv_ts`, `roots` → candidates for cuFile
- Columns needed by **both CPU and GPU**: `events`, `pids` → must be on host; cudaMemcpy to GPU
- Columns needed **only by CPU**: `types`, `srcs`, `dsts`, `tags`, `match_partner`, `coll_group_id` → host only
- `match_partner`, `coll_group_id` are **computed** (not cached), so not in the file

So cuFile can directly load 4 of 7 GPU-input arrays (timestamps, end_timestamps, leave_recv_ts, roots = 28B/event). The other 3 (events, pids, match_partner) must go through host.

**Streaming pipeline (revised)**:
```
   CPU                                          GPU
┌──────────────────────────────────┐  ┌─────────────────────────────┐
│ 1. fread batch from .soa file   │  │                             │
│    (all columns, host memory)   │  │                             │
│ 2. P2P matching → match_partner │  │                             │
│ 3. Collective grouping → CSR    │  │                             │
│ 4. cudaMemcpy: events, pids,    │→→│ 4'. Receive events, pids,   │
│    match_partner (12B/evt)      │  │     match_partner            │
│                                  │  │ 5. cuFileRead: timestamps,  │
│                                  │  │    end_ts, leave_recv,      │
│                                  │  │    roots (28B/evt)          │
│                                  │  │ 6. Kernel execution         │
│                                  │  │ 7. D2H result copy          │
└──────────────────────────────────┘  └─────────────────────────────┘
```

**Pros**:
- Reduces host→GPU transfer by 70% (28 of 40 bytes via cuFile)
- Double-buffering hides I/O latency
- Handles arbitrary trace sizes (streaming, batch_size adapts to VRAM)
- Natural extension of existing batched architecture

**Cons**:
- CPU preprocessing is still the bottleneck (10.6s of 24.4s for n1024), and it still needs host data
- Complex synchronization: cuFile reads and cudaMemcpy must coordinate on separate streams
- cuFile benefit is partial — host copies still needed for 3 arrays
- Implementation complexity is high (double-buffering + hybrid I/O)
- GDS driver dependency

**Best for**: Squeezing maximum I/O throughput on GDS-equipped systems, after CPU preprocessing is optimized.

---

### Scheme E: mmap'd Single File (No cuFile, Minimal Change)

**Concept**: Merge all per-rank files into a single column-major file (like Scheme B), but instead of cuFile, use `mmap()` for zero-copy host access and keep the existing cudaMemcpy pipeline.

**Pipeline**:
```
Single .soa file (disk)
  ↓ mmap() — zero-copy, lazy page faults
Memory-mapped SoA columns (virtual addresses)
  ↓ shmSetupLocalSoA points directly into mmap'd region (per-rank offsets)
  ↓ P2P matching, collective grouping (CPU) — operates on mmap'd data
  ↓ cudaHostRegister (pin the mmap'd pages) or cudaMemcpy from mmap'd
GPU buffers
```

This eliminates copy #1 (fread → heap) and copy #2 (heap → SHM) from the current pipeline. The mmap'd file IS the SHM-equivalent — all ranks can mmap the same file.

**Pros**:
- **Eliminates 2 of 3 copies** without needing GDS
- Single file, easily managed
- Works on any filesystem (NFS, ext4, etc.) — no GDS driver needed
- mmap is lazy: only pages actually accessed are loaded (good for column-selective access)
- Can replace `MPI_Win_allocate_shared` with a simple `mmap()` — simpler code
- cudaHostRegister works on mmap'd regions (with `MAP_LOCKED` or after madvise)
- Transparent huge pages work with mmap (`madvise(MADV_HUGEPAGE)`)

**Cons**:
- cudaMemcpy H2D still required (no direct-to-GPU)
- mmap on NFS can have performance issues (page size, readahead behavior)
- Page faults during P2P matching may cause unpredictable latency (need prefaulting/madvise)
- cudaHostRegister on mmap'd memory may fail if the region is too large (same 10 GB threshold issue)
- Column-major layout means CPU P2P matching accesses multiple columns with different strides (cache unfriendly for the CPU — but same as current SHM layout)
- **match_partner and coll_group_id are computed, not stored**: need a separate writable buffer alongside the read-only mmap

**Best for**: Immediate improvement without hardware dependencies. Works on FUSE cluster today.

---

## 4. VRAM Overflow Analysis

### 4.1 The Problem

The GPU only has 24 GB (4090) or 80 GB (H100). The data needed for GPU analysis:
- Input: 7 arrays × N events = 40 B/event
- Output: 8 duration arrays + 8 counters = ~16 B/event (worst case)
- Total: ~56 B/event per batch

| Trace | Events | GPU Data (56B/evt) | 4090 Batches | H100 Batches |
|-------|--------|-------------------|-------------|-------------|
| n1024 | 262M | 14.7 GB | 1 | 1 |
| n2048 | 525M | 29.4 GB | 2 | 1 |
| n4096 | 1.06B | 59.4 GB | 3 | 1 |
| n8192 | ~2B | ~112 GB | 6 | 2 |

### 4.2 Already Solved — Batching

The current code already handles VRAM overflow via `shmRunBatchedGPU()`:
- Divides the global event array into batches of K ranks each
- K is chosen to fit the total events within ~20 GB VRAM
- GPUMemoryPool is allocated once for the largest batch
- Results are accumulated across batches

This batching is **orthogonal to the cache file format** — any scheme above works with batching.

### 4.3 cuFile + VRAM Overflow

With cuFile, the solution is natural: **partial reads**.

```cpp
// For each batch:
size_t batch_start = batch_event_offset;
size_t batch_count = batch_event_count;
for (int col : gpu_columns) {
    size_t file_off = column_offsets[col] + batch_start * elem_size[col];
    size_t nbytes = batch_count * elem_size[col];
    cuFileRead(fh, pool.d_array[col], nbytes, file_off, 0);
}
```

Each batch reads only its slice of each column. The file itself can be arbitrarily large — only `batch_count * 40 bytes` of VRAM is used at any time.

### 4.4 mmap + VRAM Overflow

With mmap, the existing batching works directly: each batch's `TraceDataSoA` points into the mmap'd region at the batch offset, and cudaMemcpy copies only the batch's data to GPU.

### 4.5 Summary

VRAM overflow is **not a blocking concern** for any scheme — batching handles it. The question is whether cuFile's direct path provides enough bandwidth improvement to justify the complexity.

---

## 5. Comparison Matrix

| Criterion | A: Concat+Index | B: Column-Major | C: Hybrid 2-file | D: Streaming cuFile | E: mmap |
|-----------|:-:|:-:|:-:|:-:|:-:|
| **Single file** | Yes | Yes | No (2 files) | Yes | Yes |
| **cuFile compatible** | Partial | Full | Full (data file) | Full | No |
| **Eliminates host copies** | 0 of 3 | 2 of 3 (GPU cols) | 2 of 3 (GPU cols) | 2 of 3 (4 cols) | 2 of 3 |
| **CPU preprocess works** | Yes | Needs host read | Needs host read | Needs host read | Yes (mmap) |
| **Works on NFS/FUSE** | Yes | No (GDS) | Data: No, Meta: Yes | No (GDS) | Yes |
| **Implementation effort** | Low | Medium | Medium | High | Medium |
| **Handles VRAM overflow** | Batching | cuFile partial | cuFile partial | Double-buffer | Batching |
| **Write complexity** | Low (concat) | High (MPI-IO) | High (MPI-IO) | High (MPI-IO) | High (MPI-IO) |
| **Expected read speedup** | 1x (same) | 2-3x (GDS) | 2-3x (GDS) | 2-4x (overlap) | 1.5-2x (no copies) |
| **Variable-length data** | In-file | Metadata section | Separate file | Metadata section | Metadata section |
| **Filesystem dependency** | None | GDS driver | GDS driver | GDS driver | None |

---

## 6. Recommendation

### Short-term (works on FUSE today): Scheme E (mmap)

The biggest wins with the least risk:
1. Merge per-rank files into **one column-major file** (shared format for Schemes B-E)
2. Use `mmap()` instead of fread + SHM, eliminating 2 host-side copies
3. Keep existing cudaMemcpy + batching pipeline for GPU transfer
4. Existing cudaHostRegister/prefault logic adapts naturally to mmap'd pages

**Expected improvement**: ~1.5-2x on cache read phase (from 2.2s to ~1.1s for n1024), since we eliminate heap allocation + memcpy into SHM.

### Medium-term (when GDS is available): Upgrade to Scheme B/D

Once FUSE cluster gets GDS-compatible storage (local NVMe on compute nodes):
1. The same column-major file format from Scheme E becomes cuFile-readable
2. Add cuFileRead path for the 4 GPU-only columns (timestamps, end_timestamps, leave_recv_ts, roots)
3. Keep mmap path for CPU-accessed columns
4. Double-buffer for streaming if needed

**Expected improvement**: Additional 2-3x on the GPU transfer phase.

### Long-term: Full GPU pipeline

Move P2P matching and collective grouping to GPU. Then cuFile can load ALL data directly to GPU with zero host involvement. This is the ultimate goal but requires major rearchitecture.

---

## 7. cuFile API Usage Details

### 7.1 Minimal cuFile Example (for reference)

```cpp
#include <cufile.h>
#include <fcntl.h>

// Initialize cuFile driver (once per process)
CUfileError_t status = cuFileDriverOpen();

// Open file
int fd = open("cache.soa", O_RDONLY | O_DIRECT);
CUfileDescr_t cf_descr = {};
cf_descr.handle.fd = fd;
cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
CUfileHandle_t fh;
cuFileHandleRegister(&fh, &cf_descr);

// Register GPU buffer
cuFileBufRegister(d_timestamps, batch_size * sizeof(uint64_t), 0);

// Read directly to GPU
cuFileRead(fh, d_timestamps, batch_size * sizeof(uint64_t),
           file_offset,  // must be 4KB-aligned
           0);           // device buffer offset

// Cleanup
cuFileBufDeregister(d_timestamps);
cuFileHandleDeregister(fh);
close(fd);
cuFileDriverClose();
```

### 7.2 Alignment Requirements

| Parameter | Requirement |
|-----------|------------|
| File offset | Multiple of **4 KB** (filesystem block size) |
| GPU buffer alignment | At least 512 B (4 KB recommended) |
| Transfer size | Multiple of 4 KB for best throughput |
| O_DIRECT flag | Required on file descriptor |
| GPU buffer | Must be registered with `cuFileBufRegister` |

For our column-major format, ensuring each column starts at a 4KB boundary satisfies the file offset constraint. `cudaMalloc` returns 256-byte-aligned pointers; for cuFile, we should use `cudaMallocAligned` or over-allocate and align manually.

### 7.3 Build Requirements

```cmake
find_library(CUFILE_LIB cufile HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
target_link_libraries(gpu_analyzer PRIVATE ${CUFILE_LIB})
```

Requires: CUDA 11.4+, GDS driver (nvidia-gds package), GDS-compatible filesystem.

### 7.4 Fallback Strategy

Since FUSE cluster may not support GDS, implement a runtime check:
```cpp
CUfileError_t err = cuFileDriverOpen();
bool gds_available = (err.err == CU_FILE_SUCCESS);
if (gds_available) {
    // cuFile path
} else {
    // fallback to mmap + cudaMemcpy (Scheme E)
}
```

This allows a single binary to work on both GDS-equipped and non-GDS systems.

---

## 8. Appendix: Cache File Size Estimates (Column-Major Single File)

For a column-major single file, the file size is:
```
header + sum(N_total × element_size per column) + metadata
```

| Trace | Events | Fixed Data (52B/evt) | Metadata (est.) | Total File |
|-------|--------|---------------------|----------------|-----------|
| 16q | 4.1M | 213 MB | ~1 KB | ~213 MB |
| n1024 | 262M | 13.6 GB | ~50 KB | ~13.6 GB |
| n2048 | 525M | 27.3 GB | ~100 KB | ~27.3 GB |
| n4096 | 1.06B | 55.1 GB | ~200 KB | ~55.1 GB |
| n8192 | ~2B | ~104 GB | ~400 KB | ~104 GB |

Note: These are slightly larger than the sum of per-rank files because of 4KB alignment padding between columns (at most 10 × 4KB = 40 KB overhead, negligible).

---

## 9. Key Insight: CPU Preprocessing is the Fundamental Constraint

The most important finding from this analysis is that **cuFile's direct-to-GPU reads cannot fully bypass host memory** because the current pipeline requires CPU-side preprocessing:

1. **P2P Matching** (CPU, 617 ms for n1024): Matches send/recv events by scanning sorted event lists per location. Output: `match_partner[]` array — must be computed before GPU analysis.
2. **Collective Grouping** (CPU, 2,875 ms for n1024): Groups collective events by communicator. Output: CSR structure — must be computed before GPU analysis.
3. **Timestamp Correction** (CPU, <57 ms): Adjusts for clock skew — must run before GPU analysis.

These CPU stages need `events`, `types`, `pids`, `srcs`, `dsts`, `tags` on the host. So even with cuFile, at least these columns must be host-readable.

**The staged approach**:
1. **Phase 1**: Read ALL columns to host (mmap or fread) — CPU preprocessing runs here
2. **Phase 2**: cuFile reads the 4 "GPU-only" columns (timestamps, end_timestamps, leave_recv_ts, roots) directly to GPU — 70% of GPU input bytes
3. **Phase 2'**: cudaMemcpy the 3 "CPU+GPU" columns (events, pids, match_partner) — 30% of GPU input bytes

Only if/when P2P matching and collective grouping move to GPU can cuFile fully eliminate host I/O.

---

## 10. Related Documentation

- `docs/13-PERFORMANCE-ENHANCEMENT-PLAN.md`: Enhancement 1c (pre-converted binary) and 1d (GPUDirect Storage) — high-level roadmap items. This report expands them into concrete schemes.
- `docs/05-OTF2-READER.md`: Binary SoA cache format and performance numbers.
- `docs/08-CUDA-KERNELS.md`: GPU memory allocation, 7 input arrays (32-40 B/event).
- `docs/04-DATA-STRUCTURES.md`: TraceDataSoA (14 arrays, 60 B/event).
- `docs/PROBLEMS.md` TODO list item about OTF2 reading bottleneck (item for GDS).
- `chat-history/260406-1107-binary-soa-cache.md`: Implementation log for the current per-rank cache.
