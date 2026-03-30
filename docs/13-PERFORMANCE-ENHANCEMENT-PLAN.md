# Performance Enhancement Plan

This document outlines potential optimizations for the GPU trace analyzer, prioritized by expected impact. Timing data is from the latest SHM-optimized builds (2026-03-29) on LAMMPS n1024 (262M events, 9.9GB) with 16 MPI reader ranks on a single fuse node.

## Current Timing Baseline (LAMMPS n1024)

| Phase | Time (ms) | % of Total |
|-------|----------|-----------|
| OTF2 Read | 31,403 | 66.5% |
| Preprocess (CPU) | 10,600 | 22.4% |
| GPU Analysis | 1,237 | 2.6% |
| Cleanup | 2,476 | 5.2% |
| Statistics | 1,533 | 3.2% |
| **Total** | **47,249** | **100%** |

### Preprocess sub-phases (n1024)

| Sub-phase | Time (ms) | % of Preprocess |
|-----------|----------|----------------|
| fill_soa (page faults + copy) | 5,061 | 48% |
| coll_grouping | 2,875 | 27% |
| pinning (cudaHostRegister) | 1,990 | 19% |
| p2p_matching | 617 | 6% |
| ts_correction + misc | 57 | <1% |

### GPU sub-phases (n1024)

| Sub-phase | Time (ms) |
|-----------|----------|
| batch_prep (remap + CSR build) | 523 |
| H2D transfer (pinned DMA) | 447 |
| Kernels (P2P=9ms, Coll=88ms) | 97 |
| D2H + other | 170 |

---

## Priority 1: OTF2 Reading (66.5% of total — 31.4s)

The single largest bottleneck. The OTF2 library's per-location sequential access pattern limits throughput to ~315 MB/s across 16 reader ranks.

### Enhancement 1a: Increase MPI Reader Ranks

**Effort**: Minimal (config change)
**Expected impact**: 1.5-2× read speedup

Current: 16 reader ranks. Measured speedup scales to ~32 ranks (44× over serial). Simply increasing to 32-64 ranks should nearly halve read time on a 128-thread node.

**Caveats**: Diminishing returns beyond ~64 ranks due to OTF2 library overhead and disk I/O saturation. Memory usage increases linearly with ranks.

### Enhancement 1b: Multi-threaded Reading Within Each Rank

**Effort**: Medium (OTF2 API supports per-location readers)
**Expected impact**: 2-4× read speedup

Each MPI rank currently reads its assigned locations sequentially. Using pthreads or OpenMP, each rank could read multiple locations concurrently. OTF2's thread-safe per-location reader API supports this.

**Implementation sketch**:

```
For each rank:
  Pass 1: Spawn T threads, each handles N/T locations → count events
  Barrier
  Pass 2: Spawn T threads, each fills its subset of SoA
```

**Caveats**: OTF2 internal file handles may contend. Need benchmark to find optimal T.

### Enhancement 1c: Pre-converted Binary Format

**Effort**: High (new tool + format)
**Expected impact**: 5-10× read speedup

Convert OTF2 to a simple SoA binary format (one-time preprocessing). The analyzer reads the binary directly via `mmap()` or `read()`, bypassing OTF2 deserialization.

**Format**: One file per SoA array (e.g., `timestamps.bin`, `event_types.bin`). Header contains event count, location offsets.

**Caveats**: One-time conversion cost. Must be re-done when trace changes. Ties the tool to a specific binary format.

### Enhancement 1d: GPUDirect Storage (cuFile API)

**Effort**: High (requires GDS support on storage)
**Expected impact**: 2-3× for GPU-bound phases (minimal for overall if reading is the bottleneck)

Load trace data directly from NVMe SSD to GPU memory, bypassing CPU entirely. Requires NVIDIA GPUDirect Storage API and compatible storage controller.

**Caveats**: Only helps if the data is in a GPU-friendly format (see 1c). FUSE cluster storage may not support GDS.

---

## Priority 2: fill_soa Page Faults (48% of preprocess — 5.1s)

### Enhancement 2a: SHM Prefaulting

**Effort**: Low (few lines of code)
**Expected impact**: 2-3× fill_soa speedup

After `MPI_Win_allocate_shared`, use `madvise(MADV_WILLNEED)` or explicit `memset()` on the window to pre-fault physical pages before writing OTF2 data. This trades one pass of page faults during fill for a more efficient bulk prefault.

**Implementation**: Add `memset(shm_ptr, 0, total_bytes)` before `readerFillSoA()`. The zeroing cost is ~1-2s for 16GB (128GB/s memory bandwidth), but eliminates the ~5s of scattered page faults during fill.

### Enhancement 2b: Huge Pages (2MB THP)

**Effort**: Low (kernel config or `madvise(MADV_HUGEPAGE)`)
**Expected impact**: 1.3-2× fill_soa speedup

Standard 4KB pages require 4M TLB entries for 16GB of data. With 2MB huge pages, only 8K entries are needed. This reduces TLB miss rate dramatically during the scattered write pattern of fill_soa.

**Implementation**: `madvise(shm_ptr, total_bytes, MADV_HUGEPAGE)` after SHM allocation.

---

## Priority 3: Collective Grouping (27% of preprocess — 2.9s)

### Enhancement 3a: Hash-Based Comm-Set Lookup

**Effort**: Low-Medium
**Expected impact**: 2-5× grouping speedup for traces with many communicators

Replace the linear scan + vector equality comparison with a hash map keyed by a comm_set hash. Pre-compute a deterministic hash of the sorted communicator member set (e.g., FNV-1a or XXHash over the sorted member IDs).

**Implementation sketch**:
```cpp
// Instead of linear scan over pending groups:
uint64_t cs_hash = hashCommSet(comm_set);
auto it = pending_by_hash.find({event_type, cs_hash});
```

**Caveats**: Hash collisions require fallback to vector comparison. For typical MPI programs with 1-3 communicators, the impact is minimal.

### Enhancement 3b: Communicator ID Caching

**Effort**: Low
**Expected impact**: Eliminates redundant sort in normalizeCommSet()

Cache the normalized (sorted) comm_set by communicator ID during OTF2 reading. Since most MPI programs use a small number of communicators, the cache hit rate would be ~99%+.

### Enhancement 3c: Parallelize Across Event Types

**Effort**: Medium
**Expected impact**: Up to 6× if dominant type is one of {Bcast, Reduce, AllReduce, Barrier, AlltoAll, Scan}

Since collective grouping per event type is independent, process all types in parallel using threads. Partition the event stream by type, then group each partition independently.

---

## Priority 4: cudaHostRegister Pinning Cost (19% of preprocess — 2.0s)

### Enhancement 4a: Persistent Pinned Allocator

**Effort**: Medium
**Expected impact**: Eliminate 2.0s pinning + 2.5s cleanup = 4.5s saved

Instead of `MPI_Win_allocate_shared` + `cudaHostRegister`, use a custom allocator that allocates pinned memory from the start (`cudaHostAlloc` or `cudaMallocHost`). This eliminates the separate pinning step entirely.

**Caveats**: Pinned memory is limited by the OS. For >16GB, this may fail or degrade performance. The adaptive pinning heuristic already handles the >10GB case with per-batch pinning.

### Enhancement 4b: Lazy Per-Batch Pinning (Already Implemented for >10GB)

The current code already uses per-batch pinning for traces >10GB. For smaller traces, full-window pinning is used because it's simpler and the 2s cost is acceptable relative to the 31s read time.

---

## Priority 5: GPU Kernel Optimization (2.6% of total — 1.2s)

The GPU phase is already very fast. Optimizations here have diminishing returns on total time.

### Enhancement 5a: Warp-Level Parallel Reduction for Collectives

**Effort**: Medium
**Expected impact**: 2-4× kernel time reduction for collective kernels (88ms → ~25ms)

Currently, collective kernels launch one block per group with only thread 0 active. For large groups (1024+ members), using warp-level parallel reduction (warp shuffle `__shfl_down_sync`) for max/min computations would parallelize the inner loop.

**Implementation**: Replace the thread-0 serial loop with a warp-cooperative scan/reduction. Each thread processes `ceil(group_size / 32)` members, then warp-reduce to find max timestamp.

### Enhancement 5b: Inter-Batch Pipelining

**Effort**: Medium
**Expected impact**: Hide 523ms batch_prep behind GPU execution

Use double-buffering: while batch K runs on the GPU, CPU prepares batch K+1 (remap match_partner, build CSR). Requires two sets of host-side buffers and careful synchronization.

**Caveats**: The GPU is only busy for ~97ms per batch (kernels + H2D + D2H), while CPU prep takes ~523ms. So pipelining would make the GPU wait for CPU, not the other way around. The benefit is limited unless CPU prep is also parallelized.

### Enhancement 5c: Fused P2P + Collective Kernel

**Effort**: Low
**Expected impact**: Marginal (saves kernel launch overhead)

Merge the 5 separate kernel launches (late_sender_receiver, barrier_wait, barrier_completion, early_reduce_late_broadcast, wait_nxn_nxn_completion) into fewer launches by using runtime branching on event type. Each event determines which analysis to apply.

**Caveats**: Branch divergence within warps may negate the benefit. Current 5-kernel approach has clean separation and no divergence.

---

## Priority 6: Statistics (3.2% of total — 1.5s)

### Enhancement 6a: GPU Radix Sort for Quantiles

**Effort**: Medium
**Expected impact**: 2-3× statistics speedup (1.5s → 0.5s)

Use CUB `DeviceRadixSort` to sort duration arrays on GPU, then read quantile values directly. Radix sort of 45M doubles on RTX 4090 takes ~50ms (vs 1.5s CPU nth_element).

**Caveats**: Requires keeping duration arrays on GPU (currently they're D2H'd). Could integrate with the kernel output phase.

### Enhancement 6b: Approximate Quantiles (T-Digest / Count-Min)

**Effort**: Medium
**Expected impact**: Near-zero statistics time

Use a streaming approximate quantile algorithm (T-Digest) that computes during the GPU output phase. No sorting needed. Accuracy within 1% for practical purposes.

**Caveats**: Approximate, not exact. May not match Scalasca's exact statistics.

---

## Priority 7: Large-Scale Trace Support (n4096+)

### Enhancement 7a: Fix redistributeCollectives Integer Overflow (TODO 22)

**Effort**: Low (type change)
**Expected impact**: Unblocks n4096+ analysis

Change `int` types in `redistributeCollectives` to `int64_t`/`size_t` for counts and displacements. This fixes the crash that currently prevents analyzing traces with 4096+ ranks.

### Enhancement 7b: Communicator Deduplication

**Effort**: Medium
**Expected impact**: 10-100× reduction in flat_comm_sets memory for large traces

Instead of flattening the full communicator member list for each collective event, maintain a communicator-to-members mapping (deduplication). Each event stores only a communicator ID. For n4096 with `MPI_COMM_WORLD`, this reduces 4096 repeated members per event to a single ID lookup.

### Enhancement 7c: Multi-Node Distributed Analysis

**Effort**: High
**Expected impact**: Linear scaling with nodes for CPU-bound phases

For traces with 4096+ ranks where CPU preprocessing dominates, distribute the analysis across multiple nodes. Each node handles a subset of ranks (reading + preprocessing + local GPU analysis), with only final statistics gathered to one node.

This is essentially the TileTrace approach but with GPU kernels instead of CPU analysis per node.

---

## Summary: Expected Impact by Priority

| Priority | Enhancement | Time Saved (n1024) | Effort |
|----------|------------|-------------------|--------|
| 1a | More reader ranks (32-64) | ~15s (50% of read) | Minimal |
| 1b | Multi-threaded per-rank reading | ~20s (65% of read) | Medium |
| 2a | SHM prefaulting | ~3s (60% of fill_soa) | Low |
| 2b | Huge pages | ~1.5s (30% of fill_soa) | Low |
| 3a | Hash-based comm_set lookup | ~1.5s (50% of grouping) | Low-Medium |
| 4a | Persistent pinned allocator | ~4.5s (pin + cleanup) | Medium |
| 5a | Warp-level collective kernels | ~0.06s (marginal) | Medium |
| 6a | GPU radix sort for stats | ~1.0s | Medium |
| 7a | Fix int overflow for n4096+ | Unblocks large traces | Low |

**Quick wins** (low effort, significant impact): 1a, 2a, 2b, 3a, 7a
**Medium effort, high impact**: 1b, 4a
**High effort, transformative**: 1c, 7c
