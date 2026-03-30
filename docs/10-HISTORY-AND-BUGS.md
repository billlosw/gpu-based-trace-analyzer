# Development History and Bug Fixes

This document chronicles the key bugs encountered and fixed during development, providing context for future debugging.

## Timeline

| Date | Event |
|------|-------|
| 2026-03-13 | Initial implementation with GPU hash table matching |
| 2026-03-14 (AM) | Fixed empty results: MPI_Init, P2P matching, collective grouping |
| 2026-03-14 (PM) | Added MPI-parallel reading (7.5x speedup to 44x) |
| 2026-03-14 (PM) | Discovered and fixed TileTrace's tiled replay bug |
| 2026-03-15 (AM) | Fixed recv-side timestamp semantics (Enter vs point-event) |
| 2026-03-15 (PM) | Fixed send-side timestamp semantics (Enter vs point-event) |
| 2026-03-15 (PM) | Achieved exact match with Scalasca on 7/8 analyses (late_receiver pending) |
| 2026-03-16 | Added timestamp mode selector (SCALASCA / TILETRACE via CMake) |
| 2026-03-17 | Fixed late_receiver: independent check + correct Scalasca timestamps → 8/8 exact match |
| 2026-03-26 | Fixed Irecv enter timestamp per-request_id tracking; performance comparison on large traces |
| 2026-03-29 | CLC timestamp correction v4 (blocking-only neutralization) |
| 2026-03-29 | MPI shared memory window optimization (1.6-2.6x analysis speedup) |
| 2026-03-29 | Direct-to-SHM reading (50% memory reduction) |
| 2026-03-29 | GPUMemoryPool + async stream (eliminates per-batch cudaMalloc) |
| 2026-03-29 | Statistics: `std::sort` → `std::nth_element` (4-5x faster) |
| 2026-03-29 | P2PMatching.cu → .cpp rename (4.5x faster: no CUDA runtime init) |
| 2026-03-29 | GPU sort investigation for P2P matching (rejected: multi-process OOM) |

## Bug 1: Silent CUDA Kernel Failure (CUDA Version Mismatch)

**Symptom**: Kernels launch without error, `cudaMemcpy` works, but all output arrays contain zeros.

**Root Cause**: CUDA toolkit version (12.9) exceeded GPU driver version (12.8) on the compute node. PTX compiled with toolkit 12.9 cannot be JIT-compiled by the 12.8 driver. The error "PTX compiled with unsupported toolchain" only appears in `cuda-memcheck` or `CUDA_LAUNCH_BLOCKING=1` output.

**Fix**: Use `cuda@12.8.0` (not 12.9). The `helper.sh` was updated to auto-detect and use `spack location -i cuda@12.8.0`.

**Lesson**: CUDA toolkit version must be <= driver version. This is a host-side constraint, not a compute capability issue.

## Bug 2: GPU Hash Table P2P Matching

**Symptom**: P2P analyses returned incorrect results or no results.

**Root Cause**: Three issues with the GPU open-addressing hash table approach:
1. **No temporal ordering**: The same `(src, dst, tag)` key can appear multiple times. GPU threads matched arbitrarily, not by temporal order, causing cross-iteration matches.
2. **Probing chain breaks**: With concurrent insertion, linear probing chains had gaps (empty slots between entries with the same hash), causing lookups to stop early.
3. **Race conditions**: Multiple threads atomically claimed the same match target.

**Fix**: Replaced with CPU timestamp-sorted FIFO queue matching (same as TileTrace). This is O(N log N) and takes ~255 ms for 2.5M events — only 1% of total time.

**Lesson**: Fine-grained GPU parallelism doesn't work for order-dependent matching. The CPU approach is simple, correct, and fast enough.

## Bug 3: Missing MPI_Init

**Symptom**: OTF2 reader hangs or crashes when reading real trace files.

**Root Cause**: `main.cu` never called `MPI_Init()`. The program is linked against MPI (through otf2xx), and OTF2/otf2xx requires MPI to be initialized even in serial mode.

**Fix**: Added `MPI_Init(&argc, &argv)` at program start, `MPI_Finalize()` at all exit paths.

## Bug 4: Incorrect Collective Grouping Key

**Symptom**: Collective analyses returned wrong counts and values.

**Root Cause**: Original grouping key was `(event_type - TT_MPI_Bcast) * 1000000 + root`. This didn't distinguish:
- Multiple concurrent collectives of the same type with the same root
- Different sub-communicators vs MPI_COMM_WORLD
- Completed groups weren't being removed from the pending map

**Fix**: Rewrote using `(event_type, sorted_comm_set)` as the group key with `std::set<id_t>` tracking needed PIDs. This correctly handles all cases.

## Bug 5: Two-Pass OTF2 Reading

**Symptom**: Trace reading took ~210 seconds (for CG-C).

**Root Cause**: The original reader did two passes over the trace: pass 1 counted events, pass 2 filled arrays. This doubled I/O time.

**Fix**: Replaced with single-pass reading using `std::vector` dynamic arrays, then `memcpy` to `TraceDataSoA`.

## Bug 6: Recv-Side Timestamp Semantics

**Symptom**: 0 late_sender events and inflated late_receiver (all 1.2M pairs classified as late_receiver).

**Root Cause**: The `mpi_receive` point-event timestamp is the **completion time** (when data arrived), not the **enter time** (when the process started waiting). Similarly, `mpi_ireceive_complete` fires inside `MPI_Wait` at completion time. Using completion timestamps made `recv_ts > send_ts` almost always true.

**Fix**: Changed recv handlers to use `m_last_enter_ts[pid]`:
- Blocking `MPI_Recv`: uses `Enter(MPI_Recv)` timestamp
- Non-blocking `MPI_Irecv` + `MPI_Wait`: uses `Enter(MPI_Wait)` timestamp

### Failed Fix Attempt: Enter(MPI_Irecv)

An intermediate attempt used `Enter(MPI_Irecv)` for non-blocking receives. This was wrong because for CG's pattern `MPI_Irecv → MPI_Send → MPI_Wait`, `Enter(MPI_Irecv)` is when the receive is **posted** (before the send), not when the process starts **waiting**. This made `recv_enter < send_enter` essentially always true, producing all late_sender and 0 late_receiver — the opposite extreme.

**Correct approach**: Use `Enter(MPI_Wait)`, which is when the process actually starts waiting for the data.

## Bug 7: Send-Side Timestamp Asymmetry

**Symptom**: late_sender count was 3-6% higher than Scalasca (e.g., 408,375 vs 396,922 for CG-C).

**Root Cause**: After fixing recv timestamps, send handlers still used the `mpi_send` point-event timestamp instead of `Enter(MPI_Send)`. The gap between `Enter(MPI_Send)` and `mpi_send` includes Score-P's region-entry instrumentation overhead (nanoseconds to microseconds).

For pairs near the threshold where `Enter(MPI_Recv) ≈ Enter(MPI_Send)`:
```
Scalasca:     Enter(MPI_Send)=1000 vs Enter(MPI_Recv)=1002 → NOT late_sender
GPU Analyzer: mpi_send_event=1003  vs Enter(MPI_Recv)=1002 → late_sender (false positive!)
```

**Fix**: Changed `mpi_send` and `mpi_isend_request` handlers to use `m_last_enter_ts[pid]` instead of `event.timestamp()`.

**Result**: late_sender now matches Scalasca exactly.

## Bug 8: TileTrace Tiled Replay Bug

**Symptom**: When running TileTrace with `nprocs == nlocs` (64 procs for 64 locations), P2P analysis results were dramatically wrong (0 late_sender, 15k late_receiver vs expected 216k and 1M).

**Root Cause**: In TileTrace's `TraceReplay::forwardReplay()`, the `calculateReplayPid` ownership filter skipped tiles that belonged to remote locations. Recv events stored in `tile[receiver_pid]` on the sender's rank weren't replayed because `calculateReplayPid(receiver_pid)` mapped to the receiver's rank.

**Fix**: Removed the `calculateReplayPid` ownership check from the replay loop. Each rank now iterates all its local tiles unconditionally.

**Note**: This was a bug in TileTrace, not in the GPU analyzer. It was discovered because the GPU analyzer's correct results revealed TileTrace's discrepancy.

## Bug 9: OTF2 3.1 vs 3.0.3 Compatibility

**Symptom**: Build failure: OTF2 version mismatch.

**Root Cause**: The otf2xx submodule was updated to require OTF2 3.1 (`find_package(OTF2 3.1 EXACT)`), but the server only has 3.0.3.

**Fix**: Changed version requirement in otf2xx's CMakeLists and moved compile tests that used 3.1-specific enums behind the test guard.

## Bug 10: Late Receiver Algorithm (Two Bugs)

**Symptom**: late_receiver count was 3-4x higher than Scalasca (818,613 vs 250,643 for CG-B).

**Root Cause**: Two distinct bugs:

1. **Mutually exclusive branching**: The CUDA kernel had late_sender and late_receiver in `if/else` branches. Scalasca checks them independently in separate replay callbacks (`post_recv` for late_sender, `post_send_bws` for late_receiver).

2. **Wrong timestamps**: The kernel used `Enter(MPI_Wait)` as the recv timestamp and didn't check the `Leave(MPI_Send)` blocking condition. Scalasca uses `Enter(MPI_Irecv)` (when the receive request was posted) and requires `Leave(MPI_Send) > Enter(MPI_Irecv)` (sender still blocked when recv was posted).

**Fix**:
- Restructured `kernelLateSenderReceiver`: late_receiver is now an independent check, not mutually exclusive with late_sender.
- Under `#ifdef USE_SCALASCA_TIMESTAMPS`, the late_receiver condition is `Leave(MPI_Send) > Enter(MPI_Irecv) > Enter(MPI_Send)`, with duration `Enter(MPI_Irecv) - Enter(MPI_Send)`.
- Added three new OTF2 reader event handlers: `leave` (to capture `Leave(MPI_Send)`), `mpi_ireceive_request` (to capture `Enter(MPI_Irecv)`), plus `m_last_send_soa_idx` tracking.
- Added three new member variables: `m_last_leave_ts`, `m_irecv_enter_ts`, `m_last_send_soa_idx`.

**Result**: All 8 analyses now match Scalasca exactly (count, sum, max) on both CG-B and CG-C.

**Lesson**: Scalasca uses `pearl::timestamp_t = double` (signed), so subtractions can go negative and filter naturally. Our `uint64_t` timestamps require explicit `>` guards to prevent unsigned underflow.

## Bug 11: Irecv Enter Timestamp Per-PID Overwrite

**Symptom**: `late_receiver` counts differed from Scalasca on large traces with many concurrent non-blocking receives (NPB CG 1024: 8,985,237 vs 7,731,027).

**Root Cause**: `m_irecv_enter_ts` was keyed by PID (location ID) instead of `request_id`. When multiple `MPI_Irecv` operations are outstanding on the same location (common in NPB CG where 96% of recvs are non-blocking), only the LAST one's `Enter(MPI_Irecv)` timestamp was stored. Earlier Irecvs received incorrect (overwritten) timestamps.

**Fix**: Changed `m_irecv_enter_ts` from `unordered_map<id_t, timestamp_t>` (per-PID) to `unordered_map<uint64_t, timestamp_t>` (per-request_id). Look up by `event.request_id()` in both `mpi_ireceive_request` and `mpi_ireceive_complete`, with erase-after-use.

**Result**: Counts changed (CG 1024: 8,985,237 → 11,115,347). The increase is expected: correct earlier timestamps change which events satisfy the late_receiver condition. The remaining discrepancy with Scalasca is a separate algorithmic issue (see PROBLEMS.md TODO 19).

## Optimization 1: MPI Shared Memory Window (2026-03-29)

**Problem**: The Architecture C `streamBatchAnalysis()` spent 91-94% of GPU phase time on sequential MPI Send/Recv data transfer to rank 0 (13s for n1024, 38s for n2048), even though all ranks ran on the same node.

**Solution**: Replaced MPI Send/Recv with `MPI_Win_allocate_shared`. All ranks write data directly into a contiguous shared memory window. After `MPI_Win_fence`, rank 0 reads the data directly — no transfer needed.

**Results**: Analysis phase 14.7s → 9.3s (n1024, 1.58x), 40.5s → 15.7s (n2048, 2.58x).

## Optimization 2: Direct-to-SHM Reading (2026-03-29)

**Problem**: The original flow had double memory: local TraceDataSoA (calloc'd) + shared memory window (memcpy'd). For n1024: 16 GB + 16 GB = 32 GB peak.

**Solution**: Split-phase reader API. `readOTF2TracePhase1()` returns event count without allocating SoA. Shared window is allocated, then `readerFillSoA()` writes vectors directly into the window. Eliminates the intermediate calloc'd SoA entirely.

**Results**: Peak memory halved (32 GB → 16 GB for n1024, 64 GB → 32 GB for n2048). Wall time similar (page faults during first-touch write replace the zeroing + memcpy cost).

## Optimization 3: GPUMemoryPool + Async Stream (2026-03-29)

**Problem**: Per-batch `cudaMalloc`/`cudaFree` (26 pairs) cost ~0.5-1s per batch.

**Solution**: `GPUMemoryPool` pre-allocates all device arrays once, reused across batches. `runAnalysisKernelsAsync()` uses `cudaMemcpyAsync` with a `cudaStream_t`.

**Results**: GPU batches 1,328ms → 1,158ms (n1024), pool alloc = 2.2ms one-time.

## Optimization 4: Statistics nth_element (2026-03-29)

**Problem**: `std::sort` on 45M doubles for quartile computation took 6.7s (n1024).

**Solution**: Replaced `std::sort` with `std::nth_element` — O(n) average for each quantile. Also merged 3 data passes into 2.

**Results**: Statistics 6.7s → 1.5s (n1024, 4.3x), 13.1s → 3.0s (n2048, 5.1x).

## Optimization 5: P2PMatching.cu → .cpp (2026-03-29)

**Problem**: nvcc compilation of P2PMatching.cu caused CUDA runtime initialization even though no GPU code ran.

**Solution**: Renamed to `.cpp`, removed unnecessary `#include "common/cuda_check.h"`.

**Results**: P2P matching 2,799ms → 617ms (4.5x faster for n1024). Pure C++ avoids CUDA runtime init overhead.

## CLC Timestamp Correction v4 (2026-03-29)

**Design**: Blocking-only neutralization. For each recv with clock violation (`send_leave > recv_enter`):
- If MPI_Recv (blocking): set `end_timestamps[i] = send_leave` to neutralize false late_receiver
- If MPI_Irecv (non-blocking): leave unchanged (genuine late_receiver)

**Results**: EXACT match on 16q/32q/128q LAMMPS traces, <0.5% on 64q, +0.69% on n1024 late_receiver. Supersedes v1-v3 and v8 cascade approaches.
