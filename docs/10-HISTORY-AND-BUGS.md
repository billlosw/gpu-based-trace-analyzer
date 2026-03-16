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
| 2026-03-15 (PM) | Achieved exact match with Scalasca on 7/8 analyses |

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
