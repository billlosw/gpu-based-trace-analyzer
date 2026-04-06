# Known Problems, Potential Issues, and TODOs

## Bugs / Correctness Issues

### ~~TODO 1: Late Receiver Over-Counting (Known Semantic Mismatch)~~ [SOLVED 2026-03-17]

**File**: `src/analysis/AnalysisKernels.cu`, `kernelLateSenderReceiver`

**Issue**: Late_receiver counts were 3-4x higher than Scalasca. The GPU analyzer classified ALL matched pairs where `recv_enter > send_enter` as late_receiver. Scalasca only counts pairs where the sender was still blocked when the receive was posted.

**Root Cause**: Two bugs: (1) late_sender and late_receiver were in mutually exclusive `if/else` branches, but Scalasca checks them independently in separate replay callbacks; (2) the wrong timestamps were used — `Enter(MPI_Wait)` instead of `Enter(MPI_Irecv)` for the recv side, and no check on `Leave(MPI_Send)` for the blocking condition.

**Solution**: Restructured the kernel so late_receiver is an independent check (not mutually exclusive with late_sender). Under `#ifdef USE_SCALASCA_TIMESTAMPS`, the late_receiver now uses `Enter(MPI_Irecv)` and checks `Leave(MPI_Send) > Enter(MPI_Irecv) > Enter(MPI_Send)`, matching Scalasca's algorithm exactly. Added three new OTF2 reader event handlers (`leave`, `mpi_ireceive_request`) and member variables to populate the required timestamps. All 8 analyses now match Scalasca exactly. See `chat-history/260317-1000-late-receiver-fix.md`.

---

### TODO 2: P2P Key Bit Packing Overflow Risk

**File**: `src/matching/P2PMatching.cpp`, line 37-39

```cpp
auto makeKey = [](id_t a, id_t b, id_t c) -> uint64_t {
    return ((uint64_t)a << 42) | ((uint64_t)b << 21) | (uint64_t)c;
};
```

**Issue**: This packing uses 22 bits for `a`, 21 bits for `b`, and 21 bits for `c`. This limits:
- Rank IDs to ~4 million (22 bits)
- Tags to ~2 million (21 bits)

If an MPI application uses rank IDs >= 2^22 or tags >= 2^21, keys will collide, causing incorrect matching.

**Impact**: Low for current HPC workloads (most use < 100K ranks and small tags), but could be a problem for extremely large-scale runs or applications using large tag values.

**Possible fix**: Use a proper hash function (e.g., multiply-shift) or use `std::tuple<id_t, id_t, id_t>` with a custom hash.

**Priority**: Low.

---

### TODO 3: Comm-Set to Event Index Mapping Fragility

**File**: `src/matching/CollectiveGrouping.cpp`, lines 27-39

```cpp
size_t cs_idx = 0;
for (size_t i = 0; i < n; i++) {
    if (ev >= TT_MPI_Bcast && ev <= TT_MPI_AlltoAll) {
        soa_to_commset[i] = cs_idx;
        cs_idx++;
    }
}
```

**Issue**: This assumes collective events in the SoA appear in exactly the same order as comm_sets in the `ReaderOutput::comm_sets` vector. This invariant holds in single-process mode (same callback populates both), but in **MPI-parallel mode**, events are gathered via `MPI_Gatherv` in rank order, while comm_sets are gathered separately. If rank ordering differs between the two gathers, the mapping breaks.

**Impact**: Could produce incorrect collective groups when using multi-rank reading. In practice this hasn't manifested because the MPI_Gatherv uses the same rank ordering for both, but it's a fragile coupling.

**Possible fix**: Store the comm_set index as a field in the SoA (e.g., add a `comm_set_idx` array) so the mapping is explicit and survives reordering.

**Priority**: Medium. Should be fixed before adding any event sorting or filtering.

---

### ~~TODO 4: README Contains Wrong CUDA Version~~ [SOLVED 2026-03-17]

**File**: `README.md`, line 9

```
- CUDA Toolkit (12.x recommended, tested with 12.9.0)
```

**Issue**: The README says "tested with 12.9.0" but 12.9 causes **silent kernel failure** on the FUSE cluster (driver is 12.8). The correct version is 12.8.0. The `helper.sh` correctly uses 12.8.0, but the README is misleading.

**Priority**: High (misleading for new developers).

---

## Performance Issues

### ~~TODO 5: No Chunked Processing for Large Traces~~ [ADDRESSED 2026-03-29]

**File**: `src/analysis/AnalysisKernels.cu`, `runAnalysisKernelsAsync()`

**Issue**: Previously, the entire trace was allocated on GPU in one shot. With 24GB VRAM on RTX 4090, this limited processable trace size.

**Solution**: Architecture C adaptive batch streaming is now implemented in `sharedMemoryDirectAnalysis()`. The system computes `K = min(P, VRAM_budget / per_rank_data)` ranks per GPU batch. `GPUMemoryPool` pre-allocates device memory once, and each batch processes K ranks' data with H2D → kernels → D2H. Memory usage is O(K×N/P), not O(N). This was validated on traces up to n2048 (524M events).

**Remaining concern**: Very large per-rank event counts could still exceed per-batch VRAM budget. The adaptive K computation handles this, but has not been tested at extreme scale.

**Priority**: Low (the adaptive batching mechanism is in place).

---

### TODO 6: Collective Kernels Use Only Thread 0

**File**: `src/analysis/AnalysisKernels.cu`, all collective kernels

```cuda
// Launched with <<<num_groups, 1>>>
if (threadIdx.x != 0) return;
```

**Issue**: Each collective kernel launches one block per group but only uses thread 0. For groups with many members (e.g., AllReduce on 1024 ranks), this is sequential within each group.

**Impact**: Negligible for current traces (CG uses 64-member groups, processing takes microseconds). Could matter for traces with thousands of processes.

**Possible fix**: Use warp-level or block-level parallel reduction for max/min computations within each group. Standard reduction pattern: each thread processes a subset of members, warp shuffle to combine.

**Priority**: Low.

---

### TODO 7: P2P Output Array Over-Allocation

**File**: `src/analysis/AnalysisKernels.cu`, lines 313-314

```cpp
CUDA_CHECK(cudaMalloc(&d_ls_out, n * sizeof(double)));
CUDA_CHECK(cudaMalloc(&d_lr_out, n * sizeof(double)));
```

**Issue**: Allocates `n * sizeof(double)` for each P2P output (late_sender, late_receiver), where `n` is the total number of events. In reality, only ~50% of events are P2P (the rest are collectives, init, finalize), and only matched recv events produce output. For CG with 2.5M events, this allocates 2 * 2.5M * 8 = 40MB instead of the needed ~10MB.

**Impact**: Wastes ~30MB of GPU memory. Negligible for current traces but adds up for very large traces.

**Possible fix**: Count P2P events on CPU and allocate only the needed amount. Or use a more conservative estimate like `n/2 * sizeof(double)`.

**Priority**: Low.

---

### ~~TODO 8: Statistics Sort on CPU Is Sequential~~ [ADDRESSED 2026-03-29]

**File**: `src/analysis/Statistics.cpp`

**Issue**: Previously used `std::sort` (O(N log N)) for quartile computation, taking ~6.7s for n1024.

**Solution**: Replaced with `std::nth_element` (O(N) average) for each quantile point (median, Q25, Q75). Also merged 3 data passes into 2. Results: 6.7s → 1.5s (n1024, 4.3×), 13.1s → 3.0s (n2048, 5.1×).

**Remaining opportunity**: GPU-accelerated radix sort (CUB/Thrust) could further reduce statistics time, especially for very large traces. See performance enhancement plan.

**Priority**: Low (nth_element is already 4-5× faster).

---

## Code Quality Issues

### ~~TODO 9: P2PMatching.cu Is Not Actually GPU Code~~ [FIXED 2026-03-29]

**File**: `src/matching/P2PMatching.cpp` (renamed from `.cu`)

**Issue**: The `.cu` extension caused nvcc to compile the file, triggering CUDA runtime initialization even though it contained pure CPU code. This added 4.5× overhead (2,799ms → 617ms after rename).

**Solution**: Renamed to `.cpp` and updated `CMakeLists.txt`. No code changes needed.

**Note**: The old `P2PMatching.cu` file still exists on disk as stale dead code. It should be deleted (see TODO 24).

**Priority**: Fixed.

---

### TODO 10: No Integration Test for MPI-Parallel Reading

**Issue**: The test suite only tests P2P matching, analysis kernels, and statistics. There are no tests for:
- MPI-parallel reading correctness (does gathering produce the same result as serial reading?)
- OTF2 reader event mapping (correct timestamp selection, correct field mapping)
- End-to-end integration with real traces

**Possible fix**: Add a test that reads a small trace with 1 and N MPI ranks and compares results. This requires a small OTF2 trace file committed to the repo.

**Priority**: Medium.

---

### ~~TODO 11: `const_cast` in `gatherEventsToRank0`~~ [OBSOLETE 2026-03-29]

**Issue**: This function no longer exists. The SHM-based architecture (`sharedMemoryDirectAnalysis()`) eliminated the need for `gatherEventsToRank0` — each rank writes directly into the shared memory window, and rank 0 reads from it with zero copy. No data gathering is needed.

**Priority**: N/A (code removed).

---

### TODO 12: No Graceful Handling of GPU-less Environments

**File**: `src/main.cu`, line 18-33

```cpp
static void printGpuInfo() {
    int device;
    cudaGetDevice(&device);  // No error check
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);  // No error check
```

**Issue**: If no GPU is available, `cudaGetDevice()` will fail but the error is not checked. The program will likely crash later in `cudaMalloc` with a less helpful error message.

**Possible fix**: Check `cudaGetDeviceCount()` at startup and exit with a clear error message if no GPU is found.

**Priority**: Low (the tool is always run on GPU nodes).

---

## Missing Features

### TODO 13: CG Class D Benchmark Pending

**Issue**: CG Class D trace (128 GB, ~40M events) was generated but benchmarking was not completed due to FUSE cluster downtime. This is the key benchmark that should show the GPU analyzer's advantage over TileTrace for large traces.

**Action**: Run when cluster recovers:
```bash
srun --mpi=pmix -n 8 --gres=gpu:5090:1 ./build/gpu_analyzer ~/claude/TileTraceClaude/exp/traces/cg.D/traces.otf2
srun --mpi=pmix -n 64 scout.mpi ~/claude/TileTraceClaude/exp/traces/cg.D/traces.otf2
```

**Priority**: High.

---

### TODO 14: Missing Benchmarks for Other NPB Programs

**Issue**: Only CG has been tested. The paper evaluates BT, CG, EP, FT, LU, MG, SP, LAMMPS, Sweep3D, and HPCG. At minimum, BT, FT, LU, and SP should be tested to verify correctness across different communication patterns.

**Priority**: High (for paper validation).

---

### ~~TODO 15: No CUDA Stream Overlapping~~ [PARTIALLY ADDRESSED 2026-03-29]

**Issue**: The pipeline was strictly sequential: H2D → kernels → D2H. Using CUDA streams, these could be overlapped.

**Solution**: `runAnalysisKernelsAsync()` now uses `cudaMemcpyAsync` with a `cudaStream_t`, and `GPUMemoryPool` eliminates per-batch `cudaMalloc`/`cudaFree`. The async path is used within each batch.

**Remaining opportunity**: Inter-batch overlapping (preparing batch K+1 on CPU while batch K runs on GPU) is not yet implemented. Since GPU kernel time is only ~97ms for n1024 vs ~523ms batch prep time, this could hide most of the kernel latency. However, the total GPU phase is only ~1.2s for n1024 — the bigger bottleneck is CPU preprocessing (10.6s) and OTF2 reading (31.4s).

**Priority**: Low (GPU phase is a small fraction of total time).

---

### TODO 16: Comm-Set Comparison Uses Vector Equality (Potential Performance Issue)

**File**: `src/matching/CollectiveGrouping.cpp`, `PendingGroup::comm_set_key`

```cpp
std::vector<uint64_t> comm_set_key;    // Sorted communicator members (for comparison)
```

**Issue**: The collective grouping algorithm matches events to pending groups by comparing `comm_set_key` vectors element-by-element. For large communicators (e.g., 1024+ members), this O(N) comparison runs for every event against every pending group of the same type. With many concurrent collectives, this becomes O(events * pending_groups * comm_size).

**Impact**: Negligible for current traces (CG uses 64-member communicators with few concurrent collectives). Could become a bottleneck for traces with thousands of processes and many sub-communicators.

**Possible fix**: Replace the sorted vector comparison with a hash of the communicator member set. Pre-compute a hash (e.g., using a commutative hash function like XOR of hashed members, or `std::hash` over the sorted vector) and compare hashes first, with full vector comparison only on hash collision. This reduces the comparison from O(N) to O(1) amortized.

**Priority**: Low (optimization for large-scale traces).

---

### TODO 17: No Per-Event Detail for Max/Min (Missing Scalasca Feature)

**Issue**: Scalasca reports per-cnode detail information for the maximum and minimum events of each metric:

```
mpi_latesender  719547 0.0071663 0.2515962 0.0000000005 1677.3065939138 ...
- cnode 24 enter: 41.2452562001 exit: 1718.5532726106 duration: 1677.3065939138 rank: 62
- cnode 27 enter: 1774.8533962698 exit: 1774.8590713707 duration: 0.0056734995 rank: 60
```

The GPU analyzer only reports aggregate statistics (count, sum, mean, min, max, median, q25, q75, variance). It does not track which event (cnode, rank, enter/exit timestamps) produced the max or min value for each metric. This information is useful for identifying the specific callsite and rank responsible for the worst performance bottleneck.

**Missing components**:
1. **No cnode ID in `TraceDataSoA`**: The OTF2 call-tree node concept is not captured during reading.
2. **Kernels output only durations**: The CUDA kernels write `double` durations to output arrays without the source event index, rank, or cnode.
3. **`computeStatistics()` does not track min/max index**: When finding min/max, only the value is saved, not which event produced it.
4. **Output format lacks detail lines**: `printResult()` only prints aggregate stats.

**Possible fix**: (1) Add a `cnode_id` array to `TraceDataSoA`, populated from OTF2 region/callpath definitions during reading. (2) Have kernels output `(duration, event_index)` pairs instead of just durations (e.g., using a struct or parallel index array). (3) In `computeStatistics()`, track the index of min/max values and look up the corresponding cnode, rank, enter, exit from `TraceDataSoA`. (4) Add per-cnode detail lines to the output.

**Priority**: Medium (useful for practical performance analysis, but aggregate stats are sufficient for validation).

---

### TODO 18: Reader Crashes When MPI Ranks > Trace Locations

**File**: `src/reader/OTF2SoAReader.cpp`, `traceRange()`

```cpp
size_t traceRange(size_t nlocs, int nprocs, int rank) {
    size_t chunk = nlocs / nprocs;
    size_t remain = nlocs % nprocs;
    // ...
}
```

**Issue**: When the number of MPI reader ranks exceeds the number of trace locations (e.g., 64 ranks for a 16-location trace), excess ranks get `start == end`, register no locations with the OTF2 reader, and then `rdr.read_events()` throws an OTF2 exception "Parameter value out of range".

**Reproduction**: `srun -n 64 ./gpu_analyzer ./__16_qtraces/traces.otf2` (16 locations, 64 ranks)

**Workaround**: Use `min(64, nlocs)` MPI ranks.

**Possible fix**: In `main.cu`, detect when `nprocs > nlocs` and have excess ranks skip reading entirely (no `register_location`, no `read_events`), then participate in the `MPI_Gatherv` with zero-length contributions. Or auto-cap reader ranks to `nlocs` and reassign excess ranks to idle.

**Priority**: Medium (affects usability for small traces).

---

### TODO 19: late_receiver Count Discrepancy (Systematic Algorithmic Difference)

**Issue**: The GPU analyzer consistently reports more `late_receiver` events than Scalasca across ALL tested traces. After fixing the Irecv request_id tracking bug (see TODO 23), the discrepancy increased because correct per-request timestamps enable more true detections:

| Trace | Scalasca Count | GPU Count | Difference |
|-------|---------------|-----------|-----------|
| LAMMPS 16 | 409,789 | 536,530 | +30.9% |
| LAMMPS 32 | 1,060,184 | 1,197,342 | +12.9% |
| LAMMPS 128 | 2,559,435 | 3,561,329 | +39.2% |
| LAMMPS 512 | 10,877,100 | 13,172,054 | +21.1% |
| NPB CG 1024 | 7,731,027 | 11,115,347 | +43.8% |
| NPB CG 2048 | 11,961,014 | 16,633,770 | +39.1% |

**Root Cause**: Two contributing factors:
1. **LAMMPS**: Scalasca performs timestamp correction (`--time-correct`) that removes reversed clock orderings. This reclassifies some events.
2. **NPB CG**: Even without timestamp correction, a ~40% discrepancy persists. This points to a **systematic algorithmic difference** in how Scalasca SCOUT computes `late_receiver` vs our implementation. Extensive investigation ruled out GPU kernel bugs (CPU validation matches), send timestamp corruption (all sends have correct Leave > Enter), and matching errors (late_sender matches exactly for NPB CG). The exact cause is unknown without Scalasca's source code.

**Verified correct**:
- `late_sender`: Exact match for NPB CG (no TS correction), <1% diff for LAMMPS (TS correction effect)
- All 6 collective metrics: Exact match for all traces
- P2P matching: Correct (proven by late_sender exact match)

**Possible causes**:
- Scalasca may use a different timestamp source for `enter_sendcmp` (point event timestamp vs Enter region)
- Different handling of edge cases or minimum duration thresholds
- The condition `send_leave > recv_req_enter > send_enter` may be evaluated differently in SCOUT's replay

**Priority**: Medium (all other metrics are correct; late_receiver discrepancy is consistent and documented).

---

### TODO 20: Missing early_scan Pattern Detection

**Issue**: The GPU analyzer does not implement `early_scan` pattern detection. Scalasca reports this metric for all traces:

| Trace | early_scan Count | early_scan Sum |
|-------|-----------------|---------------|
| LAMMPS 16 | 13 | 0.0010 |
| LAMMPS 32 | 27 | 0.0025 |
| LAMMPS 128 | 105 | 0.0008 |
| LAMMPS 512 | 484 | 0.0072 |
| NPB CG 1024 | 1,011 | 0.1916 |
| NPB CG 2048 | 2,033 | 0.4217 |
| NPB CG 8192 | 8,180 | 14.2724 |

The early_scan metric becomes more significant at higher rank counts (14.3s aggregate wait time at 8192 ranks).

**Possible fix**: Add an `early_scan` detection kernel similar to `early_reduce`. Requires identifying MPI_Scan collective operations in the trace and checking if processes arrive early relative to the root.

**Priority**: Low-Medium (small impact on small traces, but growing significance at scale).

---

### TODO 21: OTF2 Reading is Performance Bottleneck (67-85% of Total Time)

**Issue**: OTF2 reading dominates total execution time across all traces:

| Trace | OTF2 Read Time | % of Total | Trace Size |
|-------|---------------|-----------|-----------|
| LAMMPS 1024 | 31.4s | 66.5% | ~9.9 GB |
| LAMMPS 2048 | 213.7s | 85.3% | ~20 GB |

The read throughput is approximately 55-65 MB/s, far below the NVMe SSD capability. This is due to the OTF2 library's per-location sequential reading and the 2-pass reader design (discovery + data).

**Root Cause**: The OTF2 library is designed for single-location sequential access. The 64 MPI reader ranks help, but each rank still reads sequentially through its assigned locations. In contrast, Scalasca reads data in 1-2s because each of its N ranks reads only one location in parallel (N-way parallelism).

**Possible enhancements**:
1. Multi-threaded reading within each MPI rank (thread per location)
2. POSIX direct I/O or memory-mapped reading bypassing OTF2 library
3. GPUDirect Storage (cuFile API) for direct GPU memory loading
4. Pre-conversion to GPU-friendly binary format (one-time preprocessing cost)
5. Multi-node distributed reading with single-GPU analysis (hybrid approach)

**Priority**: High (single biggest bottleneck, limits competitiveness with Scalasca on large traces).

---

### ~~TODO 22: Crash on Large Traces (n4096+) During Collective Redistribution~~ [FIXED 2026-04-04]

**Issue**: The GPU analyzer crashes with `std::length_error: vector::_M_default_append` when processing the NPB CG n4096 trace (41GB, 4096 ranks) with 64 MPI reader ranks on fuse2 (251GB RAM).

**Error output**:
```
[Reader] Pass 2 done in 233689 ms
terminate called after throwing an instance of 'std::length_error'
  what():  vector::_M_default_append
```

**Stack trace**: The crash occurs in the `redistributeCollectives` function after pass 2 completes.

**Root Cause Analysis**: The `redistributeCollectives` function uses `int` for event counts and displacements. With 4096 ranks, each collective on `MPI_COMM_WORLD` has a communicator set of 4096 members. The `flat_comm_sets` buffer stores `(1 + comm_size)` entries per collective event. If many collective events are redistributed, the total flat buffer size or the `total_cs` counter could overflow `int` (>2.1 billion), causing `resize()` to receive a negative value cast to a huge `size_t`, triggering `length_error`.

Alternatively, with 4096-member communicators, the `flat_comm_sets` buffer per target could exceed available memory:
- ~130 collective operations × 4096 rank-events × (1+4096) entries × 8 bytes ≈ large
- This compounds across the 64-iteration loop over `target` in `redistributeCollectives`

**Reproduction**: 
```bash
srun -N 1 -n 64 -w fuse2 -p Long --gres=gpu:5090:1 ./build/gpu_analyzer /home/luosw22/trace_data/scorep_2024-09-20_20-30-03_n4096/traces.otf2
```

**Impact**: Cannot analyze traces with 4096+ ranks. NPB CG n8192 (107GB, 8192 ranks) would also crash.

**Possible fix**:
1. Change `int` types in `redistributeCollectives` to `int64_t`/`size_t` for counts and displacements
2. Use `MPI_Gatherv(..., MPI_INT64_T...)` instead of `MPI_INT` for large counts
3. For flat_comm_sets: instead of flattening the full communicator member list for each event, use a communicator ID and maintain a separate communicator-to-members mapping (deduplication)
4. Process redistributions in chunks to bound peak memory per target

**Priority**: Fixed. Changed all counts/displacements in `redistributeCollectives` from `int` to `int64_t`. Added `safeGatherv<T>()` helper that uses MPI_Gatherv when data fits in int, falls back to chunked MPI_Send/MPI_Recv otherwise. Also fixed `int` overflow in main.cu event count accumulators.

---

### ~~TODO 23: Irecv Enter Timestamp Tracked Per-PID Instead of Per-Request~~ [FIXED 2026-03-26]

**File**: `src/reader/OTF2SoAReader.cpp`

**Issue**: `m_irecv_enter_ts` was keyed by PID (location ID) rather than `request_id`. When multiple non-blocking `MPI_Irecv` operations are outstanding on the same location, only the LAST one's `Enter(MPI_Irecv)` timestamp was stored. Earlier Irecvs received incorrect (overwritten) timestamps.

**Root Cause**: NPB CG traces have 96% non-blocking receives with many concurrent Irecvs per location. The per-PID map only stored one entry per location, so when `MPI_Irecv → MPI_Irecv → MPI_Wait → MPI_Wait` occurred, the first Irecv's enter timestamp was lost.

**Fix**: Changed `m_irecv_enter_ts` from `unordered_map<id_t, timestamp_t>` (per-PID) to `unordered_map<uint64_t, timestamp_t>` (per-request_id). The `mpi_ireceive_request` handler stores by `event.request_id()`, and `mpi_ireceive_complete` looks up by `event.request_id()` with erase-after-use.

**Impact on counts**: The fix INCREASED late_receiver counts (e.g., NPB CG 1024: 8,985,237 → 11,115,347) because correct earlier timestamps change which events satisfy the `send_leave > recv_req_enter > send_enter` condition. This is correct behavior. See TODO 19 for the remaining Scalasca discrepancy.

**Priority**: Fixed.

---

### TODO 24: Stale P2PMatching.cu File on Disk

**File**: `src/matching/P2PMatching.cu`

**Issue**: After the `.cu` → `.cpp` rename (TODO 9), the old `P2PMatching.cu` file was not deleted. It sits alongside the active `P2PMatching.cpp` and is not referenced by `CMakeLists.txt`. This is confusing for developers who may edit the wrong file.

**Possible fix**: Delete `src/matching/P2PMatching.cu`.

**Priority**: Low (cosmetic, no build impact).

---

### TODO 25: SHM fill_soa Page Fault Cost

**File**: `src/main.cu`, `sharedMemoryDirectAnalysis()` → `readerFillSoA()`

**Issue**: The `fill_soa` phase (writing OTF2 data into the shared memory window) takes 5.1s for n1024 (48% of the 10.6s preprocess phase). This cost comes from first-touch page faults on the SHM window — `MPI_Win_allocate_shared` allocates virtual memory but physical pages are only mapped on first write.

**Possible mitigations**:
1. **Explicit mmap prefaulting**: Use `MAP_POPULATE` or `madvise(MADV_WILLNEED)` on the SHM region before writing
2. **NUMA-aware allocation**: Pin SHM pages to the local NUMA node
3. **Parallel fill**: Use OpenMP threads within each rank to parallelize the first-touch writes

**Priority**: Medium (5.1s is significant in the 15.8s analysis phase).

---

### TODO 26: Collective Grouping Scalability

**File**: `src/matching/CollectiveGrouping.cpp`

**Issue**: Collective grouping takes 2.9s for n1024 (27% of preprocess) and likely scales super-linearly for larger traces. Two algorithmic bottlenecks:
1. **Linear scan over pending groups**: For each event, the algorithm linearly scans all pending groups of the same type and performs O(comm_size) vector equality comparison. With many concurrent collectives of different communicators, this becomes O(events × pending_groups × comm_size).
2. **Redundant comm_set normalization**: `normalizeCommSet()` copies and sorts the communicator member vector for every collective event, even when the same communicator appears thousands of times.

**Possible fixes**:
1. Hash the sorted comm_set for O(1) amortized lookup instead of linear scan with vector comparison
2. Cache normalized comm_sets by communicator ID to avoid redundant sorts
3. Parallelize grouping across event types (they are independent)

**Priority**: Medium (becomes bottleneck for n2048+ traces).

---

### TODO 27: SHM Cleanup Cost

**Issue**: The cleanup phase (`cudaHostUnregister` + `MPI_Win_free`) takes 2.5s for n1024 and 3.8s for n2048. `cudaHostUnregister` is expensive because it must synchronize the CUDA context and unpin physical pages.

**Possible mitigations**:
1. **Defer cleanup**: Let `MPI_Finalize` handle window freeing (acceptable for single-use tools)
2. **Reduce pinned region**: Only pin the subset of SHM needed for the current GPU batch, not the entire window
3. **Adaptive pinning**: Already partially implemented — for >10GB data, per-batch pinning is used instead of full-window pinning

**Priority**: Low (cleanup is not on the critical path for analysis correctness).

---

### TODO 28: CLC Timestamp Correction Limitations (Distributed Reading)

**Issue**: The CLC v4 blocking-only neutralization achieves exact match on small LAMMPS traces (16q/32q/128q) but has <1% gap on larger traces (64q: <0.5%, n1024: +0.69% late_receiver). This comes from the distributed reading architecture: each rank only sees its local subset of events, while Scalasca's CLC operates on the full per-location timeline.

Specifically, clock violations where the sender's Leave timestamp from a remote rank's timeline is needed for local correction cannot be resolved without cross-rank communication. The current design applies corrections locally per-rank, which is a practical approximation.

**Possible improvements**:
1. After P2P matching, exchange violation information across ranks via MPI
2. Implement multi-pass correction (correct violations that create new violations)
3. Accept the <1% gap as within acceptable tolerance

**Priority**: Low (the gap is small and documented; exact CLC would require Scalasca's full multi-pass amortization algorithm).