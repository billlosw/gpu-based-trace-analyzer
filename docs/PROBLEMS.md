# Known Problems, Potential Issues, and TODOs

## Bugs / Correctness Issues

### ~~TODO 1: Late Receiver Over-Counting (Known Semantic Mismatch)~~ [SOLVED 2026-03-17]

**File**: `src/analysis/AnalysisKernels.cu`, `kernelLateSenderReceiver`

**Issue**: Late_receiver counts were 3-4x higher than Scalasca. The GPU analyzer classified ALL matched pairs where `recv_enter > send_enter` as late_receiver. Scalasca only counts pairs where the sender was still blocked when the receive was posted.

**Root Cause**: Two bugs: (1) late_sender and late_receiver were in mutually exclusive `if/else` branches, but Scalasca checks them independently in separate replay callbacks; (2) the wrong timestamps were used — `Enter(MPI_Wait)` instead of `Enter(MPI_Irecv)` for the recv side, and no check on `Leave(MPI_Send)` for the blocking condition.

**Solution**: Restructured the kernel so late_receiver is an independent check (not mutually exclusive with late_sender). Under `#ifdef USE_SCALASCA_TIMESTAMPS`, the late_receiver now uses `Enter(MPI_Irecv)` and checks `Leave(MPI_Send) > Enter(MPI_Irecv) > Enter(MPI_Send)`, matching Scalasca's algorithm exactly. Added three new OTF2 reader event handlers (`leave`, `mpi_ireceive_request`) and member variables to populate the required timestamps. All 8 analyses now match Scalasca exactly. See `chat-history/260317-1000-late-receiver-fix.md`.

---

### TODO 2: P2P Key Bit Packing Overflow Risk

**File**: `src/matching/P2PMatching.cu`, line 37-39

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

### TODO 5: No Chunked Processing for Large Traces

**File**: `src/analysis/AnalysisKernels.cu`, `runAnalysisKernels()`

**Issue**: The entire trace is allocated on GPU in one shot. With 24GB VRAM on RTX 4090 and ~60 bytes per event (full SoA) or ~32 bytes (GPU-transferred subset), the maximum is ~400M-750M events. CG Class D has ~40M events, which fits easily, but very large traces (e.g., LAMMPS with millions of timesteps) could exceed GPU memory.

**Current behavior**: `cudaMalloc` will fail and `CUDA_CHECK` will `exit(EXIT_FAILURE)`.

**Possible fix**: Implement chunked processing: split events into GPU-sized chunks, process each chunk, and merge results on CPU. P2P matching (which needs global access to paired events) is the main challenge for chunking.

**Priority**: Low (no traces have exceeded GPU memory yet).

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

### TODO 8: Statistics Sort on CPU Is Sequential

**File**: `src/analysis/Statistics.cpp`, line 37-43

```cpp
std::vector<double> sorted(durations.size());
// copy and divide by scaling factor
std::sort(sorted.begin(), sorted.end());
```

**Issue**: Sorting happens on CPU, which is O(N log N). For very large traces with millions of results, this could take significant time.

**Impact**: For CG traces (~300K-1M results per analysis), sorting takes ~50-100ms per analysis. Total statistics phase is ~120ms.

**Possible fix**: Use CUB or Thrust radix sort on GPU before copying back. Or only compute approximate quartiles using GPU-based selection algorithms.

**Priority**: Low.

---

## Code Quality Issues

### TODO 9: P2PMatching.cu Is Not Actually GPU Code

**File**: `src/matching/P2PMatching.cu`

**Issue**: Despite the `.cu` extension, this file contains pure CPU code (STL containers, no CUDA kernels). The `.cu` extension was kept for CUDA linker compatibility, but it's confusing for developers.

**Possible fix**: Rename to `.cpp` and adjust CMakeLists.txt. The CUDA separable compilation should still work if the `.cpp` file doesn't contain device code. Alternatively, add a comment at the top of the file explaining why it's `.cu`.

**Priority**: Low (cosmetic).

---

### TODO 10: No Integration Test for MPI-Parallel Reading

**Issue**: The test suite only tests P2P matching, analysis kernels, and statistics. There are no tests for:
- MPI-parallel reading correctness (does gathering produce the same result as serial reading?)
- OTF2 reader event mapping (correct timestamp selection, correct field mapping)
- End-to-end integration with real traces

**Possible fix**: Add a test that reads a small trace with 1 and N MPI ranks and compares results. This requires a small OTF2 trace file committed to the repo.

**Priority**: Medium.

---

### TODO 11: `const_cast` in `gatherEventsToRank0`

**File**: `src/reader/OTF2SoAReader.cpp`, line 377

```cpp
auto &local_cs = const_cast<SoAReaderCallback &>(cb).getCommSets();
```

**Issue**: The function takes `const SoAReaderCallback &cb` but needs mutable access to `getCommSets()`. This is a code smell — the function should either take a non-const reference or `getCommSets()` should have a const overload.

**Priority**: Low (functionally correct but poor style).

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

### TODO 15: No CUDA Stream Overlapping

**Issue**: The pipeline is strictly sequential: H2D transfer → P2P kernel → D2H → H2D (CSR) → collective kernels → D2H. Using CUDA streams, the H2D transfer and kernel execution could be overlapped, potentially hiding transfer latency.

**Impact**: For CG traces, the H2D transfer is ~20ms and kernels are ~100ms. Overlapping could save ~20ms (16% of GPU phase). Since GPU phase is only ~0.5% of total time, the overall impact is minimal.

**Priority**: Low.

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
