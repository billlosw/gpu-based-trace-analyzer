# Testing and Validation

## Test Suite

### Test Files

| Test | File | What It Tests |
|------|------|---------------|
| P2P Matching | `test/test_p2p_matching.cu` | Send-recv matching correctness |
| Analysis Kernels | `test/test_analysis_kernels.cu` | All 8 analysis kernels with synthetic data |
| Statistics | `test/test_statistics.cpp` | Summary statistics computation |

All tests use `assert()` for validation. They exit with code 0 on success, nonzero on failure.

### Test: P2P Matching

4 test cases:

1. **`test_simple_matching`**: 4 events (2 send-recv pairs). Verifies bidirectional matching: `match_partner[send] == recv` and `match_partner[recv] == send`.

2. **`test_unmatched`**: 3 events (1 matched pair + 1 orphan send). Verifies orphan keeps `match_partner == -1`.

3. **`test_nonblocking_matching`**: `MPI_Isend` + `MPI_Irecv` pair. Verifies non-blocking variants match correctly.

4. **`test_empty`**: Zero events. Verifies no crash on empty input.

### Test: Analysis Kernels

8 test cases, one per analysis:

1. **`test_late_sender`**: Send ts=5000, Recv ts=2000. Expects late_sender=3000.
2. **`test_late_receiver`**: Send ts=1000, Recv ts=4000. Expects late_receiver=3000.
3. **`test_barrier_wait`**: 3 members with ts={1000, 2000, 5000}. Expects 2 wait values summing to 7000.
4. **`test_barrier_completion`**: 3 members with end_ts={6000, 8000, 10000}. Expects 2 completion values summing to 6000.
5. **`test_early_reduce`**: Root ts=1000, members ts={3000, 5000}. Expects early_reduce=4000.
6. **`test_late_broadcast`**: Root ts=5000, members ts={1000, 3000}. Expects 2 values summing to 6000.
7. **`test_nxn_wait`**: Same as barrier_wait but with AllReduce type.
8. **`test_nxn_completion`**: Same as barrier_completion but with AlltoAll type.

Each test creates synthetic `TraceDataSoA` and (for collective tests) a `CollectiveGroupCSR` using the `makeCSR()` helper, then calls `runAnalysisKernels()` and validates the output.

### Test: Statistics

3 test cases:

1. **`test_basic_statistics`**: 4 values {1e12, 2e12, 3e12, 4e12} ps → {1, 2, 3, 4} seconds. Validates count, sum, mean, min, max, median, q25, q75, variance.

2. **`test_empty`**: Empty vector. Validates count=0, sum=0.

3. **`test_single_value`**: 1 value {5e12}. Validates statistics for single-element input.

### Running Tests

```bash
# On GPU node
srun --gres=gpu:4090:1 ./build/test/test_p2p_matching
srun --gres=gpu:4090:1 ./build/test/test_analysis_kernels
srun --gres=gpu:4090:1 ./build/test/test_statistics
```

Expected output:
```
=== P2P Matching Tests ===
[test_simple_matching] PASSED
[test_unmatched] PASSED
[test_nonblocking_matching] PASSED
[test_empty] PASSED
All P2P matching tests passed!
```

## Validation Against Scalasca

The primary validation method is comparing output against Scalasca (the industry-standard MPI analysis tool):

### Validated Traces

| Trace | Locations | Events | Validated? |
|-------|-----------|--------|-----------|
| CG Class B | 64 | ~2.5M | Yes - exact match on 8/8 |
| CG Class C | 64 | ~2.5M | Yes - exact match on 8/8 |
| CG Class D | 64 | ~40M | Pending (nodes were down) |

### Per-Analysis Match Status

| Analysis | Match | Notes |
|----------|-------|-------|
| late_sender | **EXACT** | Count exact, sum within 1 ps |
| late_receiver | **EXACT** | Count and sum exact (fixed 2026-03-17) |
| barrier_wait | **EXACT** | Count and sum exact |
| barrier_completion | **EXACT** | Count and sum exact |
| early_reduce | **EXACT** | Both report 0 for CG traces |
| late_broadcast | **EXACT** | Count and sum exact |
| wait_nxn | **EXACT** | Both report 0 for CG traces |
| nxn_completion | **EXACT** | Both report 0 for CG traces |

### CG Class C Detailed Results (post late_receiver fix, 2026-03-17)

| Metric | Scalasca | GPU Analyzer | Match |
|--------|----------|--------------|-------|
| late_sender Count | 429,617 | 429,617 | EXACT |
| late_sender Sum | 35.4118 | 35.4118 | EXACT |
| late_sender Max | 0.2437714022 | 0.2437714022 | EXACT |
| late_receiver Count | 254,271 | 254,271 | EXACT |
| late_receiver Sum | 23.7791 | 23.7791 | EXACT |
| late_receiver Max | 0.2525817683 | 0.2525817683 | EXACT |
| barrier_wait Count | 63 | 63 | EXACT |
| barrier_wait Sum | 0.0008904243 | 0.0008904243 | EXACT |
| barrier_completion Count | 63 | 63 | EXACT |
| barrier_completion Sum | 0.0005550188 | 0.0005550188 | EXACT |
| latebroadcast Count | 63 | 63 | EXACT |
| latebroadcast Sum | 0.0272760182 | 0.0272760182 | EXACT |

### CG Class B Detailed Results (post late_receiver fix, 2026-03-17)

| Metric | Scalasca | GPU Analyzer | Match |
|--------|----------|--------------|-------|
| late_sender Count | 332,726 | 332,726 | EXACT |
| late_sender Sum | 9.6265 | 9.6265 | EXACT |
| late_sender Max | 0.0323421670 | 0.0323421670 | EXACT |
| late_receiver Count | 211,175 | 211,175 | EXACT |
| late_receiver Sum | 3.7667 | 3.7667 | EXACT |
| late_receiver Max | 0.0230877752 | 0.0230877752 | EXACT |
| All collectives | Exact | Exact | EXACT |

### Validation Also Against TileTrace

After fixing TileTrace's tiled replay bug (see [10-HISTORY-AND-BUGS.md](./10-HISTORY-AND-BUGS.md)), TileTrace and the GPU analyzer produce identical results for all 8 analyses. The GPU analyzer's results can be independently verified against both Scalasca and TileTrace.

## How to Run Comparison

```bash
# Run GPU analyzer
srun --mpi=pmix -n 32 --gres=gpu:4090:1 \
    ./build/gpu_analyzer /path/to/traces.otf2 2>&1 | tee gpu_results.log

# Run Scalasca
srun --mpi=pmix -n 64 scout.mpi /path/to/traces.otf2
# Use square to view results

# Run TileTrace
srun --mpi=pmix -n 64 ./analysis_integration_test /path/to/traces.otf2 \
    2>&1 | tee tiletrace_results.log

# Quick diff (compare key metrics)
grep -E "^(Count|Mean|Sum|Minimum|Maximum):" gpu_results.log > gpu.txt
grep -E "^(Count|Mean|Sum|Minimum|Maximum):" tiletrace_results.log > ref.txt
diff gpu.txt ref.txt
```

## What's NOT Tested

- **Large trace handling**: No test for traces exceeding GPU memory
- **MPI-parallel reading**: Tests don't exercise multi-rank reading or MPI_Gatherv
- **OTF2 reader**: No unit test for the reader itself (tested implicitly via integration)
- **Edge cases in collective grouping**: No test for sub-communicators, overlapping collectives, or incomplete groups
- **Performance regression**: No automated benchmarks
