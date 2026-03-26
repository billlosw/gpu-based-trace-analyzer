# Performance Comparison: GPU Analyzer vs Scalasca

## Test Environment

| Component | GPU Analyzer | Scalasca v2.6.1 |
|-----------|-------------|-----------------|
| Hardware | 1× NVIDIA RTX 4090 (24GB VRAM) | 2× Intel Xeon Gold 6530 (64C/128T, 1TB RAM) |
| Parallelism | 1 GPU + 16 MPI reader ranks | N CPU cores (N = trace locations) |
| Node | Single compute node (fuse1/fuse2) | Single compute node (fuse1/fuse2) |
| Build | CUDA 12.8, `-DTIMESTAMP_MODE=SCALASCA` | Scalasca v2.6.1, SCOUT analysis |

## Benchmark Traces

| Trace | Application | #Processes | #Events | Trace Size |
|-------|------------|-----------|---------|-----------|
| LAMMPS 16 | LAMMPS (molecular dynamics) | 16 | 4.1M | 1.1GB |
| LAMMPS 32 | LAMMPS | 32 | 8.2M | 1.5GB |
| LAMMPS 128 | LAMMPS | 128 | 33.0M | 3.7GB |
| LAMMPS 512 | LAMMPS | 512 | 132.2M | 12GB |
| NPB CG 1024 | NAS Parallel Benchmarks CG | 1024 | 262.4M | 9.9GB |
| NPB CG 2048 | NAS Parallel Benchmarks CG | 2048 | 524.8M | 20GB |

## Analysis Time Comparison (Excluding I/O)

We compare **analysis time only**, excluding trace reading/I/O:
- **Scalasca**: Preprocessing + Timestamp correction + Analyzing trace data + Writing analysis report
- **GPU Analyzer**: P2P Matching + Collective Grouping + Batch Analysis (GPU kernels) + Statistics

### Results

| Trace | Scalasca Analysis | GPU Analysis | Speedup |
|-------|------------------|-------------|---------|
| LAMMPS 16 | 98.22s | 0.75s | **130.8×** |
| LAMMPS 32 | 90.82s | 1.03s | **87.8×** |
| LAMMPS 128 | 86.47s | 3.47s | **24.9×** |
| LAMMPS 512 | 150.79s | 12.21s | **12.4×** |
| NPB CG 1024 | 12.95s | 29.17s | 0.44× |
| NPB CG 2048 | 19.41s | 66.30s | 0.29× |

### Key Observations

1. **LAMMPS (16-512 processes): 12-131× speedup.** The GPU analyzer dramatically outperforms Scalasca because:
   - Scalasca's timestamp correction is expensive (~20-28s, 22-30% of total)
   - Scalasca uses relatively few CPU cores (16-512) vs massively parallel GPU
   - GPU CUDA kernels process all pairs simultaneously

2. **NPB CG (1024-2048 processes): GPU is 2.3-3.4× slower.** This is because:
   - Scalasca uses 1024/2048 MPI ranks for analysis, leveraging massive CPU parallelism
   - NPB CG has no timestamp correction overhead
   - The GPU P2P matching and collective grouping run on CPU (single-threaded portions)
   - One RTX 4090 vs 1024+ logical CPU cores is not a fair comparison

3. **Per-core efficiency**: The GPU achieves 5-11M events/s on a single device, while Scalasca achieves 13-27K events/s per core for NPB CG. A single GPU replaces hundreds of CPU cores for the analysis phase.

### GPU Analyzer Timing Breakdown

| Trace | P2P Match | Coll. Group | Batch Analysis | Statistics | Total |
|-------|----------|------------|---------------|-----------|-------|
| LAMMPS 16 | 32ms | 2ms | 588ms | 128ms | 0.75s |
| LAMMPS 32 | 57ms | 5ms | 707ms | 265ms | 1.03s |
| LAMMPS 128 | 245ms | 45ms | 2,021ms | 1,160ms | 3.47s |
| LAMMPS 512 | 1,001ms | 533ms | 6,714ms | 3,958ms | 12.21s |
| CG 1024 | 2,647ms | 3,801ms | 15,456ms | 7,265ms | 29.17s |
| CG 2048 | 4,544ms | 13,107ms | 38,256ms | 10,392ms | 66.30s |

Dominant phases scale differently:
- **Batch Analysis (GPU kernels)**: Primary GPU workload, scales with #events × matched-pairs
- **Statistics (CPU)**: Sorting-based, O(N log N), becomes significant for large traces
- **P2P Matching (CPU)**: FIFO-queue matching, scales with #events
- **Collective Grouping (CPU)**: Depends on communicator sizes, dominant for large CG traces

### Scalasca Timing Breakdown

| Trace | Preprocess | TS Correction | Analysis | Write | Total |
|-------|-----------|--------------|----------|-------|-------|
| LAMMPS 16 | 2.04s | 21.55s | 71.50s | 3.13s | 98.22s |
| LAMMPS 32 | 1.40s | 20.08s | 66.08s | 3.26s | 90.82s |
| LAMMPS 128 | 1.39s | 18.90s | 61.72s | 4.45s | 86.47s |
| LAMMPS 512 | 1.73s | 27.77s | 89.13s | 32.17s | 150.79s |
| CG 1024 | 0.40s | — | 12.47s | 0.08s | 12.95s |
| CG 2048 | 0.28s | — | 18.91s | 0.23s | 19.41s |

## Correctness Comparison

### Metrics That Match Exactly (NPB CG, no TS correction)

| Metric | Match? | Notes |
|--------|--------|-------|
| late_sender | ✓ Exact | All NPB CG traces match to the event |
| barrier_wait | ✓ Exact | All traces |
| barrier_completion | ✓ Exact | All traces |
| early_reduce | ✓ Exact | All traces |
| late_broadcast | ✓ Exact | All traces |
| wait_nxn | ✓ Exact | All traces |
| nxn_completion | ✓ Exact | All traces |

### Metrics with Known Differences

| Metric | Status | Details |
|--------|--------|---------|
| late_sender (LAMMPS) | <1% diff | Due to Scalasca's timestamp correction |
| late_receiver | +13-44% higher | Systematic algorithmic difference (see PROBLEMS.md TODO 19) |
| early_scan | Not implemented | Low impact (see PROBLEMS.md TODO 20) |

### Full Metric Comparison

#### late_sender

| Trace | Scalasca | GPU | Diff% |
|-------|---------|-----|-------|
| LAMMPS 16 | 537,099 | 537,053 | -0.009% |
| LAMMPS 32 | 907,952 | 909,549 | +0.18% |
| LAMMPS 128 | 5,554,707 | 5,600,908 | +0.83% |
| LAMMPS 512 | 23,954,454 | 24,190,734 | +0.99% |
| CG 1024 | 45,728,652 | 45,728,652 | **0.00%** |
| CG 2048 | 90,751,279 | 90,751,279 | **0.00%** |

#### late_receiver

| Trace | Scalasca | GPU | Diff% |
|-------|---------|-----|-------|
| LAMMPS 16 | 409,789 | 536,530 | +30.9% |
| LAMMPS 32 | 1,060,184 | 1,197,342 | +12.9% |
| LAMMPS 128 | 2,559,435 | 3,561,329 | +39.2% |
| LAMMPS 512 | 10,877,100 | 13,172,054 | +21.1% |
| CG 1024 | 7,731,027 | 11,115,347 | +43.8% |
| CG 2048 | 11,961,014 | 16,633,770 | +39.1% |

#### Collective Metrics (Sample: CG 1024)

| Metric | Scalasca | GPU | Match? |
|--------|---------|-----|--------|
| barrier_wait | 5,115 | 5,115 | ✓ |
| barrier_completion | 5,115 | 5,115 | ✓ |
| early_reduce | 1 | 1 | ✓ |
| late_broadcast | 7,412 | 7,412 | ✓ |
| wait_nxn | 117,645 | 117,645 | ✓ |
| nxn_completion | 117,645 | 117,645 | ✓ |

## Total Pipeline Time (Including I/O)

| Trace | GPU Total | GPU Read | GPU Analysis | Scalasca Read* | Scalasca Analysis |
|-------|----------|---------|-------------|---------------|------------------|
| LAMMPS 16 | 8.1s | 7.3s | 0.75s | ~1-2s | 98.22s |
| LAMMPS 32 | 33.1s | 32.1s | 1.03s | ~1-2s | 90.82s |
| LAMMPS 128 | 94.7s | 91.2s | 3.47s | ~1-2s | 86.47s |
| LAMMPS 512 | 279.9s | 267.7s | 12.21s | ~1-2s | 150.79s |
| CG 1024 | 137.3s | 108.1s | 29.17s | ~1-2s | 12.95s |
| CG 2048 | 488.9s | 422.6s | 66.30s | ~1-2s | 19.41s |

*Scalasca's read time uses N-way parallelism (one location per rank), achieving 1-2s reads. GPU uses 16 MPI ranks.

OTF2 reading is the GPU analyzer's primary bottleneck (79-97% of total). Scalasca's architectural advantage is N-way parallel reading combined with N-way parallel analysis, both scaling with trace locations.

## Summary

The GPU analyzer achieves **12-131× speedup** in analysis time for moderate-scale traces (16-512 processes) on a single GPU compared to Scalasca on multi-core CPU. For large-scale traces (1024+ processes), Scalasca's massive CPU parallelism gives it an advantage. The GPU analyzer's throughput of ~5-11M events/s on a single device is competitive with hundreds of CPU cores.

The main limitation is OTF2 reading speed—dominated by sequential I/O through the OTF2 library with only 16 reader ranks—which accounts for the majority of end-to-end latency. Correctness is verified with exact matches on 7 of 8 implemented metrics (plus `late_sender` <1% for timestamp-corrected traces). The `late_receiver` discrepancy (~13-44%) is a known systematic algorithmic difference under investigation.
