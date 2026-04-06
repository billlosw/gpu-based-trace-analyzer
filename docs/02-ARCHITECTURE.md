# Architecture

## Directory Structure

```
gpu-analyzer/
├── CMakeLists.txt                      # Top-level build config
├── helper.sh                           # Build/env/clean helper script
├── include/
│   ├── common/
│   │   ├── types.h                     # Shared enums, typedefs
│   │   ├── cuda_check.h               # CUDA_CHECK error macro
│   │   └── SHA256.h                   # sha256_file / sha256_buf (no external dep)
│   ├── data/
│   │   ├── TraceDataSoA.h             # SoA data structure + CSR struct
│   │   └── AnalysisResults.h          # Result struct for statistics
│   ├── reader/
│   │   ├── OTF2SoAReader.h            # Reader interface (split-phase API)
│   │   ├── SoACache.h                 # Per-rank binary SoA cache format + API
│   │   └── ColumnMajorCache.h         # Column-major single-file cache + mmap reader
│   ├── matching/
│   │   ├── P2PMatching.h              # P2P matching interface
│   │   └── CollectiveGrouping.h       # Collective grouping interface
│   └── analysis/
│       ├── AnalysisKernels.h          # CUDA kernel + GPUMemoryPool interface
│       ├── Statistics.h               # Statistics computation interface
│       └── TimestampCorrection.h      # CLC timestamp correction interface
├── src/
│   ├── main.cu                        # Entry point, pipeline orchestration
│   ├── common/
│   │   └── SHA256.cpp                 # Minimal SHA-256 (no external dependency)
│   ├── reader/
│   │   ├── OTF2SoAReader.cpp          # OTF2 -> SoA reader implementation
│   │   ├── SoACache.cpp               # Per-rank binary SoA cache (read/write)
│   │   └── ColumnMajorCache.cpp       # Column-major cache writer (MPI) + mmap reader
│   ├── matching/
│   │   ├── P2PMatching.cpp            # CPU-based P2P matching (pure C++)
│   │   └── CollectiveGrouping.cpp     # CPU collective grouping
│   └── analysis/
│       ├── AnalysisKernels.cu         # 5 CUDA kernels (8 analyses) + GPUMemoryPool
│       ├── Statistics.cpp             # Summary statistics on CPU (nth_element)
│       └── TimestampCorrection.cpp    # CLC blocking-only timestamp neutralization
├── test/
│   ├── CMakeLists.txt
│   ├── test_p2p_matching.cu
│   ├── test_analysis_kernels.cu
│   └── test_statistics.cpp
└── third_party/
    └── otf2xx/                        # Symlink -> TileTrace's otf2xx
```

## Module Dependency Graph

```
                main.cu
               /   |    \     \
              /    |     \     \
             v     v      v     v
    OTF2SoAReader  P2PMatching  CollectiveGrouping  TimestampCorrection
         |              |              |                    |
      SoACache          |              |                    |
         |              |              |                    |
         v              v              v                    v
     TraceDataSoA  TraceDataSoA  CollectiveGroupCSR   TraceDataSoA
         |              |              |                    |
         +------+-------+------+-------+-------------------+
                |              |
                v              v
         AnalysisKernels   Statistics
         (+ GPUMemoryPool)
                |              |
                v              v
         RawAnalysisOutput  AnalysisResult
```

## Module Responsibilities

### `main.cu` - Pipeline Orchestrator

**Location**: `src/main.cu`

Coordinates the distributed pipeline with three primary code paths:

**Single-rank path** (`mpi_size == 1`):
1. `readOTF2TracePhase1()` — two-pass reading, returns event count + opaque handle
2. Allocate SoA, `readerFillSoA()` to populate, `readerRelease()` to free reader
3. `runP2PMatching()` — CPU FIFO queue matching
4. `buildCollectiveGroups()` — CPU CSR construction
5. If `--time-correct`: `applyTimestampCorrection()` — CLC blocking-only neutralization
6. `runAnalysisKernels()` — synchronous GPU analysis
7. `computeStatistics()` + print results

**Column-major mmap path** (fastest — `colmajorDirectAnalysis()`, selected when colmajor cache exists):
1. All ranks: `readOTF2TracePhase1()` — detects colmajor cache, all ranks mmap the file
2. `madvise(MADV_HUGEPAGE)` + distributed prefault (each rank touches its pages)
3. `MPI_Win_allocate_shared` — small window for writable arrays only (6 arrays)
4. Set `TraceDataSoA` pointers: read-only → mmap, writable → SHM
5. All ranks: `runP2PMatching()`, `buildCollectiveGroups()`, optional `applyTimestampCorrection()`
6. Remap `match_partner` to global indices, `MPI_Win_fence`
7. Rank 0: `cudaHostRegister` mmap'd columns for DMA H2D
8. Rank 0: `shmGatherCSR()`, batched GPU analysis with `GPUMemoryPool`
9. Rank 0: `computeStatistics()` + print results

**Multi-rank, same-node path** (`sharedMemoryDirectAnalysis()`, fallback when no colmajor cache):
1. All ranks: `readOTF2TracePhase1()` — distributed two-pass reading
2. `MPI_Allgather` event counts, compute offsets
3. `MPI_Win_allocate_shared` — single contiguous shared memory window
4. All ranks: `readerFillSoA()` directly into shared window, `readerRelease()`
5. All ranks: `runP2PMatching()` on local slice
6. All ranks: `buildCollectiveGroups()` on local slice
7. If `--time-correct`: `applyTimestampCorrection()` on local slice
8. Remap `match_partner` to global indices, `MPI_Win_fence`
9. Rank 0: `cudaHostRegister` for pinned DMA (adaptive: skip if >10 GB)
10. Rank 0: `shmGatherCSR()` via `MPI_Gatherv`
11. Rank 0: `shmRunBatchedGPU()` — `GPUMemoryPool` + `runAnalysisKernelsAsync()` with CUDA stream
12. Rank 0: `computeStatistics()` + print results

**Multi-rank, multi-node fallback** (`streamBatchAnalysis()`):
- Falls back to Architecture C MPI Send/Recv gather when `shm_size != nprocs`

**Important**: `MPI_Init()` and `MPI_Finalize()` are called by all ranks. The program must be run with `srun` or `mpirun`. Accepts `--time-correct` as optional argument.

### `OTF2SoAReader.cpp` - Trace Reader (Distributed Two-Pass, Split-Phase API, SoA Cache)

**Location**: `src/reader/OTF2SoAReader.cpp`

The largest and most complex module. Uses TileTrace-style two-pass distributed reading with a **split-phase API** that allows the caller to choose where data is stored. Includes **binary SoA cache** (`SoACache.cpp`) and **column-major mmap cache** (`ColumnMajorCache.cpp`):

- `readOTF2TracePhase1()` — First checks for column-major cache (single mmap'd file, fastest). Then checks per-rank binary cache (SHA-256 fingerprint match). On any cache hit, returns immediately. On full miss, runs pass1 + pass2 + redistribution, then writes both cache types. Returns `ReaderPhase1Output` with event count, comm_sets, opaque handle, and `colmajor_mmap` pointer (non-null on colmajor hit)
- `readerFillSoA(handle, data)` — Copies vectors from the handle into pre-set SoA pointers (dispatches via tagged `ReaderHandle`; no-op for `HANDLE_COLMAJOR`)
- `readerRelease(handle)` — Frees the opaque handle

Internal components:
- `Pass1DiscoveryCallback`: Lightweight first pass that discovers communication partners
- `Pass2DataCallback`: Full data loading with root-based collective routing
- `redistributeCollectives()`: MPI exchange of non-local collective events

**Key responsibilities**:
- Assign contiguous location blocks to MPI ranks (`traceRange`)
- Pass 1: Discover related locations (receivers of local sends)
- Pass 2: Load events for own + related locations into internal vectors
- Route collective events by root location (local → store, remote → buffer for MPI exchange)
- Redistribute buffered collective events via `MPI_Gatherv`
- Fill any TraceDataSoA target (local calloc or shared window)

### `P2PMatching.cpp` - Send-Recv Matching

**Location**: `src/matching/P2PMatching.cpp`

**Pure C++ code** (no CUDA dependency). Previously named `.cu` — renamed to `.cpp` because nvcc compilation overhead caused 4.5x slowdown from CUDA runtime initialization even on CPU-only code.

Implements timestamp-sorted FIFO queue matching (same algorithm as TileTrace's `InteractionPattern::analysis()`).

### `CollectiveGrouping.cpp` - Collective Event Grouping

**Location**: `src/matching/CollectiveGrouping.cpp`

Groups collective events (barriers, reduces, broadcasts, etc.) into complete groups where all communicator members have participated. Outputs CSR format for GPU consumption.

### `AnalysisKernels.cu` - CUDA Analysis Kernels

**Location**: `src/analysis/AnalysisKernels.cu`

Contains 5 CUDA kernels implementing 8 analyses, plus two host orchestration functions:
- `runAnalysisKernels()` — synchronous, allocates/frees device memory per call (single-rank and fallback paths)
- `runAnalysisKernelsAsync()` — uses pre-allocated `GPUMemoryPool` and `cudaStream_t` with `cudaMemcpyAsync` (SHM batched GPU path)

Also defines `GPUMemoryPool::allocate()` and `GPUMemoryPool::deallocate()` for one-time device memory allocation reused across batches.

### `Statistics.cpp` - Summary Statistics

**Location**: `src/analysis/Statistics.cpp`

Computes count, sum, mean, variance, median, Q25, Q75, min, max from raw duration arrays. Uses `std::nth_element` (O(n) average) instead of `std::sort` (O(n log n)) for quartile computation — 4-5x faster for large result vectors.

### `TimestampCorrection.cpp` - CLC Timestamp Correction

**Location**: `src/analysis/TimestampCorrection.cpp`
**Header**: `include/analysis/TimestampCorrection.h`

Implements blocking-only CLC (Controlled Logical Clock) timestamp neutralization. For each recv event with a clock violation (`send_leave > recv_enter`):
- If `MPI_Recv` (blocking): sets `end_timestamps[i] = send_leave` to neutralize false late_receiver
- If `MPI_Irecv` (non-blocking): leaves unchanged (genuine late_receiver)

Activated by `--time-correct` command-line flag (compile-time `USE_SCALASCA_TIMESTAMPS` must also be set).

## Data Flow Between Modules

```
OTF2SoAReader::readOTF2TracePhase1()
    → [Cache check order: 1. colmajor mmap, 2. per-rank binary, 3. OTF2]
    → On colmajor HIT: all ranks mmap single file, ReaderPhase1Output.colmajor_mmap != null
    → On per-rank HIT: reads binary soa_cache_r<rank>_n<nprocs>.bin (100-780x faster)
    → On MISS: pass1 + pass2 + redistribution + writes BOTH caches
    → ReaderPhase1Output { event_count, comm_sets, opaque handle, colmajor_mmap }

[Colmajor path: colmajorDirectAnalysis()]
    → madvise(MADV_HUGEPAGE), distributed prefault
    → Small SHM window for writable arrays only
    → TraceDataSoA read-only fields → mmap, writable fields → SHM

[SHM path: sharedMemoryDirectAnalysis()]
    → readerFillSoA(handle, data) into MPI SHM window
    → readerRelease(handle)

[Common pipeline (both paths):]
P2PMatching::runP2PMatching(data)
    → Fills data.match_partner[i] in-place

CollectiveGrouping::buildCollectiveGroups(data, comm_sets)
    → CollectiveGroupCSR { offsets, members, group_types, group_roots }

TimestampCorrection::applyTimestampCorrection(data)
    → Modifies data.end_timestamps[i] for blocking recv CLC violations

AnalysisKernels::runAnalysisKernelsAsync(data, csr, pool, stream)
    → RawAnalysisOutput { 8 vectors of durations (picoseconds), sub-phase timings }

Statistics::computeStatistics(durations)
    → AnalysisResult { count, mean, median, min, max, sum, variance, q25, q75 }
```

## Build System

The project builds as a static library `gpu_analyzer_lib` plus a `gpu_analyzer` executable:

```cmake
add_library(gpu_analyzer_lib STATIC
    src/common/SHA256.cpp                # Minimal SHA-256 (no external dep)
    src/reader/OTF2SoAReader.cpp        # Pure C++ with MPI
    src/reader/SoACache.cpp             # Per-rank binary SoA cache (read/write)
    src/reader/ColumnMajorCache.cpp     # Column-major single-file cache + mmap
    src/matching/P2PMatching.cpp         # Pure C++ (was .cu, renamed for perf)
    src/matching/CollectiveGrouping.cpp
    src/analysis/AnalysisKernels.cu      # CUDA kernels + GPUMemoryPool
    src/analysis/Statistics.cpp          # Pure C++ with nth_element
    src/analysis/TimestampCorrection.cpp # Pure C++ CLC correction
)
target_link_libraries(gpu_analyzer_lib PUBLIC otf2xx::otf2xx MPI::MPI_CXX)
```

Tests link against the same static library:
```cmake
target_link_libraries(${test_name} PRIVATE gpu_analyzer_lib)
```

Timestamp mode configured via CMake cache variable:
```cmake
set(TIMESTAMP_MODE "SCALASCA" CACHE STRING "Timestamp mode: SCALASCA or TILETRACE")
# Defines USE_SCALASCA_TIMESTAMPS or USE_TILETRACE_TIMESTAMPS
```

### Third-Party Dependencies

- **otf2xx**: C++ header-heavy wrapper around the OTF2 C library. Symlinked from `specifications/TileTrace/third_party/otf2xx/`. Provides `otf2::reader::reader` and event callback system.
- **OTF2**: Low-level C library for reading/writing Open Trace Format 2 files. Found via otf2xx's `FindOTF2.cmake` which calls `otf2-config`.

## Concurrency Model

```
Time →

=== Cached Path (SoA Binary Cache Hit) ===

All Ranks:  [= SHA-256 check =][= fread() binary cache =]
                         ↓ (24 ms for 16q, 2.2s for n1024)
            → continues to SHM Analysis below

=== Same-Node Path (Primary, sharedMemoryDirectAnalysis) ===

All Ranks:  [= Pass 1 =][== Pass 2 ==][= Redist =][= write cache =]
                         ↓
            [== fill_soa into SHM ==][= P2P =][= Coll =][= TS Correct =]
                         ↓
            [= remap match_partner =][= MPI_Win_fence =]

Rank 0:     [= pin =][= CSR gather =][= Batch 1: H2D→GPU→D2H =][= Batch 2: ... =][= Stats =]
Other Ranks: exit after fence

=== Multi-Node Fallback (streamBatchAnalysis) ===

Rank 0:  [= Pass 1 =][== Pass 2 ==][= Redist =][= P2P =][= Coll =][= recv+GPU per K ranks =][= Stats =]
Rank 1:  [= Pass 1 =][== Pass 2 ==][= Redist =][= P2P =][= Coll =][= Send data to R0 =]
...
Rank N:  [= Pass 1 =][== Pass 2 ==][= Redist =][= P2P =][= Coll =][= Send data to R0 =]
```

**Same-node path**: All MPI ranks share a single MPI shared memory window. Each rank fills its slice, does local P2P matching and collective grouping. Rank 0 reads the full window directly — no MPI data transfer needed. This eliminates the 8-13s MPI gather bottleneck from the Architecture C approach.

**Multi-node fallback**: Used when ranks span multiple physical nodes (`shm_size != nprocs`). Falls back to Architecture C (MPI Send/Recv gather to rank 0).

**GPU analysis on rank 0**: Uses `GPUMemoryPool` (one-time allocation) and `runAnalysisKernelsAsync()` with a `cudaStream_t`. Adaptive batching: K ranks per GPU batch, K computed from available VRAM. For most traces, K=P (all fit in one batch). For large traces (>24 GB VRAM), K < P and multiple batches are used.

**Adaptive pinning**: `cudaHostRegister` is applied to the 6 GPU-accessed SoA arrays for DMA H2D transfer. For buffers >10 GB, global pinning is skipped and per-batch pinning is used instead (pinning 16+ GB costs more than the H2D savings).
