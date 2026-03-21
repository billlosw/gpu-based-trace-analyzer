# Architecture

## Directory Structure

```
gpu-analyzer/
├── CMakeLists.txt                      # Top-level build config
├── helper.sh                           # Build/env/clean helper script
├── include/
│   ├── common/
│   │   ├── types.h                     # Shared enums, typedefs
│   │   └── cuda_check.h               # CUDA_CHECK error macro
│   ├── data/
│   │   ├── TraceDataSoA.h             # SoA data structure + CSR struct
│   │   └── AnalysisResults.h          # Result struct for statistics
│   ├── reader/
│   │   └── OTF2SoAReader.h            # Reader interface (ReaderOutput)
│   ├── matching/
│   │   ├── P2PMatching.h              # P2P matching interface
│   │   └── CollectiveGrouping.h       # Collective grouping interface
│   └── analysis/
│       ├── AnalysisKernels.h          # CUDA kernel interface
│       └── Statistics.h               # Statistics computation interface
├── src/
│   ├── main.cu                        # Entry point, pipeline orchestration
│   ├── reader/
│   │   └── OTF2SoAReader.cpp          # OTF2 -> SoA reader implementation
│   ├── matching/
│   │   ├── P2PMatching.cu             # CPU-based P2P matching (in .cu for linking)
│   │   └── CollectiveGrouping.cpp     # CPU collective grouping
│   └── analysis/
│       ├── AnalysisKernels.cu         # 5 CUDA kernels (8 analyses)
│       └── Statistics.cpp             # Summary statistics on CPU
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
               /   |   \
              /    |    \
             v     v     v
    OTF2SoAReader  P2PMatching  CollectiveGrouping
         |              |              |
         v              v              v
     TraceDataSoA  TraceDataSoA  CollectiveGroupCSR
         |              |              |
         +------+-------+------+-------+
                |              |
                v              v
         AnalysisKernels   Statistics
                |              |
                v              v
         RawAnalysisOutput  AnalysisResult
```

## Module Responsibilities

### `main.cu` - Pipeline Orchestrator

**Location**: `src/main.cu`

Coordinates the 6-step distributed pipeline:
1. All ranks call `readOTF2Trace()` — distributed two-pass reading
2. All ranks call `runP2PMatching()` on their local data
3. All ranks call `buildCollectiveGroups()` on their local data
4. All ranks call `runAnalysisKernels()` on their local data (GPU)
5. `gatherRawResults()` gathers duration vectors from all ranks to rank 0
6. Rank 0 computes statistics and prints results

**Important**: `MPI_Init()` and `MPI_Finalize()` are called by all ranks. The program must be run with `srun` or `mpirun`.

### `OTF2SoAReader.cpp` - Trace Reader (Distributed Two-Pass)

**Location**: `src/reader/OTF2SoAReader.cpp`

The largest and most complex module. Uses TileTrace-style two-pass distributed reading:
- `Pass1DiscoveryCallback`: Lightweight first pass that discovers communication partners
- `Pass2DataCallback`: Full data loading with root-based collective routing
- `redistributeCollectives()`: MPI exchange of non-local collective events
- `readOTF2Trace()`: Public entry point orchestrating both passes

**Key responsibilities**:
- Assign contiguous location blocks to MPI ranks (`traceRange`)
- Pass 1: Discover related locations (receivers of local sends)
- Pass 2: Load events for own + related locations (SoA format)
- Route collective events by root location (local → store, remote → buffer for MPI exchange)
- Redistribute buffered collective events via `MPI_Gatherv`

### `P2PMatching.cu` - Send-Recv Matching

**Location**: `src/matching/P2PMatching.cu` (99 lines)

Despite the `.cu` extension, this is **pure CPU code**. The `.cu` extension is used for CUDA linker compatibility.

Implements timestamp-sorted FIFO queue matching (same algorithm as TileTrace's `InteractionPattern::analysis()`).

### `CollectiveGrouping.cpp` - Collective Event Grouping

**Location**: `src/matching/CollectiveGrouping.cpp` (190 lines)

Groups collective events (barriers, reduces, broadcasts, etc.) into complete groups where all communicator members have participated. Outputs CSR format for GPU consumption.

### `AnalysisKernels.cu` - CUDA Analysis Kernels

**Location**: `src/analysis/AnalysisKernels.cu` (511 lines)

Contains 5 CUDA kernels implementing 8 analyses, plus the host orchestration function `runAnalysisKernels()`. Handles all GPU memory allocation, H2D/D2H transfers, kernel launches, and timing.

### `Statistics.cpp` - Summary Statistics

**Location**: `src/analysis/Statistics.cpp` (48 lines)

Computes count, sum, mean, variance, median, Q25, Q75, min, max from raw duration arrays. Sorts data on CPU for quartile computation.

## Data Flow Between Modules

```
OTF2SoAReader::readOTF2Trace()
    → ReaderOutput { TraceDataSoA data, vector<vector<uint64_t>> comm_sets }

P2PMatching::runP2PMatching(data)
    → Fills data.match_partner[i] in-place

CollectiveGrouping::buildCollectiveGroups(data, comm_sets)
    → CollectiveGroupCSR { offsets, members, group_types, group_roots }

AnalysisKernels::runAnalysisKernels(data, csr)
    → RawAnalysisOutput { 8 vectors of durations (picoseconds), sub-phase timings }

Statistics::computeStatistics(durations)
    → AnalysisResult { count, mean, median, min, max, sum, variance, q25, q75 }
```

## Build System

The project builds as a static library `gpu_analyzer_lib` plus a `gpu_analyzer` executable:

```cmake
add_library(gpu_analyzer_lib STATIC
    src/reader/OTF2SoAReader.cpp    # Pure C++ with MPI
    src/matching/P2PMatching.cu      # CPU code in .cu for CUDA link compat
    src/matching/CollectiveGrouping.cpp
    src/analysis/AnalysisKernels.cu  # CUDA kernels
    src/analysis/Statistics.cpp
)
target_link_libraries(gpu_analyzer_lib PUBLIC otf2xx::otf2xx MPI::MPI_CXX)
```

Tests link against the same static library:
```cmake
target_link_libraries(${test_name} PRIVATE gpu_analyzer_lib)
```

### Third-Party Dependencies

- **otf2xx**: C++ header-heavy wrapper around the OTF2 C library. Symlinked from `specifications/TileTrace/third_party/otf2xx/`. Provides `otf2::reader::reader` and event callback system.
- **OTF2**: Low-level C library for reading/writing Open Trace Format 2 files. Found via otf2xx's `FindOTF2.cmake` which calls `otf2-config`.

## Concurrency Model

```
Time →

Rank 0:  [= Pass 1 =][== Pass 2 ==][= Redist =][= P2P =][= Coll =][= GPU =][= Gather =][= Stats =]
Rank 1:  [= Pass 1 =][== Pass 2 ==][= Redist =][= P2P =][= Coll =][= GPU =][= Gather =]
...
Rank N:  [= Pass 1 =][== Pass 2 ==][= Redist =][= P2P =][= Coll =][= GPU =][= Gather =]

← All ranks participate in reading, matching, analysis, and result gathering →
← Only rank 0 computes final statistics and prints results →
```

All MPI ranks participate in the full pipeline up to result gathering. Each rank processes its local portion of the trace independently, then duration vectors are gathered to rank 0 for global statistics.
