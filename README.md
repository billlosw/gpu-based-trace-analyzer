# GPU-Based Trace Analyzer

A CUDA-accelerated MPI trace analyzer that reads OTF2 trace files and performs 8 wait-state analyses on GPU, producing the same metrics as TileTrace/Scalasca.

## Prerequisites

- CUDA Toolkit (12.x recommended, tested with 12.9.0)
- CMake >= 3.24
- MPI implementation (OpenMPI 4.x)
- OTF2 library (3.0.3)
- NVIDIA GPU with compute capability >= 8.9 (RTX 4090 targeted)

On the FUSE cluster, load dependencies with:
```bash
source /home/spack/spack/share/spack/setup-env.sh
spack load cuda@12.9.0
spack load cmake
spack load openmpi
spack load otf2@3.0.3
```

## Project Structure

```
gpu-analyzer/
├── CMakeLists.txt              # Build configuration
├── include/
│   ├── common/
│   │   ├── types.h             # Event enums (same as TileTrace), timestamp_t, id_t
│   │   └── cuda_check.h        # CUDA_CHECK error-handling macro
│   ├── data/
│   │   ├── TraceDataSoA.h      # SoA data layout (12 base arrays + matching arrays)
│   │   └── AnalysisResults.h   # Analysis result struct (count, mean, median, etc.)
│   ├── reader/
│   │   └── OTF2SoAReader.h     # CPU-side OTF2 reader outputting SoA directly
│   ├── matching/
│   │   ├── P2PMatching.h       # GPU-based send-recv hash matching
│   │   └── CollectiveGrouping.h # CPU-side collective event grouping (CSR format)
│   └── analysis/
│       ├── AnalysisKernels.h   # CUDA kernels for all 8 analyses
│       └── Statistics.h        # Summary statistics computation
├── src/
│   ├── reader/OTF2SoAReader.cpp     # OTF2 reader using otf2xx library
│   ├── matching/
│   │   ├── P2PMatching.cu           # GPU hash-table matching kernels
│   │   └── CollectiveGrouping.cpp   # CPU collective grouping
│   ├── analysis/
│   │   ├── AnalysisKernels.cu       # 8 analysis CUDA kernels
│   │   └── Statistics.cpp           # CPU statistics (sort for quartiles)
│   └── main.cu                      # Entry point, pipeline orchestration
├── test/
│   ├── CMakeLists.txt
│   ├── test_p2p_matching.cu         # P2P matching correctness tests
│   ├── test_analysis_kernels.cu     # All 8 analysis kernels tests
│   └── test_statistics.cpp          # Statistics computation tests
├── scripts/
│   └── run_npb.sh                   # NPB benchmark runner
└── third_party/
    └── otf2xx/                      # Symlink to TileTrace's otf2xx library
```

## Building

```bash
cd gpu-analyzer
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make    # NOTE: Do NOT use make -j on the FUSE cluster (OOM risk)
```

This produces:
- `build/gpu_analyzer` - Main executable
- `build/test/test_p2p_matching` - P2P matching test
- `build/test/test_analysis_kernels` - Analysis kernels test
- `build/test/test_statistics` - Statistics test

## Running

### Basic Usage

```bash
./gpu_analyzer <path/to/traces.otf2>
```

### On FUSE Cluster (with SLURM)

```bash
# Allocate a GPU node and run
srun --gres=gpu:4090:1 -n 1 ./build/gpu_analyzer /path/to/scorep_cg_64_trace/traces.otf2
```

### Running Tests

```bash
# On a machine with GPU:
./build/test/test_p2p_matching
./build/test/test_analysis_kernels
./build/test/test_statistics

# On FUSE cluster:
srun --gres=gpu:4090:1 ./build/test/test_p2p_matching
srun --gres=gpu:4090:1 ./build/test/test_analysis_kernels
srun --gres=gpu:4090:1 ./build/test/test_statistics
```

### Using the NPB Script

```bash
# Make sure trace data exists first
./scripts/run_npb.sh cg B 64
```

## Output Format

The analyzer outputs results identical in format to TileTrace's `analysis_integration_test`:

```
------------------ late_sender -------------------
Count: 1234
Mean: 0.0001234567
Median: 0.0000987654
Minimum: 0.0000012345
Maximum: 0.0012345678
Sum: 0.1523456789
Variance: 0.0000000123
Quartile 25: 0.0000456789
Quartile 75: 0.0001567890
```

This is printed for all 8 analyses:
1. **late_sender** - Sender arrived after receiver in P2P communication
2. **late_receiver** - Receiver arrived after sender in P2P communication
3. **barrier_wait** - Wait time at MPI_Barrier (max_enter - each_enter)
4. **barrier_completion** - Completion imbalance at MPI_Barrier (each_end - min_end)
5. **earlyreduce** - Root arrived early at MPI_Reduce/Gather (max_member - root)
6. **latebroadcast** - Root arrived late at MPI_Bcast/Scatter (root - early_members)
7. **wait_nxn** - Wait time at NxN collectives (AllReduce, AlltoAll, etc.)
8. **nxn_completion** - Completion imbalance at NxN collectives

Plus a timing summary showing per-phase durations.

## Comparing with TileTrace

Run TileTrace on the same trace:
```bash
srun -n 4 ./analysis_integration_test /path/to/traces.otf2
```

Run GPU analyzer:
```bash
srun --gres=gpu:4090:1 -n 1 ./gpu_analyzer /path/to/traces.otf2
```

Compare the Count/Mean/Sum/Min/Max values for each of the 8 analyses.

## Architecture

### Data Flow

```
OTF2 File (.otf2)
  → [CPU: OTF2SoAReader]        Read trace into SoA arrays
  → [CPU: CollectiveGrouping]    Group collective events into CSR
  → [cudaMemcpy H2D]            Transfer to GPU
  → [GPU: P2PMatching]          Hash-table send-recv matching
  → [GPU: AnalysisKernels]      Run 8 analysis kernels
  → [cudaMemcpy D2H]            Copy results back
  → [CPU: Statistics]           Compute summary statistics
  → Output
```

### Key Design Decisions

| Aspect | TileTrace (CPU) | GPU Analyzer |
|--------|-----------------|--------------|
| Parallelism | MPI across nodes | CUDA threads on single GPU |
| Data Layout | AoS (struct Event) | SoA (12 contiguous arrays) |
| Matching | CPU hash map | GPU open-addressing hash table |
| Collective Groups | Inline with matching | Separate CPU preprocessing |
| Event Linking | Pointer-based | Index-based (int32_t) |

### SoA Memory Layout

Each event occupies ~56 bytes across 12 arrays (plus matching metadata):
- `events[]` (event_t, 4B) - MPI event type
- `timestamps[]` (uint64_t, 8B) - Enter timestamp in picoseconds
- `end_timestamps[]` (uint64_t, 8B) - End timestamp (for collectives)
- `pids[]` (uint32_t, 4B) - Process ID
- `srcs[]`, `dsts[]`, `tags[]`, `roots[]` (uint32_t each) - Communication metadata
- `match_partner[]` (int32_t, 4B) - Index of matched P2P partner

With a 24GB RTX 4090, this supports ~270 million events per GPU pass.

## Generated Files After Running

| File/Output | Description |
|-------------|-------------|
| stdout | Analysis results (8 metrics) + timing summary |
| `build/gpu_analyzer` | Main executable |
| `build/test/test_*` | Test executables |

## Dependencies on Other Project Components

- `specifications/TileTrace/third_party/otf2xx/` - C++ wrapper for OTF2 (symlinked)
- `specifications/TileTrace/exp/NPB3.4.2/` - NPB benchmark sources for generating traces
- `specifications/TileTrace/test/integration_test/` - Reference analysis results for validation
