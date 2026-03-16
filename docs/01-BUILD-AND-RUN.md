# Build & Run Guide

## Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| CUDA Toolkit | **12.8.0** | Must not exceed driver version on GPU nodes (see CRITICAL below) |
| CMake | >= 3.24 | For CUDA language support |
| OpenMPI | 4.x | Built with `--with-pmix` on FUSE cluster |
| OTF2 | 3.0.3 | Interface version 10.0.0 |
| NVIDIA GPU | CC >= 8.9 | RTX 4090 (Ada Lovelace, sm_89) or 5090 (via PTX JIT) |

### CRITICAL: CUDA Version Mismatch

The CUDA *toolkit* version must be **<= the GPU driver version** on the compute nodes. On the FUSE cluster, GPU nodes have driver version 12.8.x. Using toolkit 12.9 causes **silent kernel failure**: kernels appear to launch but produce all zeros. The error "PTX compiled with unsupported toolchain" only appears in `cuda-memcheck` output. `cudaMemcpy` works fine (it's a host API), only kernel execution fails.

**Always use `cuda@12.8.0`, NOT 12.9.**

## Building on the FUSE Cluster

### Using helper.sh (Recommended)

```bash
# Step 1: Load environment (must be sourced, not executed)
. helper.sh env

# Step 2: Build
bash helper.sh build

# Step 3: Clean (if needed)
bash helper.sh clean
```

`helper.sh env` loads: `cuda@12.8.0`, `cmake`, `openmpi`, `otf2@3.0.3`, `scorep`, `scalasca`, `dyninst`

`helper.sh build` auto-detects the CUDA path via `spack location -i cuda@12.8.0` and passes `-DCMAKE_CUDA_COMPILER=...` to CMake.

### Manual Build

```bash
source /home/spack/spack/share/spack/setup-env.sh
spack load cuda@12.8.0 cmake openmpi otf2@3.0.3

cd gpu-analyzer
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=$(spack location -i cuda@12.8.0)/bin/nvcc
make -j4
```

> **Warning**: On memory-constrained nodes, avoid `make -j` with high parallelism. The otf2xx template-heavy compilation can OOM.

### Build Outputs

```
build/
  gpu_analyzer              # Main executable
  test/
    test_p2p_matching       # P2P matching unit tests
    test_analysis_kernels   # All 8 analysis kernel tests
    test_statistics         # Statistics computation tests
```

### CMake Configuration Details

In `CMakeLists.txt`:

```cmake
# sm_89 for RTX 4090. RTX 5090 uses PTX JIT from sm_89 code.
set(CMAKE_CUDA_ARCHITECTURES 89)

# CRITICAL: Must be set BEFORE add_subdirectory(third_party/otf2xx)
# Controls whether OTF2XX_HAS_MPI is defined, which guards the
# MPI reader constructor in otf2xx.
set(OTF2XX_WITH_MPI ON CACHE BOOL "Enable MPI support in otf2xx" FORCE)
```

## Running

### Basic Usage

```bash
# Single-process (serial trace reading, slower for large traces)
./build/gpu_analyzer /path/to/traces.otf2

# Multi-process (parallel trace reading, recommended)
srun --mpi=pmix -n 8 --gres=gpu:4090:1 ./build/gpu_analyzer /path/to/traces.otf2
```

### Command-Line Arguments

```
Usage: gpu_analyzer <path/to/traces.otf2>
```

Only one argument: the path to the OTF2 anchor file (`.otf2`).

### SLURM Configuration on FUSE

```bash
# CRITICAL: --mpi=pmix is required on FUSE
# OpenMPI was built with --with-pmix, not SLURM PMI
# Without it: "OMPI was not compiled with support for SLURM PMI"

srun --mpi=pmix \     # Required for OpenMPI on FUSE
     -n 8 \           # Number of MPI ranks for parallel reading
     --gres=gpu:4090:1 \  # Request one GPU (4090 or 5090)
     ./build/gpu_analyzer /path/to/traces.otf2
```

The `-n` parameter controls **how many MPI ranks** participate in parallel trace reading. Only rank 0 uses the GPU for analysis. After reading, non-rank-0 processes exit.

Recommended values for `-n`:
- **1**: Serial reading (simplest, slowest)
- **8**: Good balance (7.5x speedup over serial for CG traces)
- **32**: Maximum observed speedup (~44x)

### Running Tests

```bash
# On a GPU node (tests need CUDA runtime)
srun --gres=gpu:4090:1 ./build/test/test_p2p_matching
srun --gres=gpu:4090:1 ./build/test/test_analysis_kernels
srun --gres=gpu:4090:1 ./build/test/test_statistics
```

### Generating Traces with Score-P

To create OTF2 traces from NPB benchmarks:

```bash
# 1. Load Score-P
spack load gcc@12.2.0 openmpi scorep otf2@3.0.3

# 2. Build NPB with Score-P instrumentation
cd specifications/TileTrace/exp/NPB3.4.2/NPB3.4-MPI
# In config/make.def, set:
#   MPICC = scorep mpicc
#   MPIF77 = scorep mpif77
make cg CLASS=B NPROCS=64

# 3. Run to generate trace
export SCOREP_ENABLE_TRACING=true
export SCOREP_ENABLE_PROFILING=false
export SCOREP_TOTAL_MEMORY=1G
SCOREP_EXPERIMENT_DIRECTORY=scorep_cg_B_64_trace \
    srun -p ja -N 2 -n 64 ./bin/cg.B.x

# Result: scorep_cg_B_64_trace/traces.otf2
```

### Rsync to Cluster

```bash
rsync -avzL --exclude='.git' --exclude='build' \
    gpu-analyzer/ pacman.fuse:/home/luosw22/claude/gpu-analyzer/
```

## Output Format

The analyzer prints results in the same format as TileTrace's `analysis_integration_test`:

```
=== Analysis Results ===
------------------ late_sender -------------------
Count: 396922
Mean: 0.0000357719
Median: 0.0000087654
Minimum: 0.0000000012
Maximum: 0.0561983636
Sum: 14.2017681417
Variance: 0.0000000123
Quartile 25: 0.0000023456
Quartile 75: 0.0000345678
```

Plus a timing summary:

```
=== Timing Summary ===
OTF2 Read:            25146.23 ms
P2P Matching:         255.12 ms
Coll. Grouping (CPU): 1.80 ms
Analysis (GPU):       125.45 ms
Statistics (CPU):     120.33 ms
Total:                25648.93 ms
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| All analysis results are 0 | CUDA toolkit > driver version | Use `cuda@12.8.0`, not 12.9 |
| "OMPI was not compiled with support for SLURM PMI" | Missing `--mpi=pmix` | Add `--mpi=pmix` to `srun` |
| OTF2 reader hangs | Wrong MPI rank count | Use any `-n` value (reading is round-robin) |
| "PTX compiled with unsupported toolchain" | CUDA version mismatch | Downgrade CUDA toolkit |
| Build error: "OTF2XX_HAS_MPI not defined" | `OTF2XX_WITH_MPI` set after `add_subdirectory` | Must set BEFORE `add_subdirectory(otf2xx)` |
| Build error: OTF2 version | otf2xx expects wrong version | Check `find_package(OTF2 ...)` in otf2xx CMakeLists |
