#!/bin/bash
# Run NPB benchmarks through Score-P and then analyze with GPU analyzer.
# Usage: ./run_npb.sh <benchmark> <class> <nprocs>
# Example: ./run_npb.sh cg B 64

set -e

BENCHMARK=${1:-cg}
CLASS=${2:-B}
NPROCS=${3:-64}

echo "=== Running NPB ${BENCHMARK}.${CLASS} with ${NPROCS} processes ==="

# Load modules
source /home/spack/spack/share/spack/setup-env.sh
spack load cuda@12.9.0
spack load cmake
spack load openmpi
spack load otf2@3.0.3

# Check for pre-existing trace data
TRACE_DIR="TRACE_DATA/NPB/scorep_${BENCHMARK}_${NPROCS}_trace"
OTF2_FILE="${TRACE_DIR}/traces.otf2"

if [ ! -f "${OTF2_FILE}" ]; then
    echo "Error: Trace file not found at ${OTF2_FILE}"
    echo "Please generate trace data first using Score-P instrumentation."
    echo "See specifications/TileTrace/exp/ for examples."
    exit 1
fi

echo "Trace file: ${OTF2_FILE}"
echo ""

# Run GPU analyzer
echo "=== Running GPU Trace Analyzer ==="
srun --gres=gpu:4090:1 -n 1 ./build/gpu_analyzer "${OTF2_FILE}"
