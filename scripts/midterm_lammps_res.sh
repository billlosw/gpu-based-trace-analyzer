#!/bin/bash

# Define the lammps data pairs
part1=("__64_qtraces" "__128_qtraces" "__256_qtraces" "__512_qtraces" "lammps_n1024" "lammps_n2048")
part2=("64" "128" "256" "512" "1024" "2048")

# Ensure the log directory exists
mkdir -p ~/logs/midterm-data/

# Iterate through the indices
for i in "${!part1[@]}"; do
    p1="${part1[$i]}"
    p2="${part2[$i]}"
    
    echo "Processing: $p1 (Part2: $p2)"
    
    srun -N 1 -n 64 -p Debug --mpi=pmix --gres=gpu:4090:1 \
        ./build/gpu_analyzer ~/trace_data/"$p1"/traces.otf2 --time-correct | \
        tee ~/logs/midterm-data/260330-lammps"$p2"-g4090c64.log
done
