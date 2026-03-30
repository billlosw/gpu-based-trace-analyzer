#!/bin/bash

# Define the cg data pairs
part1=("cg.B" "cg.C" "cg.D")
part2=("cgB" "cgC" "cgD")

# Ensure the log directory exists
mkdir -p ~/logs/midterm-data/

# Iterate through the indices
for i in "${!part1[@]}"; do
    p1="${part1[$i]}"
    p2="${part2[$i]}"
    
    echo "Processing: $p1 (Part2: $p2)"
    
    srun -N 1 -n 64 -p Debug --mpi=pmix --gres=gpu:4090:1 \
        ./build/gpu_analyzer ~/claude/TileTraceClaude/exp/traces/"$p1"/traces.otf2 | \
        tee ~/logs/midterm-data/260330-"$p2"-g4090c64.log
done
