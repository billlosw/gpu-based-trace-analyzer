# Architecture Research: CPU/GPU Partition for GPU Trace Analyzer

## 1. Problem Statement

The gpu-analyzer needs to analyze OTF2 MPI traces and produce 8 wait-state metrics identical to Scalasca. The hardware constraint is:

- **GPU**: one RTX 4090 (24 GB VRAM) — used for analysis acceleration
- **CPU**: multi-core/multi-node (2× Xeon Gold 6530, 1 TB RAM via MPI) — used for I/O and preprocessing
- Target traces: NPB benchmarks (Class B through E), LAMMPS, Sweep3D, HPCG — from thousands to billions of events

### The Three Iterations and Their Failures

| Iteration | Architecture | Approach | Failure Mode |
|-----------|-------------|----------|-------------|
| 1 (initial) | Centralized single-rank | One rank reads all → GPU | O(N) RAM on rank 0, reading not parallelized |
| 2 (260321-1700) | Distributed read, per-rank GPU | All ranks read + match + each calls GPU | GPU contention — 8.6× slowdown (1074ms vs 125ms) |
| 3 (260321-1920) | Distributed read, gather-then-GPU | All ranks read + match, MPI_Gatherv to rank 0, single GPU | **O(N) memory on rank 0** — scalability wall returns |

Each iteration solved one problem but reintroduced another. We need an architecture that simultaneously achieves:

1. **Parallel I/O**: multi-rank OTF2 reading (near-linear speedup)
2. **Bounded CPU memory**: no single rank holds all N events
3. **Efficient GPU utilization**: avoid contention, avoid serialized kernel launches
4. **VRAM scalability**: handle traces larger than 24 GB VRAM

### The Fundamental Observation

After the two-pass reading + P2P matching + collective grouping (all done locally per rank on CPU), each rank holds **independently analyzable** data:

- **P2P pairs**: Fully matched locally — the two-pass read co-locates send-recv pairs on the sender's rank. Each matched pair needs only `timestamps[send]`, `timestamps[recv]`, `end_timestamps[]` to compute late_sender/late_receiver. No cross-rank data needed.
- **Collective groups**: Complete after root-based redistribution — each rank holds complete groups (all members present). Each group needs only its members' timestamps. No cross-rank data needed.

**This means analysis is embarrassingly parallel across ranks.** The current `gatherTraceData()` merging is unnecessary for correctness — it was introduced solely to avoid GPU contention, but there are better solutions.

---

## 2. Pipeline Stage Analysis

### 2.1 Per-Stage Characteristics

| Stage | Location | Time Share | Memory Pattern | Parallelizable? | GPU Benefit |
|-------|----------|-----------|----------------|-----------------|-------------|
| OTF2 Read | CPU, all ranks | ~80-90% | I/O-bound, O(N/P) per rank | Yes (MPI parallel) | None |
| P2P Matching | CPU, all ranks | ~5-10% | O(N/P) per rank | Yes (local per rank) | Medium (sort-based, see §5) |
| Coll Grouping | CPU, all ranks | ~1-2% | O(C/P) per rank | Yes (local per rank) | Low |
| GPU Analysis | GPU, rank 0 | ~0.5-3% | 32 B/event VRAM | Yes (per-event/group) | High |
| Statistics | CPU, rank 0 | <0.1% | O(R) results | Easy | Low |

### 2.2 Evidence: Timing Data (CG Class C, 64 locations)

From the distributed reading experiment (260321-1700):

| Phase | n=1 | n=8 | Speedup |
|-------|-----|-----|---------|
| OTF2 Read | 442,500 ms | 106,322 ms | 4.2× |
| P2P Matching | 211 ms | 34 ms | 6.2× |
| Coll Grouping | 2.4 ms | 1.0 ms | 2.4× |
| GPU Analysis | 125 ms | 1,074 ms | 0.12× (contention) |
| **Total** | **442,906 ms** | **107,652 ms** | **4.1×** |

**Key insight**: Reading dominates at ~95-98% of total time. With 8 ranks, reading drops from 442s to 106s but still dominates. The GPU analysis (125ms at n=1) is <0.03% of the total pipeline — **the analysis phase is not the bottleneck for current trace sizes.**

### 2.3 Projected Scaling to Larger Traces

For NPB BT Class D (1024 processes, ~500M events):

| Phase | Estimated Time (n=64 ranks) | Memory Rank 0 |
|-------|---------------------------|---------------|
| OTF2 Read | ~100-200s (I/O-bound) | O(N/P) ≈ 0.5 GB |
| P2P Matching | ~2-5s (CPU, local) | O(N/P) ≈ 0.5 GB |
| Coll Grouping | ~0.1-0.5s | Negligible |
| Gather (current) | ~5-30s (network-bound) | **O(N) ≈ 32 GB** |
| GPU Analysis | ~2-5s (single launch, 500M events) | **O(N) × 32B ≈ 16 GB VRAM** |

For TB-scale traces (1B+ events): the gather approach completely breaks — 64+ GB on rank 0, exceeding VRAM.

---

## 3. Architecture Options

### Architecture A: Gather-All-to-Rank-0 (Current Implementation)

```
ALL RANKS (P processes)         RANK 0 ONLY
┌──────────────────────┐       ┌─────────────────────────┐
│ 1. Two-pass OTF2 read│       │ 4. Allocate merged SoA  │
│ 2. Local P2P matching│ ───►  │    (N events total)     │
│ 3. Local coll group  │  MPI  │ 5. Single GPU launch    │
│    O(N/P) per rank   │ Gatherv│ 6. Statistics           │
└──────────────────────┘       └─────────────────────────┘
                               Memory: O(N) CPU + O(N) VRAM
```

**Memory on rank 0**: N × 88 bytes (full SoA) + N × 32 bytes (VRAM subset) = **~120 bytes/event**
- 500M events → 60 GB RAM + 16 GB VRAM
- 1B events → 120 GB RAM + 32 GB VRAM (exceeds 4090)

**Pros**:
- Simplest implementation (already working)
- Single GPU kernel launch → maximum GPU utilization per launch
- Correct results verified against Scalasca

**Cons**:
- O(N) memory on rank 0 — **hard scalability wall**
- Network bottleneck: all P ranks send through one MPI_Gatherv
- Gathering time grows linearly with N even as P increases
- Double memory usage: local data + merged data (briefly, until local is freed)

**When to use**: Small-to-medium traces that fit in rank 0's RAM + VRAM (<500M events with 64 GB RAM + 4090).

---

### Architecture B: Streaming Pipelined Single-GPU (Recommended)

```
ALL RANKS              RANK 0 (GPU owner)                      GPU
┌─────────────┐       ┌──────────────────────────────┐     ┌─────────────────┐
│1. Read       │       │ Process own local data first │     │ Buffer A (VRAM) │
│2. Match      │       │ Then for each remote rank:   │     │ Buffer B (VRAM) │
│3. Group      │       │   Recv SoA+CSR via MPI       │◄──►│                 │
│4. Send to R0 │──────►│   H2D to buffer A/B          │────►│ Kernel launch   │
│   (on demand)│       │   Launch kernel              │     │ Compute results │
│   O(N/P)/rank│       │   Accumulate results         │◄────│ D2H results     │
└─────────────┘       │ Compute statistics            │     └─────────────────┘
                      └──────────────────────────────┘
                      Host Memory: O(N/P) × 2 (double-buffer)
                      VRAM: O(N/P) × 2 (double-buffer)
```

**Core idea**: After all ranks complete CPU preprocessing in parallel, rank 0 processes each rank's data on the GPU **one at a time** (or in small batches), using double-buffering to overlap MPI receive + H2D copy with GPU kernel execution.

**Pipeline (with double buffering and CUDA streams)**:

```
Time →

Rank 1: [send SoA+CSR to R0]
Rank 2:                       [send SoA+CSR to R0]
Rank 3:                                             [send SoA+CSR to R0]

Rank 0:
  CPU:  [recv R1 → buf_A] [recv R2 → buf_B] [recv R3 → buf_A] ...
  GPU:  [analyze local]   [H2D_A + kernel_A] [H2D_B + kernel_B] [H2D_A + kernel_A] ...
  Stream 0: ──────────────[H2D buf_A][kernel_A]                 [H2D buf_A][kernel_A]
  Stream 1: ──────────────                    [H2D buf_B][kernel_B]
```

**Memory on rank 0**:
- Own local data: N/P × 88 bytes
- Receive buffer (double): 2 × N/P × 88 bytes  
- GPU buffer (double): 2 × N/P × 32 bytes
- **Total**: ~4 × N/P × 88 ≈ **352 bytes/event per rank** (but per-rank, not total N)
- For 500M events / 64 ranks ≈ 7.8M events/rank: **2.7 GB host + 0.5 GB VRAM** — trivially fits

**Results accumulation**: Each GPU launch produces duration vectors (late_sender[], barrier_wait[], etc.). These are small (proportional to the number of detected wait patterns, typically < number of events). Rank 0 concatenates all ranks' result vectors, then computes global statistics.

**Quantitative analysis (500M events, P=64)**:

| Metric | Per-rank batch | Total |
|--------|---------------|-------|
| Events per batch | 7.8M | 500M |
| H2D per batch (32B/event) | 250 MB → ~10 ms at 25 GB/s | 640 ms |
| Kernel per batch (P2P) | ~2-10 ms (SM saturation) | 128-640 ms |
| D2H per batch (results) | < 1 ms | < 64 ms |
| Total GPU time (sequential) | ~15-20 ms | ~960-1280 ms |
| Total GPU time (pipelined) | — | ~700-900 ms |

Compare to Architecture A single launch on 500M events: H2D = 640ms + kernel = 2-5s + D2H < 10ms ≈ 2.6-5.6s. **The streaming approach is comparable or faster** because:
- GPU fully saturates SMs even with 7.8M events (256 blocks × 256 threads = 65K threads)
- Double-buffering hides H2D latency
- Avoids the 30+ GB host allocation and network bottleneck of MPI_Gatherv

**Pros**:
- **O(N/P) memory** on rank 0 — scales to arbitrary trace sizes
- No MPI_Gatherv bottleneck — point-to-point sends from each rank
- GPU utilization remains high with double-buffering
- VRAM usage bounded at O(N/P), not O(N)
- Works with 1 GPU, 2 GPUs, or any number
- Kernel launch overhead P × 10-50 μs is negligible even for P=256

**Cons**:
- P sequential GPU batches (mitigated by pipelining)
- Implementation more complex than Architecture A (CUDA streams, double-buffer management)
- MPI point-to-point management needed (rank ordering, buffer lifecycle)
- Results must be concatenated across batches (minor bookkeeping)

**When to use**: Medium-to-large traces, or any scale where O(N) memory on rank 0 is prohibitive. **This is the recommended default architecture.**

---

### Architecture C: Adaptive Batch Streaming (Extension of B)

Instead of processing one rank's data at a time, process K ranks' data in one GPU launch:

```
K = min(P, floor(VRAM_available / per_rank_GPU_data))
```

| Trace Scale | N events | P ranks | N/P | K (24GB VRAM) | Batches |
|-------------|----------|---------|-----|---------------|---------|
| CG Class C | 1M | 8 | 125K | 8 (all fit) | 1 |
| CG Class D | 40M | 64 | 625K | 64 (all fit) | 1 |
| BT Class D | 500M | 128 | 3.9M | 128 (all fit) | 1 |
| Extreme | 5B | 256 | 19.5M | 38 | 7 |
| Extreme | 10B | 512 | 19.5M | 38 | 14 |

**Key observation**: For the vast majority of practical cases, K = P and the entire trace fits in one GPU launch (same as Architecture A, but achieved via batched gather instead of global gather). The streaming overhead only kicks in for truly extreme traces.

**Memory on rank 0**: K × (N/P) × 88 bytes (host) + K × (N/P) × 32 bytes (VRAM)
- Unlike Architecture A, this is **bounded by VRAM capacity**, not by N

**Implementation**: Same as Architecture B, but the "receive" step gathers K ranks' data at once (via K concurrent MPI_Irecv + MPI_Waitall or MPI_Gatherv to a sub-communicator).

**Pros**: All of Architecture B's pros, plus better GPU utilization for small-to-medium traces (fewer kernel launches)  
**Cons**: Slightly more complex batch management; host memory for K ranks may be non-trivial for large K

**When to use**: When maximum GPU throughput is desired and trace sizes vary widely. This naturally degrades to Architecture A (K=P) for small traces and to Architecture B (K=1) for extreme traces.

---

### Architecture D: Fully Distributed CPU Analysis (No GPU for Main Analysis)

```
ALL RANKS (P processes)            RANK 0
┌───────────────────────────┐     ┌────────────────────────┐
│ 1. Two-pass OTF2 read     │     │ Receive result vectors  │
│ 2. Local P2P matching     │     │ Concatenate durations   │
│ 3. Local coll grouping    │     │ Compute statistics      │
│ 4. Local ANALYSIS (CPU)   │────►│ Print results           │
│ 5. Send result vectors    │ MPI │                         │
│    O(N/P) per rank        │     │ Network: O(R) << O(N)   │
└───────────────────────────┘     └────────────────────────┘
```

**Core idea**: Perform the analysis entirely on CPU, per rank, then gather only the results (duration vectors, not raw events). This completely eliminates the CPU→GPU data transfer and the gather-all bottleneck.

**Memory**: O(N/P) per rank. No rank ever holds more than its share. Result vectors are typically <<N in size (only wait-state instances, not all events).

**CPU analysis implementation**: The analysis kernels are simple timestamp comparisons:
```cpp
// Late sender (CPU version):
for (size_t i = 0; i < n; i++) {
    if (is_recv(events[i]) && match_partner[i] >= 0) {
        if (timestamps[match_partner[i]] > timestamps[i])
            late_sender.push_back(timestamps[match_partner[i]] - timestamps[i]);
    }
}
```

**Performance estimate (500M events, P=64)**:
- Per rank: 7.8M events, single-threaded scan → ~10-50 ms (cache-friendly SoA)
- Total per rank (all 8 analyses): ~50-200 ms
- Compare with GPU: ~125 ms for 1M events → GPU advantage exists but is modest for small per-rank data

**Pros**:
- **Maximum scalability** — O(N/P) per rank, O(R) network transfer
- **Simplest architecture** — no CUDA streams, no GPU memory management, no gather
- **No VRAM limit** — can analyze arbitrarily large traces
- Works on systems without GPUs
- Matches TileTrace's distributed analysis philosophy

**Cons**:
- Slower per-rank analysis (CPU vs GPU) — but analysis is already <3% of total time
- Does not exercise the GPU at all for the main analysis (defeats project's stated goal)
- Loses the potential for GPU-accelerated sorting (future optimization)

**When to use**: When scalability is the primary concern and GPU acceleration is not required. Also useful as a **fallback mode** when GPU is unavailable or traces exceed VRAM even in streaming mode.

---

### Architecture E: Hybrid Distributed CPU + Selective GPU

```
ALL RANKS               RANK 0 (GPU owner)
┌──────────────────┐   ┌─────────────────────────────────────┐
│ 1. Read           │   │ 4a. Receive P2P pairs from all ranks│
│ 2. Match          │   │ 4b. GPU: analyze P2P (batched)      │
│ 3. Group          │   │                                     │
│ 4. LOCAL: analyze │   │ 5. Receive coll results (tiny)      │
│    collectives    │   │ 6. Merge P2P + coll results         │
│    (CPU per rank) │   │ 7. Statistics                        │
│ 5. Send results   │──►│                                     │
│    + P2P pairs    │   │ Memory: O(N_p2p) for P2P only       │
└──────────────────┘   └─────────────────────────────────────┘
```

**Core idea**: Split the analysis workload based on compute characteristics:
- **P2P analysis on GPU**: Millions of events, embarrassingly parallel, benefits most from GPU SIMD
- **Collective analysis on CPU**: Fewer events, sequential per-group, minimal GPU benefit

Only P2P-matched pairs (send_idx → recv_idx data) are streamed to rank 0 for GPU analysis. Collective results are computed locally on each rank's CPU and only the small result vectors are sent.

**What is transferred to rank 0**:
- P2P data: For each matched P2P pair, only 4 values needed: `timestamps[send]`, `timestamps[recv]`, `end_timestamps[send]`, `end_timestamps[recv]` = 32 bytes/pair. Much less than the full SoA (88 bytes/event including unmatched events and collectives).
- Collective results: Just duration vectors (doubles), size proportional to collective count.

**Memory**: O(N_p2p) on rank 0 for P2P data, where N_p2p < N (only matched pairs, not all events). Typically N_p2p ≈ 30-70% of N for P2P-heavy benchmarks like CG.

**Pros**:
- Transfers less data than Architecture B (P2P pairs only, not full SoA)
- GPU used where it matters most (P2P analysis = majority of events)
- Collective analysis doesn't need GPU (small groups processed sequentially)
- Good balance of GPU utilization and distributed scalability

**Cons**:
- Two different analysis pathways (GPU for P2P, CPU for collective) — more complex
- Must extract P2P-pair-specific data into a compact transfer format
- Loses the ability to do whole-trace GPU operations in the future

**When to use**: When network bandwidth is the limiting factor and minimizing data transfer is important.

---

## 4. Detailed Comparison

### 4.1 Memory Scalability

| Architecture | Rank 0 Host Memory | Rank 0 VRAM | Other Ranks |
|-------------|-------------------|-------------|-------------|
| A: Gather-All | O(N) | O(N) | O(N/P) |
| B: Streaming | O(N/P) × 2 | O(N/P) × 2 | O(N/P) |
| C: Batch | O(K×N/P) | O(K×N/P) | O(N/P) |
| D: CPU-only | O(R) ≪ O(N) | None | O(N/P) |
| E: Hybrid | O(N_p2p) | O(N_p2p/P) × 2 | O(N/P) |

Where R = total result durations, N_p2p = matched P2P events, K = batch size.

### 4.2 Network Transfer Volume

| Architecture | Total bytes from all non-zero ranks to rank 0 |
|-------------|----------------------------------------------|
| A: Gather-All | N × 88 bytes (full SoA + CSR) |
| B: Streaming | N × 88 bytes (same total, but streamed) |
| C: Batch | N × 88 bytes (same total, in K-rank batches) |
| D: CPU-only | R × 8 bytes (only result doubles) ≪ N |
| E: Hybrid | N_p2p × 32 bytes (P2P pairs) + R_coll × 8 (coll results) |

### 4.3 GPU Utilization

| Architecture | Kernel Launches | Events per Launch | GPU Idle Time |
|-------------|----------------|-------------------|---------------|
| A: Gather-All | 5 (one set) | N | Idle during gather |
| B: Streaming | 5 × P | N/P | Low (pipelined) |
| C: Batch | 5 × ceil(P/K) | K×N/P | Very low |
| D: CPU-only | 0 | — | 100% idle |
| E: Hybrid | 5 × batches (P2P only) | N_p2p per batch | Moderate |

### 4.4 Implementation Complexity

| Architecture | Complexity | New Components Needed |
|-------------|-----------|----------------------|
| A: Gather-All | Low (done) | None (already implemented) |
| B: Streaming | Medium | CUDA streams, double-buffer, MPI send/recv scheduling |
| C: Batch | Medium | Same as B + batch size computation |
| D: CPU-only | Low | CPU analysis kernels (trivial port) |
| E: Hybrid | Medium-High | Split analysis paths, P2P compaction, merge logic |

### 4.5 Correctness Risk

| Architecture | Risk | Notes |
|-------------|------|-------|
| A: Gather-All | None | Already validated against Scalasca |
| B: Streaming | Low | Same kernels, just batched; verify concatenated results match |
| C: Batch | Low | Same as B |
| D: CPU-only | Low | Must replicate exact same timestamp comparisons as GPU kernels |
| E: Hybrid | Medium | Two analysis paths must produce identical results |

---

## 5. Analysis: Where Should Each Stage Run?

### 5.1 OTF2 Reading — CPU, Distributed (All Architectures)

**Verdict**: CPU only. No alternative.

OTF2 reading is callback-driven, file-system bound, and requires the otf2xx library (CPU-only). Parallelism comes from MPI (multiple ranks reading different locations), not from GPU. Demonstrated 4.2× speedup with 8 ranks.

### 5.2 P2P Matching — CPU, Per-Rank (Current); GPU Possible (Future)

**Current**: CPU, sequential timestamp-sorted FIFO queues. O(N/P log(N/P)) for sort + O(N/P) for matching.

**GPU alternative (sort-based)**:
1. Separate sends and recvs into two arrays (GPU scatter: O(N/P))
2. Sort each by composite key (src, dst, tag, timestamp) using CUB RadixSort: O(N/P log(N/P)) but ~10-30× faster than CPU
3. Merge-join within key groups: O(N/P) GPU work

**Estimate**: CPU std::sort on 7.8M events ≈ 100-200ms. CUB RadixSort ≈ 5-20ms. Significant speedup for the sort stage. However, this optimization is **orthogonal to the architecture choice** — it applies to any of the five architectures.

**Verdict**: Keep on CPU for now (correct, simple), optimize to GPU later as a Phase 2 improvement.

### 5.3 Collective Grouping — CPU, Per-Rank (All Architectures)

**Verdict**: CPU only. 

Collective events are typically 1-5% of total events. The grouping algorithm is sequential (pending-group scan). GPU would add kernel launch overhead for minimal compute. The CPU time is already negligible (~1ms).

### 5.4 Wait-State Analysis — GPU (Architectures A/B/C/E) or CPU (Architecture D)

This is the core question. The analysis consists of:
- **P2P kernel**: grid-stride loop, one thread per event, ~32B read per event
- **Collective kernels**: one block per group, thread 0 sequential scan, ~16-32B per member

**P2P analysis** (Late Sender / Late Receiver):
- Embarrassingly parallel — each event pair is independent
- GPU advantage: for 7.8M events, GPU can process in ~2-10ms. CPU takes ~20-100ms.
- **GPU is 5-10× faster per batch**

**Collective analysis** (6 metrics):
- Per-group sequential scan (thread 0 only, other threads wasted)
- Groups are small (8-256 members typically)
- For 1000 groups × 64 members: GPU kernel ~0.5ms, CPU ~2ms
- **GPU advantage is modest (2-4×)**

**Verdict for analysis**: GPU provides meaningful speedup for P2P analysis. Marginal for collectives. The choice between GPU and CPU for analysis depends on whether total analysis time is significant relative to other stages.

### 5.5 Statistics — CPU (All Architectures)

**Verdict**: CPU only.

Statistics involve sorting result vectors (for quartiles) and computing sum/mean/variance. Result vectors are small (thousands to millions of doubles). CPU sort is <10ms. Not worth a GPU kernel launch.

---

## 6. Recommendations

### Primary Recommendation: Architecture C (Adaptive Batch Streaming)

Architecture C is recommended because it **naturally adapts** to all trace sizes:

- **Small traces** (CG Class B/C): K = P, all data fits in one GPU launch. Behavior identical to Architecture A but without the permanent O(N) host allocation (data is streamed in, not gathered).
- **Medium traces** (CG/BT Class D): K = P typically still works (40-500M events with 64 ranks → 0.6-7.8M events/rank → 250MB GPU per rank → all fit in 24 GB).
- **Large traces** (1B+ events): K < P, multiple batches. Memory bounded by K × N/P, not N.
- **Extreme traces** (10B+ events): K = 1, pure streaming. O(N/P) memory, always fits.

**Implementation plan**:
1. After all ranks complete read + match + group, rank 0 computes K based on available VRAM
2. For each batch of K ranks:
   a. K ranks MPI_Send their local SoA + CSR to rank 0 (or MPI_Gatherv within sub-communicator)
   b. Rank 0 receives into host buffer(s)
   c. H2D transfer (using CUDA stream for pipelining with next receive)
   d. Launch all 5 kernels on the batch data
   e. D2H transfer of result durations
   f. Append results to global vectors
   g. Free host buffer, GPU buffer
3. Rank 0 computes statistics on concatenated global result vectors

### Secondary Recommendation: Architecture D as Fallback

Implement a CPU-only analysis path (trivial code change — the analysis formulas are simple timestamp comparisons). This serves as:
- Fallback when no GPU is available
- Reference implementation for correctness validation
- Scalability benchmark (pure O(N/P) per rank, O(R) network)

### Future Enhancement: Architecture E for Network-Constrained Scenarios

When network bandwidth is the bottleneck (e.g., multi-node analysis of very large traces), the hybrid approach transfers less data by sending only P2P pair timestamps rather than full SoA arrays.

---

## 7. Key Design Decisions with Evidence

### 7.1 Why Not Centralized Gather (Architecture A) Long-Term?

**Evidence**: Memory math is unforgiving.

| Trace | Events | Rank 0 Host Memory (88B/event) | Rank 0 VRAM (32B/event) |
|-------|--------|-------------------------------|------------------------|
| CG.C | 1M | 88 MB | 32 MB |
| CG.D | 40M | 3.5 GB | 1.3 GB |
| BT.D | 500M | 44 GB | 16 GB |
| LAMMPS | 2B | 176 GB | 64 GB (exceeds 4090) |

For BT.D with 44 GB on rank 0 host, this requires nodes with large RAM but is feasible on the 1 TB server. However, the VRAM limit (24 GB on 4090) becomes the hard wall earlier — BT.D at 16 GB is tight when accounting for P2P output arrays (additional 16B/event = 8 GB for allocating n-sized output buffers).

### 7.2 Why Streaming Doesn't Hurt GPU Performance

**Concern**: P separate kernel launches (one per rank) might be slower than one large launch.

**Evidence**: CUDA kernel launch overhead is 5-20 μs. For P=128: 128 × 20μs = 2.5ms. Compare with kernel execution time of 2-5s for 500M events — launch overhead is <0.1%.

Furthermore, GPU SM (Streaming Multiprocessor) saturation occurs at surprisingly low event counts. The 4090 has 128 SMs. With 256 threads per block and grid-stride loops:
- 128 SMs × 4 blocks/SM (occupancy) = 512 active blocks
- Each block processes 256 events minimum → 131K events for full SM saturation
- Even 1M events (smallest practical batch) provides 8× more work than needed

### 7.3 Why P2P Analysis Is Independent Across Ranks

**Proof**: After the two-pass read, rank R holds:
- All send events from locations in `[start_R, end_R)`
- All recv events whose **sender** is in `[start_R, end_R)` (co-located by design)

A P2P match consists of a send `S` (from location `src`) and a recv `R` (at location `dst`). The match key is `(src, dst, tag)`. After two-pass read + matching:
- `S` is on rank `owningRank(src)` (it's a send from a local location)
- `R` is also on rank `owningRank(src)` (related location from pass 1)
- Therefore both events of every matched pair are on the same rank
- The analysis only needs `timestamps[S]`, `timestamps[R]`, `end_timestamps[S]`, `end_timestamps[R]`
- All four values are available locally → **analysis is local, no cross-rank dependency**

### 7.4 Why Collective Analysis Is Independent Across Ranks

**Proof**: After root-based collective redistribution, rank R holds all collective events whose root location is in `[start_R, end_R)`. Each collective group has one root, so all members of any group are co-located on the root's rank. The analysis only needs member timestamps, which are all present locally.

### 7.5 The Double-Buffer VRAM Budget

For Architecture C with double buffering:
- Buffer A (active kernel): K × N/P × 32 bytes (trace data) + K × N/P × 16 bytes (output arrays)
- Buffer B (receiving H2D): same as Buffer A
- CSR overhead: negligible (groups are small fraction of events)
- **Total VRAM**: 2 × K × N/P × 48 bytes

For 24 GB VRAM:
- Maximum K × N/P = 24 GB / 96 ≈ 250M events per double-buffered batch
- With P=64 and K=P: N/P × 64 = 250M → N ≤ 250M (fits CG.D easily, tight for BT.D)
- With K=32 (half of P=64): 32 × N/P ≤ 250M → N ≤ 500M (fits BT.D)
- With K=1 (full streaming): N/P ≤ 250M → N ≤ 16B (fits anything)

---

## 8. Complete Architecture Decision Table

| Criterion | A: Gather | B: Stream | C: Batch (rec.) | D: CPU-only | E: Hybrid |
|-----------|----------|-----------|-----------------|-------------|-----------|
| Memory scalability | Poor (O(N)) | Good (O(N/P)) | Good (O(KN/P)) | Best (O(N/P)) | Good |
| GPU utilization | Best (1 launch) | Good (pipelined) | Very good | None | Good |
| Network efficiency | Poor (gather all) | Same total | Same total | Best (results only) | Good (P2P only) |
| Implementation effort | Done | Medium | Medium | Easy | High |
| Correctness risk | None (validated) | Low | Low | Low | Medium |
| Max trace size (4090) | ~300M events | Unlimited | Unlimited | Unlimited | Unlimited |
| Max trace size (1TB RAM) | ~10B events | Unlimited | Unlimited | Unlimited | Unlimited |
| Analysis speedup vs CPU | 5-10× | 5-10× | 5-10× | 1× (CPU) | 3-7× |

---

## 9. Recommended Implementation Roadmap

### Phase 1: Validate Independent-Analysis Property

Before implementing streaming, verify that processing each rank's data independently on GPU produces identical results to the current gather-all approach. This requires:
1. Run current implementation (Architecture A) on CG.C
2. Modify main.cu so rank 0 processes each rank's data separately in a loop
3. Concatenate result vectors and compare with Architecture A output

If results match, the independence property is confirmed.

### Phase 2: Implement Architecture C (Adaptive Batch)

1. Add VRAM query utility: `cudaMemGetInfo(&free, &total)` to determine available VRAM
2. Compute optimal K based on per-rank event count and VRAM
3. Replace `gatherTraceData()` with `streamBatchToGPU()`
4. Add CUDA stream management for double-buffered operation
5. Accumulate result vectors across batches

### Phase 3: Add Architecture D (CPU Fallback)

1. Implement `runAnalysisCPU()` — simple loop-based versions of the 5 kernels
2. Add `--cpu-only` command-line flag
3. Use as correctness reference and GPU-free fallback

### Phase 4: Optimize P2P Matching on GPU (Optional)

1. GPU sort-based matching using CUB RadixSort
2. Can be combined with Architecture C (sort + match + analyze on GPU in one pipeline)
3. Eliminates the P2P matching CPU-GPU boundary

---

## 10. Appendix: Reference Numbers

### Hardware Specs

| Spec | RTX 4090 | 2× Xeon Gold 6530 |
|------|----------|-------------------|
| Compute units | 128 SMs, 16384 CUDA cores | 64 cores (128 threads) |
| Memory | 24 GB GDDR6X | 1 TB DDR5 |
| Bandwidth | 1 TB/s (VRAM) | ~100 GB/s (DDR5) |
| PCIe | 4.0 x16 (25 GB/s) | — |
| TDP | 450W | 2 × 300W |

### Data Transfer Rates

| Transfer | Rate | Time for 500M events (32B/event = 16 GB) |
|----------|------|------------------------------------------|
| PCIe 4.0 H2D | ~25 GB/s | 640 ms |
| InfiniBand HDR (node-to-node) | ~25 GB/s | 640 ms |
| NVMe SSD read | ~7 GB/s | 2.3 s |
| VRAM bandwidth | ~1 TB/s | 16 ms |

### Per-Event Memory Footprint

| Context | Bytes/Event |
|---------|-------------|
| Full SoA (CPU, 14 arrays) | 88 B |
| GPU-transferred subset (6 arrays) | 32 B |
| GPU output allocation (worst case) | 16 B |
| P2P pair data only (4 timestamps) | 32 B |
| TileTrace AoS (for comparison) | ~120+ B |
