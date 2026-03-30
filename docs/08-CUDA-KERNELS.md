# CUDA Kernels

**Source**: `src/analysis/AnalysisKernels.cu`
**Header**: `include/analysis/AnalysisKernels.h`

## Overview

5 CUDA kernels implement 8 wait-state analyses. Two host functions orchestrate GPU memory management, transfers, kernel launches, and result retrieval:

- `runAnalysisKernels()` — Synchronous variant. Allocates/frees device memory per call. Used by single-rank path and multi-node fallback.
- `runAnalysisKernelsAsync()` — Uses pre-allocated `GPUMemoryPool` and `cudaStream_t` with `cudaMemcpyAsync`. Used by the shared-memory batched GPU path. Eliminates per-batch allocation overhead.

## Kernel Designs

### Kernel 1: `kernelLateSenderReceiver` (P2P Analyses)

```cuda
__global__ void kernelLateSenderReceiver(
    const event_t *events,
    const timestamp_t *timestamps,
    const timestamp_t *end_timestamps,
    const int32_t *match_partner,
    size_t n,
    double *late_sender_out,    unsigned int *late_sender_cnt,
    double *late_receiver_out,  unsigned int *late_receiver_cnt)
```

**Strategy**: One thread per event, grid-stride loop.

**Launch Config**: `<<<min((n+255)/256, 1024), 256>>>`

**Logic**:
```
for each event i (grid-stride):
    if events[i] is not Recv/Irecv → skip
    if match_partner[i] == -1 → skip (unmatched)

    send_idx = match_partner[i]
    recv_enter = timestamps[i]
    send_enter = timestamps[send_idx]

    // Late sender (independent check)
    if send_enter > recv_enter:
        pos = atomicAdd(late_sender_cnt, 1)
        late_sender_out[pos] = (double)(send_enter - recv_enter)

    // Late receiver (independent check, NOT mutually exclusive with late sender)
    #ifdef USE_SCALASCA_TIMESTAMPS:
        send_leave = end_timestamps[send_idx]   // Leave(MPI_Send)
        recv_req_enter = end_timestamps[i]      // Enter(MPI_Irecv)
        if send_leave > recv_req_enter AND recv_req_enter > send_enter:
            pos = atomicAdd(late_receiver_cnt, 1)
            late_receiver_out[pos] = (double)(recv_req_enter - send_enter)
    #else:
        if recv_enter > send_enter:
            pos = atomicAdd(late_receiver_cnt, 1)
            late_receiver_out[pos] = (double)(recv_enter - send_enter)
```

**Trick**: Only iterates Recv/Irecv events to avoid double-counting. Each send-recv pair has exactly one Recv event, so each pair is processed exactly once.

**Atomic output**: Uses `atomicAdd` on a device counter to get the insertion position. This is safe because each event produces at most one output, and the counter is monotonically increasing. The output array is NOT position-ordered, but that doesn't matter because `computeStatistics` sorts later.

### Kernel 2: `kernelBarrierWaitCompletion` (Barrier Analyses)

```cuda
__global__ void kernelBarrierWaitCompletion(
    const timestamp_t *timestamps,
    const timestamp_t *end_timestamps,
    const int32_t *coll_offsets,
    const int32_t *coll_members,
    const event_t *group_types,
    size_t num_groups,
    double *barrier_wait_out,        unsigned int *barrier_wait_cnt,
    double *barrier_completion_out,  unsigned int *barrier_completion_cnt)
```

**Strategy**: One block per group. Thread 0 does all work sequentially.

**Launch Config**: `<<<num_groups, 1>>>`

**Why `blockDim=1`?** Collective groups are small (typically 2-256 members for HPC communicators). Using just thread 0 avoids warp divergence and synchronization overhead. For groups with thousands of members (rare), a parallelized reduction within the block would be faster. See [PROBLEMS.md](./PROBLEMS.md) for this optimization opportunity.

**Logic**:
```
if group_types[gid] != MPI_Barrier → return
if group_size < 2 → return

// Pass 1: Find max enter time and min end time
for j in [offsets[gid], offsets[gid+1]):
    max_enter = max(max_enter, timestamps[members[j]])
    min_end   = min(min_end, end_timestamps[members[j]])

// Pass 2: Compute each member's wait and completion
for j in [offsets[gid], offsets[gid+1]):
    if max_enter > timestamps[members[j]]:
        barrier_wait = max_enter - timestamps[members[j]]
    if end_timestamps[members[j]] > min_end:
        barrier_completion = end_timestamps[members[j]] - min_end
```

### Kernel 3: `kernelEarlyReduce` (Reduce/Gather)

```cuda
__global__ void kernelEarlyReduce(
    const timestamp_t *timestamps,
    const id_t *pids, const id_t *roots,
    const int32_t *coll_offsets, const int32_t *coll_members,
    const event_t *group_types, const id_t *group_roots,
    size_t num_groups,
    double *early_reduce_out, unsigned int *early_reduce_cnt)
```

**Strategy**: One block per group, thread 0 only.

**Logic**:
```
if type not in {Reduce, Gather, Gatherv} → return

Find root_ts (timestamp of member where pid == root_pid)
Find max_ts (max timestamp of all members)
if max_ts > root_ts:
    early_reduce = max_ts - root_ts  (one value per group)
```

**Key difference from barrier**: Produces at most **one** value per group (the root's wait time), not one per member.

### Kernel 4: `kernelLateBroadcast` (Bcast/Scatter)

**Strategy**: One block per group, thread 0 only.

**Logic**:
```
if type not in {Bcast, Scatter, Scatterv} → return

Find root_ts (timestamp of root member)
For each non-root member where member_ts < root_ts:
    late_broadcast = root_ts - member_ts
```

Produces one value per early-arriving member (potentially multiple per group).

### Kernel 5: `kernelNxNWaitCompletion` (N-to-N Collectives)

Identical logic to `kernelBarrierWaitCompletion` but filters for N-to-N collective types:
- `MPI_Reduce_Scatter`, `MPI_Reduce_Scatter_Block`
- `MPI_All_Gather`, `MPI_All_Gatherv`
- `MPI_All_Reduce`, `MPI_AlltoAll`

## Host Functions

### `runAnalysisKernels()` — Synchronous Variant

Allocates all device memory via `cudaMalloc`, copies data H2D synchronously, launches all 5 kernels, copies results D2H, and frees device memory. Used for single-rank and multi-node fallback paths.

### `runAnalysisKernelsAsync()` — Pool + Stream Variant

Uses a pre-allocated `GPUMemoryPool` (see [04-DATA-STRUCTURES.md](./04-DATA-STRUCTURES.md)) and a `cudaStream_t` for asynchronous operations. All `cudaMemcpyAsync` calls and kernel launches are ordered on the stream. Sets `gpu_alloc_ms = 0` since the pool is pre-allocated.

### GPU Memory Allocation Strategy

```
For trace data:
    d_events       = n * sizeof(event_t)      = n * 4 bytes
    d_timestamps   = n * sizeof(timestamp_t)   = n * 8 bytes
    d_end_timestamps = n * sizeof(timestamp_t) = n * 8 bytes
    d_match        = n * sizeof(int32_t)       = n * 4 bytes
    d_pids         = n * sizeof(id_t)          = n * 4 bytes
    d_roots        = n * sizeof(id_t)          = n * 4 bytes
    Total per event: 32 bytes on GPU

For P2P output:
    d_ls_out = n * sizeof(double)  // max possible = every event is late sender
    d_lr_out = n * sizeof(double)

For collective CSR:
    d_coll_offsets = (num_groups + 1) * 4 bytes
    d_coll_members = total_members * 4 bytes
    d_group_types  = num_groups * 4 bytes
    d_group_roots  = num_groups * 4 bytes

For collective output:
    Each output array = total_members * sizeof(double)
```

**Output array sizing**: P2P output arrays are allocated as `n * sizeof(double)` (worst case: every event produces a result). Collective output arrays use `total_members * sizeof(double)`. After kernel execution, only the actual count of results is copied back.

### Transfer and Execution Timeline

```
[H2D: events, timestamps, end_timestamps, match, pids, roots]
[P2P kernel: kernelLateSenderReceiver]
[D2H: P2P results]
[H2D: CSR data (offsets, members, group_types, group_roots)]
[Collective kernels: 4 kernels launched sequentially]
[D2H: collective results]
[Free all device memory]
```

### CUDA Event Timing

The function uses CUDA events for sub-phase timing:

```cpp
cudaEventRecord(ev_start);
// ... H2D transfers
cudaEventRecord(ev_h2d_done);
// ... P2P kernel
cudaEventRecord(ev_p2p_done);
// ... collective kernels
cudaEventRecord(ev_coll_done);
// ... later
cudaEventElapsedTime(&output.h2d_ms, ev_start, ev_h2d_done);
cudaEventElapsedTime(&output.p2p_kernel_ms, ev_h2d_done, ev_p2p_done);
```

### Performance Profile (LAMMPS n1024, 262M events, RTX 4090, SHM + GPUMemoryPool)

| Sub-phase | Time | Notes |
|-----------|------|-------|
| Pool allocation | ~2.2 ms | One-time, eliminates per-batch cudaMalloc/cudaFree |
| Batch prep (match_partner remap + CSR) | ~523 ms | CPU work before GPU |
| H2D transfer | ~447 ms | 32 bytes/event, pinned DMA |
| P2P kernel | ~9 ms | Grid-stride, 256 blocks × 256 threads |
| Collective kernels | ~88 ms | 186 groups × 1 thread each |
| D2H transfer | ~1 ms | Only actual results |
| **GPU total (batches)** | **~1,237 ms** | Including prep + pin overhead |

## Memory Access Patterns

### P2P Kernel (Coalesced)

Each thread in a warp accesses consecutive events:
```
thread 0: events[warp_base + 0], timestamps[warp_base + 0]
thread 1: events[warp_base + 1], timestamps[warp_base + 1]
...
thread 31: events[warp_base + 31], timestamps[warp_base + 31]
```

This produces coalesced memory transactions (128-byte aligned reads).

**Non-coalesced access**: When a Recv event accesses `timestamps[match_partner[i]]`, the partner indices are essentially random — each thread reads from a different, non-contiguous location. This is inherently non-coalesced but unavoidable for P2P matching.

### Collective Kernels (Sequential within Groups)

Only thread 0 runs, accessing contiguous member indices:
```
for j in [offsets[gid], offsets[gid+1]):
    timestamps[members[j]]  // members are sequential SoA indices
```

The `members[j]` values are not necessarily contiguous in the SoA (different processes' events are interleaved), so the actual timestamp reads are scattered.

## Error Handling

All CUDA API calls are wrapped with `CUDA_CHECK`:

```cpp
#define CUDA_CHECK(call) do {
    cudaError_t err = (call);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n",
                __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
} while(0)
```

This provides early failure with file/line information instead of silent corruption.
