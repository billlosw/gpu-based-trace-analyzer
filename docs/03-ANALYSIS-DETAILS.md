# Analysis Details

This document explains each of the 8 wait-state analyses in detail: what they measure, the formulas used, and the timestamp semantics that make them match Scalasca.

## Background: OTF2 Event Model

Score-P instruments MPI calls and records events in OTF2 format. Each MPI call produces multiple events:

```
Enter(MPI_Send)           ← Region entry (timestamp A)
  mpi_send                ← Point event inside the region (timestamp B, B >= A)
  [actual PMPI_Send call]
Leave(MPI_Send)           ← Region exit (timestamp C)
```

The gap between `Enter` and the point event includes Score-P's instrumentation overhead (typically nanoseconds).

### Timestamp Semantics (Critical for Correctness)

**Scalasca compares Enter timestamps**, not point-event timestamps. This project follows the same convention:

| MPI Call Type | Timestamp Used | Source in Code |
|---------------|---------------|----------------|
| `MPI_Send` (blocking) | `Enter(MPI_Send)` | `m_last_enter_ts[pid]` from `event(enter)` callback |
| `MPI_Isend` (non-blocking) | `Enter(MPI_Isend)` | `m_last_enter_ts[pid]` from `event(enter)` callback |
| `MPI_Recv` (blocking) | `Enter(MPI_Recv)` | `m_last_enter_ts[pid]` from `event(enter)` callback |
| `MPI_Irecv` + `MPI_Wait` | `Enter(MPI_Wait)` | `m_last_enter_ts[pid]` — the `mpi_ireceive_complete` event fires inside `Enter(MPI_Wait)/Leave(MPI_Wait)`, so `m_last_enter_ts` holds `Enter(MPI_Wait)` |
| Collective (begin/end) | `mpi_collective_begin` timestamp | Directly from the point event (begin marks when the process enters the collective) |

### Why Not Point-Event Timestamps?

Using point-event timestamps instead of Enter timestamps causes two problems (both were bugs that were fixed):

1. **Recv-side**: The `mpi_receive` point event fires AFTER the data arrives (≈ Leave time). Using it makes `recv_ts > send_ts` almost always true, producing 0 late_sender and inflated late_receiver.

2. **Send-side**: The `mpi_send` point event fires slightly after `Enter(MPI_Send)` due to instrumentation overhead. For pairs near the threshold, this creates false late_sender positives (3-6% over-count vs Scalasca).

## Analysis 1: Late Sender

**Purpose**: Detect when the sender arrived at its MPI_Send after the receiver was already waiting.

**Formula**: `send_enter - recv_enter` when `send_enter > recv_enter`

**Implementation** (in `kernelLateSenderReceiver`):
```
For each matched Recv event i:
    send_idx = match_partner[i]
    recv_enter = timestamps[i]        // Enter(MPI_Recv) or Enter(MPI_Wait)
    send_enter = timestamps[send_idx] // Enter(MPI_Send) or Enter(MPI_Isend)

    if (send_enter > recv_enter):
        late_sender_duration = send_enter - recv_enter
```

**Scalasca match**: Exact (count, sum, mean exact; sum within 1 picosecond from FP rounding).

**Trick**: Only Recv/Irecv events are iterated (not Send events) to avoid double-counting each pair.

## Analysis 2: Late Receiver

**Purpose**: Detect when the receiver arrived at its MPI_Recv/MPI_Wait after the sender had already called MPI_Send.

**Formula**: `recv_enter - send_enter` when `recv_enter > send_enter`

**Implementation**: Same kernel as Late Sender, opposite branch.

**Scalasca match**: Over-counts by 3-4x. Our tool classifies ALL pairs where `recv_enter > send_enter` as late_receiver. Scalasca only counts pairs where the MPI call actually blocked (i.e., had nonzero wait time). For CG's `MPI_Irecv + MPI_Send + MPI_Wait` pattern, many `MPI_Wait` calls return immediately because data already arrived, but our tool still counts them as late_receiver.

**Why this is acceptable**: The late_sender metric is the one that identifies genuine performance bottlenecks (the receiver was waiting and the sender was slow). Late_receiver is less actionable — it just means the receiver started waiting after the sender sent, which is often the expected/normal case.

## Analysis 3: Barrier Wait

**Purpose**: Measure how long each process waited at an MPI_Barrier for the slowest process to arrive.

**Formula**: For each member in a barrier group:
```
max_enter = max(timestamps[member] for all members in group)
barrier_wait = max_enter - timestamps[this_member]    (if > 0)
```

**Implementation** (in `kernelBarrierWaitCompletion`):
- One CUDA block per barrier group
- Thread 0 does all work (groups are small: typically 2-256 members)
- First pass: find `max_enter`
- Second pass: compute each member's wait time

**Scalasca match**: Exact (count and sum).

**Applies to**: Only `TT_MPI_Barrier` events.

## Analysis 4: Barrier Completion

**Purpose**: Measure the completion imbalance at MPI_Barrier — how much longer some processes take to finish compared to the fastest.

**Formula**: For each member in a barrier group:
```
min_end = min(end_timestamps[member] for all members in group)
barrier_completion = end_timestamps[this_member] - min_end    (if > 0)
```

**Implementation**: Same kernel as Barrier Wait, computed in the same pass.

**Scalasca match**: Exact.

## Analysis 5: Early Reduce

**Purpose**: Detect when the root process arrived early at a Reduce/Gather operation, meaning it waited idle for other processes.

**Formula**: For each Reduce/Gather/Gatherv group:
```
root_ts = timestamps[root_member]
max_ts = max(timestamps[member] for all members)
early_reduce = max_ts - root_ts    (if max_ts > root_ts)
```

One value per group (not per member).

**Implementation** (in `kernelEarlyReduce`):
- One block per group, thread 0 does work
- Iterates members to find root and max timestamps

**Scalasca match**: Exact.

**Applies to**: `TT_MPI_Reduce`, `TT_MPI_Gather`, `TT_MPI_Gatherv`.

## Analysis 6: Late Broadcast

**Purpose**: Detect when the root process arrived late at a Bcast/Scatter, making early-arriving members wait.

**Formula**: For each Bcast/Scatter/Scatterv group:
```
root_ts = timestamps[root_member]
For each non-root member where member_ts < root_ts:
    late_broadcast = root_ts - member_ts
```

Multiple values per group (one per early-arriving member).

**Implementation** (in `kernelLateBroadcast`):
- One block per group, thread 0 does work
- First pass: find root timestamp
- Second pass: compute delay for each early member

**Scalasca match**: Exact.

**Applies to**: `TT_MPI_Bcast`, `TT_MPI_Scatter`, `TT_MPI_Scatterv`.

## Analysis 7: Wait NxN

**Purpose**: Same as Barrier Wait but for N-to-N collective operations.

**Formula**: Identical to Barrier Wait.

**Applies to**: `TT_MPI_Reduce_Scatter`, `TT_MPI_Reduce_Scatter_Block`, `TT_MPI_All_Gather`, `TT_MPI_All_Gatherv`, `TT_MPI_All_Reduce`, `TT_MPI_AlltoAll`.

**Implementation** (in `kernelNxNWaitCompletion`): Same logic as barrier, different event type filter.

**Scalasca match**: Exact.

## Analysis 8: NxN Completion

**Purpose**: Same as Barrier Completion but for N-to-N collectives.

**Formula**: Identical to Barrier Completion.

**Implementation**: Same kernel as Wait NxN, computed in the same pass.

**Scalasca match**: Exact.

## Kernel-to-Analysis Mapping

5 CUDA kernels implement 8 analyses:

| Kernel | Analyses | Strategy |
|--------|----------|----------|
| `kernelLateSenderReceiver` | Late Sender + Late Receiver | One thread per event, grid-stride loop |
| `kernelBarrierWaitCompletion` | Barrier Wait + Barrier Completion | One block per group, thread 0 sequential |
| `kernelEarlyReduce` | Early Reduce | One block per group, thread 0 sequential |
| `kernelLateBroadcast` | Late Broadcast | One block per group, thread 0 sequential |
| `kernelNxNWaitCompletion` | Wait NxN + NxN Completion | One block per group, thread 0 sequential |

## Output Units

- Internal representation: **picoseconds** (`uint64_t`, from OTF2)
- Analysis output: **picoseconds** (as `double` in `RawAnalysisOutput`)
- Statistics output: **seconds** (divided by `1e12` scaling factor in `computeStatistics`)

## Collective Type Classification

```
Barrier-type:     MPI_Barrier
Root-Reduce-type: MPI_Reduce, MPI_Gather, MPI_Gatherv
Root-Bcast-type:  MPI_Bcast, MPI_Scatter, MPI_Scatterv
NxN-type:         MPI_Reduce_Scatter, MPI_Reduce_Scatter_Block,
                  MPI_All_Gather, MPI_All_Gatherv,
                  MPI_All_Reduce, MPI_AlltoAll
```
