#ifndef GPU_ANALYZER_ANALYSIS_KERNELS_H
#define GPU_ANALYZER_ANALYSIS_KERNELS_H

#include "common/types.h"
#include "data/AnalysisResults.h"
#include "data/TraceDataSoA.h"
#include <vector>

#include <cuda_runtime.h>

// Run all 8 analyses on GPU. Data must already have match_partner filled.
// Returns durations for each analysis type (in picoseconds, unscaled).
struct RawAnalysisOutput {
  std::vector<double> late_sender;
  std::vector<double> late_receiver;
  std::vector<double> barrier_wait;
  std::vector<double> barrier_completion;
  std::vector<double> early_reduce;
  std::vector<double> late_broadcast;
  std::vector<double> wait_nxn;
  std::vector<double> nxn_completion;

  // Sub-phase timing (ms) — GPU-side (cuda events)
  float h2d_ms = 0;
  float p2p_kernel_ms = 0;
  float coll_kernel_ms = 0;
  float d2h_ms = 0;
  float gpu_alloc_ms = 0;   // cudaMalloc time
  float gpu_free_ms = 0;    // cudaFree time

  // Sub-phase timing (ms) — host-side (only used in batched GPU path)
  float pin_ms = 0;         // cudaHostRegister per batch
  float unpin_ms = 0;       // cudaHostUnregister per batch
  float batch_prep_ms = 0;  // match_partner remap + CSR batch construction
};

// Pre-allocated GPU memory pool for reuse across batches
struct GPUMemoryPool {
  // Trace input arrays (sized for max batch)
  event_t *d_events = nullptr;
  timestamp_t *d_timestamps = nullptr;
  timestamp_t *d_end_timestamps = nullptr;
  int32_t *d_match = nullptr;
  id_t *d_pids = nullptr;
  id_t *d_roots = nullptr;

  // P2P output arrays
  double *d_ls_out = nullptr;
  double *d_lr_out = nullptr;
  unsigned int *d_ls_cnt = nullptr;
  unsigned int *d_lr_cnt = nullptr;

  // Collective input arrays
  int32_t *d_coll_offsets = nullptr;
  int32_t *d_coll_members = nullptr;
  event_t *d_group_types = nullptr;
  id_t *d_group_roots = nullptr;

  // Collective output arrays
  double *d_bw_out = nullptr, *d_bc_out = nullptr;
  double *d_er_out = nullptr, *d_lb_out = nullptr;
  double *d_wn_out = nullptr, *d_nc_out = nullptr;
  unsigned int *d_bw_cnt = nullptr, *d_bc_cnt = nullptr;
  unsigned int *d_er_cnt = nullptr, *d_lb_cnt = nullptr;
  unsigned int *d_wn_cnt = nullptr, *d_nc_cnt = nullptr;

  // Allocated capacities
  size_t max_events = 0;
  size_t max_coll_members = 0;
  size_t max_coll_groups = 0;

  void allocate(size_t max_n, size_t max_members, size_t max_groups);
  void deallocate();
  ~GPUMemoryPool() { deallocate(); }
};

RawAnalysisOutput runAnalysisKernels(const TraceDataSoA &data,
                                     const CollectiveGroupCSR &csr);

// Stream-based variant: uses pre-allocated GPU memory pool and a specific stream
RawAnalysisOutput runAnalysisKernelsAsync(const TraceDataSoA &data,
                                          const CollectiveGroupCSR &csr,
                                          GPUMemoryPool &pool,
                                          cudaStream_t stream);

#endif // GPU_ANALYZER_ANALYSIS_KERNELS_H
