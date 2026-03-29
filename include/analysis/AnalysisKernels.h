#ifndef GPU_ANALYZER_ANALYSIS_KERNELS_H
#define GPU_ANALYZER_ANALYSIS_KERNELS_H

#include "common/types.h"
#include "data/AnalysisResults.h"
#include "data/TraceDataSoA.h"
#include <vector>

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

RawAnalysisOutput runAnalysisKernels(const TraceDataSoA &data,
                                     const CollectiveGroupCSR &csr);

#endif // GPU_ANALYZER_ANALYSIS_KERNELS_H
