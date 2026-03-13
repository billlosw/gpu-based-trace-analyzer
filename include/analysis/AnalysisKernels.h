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
};

RawAnalysisOutput runAnalysisKernels(const TraceDataSoA &data,
                                     const CollectiveGroupCSR &csr);

#endif // GPU_ANALYZER_ANALYSIS_KERNELS_H
