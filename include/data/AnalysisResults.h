#ifndef GPU_ANALYZER_ANALYSIS_RESULTS_H
#define GPU_ANALYZER_ANALYSIS_RESULTS_H

#include <cstddef>

struct AnalysisResult {
  size_t count = 0;
  double sum = 0.0;
  double min_val = 0.0;
  double max_val = 0.0;
  double mean = 0.0;
  double variance = 0.0;
  double median = 0.0;
  double q25 = 0.0;
  double q75 = 0.0;
};

struct AllAnalysisResults {
  AnalysisResult late_sender;
  AnalysisResult late_receiver;
  AnalysisResult barrier_wait;
  AnalysisResult barrier_completion;
  AnalysisResult early_reduce;
  AnalysisResult late_broadcast;
  AnalysisResult wait_nxn;
  AnalysisResult nxn_completion;
};

#endif // GPU_ANALYZER_ANALYSIS_RESULTS_H
