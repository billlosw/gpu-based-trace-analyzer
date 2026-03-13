#ifndef GPU_ANALYZER_STATISTICS_H
#define GPU_ANALYZER_STATISTICS_H

#include "data/AnalysisResults.h"
#include <vector>

// Compute summary statistics from a vector of durations.
// scaling_factor: divide durations by this (e.g., 1e12 for ps->s).
AnalysisResult computeStatistics(const std::vector<double> &durations,
                                 double scaling_factor = 1e12);

#endif // GPU_ANALYZER_STATISTICS_H
