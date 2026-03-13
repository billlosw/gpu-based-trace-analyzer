#include "analysis/Statistics.h"

#include <algorithm>
#include <cmath>
#include <vector>

AnalysisResult computeStatistics(const std::vector<double> &durations,
                                 double scaling_factor) {
  AnalysisResult r{};
  if (durations.empty())
    return r;

  r.count = durations.size();
  r.sum = 0.0;
  r.min_val = 1e18;
  r.max_val = 0.0;

  for (double d : durations) {
    double v = d / scaling_factor;
    r.sum += v;
    if (v < r.min_val)
      r.min_val = v;
    if (v > r.max_val)
      r.max_val = v;
  }

  r.mean = r.sum / (double)r.count;

  r.variance = 0.0;
  for (double d : durations) {
    double v = d / scaling_factor;
    r.variance += (v - r.mean) * (v - r.mean);
  }
  r.variance /= (double)r.count;

  // Sort for quartiles and median
  std::vector<double> sorted(durations.size());
  for (size_t i = 0; i < durations.size(); i++)
    sorted[i] = durations[i] / scaling_factor;
  std::sort(sorted.begin(), sorted.end());

  r.q25 = sorted[sorted.size() / 4];
  r.median = sorted[sorted.size() / 2];
  r.q75 = sorted[sorted.size() * 3 / 4];

  return r;
}
