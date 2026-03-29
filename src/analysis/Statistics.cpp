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
  size_t n = r.count;
  double inv_scale = 1.0 / scaling_factor;

  // Single pass: sum, min, max
  r.sum = 0.0;
  r.min_val = durations[0] * inv_scale;
  r.max_val = r.min_val;

  for (size_t i = 0; i < n; i++) {
    double v = durations[i] * inv_scale;
    r.sum += v;
    if (v < r.min_val)
      r.min_val = v;
    if (v > r.max_val)
      r.max_val = v;
  }
  r.mean = r.sum / (double)n;

  // Variance pass
  r.variance = 0.0;
  for (size_t i = 0; i < n; i++) {
    double v = durations[i] * inv_scale - r.mean;
    r.variance += v * v;
  }
  r.variance /= (double)n;

  // Use nth_element for O(n) quartile/median computation
  // Avoids O(n log n) sort — 10-50x faster for large arrays
  std::vector<double> data(n);
  for (size_t i = 0; i < n; i++)
    data[i] = durations[i] * inv_scale;

  size_t q25_idx = n / 4;
  size_t med_idx = n / 2;
  size_t q75_idx = n * 3 / 4;

  // nth_element partitions so that the element at the given position
  // is the element that would be there if the array were sorted.
  // Do them in order to progressively narrow the range.
  std::nth_element(data.begin(), data.begin() + q25_idx, data.end());
  r.q25 = data[q25_idx];

  // After nth_element for q25, elements before q25_idx are <= data[q25_idx]
  // so we only need to search from q25_idx onwards for median
  std::nth_element(data.begin() + q25_idx, data.begin() + med_idx, data.end());
  r.median = data[med_idx];

  std::nth_element(data.begin() + med_idx, data.begin() + q75_idx, data.end());
  r.q75 = data[q75_idx];

  return r;
}
