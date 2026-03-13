#include "analysis/Statistics.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void test_basic_statistics() {
  std::cout << "[test_basic_statistics] ";

  // Durations in picoseconds: 1e12, 2e12, 3e12, 4e12
  // In seconds: 1, 2, 3, 4
  std::vector<double> durations = {1e12, 2e12, 3e12, 4e12};
  auto r = computeStatistics(durations);

  assert(r.count == 4);
  assert(fabs(r.sum - 10.0) < 1e-6);      // 1+2+3+4 = 10
  assert(fabs(r.mean - 2.5) < 1e-6);      // 10/4 = 2.5
  assert(fabs(r.min_val - 1.0) < 1e-6);   // 1
  assert(fabs(r.max_val - 4.0) < 1e-6);   // 4
  assert(fabs(r.median - 3.0) < 1e-6);    // sorted[2] = 3
  assert(fabs(r.q25 - 1.0) < 1e-6);       // sorted[1] = 2 -> sorted[4/4]=sorted[1]=2? No: size/4=1, sorted[1]=2
  // Actually: size=4, sorted[4/4]=sorted[1]=2.0
  // But wait: q25 = sorted[size/4] = sorted[1] = 2.0
  // Let me re-check...
  // sorted = {1, 2, 3, 4}, size=4
  // q25 = sorted[4/4] = sorted[1] = 2.0
  // median = sorted[4/2] = sorted[2] = 3.0
  // q75 = sorted[4*3/4] = sorted[3] = 4.0

  assert(fabs(r.q25 - 2.0) < 1e-6);
  assert(fabs(r.q75 - 4.0) < 1e-6);

  // variance = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4
  //          = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 5.0/4 = 1.25
  assert(fabs(r.variance - 1.25) < 1e-6);

  std::cout << "PASSED" << std::endl;
}

void test_empty() {
  std::cout << "[test_empty] ";

  std::vector<double> durations;
  auto r = computeStatistics(durations);
  assert(r.count == 0);
  assert(r.sum == 0.0);

  std::cout << "PASSED" << std::endl;
}

void test_single_value() {
  std::cout << "[test_single_value] ";

  std::vector<double> durations = {5e12};
  auto r = computeStatistics(durations);

  assert(r.count == 1);
  assert(fabs(r.sum - 5.0) < 1e-6);
  assert(fabs(r.mean - 5.0) < 1e-6);
  assert(fabs(r.min_val - 5.0) < 1e-6);
  assert(fabs(r.max_val - 5.0) < 1e-6);
  assert(fabs(r.variance) < 1e-6);

  std::cout << "PASSED" << std::endl;
}

int main() {
  std::cout << "=== Statistics Tests ===" << std::endl;
  test_basic_statistics();
  test_empty();
  test_single_value();
  std::cout << "All statistics tests passed!" << std::endl;
  return 0;
}
