#include "analysis/AnalysisKernels.h"
#include "data/TraceDataSoA.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// Helper to set up a basic collective group CSR with one group
static CollectiveGroupCSR makeCSR(const std::vector<int32_t> &members,
                                  event_t type, id_t root) {
  CollectiveGroupCSR csr;
  csr.num_groups = 1;
  csr.total_members = members.size();
  csr.offsets = (int32_t *)malloc(2 * sizeof(int32_t));
  csr.offsets[0] = 0;
  csr.offsets[1] = (int32_t)members.size();
  csr.members = (int32_t *)malloc(members.size() * sizeof(int32_t));
  for (size_t i = 0; i < members.size(); i++)
    csr.members[i] = members[i];
  csr.group_types = (event_t *)malloc(sizeof(event_t));
  csr.group_types[0] = type;
  csr.group_roots = (id_t *)malloc(sizeof(id_t));
  csr.group_roots[0] = root;
  return csr;
}

// Test late sender: send_ts > recv_ts
void test_late_sender() {
  std::cout << "[test_late_sender] ";

  TraceDataSoA data;
  data.allocate(2);
  data.count = 2;

  // Event 0: Send (pid=0, ts=5000)
  data.events[0] = TT_MPI_Send;
  data.timestamps[0] = 5000;
  data.pids[0] = 0;

  // Event 1: Recv (pid=1, ts=2000)
  data.events[1] = TT_MPI_Recv;
  data.timestamps[1] = 2000;
  data.pids[1] = 1;

  // Manually set matching: 0<->1
  data.match_partner[0] = 1;
  data.match_partner[1] = 0;

  CollectiveGroupCSR csr; // empty
  auto raw = runAnalysisKernels(data, csr);

  assert(raw.late_sender.size() == 1);
  assert(raw.late_receiver.size() == 0);
  assert(fabs(raw.late_sender[0] - 3000.0) < 1e-6);

  std::cout << "PASSED" << std::endl;
}

// Test late receiver: recv_ts > send_ts
void test_late_receiver() {
  std::cout << "[test_late_receiver] ";

  TraceDataSoA data;
  data.allocate(2);
  data.count = 2;

  // Event 0: Send (pid=0, ts=1000)
  data.events[0] = TT_MPI_Send;
  data.timestamps[0] = 1000;
  data.pids[0] = 0;

  // Event 1: Recv (pid=1, ts=4000)
  data.events[1] = TT_MPI_Recv;
  data.timestamps[1] = 4000;
  data.pids[1] = 1;

  data.match_partner[0] = 1;
  data.match_partner[1] = 0;

  CollectiveGroupCSR csr;
  auto raw = runAnalysisKernels(data, csr);

  assert(raw.late_sender.size() == 0);
  assert(raw.late_receiver.size() == 1);
  assert(fabs(raw.late_receiver[0] - 3000.0) < 1e-6);

  std::cout << "PASSED" << std::endl;
}

// Test barrier wait: 3 members with timestamps 1000, 2000, 5000
// max_enter=5000. wait = (5000-1000) + (5000-2000) = 4000 + 3000 = 7000
void test_barrier_wait() {
  std::cout << "[test_barrier_wait] ";

  TraceDataSoA data;
  data.allocate(3);
  data.count = 3;

  // 3 barrier events with different enter timestamps
  for (int i = 0; i < 3; i++) {
    data.events[i] = TT_MPI_Barrier;
    data.pids[i] = (id_t)i;
    data.roots[i] = 0;
    data.end_timestamps[i] = 10000; // same end
  }
  data.timestamps[0] = 1000;
  data.timestamps[1] = 2000;
  data.timestamps[2] = 5000;

  // All have same end timestamp -> no barrier completion
  auto csr = makeCSR({0, 1, 2}, TT_MPI_Barrier, 0);
  auto raw = runAnalysisKernels(data, csr);

  // barrier_wait: (5000-1000)=4000 and (5000-2000)=3000. Not for 5000-5000=0.
  assert(raw.barrier_wait.size() == 2);
  double sum = 0;
  for (auto v : raw.barrier_wait)
    sum += v;
  assert(fabs(sum - 7000.0) < 1e-6);

  std::cout << "PASSED" << std::endl;
}

// Test barrier completion: 3 members with end timestamps 6000, 8000, 10000
// min_end=6000. completion = (8000-6000) + (10000-6000) = 2000 + 4000 = 6000
void test_barrier_completion() {
  std::cout << "[test_barrier_completion] ";

  TraceDataSoA data;
  data.allocate(3);
  data.count = 3;

  for (int i = 0; i < 3; i++) {
    data.events[i] = TT_MPI_Barrier;
    data.pids[i] = (id_t)i;
    data.roots[i] = 0;
    data.timestamps[i] = 1000; // same enter
  }
  data.end_timestamps[0] = 6000;
  data.end_timestamps[1] = 8000;
  data.end_timestamps[2] = 10000;

  auto csr = makeCSR({0, 1, 2}, TT_MPI_Barrier, 0);
  auto raw = runAnalysisKernels(data, csr);

  assert(raw.barrier_completion.size() == 2);
  double sum = 0;
  for (auto v : raw.barrier_completion)
    sum += v;
  assert(fabs(sum - 6000.0) < 1e-6);

  std::cout << "PASSED" << std::endl;
}

// Test early reduce: root_ts=1000, member_ts=3000, member_ts=5000
// max=5000, early_reduce = 5000-1000=4000
void test_early_reduce() {
  std::cout << "[test_early_reduce] ";

  TraceDataSoA data;
  data.allocate(3);
  data.count = 3;

  // pid 0 is root
  data.events[0] = TT_MPI_Reduce;
  data.pids[0] = 0;
  data.roots[0] = 0;
  data.timestamps[0] = 1000;
  data.end_timestamps[0] = 10000;

  data.events[1] = TT_MPI_Reduce;
  data.pids[1] = 1;
  data.roots[1] = 0;
  data.timestamps[1] = 3000;
  data.end_timestamps[1] = 10000;

  data.events[2] = TT_MPI_Reduce;
  data.pids[2] = 2;
  data.roots[2] = 0;
  data.timestamps[2] = 5000;
  data.end_timestamps[2] = 10000;

  auto csr = makeCSR({0, 1, 2}, TT_MPI_Reduce, 0);
  auto raw = runAnalysisKernels(data, csr);

  assert(raw.early_reduce.size() == 1);
  assert(fabs(raw.early_reduce[0] - 4000.0) < 1e-6);

  std::cout << "PASSED" << std::endl;
}

// Test late broadcast: root_ts=5000, members at 1000, 3000
// late_bcast = (5000-1000) + (5000-3000) = 4000 + 2000 = 6000
void test_late_broadcast() {
  std::cout << "[test_late_broadcast] ";

  TraceDataSoA data;
  data.allocate(3);
  data.count = 3;

  // pid 0 is root
  data.events[0] = TT_MPI_Bcast;
  data.pids[0] = 0;
  data.roots[0] = 0;
  data.timestamps[0] = 5000;
  data.end_timestamps[0] = 10000;

  data.events[1] = TT_MPI_Bcast;
  data.pids[1] = 1;
  data.roots[1] = 0;
  data.timestamps[1] = 1000;
  data.end_timestamps[1] = 10000;

  data.events[2] = TT_MPI_Bcast;
  data.pids[2] = 2;
  data.roots[2] = 0;
  data.timestamps[2] = 3000;
  data.end_timestamps[2] = 10000;

  auto csr = makeCSR({0, 1, 2}, TT_MPI_Bcast, 0);
  auto raw = runAnalysisKernels(data, csr);

  assert(raw.late_broadcast.size() == 2);
  double sum = 0;
  for (auto v : raw.late_broadcast)
    sum += v;
  assert(fabs(sum - 6000.0) < 1e-6);

  std::cout << "PASSED" << std::endl;
}

// Test NxN wait: same as barrier wait but for AllReduce
void test_nxn_wait() {
  std::cout << "[test_nxn_wait] ";

  TraceDataSoA data;
  data.allocate(3);
  data.count = 3;

  for (int i = 0; i < 3; i++) {
    data.events[i] = TT_MPI_All_Reduce;
    data.pids[i] = (id_t)i;
    data.roots[i] = 0;
    data.end_timestamps[i] = 10000;
  }
  data.timestamps[0] = 1000;
  data.timestamps[1] = 2000;
  data.timestamps[2] = 5000;

  auto csr = makeCSR({0, 1, 2}, TT_MPI_All_Reduce, 0);
  auto raw = runAnalysisKernels(data, csr);

  assert(raw.wait_nxn.size() == 2);
  double sum = 0;
  for (auto v : raw.wait_nxn)
    sum += v;
  assert(fabs(sum - 7000.0) < 1e-6);

  std::cout << "PASSED" << std::endl;
}

// Test NxN completion
void test_nxn_completion() {
  std::cout << "[test_nxn_completion] ";

  TraceDataSoA data;
  data.allocate(3);
  data.count = 3;

  for (int i = 0; i < 3; i++) {
    data.events[i] = TT_MPI_AlltoAll;
    data.pids[i] = (id_t)i;
    data.roots[i] = 0;
    data.timestamps[i] = 1000;
  }
  data.end_timestamps[0] = 6000;
  data.end_timestamps[1] = 8000;
  data.end_timestamps[2] = 10000;

  auto csr = makeCSR({0, 1, 2}, TT_MPI_AlltoAll, 0);
  auto raw = runAnalysisKernels(data, csr);

  assert(raw.nxn_completion.size() == 2);
  double sum = 0;
  for (auto v : raw.nxn_completion)
    sum += v;
  assert(fabs(sum - 6000.0) < 1e-6);

  std::cout << "PASSED" << std::endl;
}

int main() {
  std::cout << "=== Analysis Kernel Tests ===" << std::endl;
  test_late_sender();
  test_late_receiver();
  test_barrier_wait();
  test_barrier_completion();
  test_early_reduce();
  test_late_broadcast();
  test_nxn_wait();
  test_nxn_completion();
  std::cout << "All analysis kernel tests passed!" << std::endl;
  return 0;
}
