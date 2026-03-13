#include "matching/P2PMatching.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

// Test: Known send-recv pairs should be correctly matched.
void test_simple_matching() {
  std::cout << "[test_simple_matching] ";

  // Create a trace with 4 events:
  // Event 0: Send from pid=0 to dst=1, tag=100, timestamp=1000
  // Event 1: Recv at pid=1 from src=0, tag=100, timestamp=2000
  // Event 2: Send from pid=1 to dst=0, tag=200, timestamp=3000
  // Event 3: Recv at pid=0 from src=1, tag=200, timestamp=4000

  TraceDataSoA data;
  data.allocate(4);
  data.count = 4;

  // Event 0: Send (pid=0 -> dst=1, tag=100)
  data.events[0] = TT_MPI_Send;
  data.types[0] = ENTER;
  data.timestamps[0] = 1000;
  data.pids[0] = 0;
  data.srcs[0] = 0;
  data.dsts[0] = 1;
  data.tags[0] = 100;

  // Event 1: Recv (pid=1, src=0, tag=100)
  data.events[1] = TT_MPI_Recv;
  data.types[1] = ENTER;
  data.timestamps[1] = 2000;
  data.pids[1] = 1;
  data.srcs[1] = 0;
  data.dsts[1] = 1;
  data.tags[1] = 100;

  // Event 2: Send (pid=1 -> dst=0, tag=200)
  data.events[2] = TT_MPI_Send;
  data.types[2] = ENTER;
  data.timestamps[2] = 3000;
  data.pids[2] = 1;
  data.srcs[2] = 1;
  data.dsts[2] = 0;
  data.tags[2] = 200;

  // Event 3: Recv (pid=0, src=1, tag=200)
  data.events[3] = TT_MPI_Recv;
  data.types[3] = ENTER;
  data.timestamps[3] = 4000;
  data.pids[3] = 0;
  data.srcs[3] = 1;
  data.dsts[3] = 0;
  data.tags[3] = 200;

  runP2PMatching(data);

  // Verify matches: 0<->1 and 2<->3
  assert(data.match_partner[0] == 1);
  assert(data.match_partner[1] == 0);
  assert(data.match_partner[2] == 3);
  assert(data.match_partner[3] == 2);

  std::cout << "PASSED" << std::endl;
}

// Test: Unmatched events should remain -1.
void test_unmatched() {
  std::cout << "[test_unmatched] ";

  TraceDataSoA data;
  data.allocate(3);
  data.count = 3;

  // Event 0: Send (pid=0 -> dst=1, tag=100)
  data.events[0] = TT_MPI_Send;
  data.types[0] = ENTER;
  data.timestamps[0] = 1000;
  data.pids[0] = 0;
  data.srcs[0] = 0;
  data.dsts[0] = 1;
  data.tags[0] = 100;

  // Event 1: Recv (pid=1, src=0, tag=100) - matches event 0
  data.events[1] = TT_MPI_Recv;
  data.types[1] = ENTER;
  data.timestamps[1] = 2000;
  data.pids[1] = 1;
  data.srcs[1] = 0;
  data.dsts[1] = 1;
  data.tags[1] = 100;

  // Event 2: Send (pid=2 -> dst=3, tag=300) - no matching recv
  data.events[2] = TT_MPI_Send;
  data.types[2] = ENTER;
  data.timestamps[2] = 3000;
  data.pids[2] = 2;
  data.srcs[2] = 2;
  data.dsts[2] = 3;
  data.tags[2] = 300;

  runP2PMatching(data);

  assert(data.match_partner[0] == 1);
  assert(data.match_partner[1] == 0);
  assert(data.match_partner[2] == -1); // unmatched

  std::cout << "PASSED" << std::endl;
}

// Test: Isend/Irecv matching
void test_nonblocking_matching() {
  std::cout << "[test_nonblocking_matching] ";

  TraceDataSoA data;
  data.allocate(2);
  data.count = 2;

  data.events[0] = TT_MPI_Isend;
  data.types[0] = ENTER;
  data.timestamps[0] = 1000;
  data.pids[0] = 0;
  data.srcs[0] = 0;
  data.dsts[0] = 1;
  data.tags[0] = 50;

  data.events[1] = TT_MPI_Irecv;
  data.types[1] = ENTER;
  data.timestamps[1] = 1500;
  data.pids[1] = 1;
  data.srcs[1] = 0;
  data.dsts[1] = 1;
  data.tags[1] = 50;

  runP2PMatching(data);

  assert(data.match_partner[0] == 1);
  assert(data.match_partner[1] == 0);

  std::cout << "PASSED" << std::endl;
}

// Test: Empty trace
void test_empty() {
  std::cout << "[test_empty] ";

  TraceDataSoA data;
  data.allocate(0);
  data.count = 0;

  runP2PMatching(data); // Should not crash

  std::cout << "PASSED" << std::endl;
}

int main() {
  std::cout << "=== P2P Matching Tests ===" << std::endl;
  test_simple_matching();
  test_unmatched();
  test_nonblocking_matching();
  test_empty();
  std::cout << "All P2P matching tests passed!" << std::endl;
  return 0;
}
