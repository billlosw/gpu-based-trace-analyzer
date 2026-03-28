#include "reader/OTF2SoAReader.h"

#include <chrono>
#include <functional>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mpi.h>
#include <otf2xx/otf2.hpp>

// ============================================================
// Helper: contiguous block assignment (handles remainder)
// ============================================================
static std::pair<id_t, id_t> traceRange(id_t nlocs, id_t nprocs, id_t rank) {
  id_t per_rank = nlocs / nprocs;
  id_t remainder = nlocs % nprocs;
  id_t start, end;
  if (rank < remainder) {
    start = rank * (per_rank + 1);
    end = start + per_rank + 1;
  } else {
    start = remainder * (per_rank + 1) + (rank - remainder) * per_rank;
    end = start + per_rank;
  }
  return {start, end};
}

// Inverse of traceRange: which rank owns a given location?
static id_t owningRank(id_t nlocs, id_t nprocs, id_t loc_id) {
  id_t per_rank = nlocs / nprocs;
  id_t remainder = nlocs % nprocs;
  id_t boundary = remainder * (per_rank + 1);
  if (loc_id < boundary) {
    return loc_id / (per_rank + 1);
  } else {
    return remainder + (loc_id - boundary) / per_rank;
  }
}

static timestamp_t extractTimestamp(const otf2::chrono::time_point &tp) {
  auto ts_ps = std::chrono::time_point_cast<otf2::chrono::picoseconds>(tp);
  return ts_ps.time_since_epoch().count();
}

// ============================================================
// Pass 1: Discovery callback — finds communication partners
// ============================================================
class Pass1DiscoveryCallback : public otf2::reader::callback {
public:
  Pass1DiscoveryCallback(otf2::reader::reader &rdr, id_t start_loc,
                         id_t end_loc,
                         std::unordered_set<id_t> &related_locs)
      : m_rdr(rdr), m_start(start_loc), m_end(end_loc),
        m_related(related_locs) {}

  void definition(const otf2::definition::location &loc) override {
    id_t lid = loc.ref().get();
    if (lid >= m_start && lid < m_end) {
      m_related.insert(lid);
      m_rdr.register_location(loc);
    }
  }

  void definitions_done(const otf2::reader::reader &) override {}

  void event(const otf2::definition::location &,
             const otf2::event::mpi_send &event) override {
    m_related.insert(event.receiver());
  }

  void event(const otf2::definition::location &,
             const otf2::event::mpi_isend_request &event) override {
    m_related.insert(event.receiver());
  }

  void events_done(const otf2::reader::reader &) override {}

private:
  otf2::reader::reader &m_rdr;
  id_t m_start, m_end;
  std::unordered_set<id_t> &m_related;
};

// ============================================================
// Collective redistribution buffers (per target rank)
// ============================================================
struct CollRedistBuffers {
  int nprocs;
  std::vector<std::vector<int>> op_types;
  std::vector<std::vector<uint64_t>> begin_ts;
  std::vector<std::vector<uint64_t>> end_ts;
  std::vector<std::vector<uint32_t>> roots;
  std::vector<std::vector<uint32_t>> pids;
  std::vector<std::vector<uint64_t>> flat_comm_sets; // [size, m0, m1, ...]

  CollRedistBuffers(int np)
      : nprocs(np), op_types(np), begin_ts(np), end_ts(np), roots(np),
        pids(np), flat_comm_sets(np) {}
};

// ============================================================
// Pass 2: Data loading callback — reads events into SoA vectors
// ============================================================
class Pass2DataCallback : public otf2::reader::callback {
public:
  Pass2DataCallback(otf2::reader::reader &rdr,
                    const std::unordered_set<id_t> &related_locs,
                    id_t start_loc, id_t end_loc, id_t nlocs, int rank,
                    int nprocs, CollRedistBuffers &redist)
      : m_rdr(rdr), m_related(related_locs), m_start(start_loc),
        m_end(end_loc), m_nlocs(nlocs), m_rank(rank), m_nprocs(nprocs),
        m_redist(redist) {}

  // --- Definition: register own + related locations ---
  void definition(const otf2::definition::location &loc) override {
    id_t lid = loc.ref().get();
    if (m_related.count(lid)) {
      m_rdr.register_location(loc);
    }
  }

  void definitions_done(const otf2::reader::reader &) override {}

#ifdef USE_SCALASCA_TIMESTAMPS
  void event(const otf2::definition::location &loc,
             const otf2::event::enter &event) override {
    id_t pid = loc.ref().get();
    m_last_enter_ts[pid] = extractTimestamp(event.timestamp());
  }

  void event(const otf2::definition::location &loc,
             const otf2::event::leave &event) override {
    id_t pid = loc.ref().get();
    m_last_leave_ts[pid] = extractTimestamp(event.timestamp());
    auto it = m_last_send_soa_idx.find(pid);
    if (it != m_last_send_soa_idx.end() && it->second >= 0) {
      m_v_end_timestamps[it->second] = m_last_leave_ts[pid];
      it->second = -1;
    }
  }
#endif

  // --- P2P: store sends only from own locations ---
  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_send &event) override {
    id_t pid = loc.ref().get();
    if (pid < m_start || pid >= m_end)
      return;

#ifdef USE_SCALASCA_TIMESTAMPS
    auto it = m_last_enter_ts.find(pid);
    timestamp_t enter_ts = (it != m_last_enter_ts.end())
                               ? it->second
                               : extractTimestamp(event.timestamp());
    timestamp_t leave_ts = extractTimestamp(event.timestamp());
    size_t soa_idx = m_v_events.size();
    pushEvent(TT_MPI_Send, ENTER, enter_ts, leave_ts, pid, pid,
              event.receiver(), event.msg_tag(), 0);
    m_last_send_soa_idx[pid] = (int64_t)soa_idx;
#else
    auto ts = extractTimestamp(event.timestamp());
    pushEvent(TT_MPI_Send, ENTER, ts, ts, pid, pid, event.receiver(),
              event.msg_tag(), 0);
#endif
    m_send_count++;
    m_blocking_send_count++;
  }

  // --- P2P: store recvs only where sender is in own range ---
  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_receive &event) override {
    if (event.sender() < m_start || event.sender() >= m_end)
      return;

    id_t pid = loc.ref().get();
#ifdef USE_SCALASCA_TIMESTAMPS
    auto it = m_last_enter_ts.find(pid);
    timestamp_t enter_ts = (it != m_last_enter_ts.end())
                               ? it->second
                               : extractTimestamp(event.timestamp());
    pushEvent(TT_MPI_Recv, ENTER, enter_ts, enter_ts, pid, event.sender(), pid,
              event.msg_tag(), 0);
#else
    auto ts = extractTimestamp(event.timestamp());
    pushEvent(TT_MPI_Recv, ENTER, ts, ts, pid, event.sender(), pid,
              event.msg_tag(), 0);
#endif
    m_recv_count++;
    m_blocking_recv_count++;
  }

  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_isend_request &event) override {
    id_t pid = loc.ref().get();
    if (pid < m_start || pid >= m_end)
      return;

#ifdef USE_SCALASCA_TIMESTAMPS
    auto it = m_last_enter_ts.find(pid);
    timestamp_t enter_ts = (it != m_last_enter_ts.end())
                               ? it->second
                               : extractTimestamp(event.timestamp());
    timestamp_t leave_ts = extractTimestamp(event.timestamp());
    size_t soa_idx = m_v_events.size();
    pushEvent(TT_MPI_Isend, ENTER, enter_ts, leave_ts, pid, pid,
              event.receiver(), event.msg_tag(), 0);
    m_last_send_soa_idx[pid] = (int64_t)soa_idx;
#else
    auto ts = extractTimestamp(event.timestamp());
    pushEvent(TT_MPI_Isend, ENTER, ts, ts, pid, pid, event.receiver(),
              event.msg_tag(), 0);
#endif
    m_send_count++;
    m_nonblocking_send_count++;
  }

#ifdef USE_SCALASCA_TIMESTAMPS
  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_ireceive_request &event) override {
    id_t pid = loc.ref().get();
    auto it = m_last_enter_ts.find(pid);
    if (it != m_last_enter_ts.end()) {
      // Key by (pid, request_id) to handle request_id collisions
      // across different locations (request_id is per-location, not global).
      // Use a struct key to avoid truncation issues with large values.
      auto key = std::make_pair(pid, event.request_id());
      m_irecv_enter_ts[key] = it->second;
    }
  }
#endif

  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_ireceive_complete &event) override {
    if (event.sender() < m_start || event.sender() >= m_end)
      return;

    id_t pid = loc.ref().get();
#ifdef USE_SCALASCA_TIMESTAMPS
    auto it = m_last_enter_ts.find(pid);
    timestamp_t enter_ts = (it != m_last_enter_ts.end())
                               ? it->second
                               : extractTimestamp(event.timestamp());
    // Look up by (pid, request_id) to get the correct Enter(MPI_Irecv)
    // timestamp even when multiple Irecvs are outstanding.
    auto key = std::make_pair(pid, event.request_id());
    auto it2 = m_irecv_enter_ts.find(key);
    timestamp_t irecv_enter_ts =
        (it2 != m_irecv_enter_ts.end()) ? it2->second : enter_ts;
    if (it2 != m_irecv_enter_ts.end())
      m_irecv_enter_ts.erase(it2);
    pushEvent(TT_MPI_Irecv, ENTER, enter_ts, irecv_enter_ts, pid,
              event.sender(), pid, event.msg_tag(), 0);
#else
    auto ts = extractTimestamp(event.timestamp());
    pushEvent(TT_MPI_Irecv, ENTER, ts, ts, pid, event.sender(), pid,
              event.msg_tag(), 0);
#endif
    m_recv_count++;
    m_nonblocking_recv_count++;
  }

  // --- Collective: begin ---
  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_collective_begin &event) override {
    id_t pid = loc.ref().get();
    if (pid < m_start || pid >= m_end)
      return;
    auto ts = extractTimestamp(event.timestamp());
    m_coll_begin_ts[pid] = ts;
    m_coll_begin_valid[pid] = true;
  }

  // --- Collective: end (with root-based routing) ---
  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_collective_end &event) override {
    id_t pid = loc.ref().get();
    if (pid < m_start || pid >= m_end)
      return;

    if (!m_coll_begin_valid[pid])
      return;
    m_coll_begin_valid[pid] = false;

    auto end_ts = extractTimestamp(event.timestamp());
    timestamp_t begin_ts = m_coll_begin_ts[pid];

    // Map OTF2 collective op type
    auto otf2_op_type = event.type();
    event_t op_type;
    int op_int = static_cast<int>(otf2_op_type);

    if (op_int == OTF2_COLLECTIVE_OP_BARRIER)
      op_type = TT_MPI_Barrier;
    else if (op_int == OTF2_COLLECTIVE_OP_REDUCE)
      op_type = TT_MPI_Reduce;
    else if (op_int == OTF2_COLLECTIVE_OP_BCAST)
      op_type = TT_MPI_Bcast;
    else if (op_int == OTF2_COLLECTIVE_OP_GATHER)
      op_type = TT_MPI_Gather;
    else if (op_int == OTF2_COLLECTIVE_OP_GATHERV)
      op_type = TT_MPI_Gatherv;
    else if (op_int == OTF2_COLLECTIVE_OP_SCATTER)
      op_type = TT_MPI_Scatter;
    else if (op_int == OTF2_COLLECTIVE_OP_SCATTERV)
      op_type = TT_MPI_Scatterv;
    else if (op_int == OTF2_COLLECTIVE_OP_REDUCE_SCATTER)
      op_type = TT_MPI_Reduce_Scatter;
    else if (op_int == OTF2_COLLECTIVE_OP_REDUCE_SCATTER_BLOCK)
      op_type = TT_MPI_Reduce_Scatter_Block;
    else if (op_int == OTF2_COLLECTIVE_OP_ALLGATHER)
      op_type = TT_MPI_All_Gather;
    else if (op_int == OTF2_COLLECTIVE_OP_ALLGATHERV)
      op_type = TT_MPI_All_Gatherv;
    else if (op_int == OTF2_COLLECTIVE_OP_ALLREDUCE)
      op_type = TT_MPI_All_Reduce;
    else if (op_int == OTF2_COLLECTIVE_OP_ALLTOALL)
      op_type = TT_MPI_AlltoAll;
    else
      return;

    // Extract communicator members
    auto comm_set = std::get<otf2::definition::comm_group>(
                        std::get<otf2::definition::comm>(event.comm()).group())
                        .members();

    // Handle sentinel root value
    auto root = event.root();
    if (root == 4294967295u) {
      root = pid;
      for (auto id : comm_set)
        root = std::min(root, (uint32_t)id);
    }

    // Route by root location
    if (root >= m_start && root < m_end) {
      // Root is local: store directly
      pushEvent(op_type, ENTER, begin_ts, end_ts, pid, 0, 0, 0, root);
      std::vector<uint64_t> cs(comm_set.begin(), comm_set.end());
      m_comm_sets.push_back(std::move(cs));
      m_coll_count++;
    } else {
      // Root is remote: buffer for redistribution
      id_t target = owningRank(m_nlocs, m_nprocs, root);
      m_redist.op_types[target].push_back((int)op_type);
      m_redist.begin_ts[target].push_back(begin_ts);
      m_redist.end_ts[target].push_back(end_ts);
      m_redist.roots[target].push_back(root);
      m_redist.pids[target].push_back(pid);
      // Flatten comm_set
      m_redist.flat_comm_sets[target].push_back(comm_set.size());
      for (auto id : comm_set)
        m_redist.flat_comm_sets[target].push_back(id);
    }
  }

  void events_done(const otf2::reader::reader &) override {}

  // --- Accessors ---
  size_t getEventCount() const { return m_v_events.size(); }
  size_t getSendCount() const { return m_send_count; }
  size_t getRecvCount() const { return m_recv_count; }
  size_t getCollCount() const { return m_coll_count; }
  size_t getBlockingSendCount() const { return m_blocking_send_count; }
  size_t getNonBlockingSendCount() const { return m_nonblocking_send_count; }
  size_t getBlockingRecvCount() const { return m_blocking_recv_count; }
  size_t getNonBlockingRecvCount() const { return m_nonblocking_recv_count; }

  // Copy vector data into pre-allocated TraceDataSoA
  void fillSoA(TraceDataSoA &data) const {
    size_t n = m_v_events.size();
    data.allocate(n);
    data.count = n;
    std::memcpy(data.events, m_v_events.data(), n * sizeof(event_t));
    std::memcpy(data.types, m_v_types.data(), n * sizeof(event_type_t));
    std::memcpy(data.timestamps, m_v_timestamps.data(),
                n * sizeof(timestamp_t));
    std::memcpy(data.end_timestamps, m_v_end_timestamps.data(),
                n * sizeof(timestamp_t));
    std::memcpy(data.pids, m_v_pids.data(), n * sizeof(id_t));
    std::memcpy(data.srcs, m_v_srcs.data(), n * sizeof(id_t));
    std::memcpy(data.dsts, m_v_dsts.data(), n * sizeof(id_t));
    std::memcpy(data.tags, m_v_tags.data(), n * sizeof(id_t));
    std::memcpy(data.roots, m_v_roots.data(), n * sizeof(id_t));
    std::memcpy(data.replay_pids, m_v_pids.data(), n * sizeof(id_t));
    std::memset(data.tids, 0, n * sizeof(id_t));
    for (size_t i = 0; i < n; i++)
      data.indices[i] = (id_t)i;
  }

  std::vector<std::vector<uint64_t>> &getCommSets() { return m_comm_sets; }

  // Append received collective events after redistribution
  void appendCollectiveEvents(const std::vector<int> &recv_op_types,
                              const std::vector<uint64_t> &recv_begin_ts,
                              const std::vector<uint64_t> &recv_end_ts,
                              const std::vector<uint32_t> &recv_roots,
                              const std::vector<uint32_t> &recv_pids,
                              const std::vector<std::vector<uint64_t>> &recv_cs) {
    for (size_t i = 0; i < recv_pids.size(); i++) {
      pushEvent((event_t)recv_op_types[i], ENTER, recv_begin_ts[i],
                recv_end_ts[i], recv_pids[i], 0, 0, 0, recv_roots[i]);
      m_comm_sets.push_back(recv_cs[i]);
      m_coll_count++;
    }
  }

private:
  otf2::reader::reader &m_rdr;
  const std::unordered_set<id_t> &m_related;
  id_t m_start, m_end, m_nlocs;
  int m_rank, m_nprocs;
  CollRedistBuffers &m_redist;

  // Dynamic vectors for SoA data
  std::vector<event_t> m_v_events;
  std::vector<event_type_t> m_v_types;
  std::vector<timestamp_t> m_v_timestamps;
  std::vector<timestamp_t> m_v_end_timestamps;
  std::vector<id_t> m_v_pids;
  std::vector<id_t> m_v_srcs;
  std::vector<id_t> m_v_dsts;
  std::vector<id_t> m_v_tags;
  std::vector<id_t> m_v_roots;

  // Comm sets for collective events
  std::vector<std::vector<uint64_t>> m_comm_sets;

  // Collective begin/end pairing
  std::unordered_map<id_t, timestamp_t> m_coll_begin_ts;
  std::unordered_map<id_t, bool> m_coll_begin_valid;

#ifdef USE_SCALASCA_TIMESTAMPS
  // Hash for (pid, request_id) pair keys
  struct PairHash {
    size_t operator()(const std::pair<id_t, uint64_t> &p) const {
      size_t h1 = std::hash<id_t>{}(p.first);
      size_t h2 = std::hash<uint64_t>{}(p.second);
      return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
  };

  std::unordered_map<id_t, timestamp_t> m_last_enter_ts;
  std::unordered_map<std::pair<id_t, uint64_t>, timestamp_t, PairHash> m_irecv_enter_ts;
  std::unordered_map<id_t, timestamp_t> m_last_leave_ts;
  std::unordered_map<id_t, int64_t> m_last_send_soa_idx;
#endif

  size_t m_send_count = 0;
  size_t m_recv_count = 0;
  size_t m_coll_count = 0;
  size_t m_blocking_send_count = 0;
  size_t m_nonblocking_send_count = 0;
  size_t m_blocking_recv_count = 0;
  size_t m_nonblocking_recv_count = 0;

  void pushEvent(event_t ev, event_type_t type, timestamp_t ts,
                 timestamp_t end_ts, id_t pid, id_t src, id_t dst, id_t tag,
                 id_t root) {
    m_v_events.push_back(ev);
    m_v_types.push_back(type);
    m_v_timestamps.push_back(ts);
    m_v_end_timestamps.push_back(end_ts);
    m_v_pids.push_back(pid);
    m_v_srcs.push_back(src);
    m_v_dsts.push_back(dst);
    m_v_tags.push_back(tag);
    m_v_roots.push_back(root);
  }
};

// ============================================================
// Collective redistribution via MPI
// ============================================================
static void redistributeCollectives(Pass2DataCallback &cb,
                                    CollRedistBuffers &redist, int rank,
                                    int nprocs) {
  for (int target = 0; target < nprocs; target++) {
    int local_count = (int)redist.pids[target].size();

    // Gather event counts to target rank
    std::vector<int> recv_counts(nprocs);
    MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               target, MPI_COMM_WORLD);

    // Compute displacements on target rank
    std::vector<int> displs(nprocs);
    int total = 0;
    if (rank == target) {
      for (int i = 0; i < nprocs; i++) {
        displs[i] = total;
        total += recv_counts[i];
      }
    }

    // Skip if no data to transfer for this target
    int global_total = 0;
    MPI_Allreduce(&local_count, &global_total, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    if (global_total == 0)
      continue;

    // Allocate receive buffers on target rank
    std::vector<int> g_op_types;
    std::vector<uint64_t> g_begin_ts, g_end_ts;
    std::vector<uint32_t> g_roots, g_pids;
    if (rank == target) {
      g_op_types.resize(total);
      g_begin_ts.resize(total);
      g_end_ts.resize(total);
      g_roots.resize(total);
      g_pids.resize(total);
    }

    // Gatherv fixed-length arrays
    MPI_Gatherv(redist.op_types[target].data(), local_count, MPI_INT,
                rank == target ? g_op_types.data() : nullptr,
                recv_counts.data(), displs.data(), MPI_INT, target,
                MPI_COMM_WORLD);
    MPI_Gatherv(redist.begin_ts[target].data(), local_count, MPI_UINT64_T,
                rank == target ? (uint64_t *)g_begin_ts.data() : nullptr,
                recv_counts.data(), displs.data(), MPI_UINT64_T, target,
                MPI_COMM_WORLD);
    MPI_Gatherv(redist.end_ts[target].data(), local_count, MPI_UINT64_T,
                rank == target ? (uint64_t *)g_end_ts.data() : nullptr,
                recv_counts.data(), displs.data(), MPI_UINT64_T, target,
                MPI_COMM_WORLD);
    MPI_Gatherv(redist.roots[target].data(), local_count, MPI_UINT32_T,
                rank == target ? g_roots.data() : nullptr, recv_counts.data(),
                displs.data(), MPI_UINT32_T, target, MPI_COMM_WORLD);
    MPI_Gatherv(redist.pids[target].data(), local_count, MPI_UINT32_T,
                rank == target ? g_pids.data() : nullptr, recv_counts.data(),
                displs.data(), MPI_UINT32_T, target, MPI_COMM_WORLD);

    // Gatherv variable-length comm_sets
    int local_cs_len = (int)redist.flat_comm_sets[target].size();
    std::vector<int> cs_lens(nprocs), cs_displs(nprocs);
    MPI_Gather(&local_cs_len, 1, MPI_INT, cs_lens.data(), 1, MPI_INT, target,
               MPI_COMM_WORLD);

    int total_cs = 0;
    if (rank == target) {
      for (int i = 0; i < nprocs; i++) {
        cs_displs[i] = total_cs;
        total_cs += cs_lens[i];
      }
    }

    std::vector<uint64_t> g_flat_cs;
    if (rank == target)
      g_flat_cs.resize(total_cs);

    MPI_Gatherv(redist.flat_comm_sets[target].data(), local_cs_len,
                MPI_UINT64_T, rank == target ? g_flat_cs.data() : nullptr,
                cs_lens.data(), cs_displs.data(), MPI_UINT64_T, target,
                MPI_COMM_WORLD);

    // On target rank: reconstruct and append
    if (rank == target && total > 0) {
      std::vector<std::vector<uint64_t>> recv_cs;
      size_t pos = 0;
      while (pos < g_flat_cs.size()) {
        size_t sz = g_flat_cs[pos++];
        recv_cs.emplace_back(g_flat_cs.begin() + pos,
                             g_flat_cs.begin() + pos + sz);
        pos += sz;
      }
      cb.appendCollectiveEvents(g_op_types, g_begin_ts, g_end_ts, g_roots,
                                g_pids, recv_cs);
    }
  }
}

// ============================================================
// Main entry point: two-pass distributed reading
// ============================================================
ReaderOutput readOTF2Trace(const std::string &trace_path) {
  ReaderOutput output;

  int rank = 0, comm_sz = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  auto t_start = std::chrono::high_resolution_clock::now();

  std::unordered_set<id_t> related_locs;
  id_t start_loc = 0, end_loc = 0, nlocs = 0;

  // ======== PASS 1: DISCOVERY ========
  {
    otf2::reader::reader rdr(trace_path, MPI_COMM_WORLD);
    nlocs = rdr.num_locations();

    auto [s, e] = traceRange(nlocs, comm_sz, rank);
    start_loc = s;
    end_loc = e;

    if (rank == 0) {
      std::cout << "[Reader] " << nlocs << " locations, " << comm_sz
                << " ranks, pass 1 discovery..." << std::endl;
    }

    Pass1DiscoveryCallback cb(rdr, start_loc, end_loc, related_locs);
    rdr.set_callback(cb);
    rdr.read_definitions();
    rdr.read_events();
  }

  auto t_pass1 = std::chrono::high_resolution_clock::now();
  double pass1_ms =
      std::chrono::duration<double, std::milli>(t_pass1 - t_start).count();

  if (rank == 0) {
    std::cout << "[Reader] Pass 1 done in " << pass1_ms << " ms" << std::endl;
  }

  // ======== PASS 2: DATA LOADING ========
  CollRedistBuffers redist(comm_sz);
  Pass2DataCallback *data_cb = nullptr;
  {
    otf2::reader::reader rdr2(trace_path, MPI_COMM_WORLD);

    data_cb =
        new Pass2DataCallback(rdr2, related_locs, start_loc, end_loc, nlocs,
                              rank, comm_sz, redist);
    rdr2.set_callback(*data_cb);
    rdr2.read_definitions();
    rdr2.read_events();
  }

  auto t_pass2 = std::chrono::high_resolution_clock::now();
  double pass2_ms =
      std::chrono::duration<double, std::milli>(t_pass2 - t_pass1).count();

  if (rank == 0) {
    std::cout << "[Reader] Pass 2 done in " << pass2_ms << " ms" << std::endl;
  }

  // ======== COLLECTIVE REDISTRIBUTION ========
  if (comm_sz > 1) {
    redistributeCollectives(*data_cb, redist, rank, comm_sz);
  }

  auto t_redist = std::chrono::high_resolution_clock::now();
  double redist_ms =
      std::chrono::duration<double, std::milli>(t_redist - t_pass2).count();

  // ======== BUILD SOA ========
  data_cb->fillSoA(output.data);
  output.comm_sets = std::move(data_cb->getCommSets());

  // Gather total counts for diagnostics
  size_t local_sends = data_cb->getSendCount(),
         local_recvs = data_cb->getRecvCount(),
         local_colls = data_cb->getCollCount();
  size_t local_bsend = data_cb->getBlockingSendCount(),
         local_nbsend = data_cb->getNonBlockingSendCount(),
         local_brecv = data_cb->getBlockingRecvCount(),
         local_nbrecv = data_cb->getNonBlockingRecvCount();
  size_t total_sends = 0, total_recvs = 0, total_colls = 0;
  size_t total_bsend = 0, total_nbsend = 0, total_brecv = 0, total_nbrecv = 0;
  MPI_Reduce(&local_sends, &total_sends, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&local_recvs, &total_recvs, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&local_colls, &total_colls, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&local_bsend, &total_bsend, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&local_nbsend, &total_nbsend, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&local_brecv, &total_brecv, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&local_nbrecv, &total_nbrecv, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);

  delete data_cb;

  auto t_end = std::chrono::high_resolution_clock::now();
  double total_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();

  if (rank == 0) {
    std::cout << "[Reader] Collective redistribution: " << redist_ms << " ms"
              << std::endl;
    std::cout << "[Reader] Total events (global): "
              << total_sends + total_recvs + total_colls
              << " (" << total_sends << " sends [" << total_bsend << " blocking, "
              << total_nbsend << " nonblocking], " << total_recvs << " recvs ["
              << total_brecv << " blocking, " << total_nbrecv << " nonblocking], "
              << total_colls << " collectives)" << std::endl;
    std::cout << "[Reader] Local events on rank 0: " << output.data.count
              << std::endl;
    std::cout << "[Reader] Total read time: " << total_ms << " ms (pass1="
              << pass1_ms << ", pass2=" << pass2_ms << ", redist=" << redist_ms
              << ")" << std::endl;
  }

  return output;
}
