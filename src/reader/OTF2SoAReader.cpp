#include "reader/OTF2SoAReader.h"

#include <chrono>
#include <iostream>
#include <tuple>
#include <vector>

#include <mpi.h>
#include <otf2xx/otf2.hpp>

// Internal callback class for otf2xx reader — single-pass using vectors
class SoAReaderCallback : public otf2::reader::callback {
public:
  SoAReaderCallback(otf2::reader::reader &rdr, int rank = 0, int size = 1)
      : m_rdr(rdr), m_rank(rank), m_size(size) {}

  // --- Definition callbacks ---
  void definition(const otf2::definition::location &loc) override {
    uint64_t loc_id = loc.ref().get();
    // Each rank registers only its assigned locations (round-robin)
    if ((int)(loc_id % m_size) == m_rank) {
      m_rdr.register_location(loc);
    }
  }

  void definitions_done(const otf2::reader::reader &) override {}

  // --- P2P event callbacks ---
  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_send &event) override {
    id_t pid = loc.ref().get();
    auto ts = extractTimestamp(event.timestamp());

    pushEvent(TT_MPI_Send, ENTER, ts, ts, pid,
              pid, event.receiver(), event.msg_tag(), 0);

    m_send_count++;
  }

  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_receive &event) override {
    id_t pid = loc.ref().get();
    auto ts = extractTimestamp(event.timestamp());

    pushEvent(TT_MPI_Recv, ENTER, ts, ts, pid,
              event.sender(), pid, event.msg_tag(), 0);

    m_recv_count++;
  }

  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_isend_request &event) override {
    id_t pid = loc.ref().get();
    auto ts = extractTimestamp(event.timestamp());

    pushEvent(TT_MPI_Isend, ENTER, ts, ts, pid,
              pid, event.receiver(), event.msg_tag(), 0);

    m_send_count++;
  }

  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_ireceive_complete &event) override {
    id_t pid = loc.ref().get();
    auto ts = extractTimestamp(event.timestamp());

    pushEvent(TT_MPI_Irecv, ENTER, ts, ts, pid,
              event.sender(), pid, event.msg_tag(), 0);

    m_recv_count++;
  }

  // --- Collective event callbacks ---
  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_collective_begin &event) override {
    id_t pid = loc.ref().get();
    auto ts = extractTimestamp(event.timestamp());
    m_coll_begin_ts[pid] = ts;
    m_coll_begin_valid[pid] = true;
  }

  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_collective_end &event) override {
    id_t pid = loc.ref().get();

    if (!m_coll_begin_valid[pid])
      return;
    m_coll_begin_valid[pid] = false;

    auto end_ts = extractTimestamp(event.timestamp());
    timestamp_t begin_ts = m_coll_begin_ts[pid];

    // Map OTF2 collective op type to our event_t
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
      return; // Unknown collective type, skip

    // Extract communicator members
    auto comm_set = std::get<otf2::definition::comm_group>(
                        std::get<otf2::definition::comm>(event.comm()).group())
                        .members();

    // Handle sentinel root value (same as TileTrace)
    auto root = event.root();
    if (root == 4294967295u) {
      root = pid;
      for (auto id : comm_set)
        root = std::min(root, (uint32_t)id);
    }

    size_t i = m_v_events.size();
    pushEvent(op_type, ENTER, begin_ts, end_ts, pid,
              0, 0, 0, root);

    // Store comm_set
    std::vector<uint64_t> cs(comm_set.begin(), comm_set.end());
    m_comm_sets_out.push_back(std::move(cs));
    m_coll_soa_indices.push_back(i);

    m_coll_count++;
  }

  void events_done(const otf2::reader::reader &) override {}

  // --- Accessors ---
  size_t getEventCount() const { return m_v_events.size(); }
  size_t getSendCount() const { return m_send_count; }
  size_t getRecvCount() const { return m_recv_count; }
  size_t getCollCount() const { return m_coll_count; }

  const std::vector<size_t> &getCollSoAIndices() const {
    return m_coll_soa_indices;
  }

  // Copy vector data into pre-allocated TraceDataSoA
  void fillSoA(TraceDataSoA &data) const {
    size_t n = m_v_events.size();
    data.allocate(n);
    data.count = n;
    std::memcpy(data.events, m_v_events.data(), n * sizeof(event_t));
    std::memcpy(data.types, m_v_types.data(), n * sizeof(event_type_t));
    std::memcpy(data.timestamps, m_v_timestamps.data(), n * sizeof(timestamp_t));
    std::memcpy(data.end_timestamps, m_v_end_timestamps.data(), n * sizeof(timestamp_t));
    std::memcpy(data.pids, m_v_pids.data(), n * sizeof(id_t));
    std::memcpy(data.srcs, m_v_srcs.data(), n * sizeof(id_t));
    std::memcpy(data.dsts, m_v_dsts.data(), n * sizeof(id_t));
    std::memcpy(data.tags, m_v_tags.data(), n * sizeof(id_t));
    std::memcpy(data.roots, m_v_roots.data(), n * sizeof(id_t));
    // tids and replay_pids: set to pids
    std::memcpy(data.replay_pids, m_v_pids.data(), n * sizeof(id_t));
    std::memset(data.tids, 0, n * sizeof(id_t));
    // indices: fill with 0..n-1
    for (size_t i = 0; i < n; i++)
      data.indices[i] = (id_t)i;
  }

  std::vector<std::vector<uint64_t>> &getCommSets() { return m_comm_sets_out; }

  // Access raw vectors for MPI gathering
  const std::vector<event_t>       &rawEvents()     const { return m_v_events; }
  const std::vector<event_type_t>  &rawTypes()      const { return m_v_types; }
  const std::vector<timestamp_t>   &rawTimestamps() const { return m_v_timestamps; }
  const std::vector<timestamp_t>   &rawEndTimestamps() const { return m_v_end_timestamps; }
  const std::vector<id_t>          &rawPids()       const { return m_v_pids; }
  const std::vector<id_t>          &rawSrcs()       const { return m_v_srcs; }
  const std::vector<id_t>          &rawDsts()       const { return m_v_dsts; }
  const std::vector<id_t>          &rawTags()       const { return m_v_tags; }
  const std::vector<id_t>          &rawRoots()      const { return m_v_roots; }

private:
  otf2::reader::reader &m_rdr;
  int m_rank;
  int m_size;

  // Dynamic vectors for single-pass reading
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
  std::vector<std::vector<uint64_t>> m_comm_sets_out;
  std::vector<size_t> m_coll_soa_indices;

  // Collective begin/end pairing
  std::unordered_map<id_t, timestamp_t> m_coll_begin_ts;
  std::unordered_map<id_t, bool> m_coll_begin_valid;

  // Counts for diagnostics
  size_t m_send_count = 0;
  size_t m_recv_count = 0;
  size_t m_coll_count = 0;

  void pushEvent(event_t ev, event_type_t type, timestamp_t ts,
                 timestamp_t end_ts, id_t pid, id_t src, id_t dst,
                 id_t tag, id_t root) {
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

  static timestamp_t
  extractTimestamp(const otf2::chrono::time_point &tp) {
    auto ts_ps =
        std::chrono::time_point_cast<otf2::chrono::picoseconds>(tp);
    return ts_ps.time_since_epoch().count();
  }
};

// Gather all events from all MPI ranks to rank 0
static void gatherEventsToRank0(
    const SoAReaderCallback &cb, int rank, int comm_sz,
    ReaderOutput &output,
    size_t &total_sends, size_t &total_recvs, size_t &total_colls) {

  int local_count = (int)cb.getEventCount();

  // Gather counts to all ranks (needed for displacements)
  std::vector<int> all_counts(comm_sz);
  MPI_Allgather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT,
                MPI_COMM_WORLD);

  int total = 0;
  std::vector<int> displs(comm_sz);
  for (int i = 0; i < comm_sz; i++) {
    displs[i] = total;
    total += all_counts[i];
  }

  // Gather send/recv/coll counts for diagnostics
  size_t local_sends = cb.getSendCount(),
         local_recvs = cb.getRecvCount(),
         local_colls = cb.getCollCount();
  MPI_Reduce(&local_sends, &total_sends, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&local_recvs, &total_recvs, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&local_colls, &total_colls, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    output.data.allocate(total);
    output.data.count = total;
  }

  // Helper: Gatherv for typed arrays
  auto gatherv_int = [&](const int *sendbuf, int *recvbuf) {
    MPI_Gatherv(sendbuf, local_count, MPI_INT,
                recvbuf, all_counts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
  };
  auto gatherv_u32 = [&](const uint32_t *sendbuf, uint32_t *recvbuf) {
    MPI_Gatherv(sendbuf, local_count, MPI_UINT32_T,
                recvbuf, all_counts.data(), displs.data(), MPI_UINT32_T,
                0, MPI_COMM_WORLD);
  };
  auto gatherv_u64 = [&](const uint64_t *sendbuf, uint64_t *recvbuf) {
    MPI_Gatherv(sendbuf, local_count, MPI_UINT64_T,
                recvbuf, all_counts.data(), displs.data(), MPI_UINT64_T,
                0, MPI_COMM_WORLD);
  };

  // Gather each SoA array
  gatherv_int((const int *)cb.rawEvents().data(),
              rank == 0 ? (int *)output.data.events : nullptr);
  gatherv_int((const int *)cb.rawTypes().data(),
              rank == 0 ? (int *)output.data.types : nullptr);
  gatherv_u64(cb.rawTimestamps().data(),
              rank == 0 ? output.data.timestamps : nullptr);
  gatherv_u64(cb.rawEndTimestamps().data(),
              rank == 0 ? output.data.end_timestamps : nullptr);
  gatherv_u32(cb.rawPids().data(),
              rank == 0 ? output.data.pids : nullptr);
  gatherv_u32(cb.rawSrcs().data(),
              rank == 0 ? output.data.srcs : nullptr);
  gatherv_u32(cb.rawDsts().data(),
              rank == 0 ? output.data.dsts : nullptr);
  gatherv_u32(cb.rawTags().data(),
              rank == 0 ? output.data.tags : nullptr);
  gatherv_u32(cb.rawRoots().data(),
              rank == 0 ? output.data.roots : nullptr);

  if (rank == 0) {
    // Fill tids, replay_pids, indices, match_partner, coll_group_id
    std::memcpy(output.data.replay_pids, output.data.pids, total * sizeof(id_t));
    std::memset(output.data.tids, 0, total * sizeof(id_t));
    for (int i = 0; i < total; i++)
      output.data.indices[i] = (id_t)i;
  }

  // --- Gather comm_sets ---
  // Each rank sends: num_comm_sets, then for each: size, members...
  // Flatten local comm_sets
  auto &local_cs = const_cast<SoAReaderCallback &>(cb).getCommSets();
  int local_num_cs = (int)local_cs.size();

  // Flatten: [size0, m0_0, m0_1, ..., size1, m1_0, ...]
  std::vector<uint64_t> flat_cs;
  for (auto &cs : local_cs) {
    flat_cs.push_back(cs.size());
    flat_cs.insert(flat_cs.end(), cs.begin(), cs.end());
  }
  int flat_cs_len = (int)flat_cs.size();

  // Gather flat_cs lengths
  std::vector<int> cs_lens(comm_sz);
  MPI_Gather(&flat_cs_len, 1, MPI_INT, cs_lens.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  std::vector<int> cs_displs(comm_sz);
  int total_flat_cs = 0;
  if (rank == 0) {
    for (int i = 0; i < comm_sz; i++) {
      cs_displs[i] = total_flat_cs;
      total_flat_cs += cs_lens[i];
    }
  }

  std::vector<uint64_t> all_flat_cs;
  if (rank == 0)
    all_flat_cs.resize(total_flat_cs);

  MPI_Gatherv(flat_cs.data(), flat_cs_len, MPI_UINT64_T,
              rank == 0 ? all_flat_cs.data() : nullptr,
              cs_lens.data(), cs_displs.data(), MPI_UINT64_T,
              0, MPI_COMM_WORLD);

  // Reconstruct comm_sets on rank 0
  if (rank == 0) {
    output.comm_sets.clear();
    size_t pos = 0;
    while (pos < all_flat_cs.size()) {
      size_t sz = all_flat_cs[pos++];
      std::vector<uint64_t> cs(all_flat_cs.begin() + pos,
                               all_flat_cs.begin() + pos + sz);
      output.comm_sets.push_back(std::move(cs));
      pos += sz;
    }
  }
}

ReaderOutput readOTF2Trace(const std::string &trace_path) {
  ReaderOutput output;

  int rank = 0, comm_sz = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  auto t_start = std::chrono::high_resolution_clock::now();

  size_t total_sends = 0, total_recvs = 0, total_colls = 0;

  {
    // Open reader with MPI for parallel location reading
    otf2::reader::reader rdr(trace_path, MPI_COMM_WORLD);
    SoAReaderCallback cb(rdr, rank, comm_sz);
    rdr.set_callback(cb);
    rdr.read_definitions();
    rdr.read_events();

    auto t_read = std::chrono::high_resolution_clock::now();
    double read_ms = std::chrono::duration<double, std::milli>(t_read - t_start).count();

    if (rank == 0) {
      std::cout << "[Reader] Local read done in " << read_ms << " ms" << std::endl;
    }

    if (comm_sz == 1) {
      // Single process: no MPI gathering needed
      total_sends = cb.getSendCount();
      total_recvs = cb.getRecvCount();
      total_colls = cb.getCollCount();
      cb.fillSoA(output.data);
      output.comm_sets = std::move(cb.getCommSets());
    } else {
      // Multi-process: gather all events to rank 0
      gatherEventsToRank0(cb, rank, comm_sz, output,
                          total_sends, total_recvs, total_colls);
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

  if (rank == 0) {
    size_t total_events = (comm_sz == 1) ? output.data.count :
                          total_sends + total_recvs + total_colls;
    std::cout << "[Reader] Read " << total_events << " events"
              << " (" << comm_sz << " MPI ranks) in " << total_ms << " ms"
              << std::endl;
    std::cout << "[Reader] Breakdown: " << total_sends << " sends, "
              << total_recvs << " recvs, "
              << total_colls << " collectives" << std::endl;
  }

  return output;
}
