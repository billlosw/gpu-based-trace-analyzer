#include "reader/OTF2SoAReader.h"

#include <chrono>
#include <iostream>
#include <tuple>
#include <vector>

#include <otf2xx/otf2.hpp>

// Internal callback class for otf2xx reader — single-pass using vectors
class SoAReaderCallback : public otf2::reader::callback {
public:
  SoAReaderCallback(otf2::reader::reader &rdr) : m_rdr(rdr) {}

  // --- Definition callbacks ---
  void definition(const otf2::definition::location &loc) override {
    m_rdr.register_location(loc);
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

private:
  otf2::reader::reader &m_rdr;

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

ReaderOutput readOTF2Trace(const std::string &trace_path) {
  ReaderOutput output;

  auto t_start = std::chrono::high_resolution_clock::now();

  // Single-pass reading using dynamic vectors
  {
    otf2::reader::reader rdr(trace_path);
    SoAReaderCallback cb(rdr);
    rdr.set_callback(cb);
    rdr.read_definitions();
    rdr.read_events();

    auto t_read = std::chrono::high_resolution_clock::now();
    double read_ms = std::chrono::duration<double, std::milli>(t_read - t_start).count();

    size_t total = cb.getEventCount();
    std::cout << "[Reader] Read " << total << " events in " << read_ms << " ms"
              << std::endl;
    std::cout << "[Reader] Breakdown: " << cb.getSendCount() << " sends, "
              << cb.getRecvCount() << " recvs, "
              << cb.getCollCount() << " collectives" << std::endl;

    // Copy from vectors to SoA
    cb.fillSoA(output.data);
    output.comm_sets = std::move(cb.getCommSets());
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  std::cout << "[Reader] Total (read + copy): " << total_ms << " ms" << std::endl;

  return output;
}
