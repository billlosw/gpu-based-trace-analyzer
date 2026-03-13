#include "reader/OTF2SoAReader.h"

#include <chrono>
#include <iostream>
#include <tuple>
#include <vector>

#include <mpi.h>
#include <otf2xx/otf2.hpp>

// Internal callback class for otf2xx reader
class SoAReaderCallback : public otf2::reader::callback {
public:
  SoAReaderCallback(otf2::reader::reader &rdr, bool counting_pass)
      : m_rdr(rdr), m_counting(counting_pass) {}

  // --- Definition callbacks ---
  void definition(const otf2::definition::location &loc) override {
    // Register all locations for reading
    m_rdr.register_location(loc);
  }

  void definitions_done(const otf2::reader::reader &) override {
    if (!m_counting) {
      m_nlocs = m_rdr.num_locations();
    }
  }

  // --- P2P event callbacks ---
  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_send &event) override {
    if (m_counting) {
      m_event_count++;
      return;
    }
    size_t i = m_write_pos++;
    id_t pid = loc.ref().get();
    auto ts = extractTimestamp(event.timestamp());

    m_data->events[i] = TT_MPI_Send;
    m_data->types[i] = ENTER;
    m_data->timestamps[i] = ts;
    m_data->end_timestamps[i] = ts;
    m_data->pids[i] = pid;
    m_data->tids[i] = 0;
    m_data->replay_pids[i] = pid;
    m_data->srcs[i] = pid;
    m_data->dsts[i] = event.receiver();
    m_data->tags[i] = event.msg_tag();
    m_data->roots[i] = 0;
    m_data->indices[i] = (id_t)i;
  }

  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_receive &event) override {
    if (m_counting) {
      m_event_count++;
      return;
    }
    size_t i = m_write_pos++;
    id_t pid = loc.ref().get();
    auto ts = extractTimestamp(event.timestamp());

    m_data->events[i] = TT_MPI_Recv;
    m_data->types[i] = ENTER;
    m_data->timestamps[i] = ts;
    m_data->end_timestamps[i] = ts;
    m_data->pids[i] = pid;
    m_data->tids[i] = 0;
    m_data->replay_pids[i] = pid;
    m_data->srcs[i] = event.sender();
    m_data->dsts[i] = pid;
    m_data->tags[i] = event.msg_tag();
    m_data->roots[i] = 0;
    m_data->indices[i] = (id_t)i;
  }

  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_isend_request &event) override {
    if (m_counting) {
      m_event_count++;
      return;
    }
    size_t i = m_write_pos++;
    id_t pid = loc.ref().get();
    auto ts = extractTimestamp(event.timestamp());

    m_data->events[i] = TT_MPI_Isend;
    m_data->types[i] = ENTER;
    m_data->timestamps[i] = ts;
    m_data->end_timestamps[i] = ts;
    m_data->pids[i] = pid;
    m_data->tids[i] = 0;
    m_data->replay_pids[i] = pid;
    m_data->srcs[i] = pid;
    m_data->dsts[i] = event.receiver();
    m_data->tags[i] = event.msg_tag();
    m_data->roots[i] = 0;
    m_data->indices[i] = (id_t)i;
  }

  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_ireceive_complete &event) override {
    if (m_counting) {
      m_event_count++;
      return;
    }
    size_t i = m_write_pos++;
    id_t pid = loc.ref().get();
    auto ts = extractTimestamp(event.timestamp());

    m_data->events[i] = TT_MPI_Irecv;
    m_data->types[i] = ENTER;
    m_data->timestamps[i] = ts;
    m_data->end_timestamps[i] = ts;
    m_data->pids[i] = pid;
    m_data->tids[i] = 0;
    m_data->replay_pids[i] = pid;
    m_data->srcs[i] = event.sender();
    m_data->dsts[i] = pid;
    m_data->tags[i] = event.msg_tag();
    m_data->roots[i] = 0;
    m_data->indices[i] = (id_t)i;
  }

  // --- Collective event callbacks ---
  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_collective_begin &event) override {
    if (m_counting)
      return; // Don't count begin events separately
    id_t pid = loc.ref().get();
    auto ts = extractTimestamp(event.timestamp());
    m_coll_begin_ts[pid] = ts;
    m_coll_begin_valid[pid] = true;
  }

  void event(const otf2::definition::location &loc,
             const otf2::event::mpi_collective_end &event) override {
    id_t pid = loc.ref().get();

    if (m_counting) {
      // Only count if we can pair with a begin
      m_event_count++;
      return;
    }

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

    size_t i = m_write_pos++;
    m_data->events[i] = op_type;
    m_data->types[i] = ENTER;
    m_data->timestamps[i] = begin_ts;
    m_data->end_timestamps[i] = end_ts;
    m_data->pids[i] = pid;
    m_data->tids[i] = 0;
    m_data->replay_pids[i] = pid;
    m_data->srcs[i] = 0;
    m_data->dsts[i] = 0;
    m_data->tags[i] = 0;
    m_data->roots[i] = root;
    m_data->indices[i] = (id_t)i;

    // Store comm_set
    std::vector<uint64_t> cs(comm_set.begin(), comm_set.end());
    m_comm_sets_out->push_back(std::move(cs));
    m_coll_soa_indices.push_back(i);
  }

  void events_done(const otf2::reader::reader &) override {}

  // --- Accessors ---
  size_t getEventCount() const { return m_event_count; }
  void setDataTarget(TraceDataSoA *data,
                     std::vector<std::vector<uint64_t>> *comm_sets) {
    m_data = data;
    m_comm_sets_out = comm_sets;
    m_write_pos = 0;
  }
  size_t getWritePos() const { return m_write_pos; }
  const std::vector<size_t> &getCollSoAIndices() const {
    return m_coll_soa_indices;
  }

private:
  otf2::reader::reader &m_rdr;
  bool m_counting;

  // Counting pass
  size_t m_event_count = 0;

  // Writing pass
  TraceDataSoA *m_data = nullptr;
  std::vector<std::vector<uint64_t>> *m_comm_sets_out = nullptr;
  size_t m_write_pos = 0;
  size_t m_nlocs = 0;

  // Collective begin/end pairing
  std::unordered_map<id_t, timestamp_t> m_coll_begin_ts;
  std::unordered_map<id_t, bool> m_coll_begin_valid;

  // Tracks which SoA indices are collective events
  std::vector<size_t> m_coll_soa_indices;

  static timestamp_t
  extractTimestamp(const otf2::chrono::time_point &tp) {
    auto ts_ps =
        std::chrono::time_point_cast<otf2::chrono::picoseconds>(tp);
    return ts_ps.time_since_epoch().count();
  }
};

ReaderOutput readOTF2Trace(const std::string &trace_path) {
  ReaderOutput output;

  // Initialize MPI if not already initialized (for otf2xx)
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    MPI_Init(nullptr, nullptr);
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  // Pass 1: Count events
  size_t total_events = 0;
  {
    otf2::reader::reader rdr(trace_path, MPI_COMM_SELF);
    SoAReaderCallback cb(rdr, /*counting_pass=*/true);
    rdr.set_callback(cb);
    rdr.read_definitions();
    rdr.read_events();
    total_events = cb.getEventCount();
  }

  std::cout << "[Reader] Pass 1 done: " << total_events << " events counted"
            << std::endl;

  // Pass 2: Read events into SoA
  output.data.allocate(total_events);
  {
    otf2::reader::reader rdr(trace_path, MPI_COMM_SELF);
    SoAReaderCallback cb(rdr, /*counting_pass=*/false);
    cb.setDataTarget(&output.data, &output.comm_sets);
    rdr.set_callback(cb);
    rdr.read_definitions();
    rdr.read_events();
    output.data.count = cb.getWritePos();
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double read_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  std::cout << "[Reader] Pass 2 done: " << output.data.count
            << " events read in " << read_ms << " ms" << std::endl;

  return output;
}
