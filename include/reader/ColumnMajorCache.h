#ifndef GPU_ANALYZER_COLUMN_MAJOR_CACHE_H
#define GPU_ANALYZER_COLUMN_MAJOR_CACHE_H

#include "common/types.h"
#include "reader/SoACache.h"
#include <cstdint>
#include <string>
#include <vector>

// Column-major single-file SoA cache format.
//
// All ranks' data is merged into a single file organized by column.
// Each column contains all events from all ranks contiguously, ordered by rank.
// Columns are 4KB-aligned for future cuFile/GDS compatibility.
//
// File layout:
//   [Header: 4KB padded]
//     magic, version, flags, nprocs, total_events, fingerprint[32]
//     num_columns, metadata_offset
//     rank_event_counts[nprocs]     : uint64
//     column_offsets[num_columns]   : uint64  (file byte offset for each column)
//     column_elem_sizes[num_columns]: uint32  (element size for each column)
//
//   [Column 0: events]       all_events[total_events]      @ 4KB-aligned
//   [Column 1: types]        all_types[total_events]       @ 4KB-aligned
//   [Column 2: timestamps]   all_timestamps[total_events]  @ 4KB-aligned
//   [Column 3: end_ts]       all_end_timestamps[total_events] @ 4KB-aligned
//   [Column 4: pids]         all_pids[total_events]        @ 4KB-aligned
//   [Column 5: srcs]         all_srcs[total_events]        @ 4KB-aligned
//   [Column 6: dsts]         all_dsts[total_events]        @ 4KB-aligned
//   [Column 7: tags]         all_tags[total_events]        @ 4KB-aligned
//   [Column 8: roots]        all_roots[total_events]       @ 4KB-aligned
//   [Column 9: leave_recv_ts] (conditional on flags)       @ 4KB-aligned
//
//   [Metadata section: variable-length]
//     Per-rank: num_comm_sets, [size:u64, members[]:u64]...,
//               coll_bytes_sent[], coll_bytes_received[]
//
// Within each column, data is ordered by rank:
//   [rank0_data | rank1_data | ... | rankP-1_data]

static constexpr uint64_t COLMAJOR_CACHE_MAGIC = 0x4C4F434D414F53ULL; // "SOAMCOL\0" approx
static constexpr uint32_t COLMAJOR_CACHE_VERSION = 1;
static constexpr uint32_t COLMAJOR_FLAG_HAS_LEAVE_RECV_TS = 1;
static constexpr int COLMAJOR_MAX_COLUMNS = 10;
static constexpr size_t COLMAJOR_HEADER_SIZE = 4096; // padded to 4KB

struct ColumnMajorHeader {
  uint64_t magic;               // COLMAJOR_CACHE_MAGIC
  uint32_t version;             // format version
  uint32_t flags;               // feature flags
  uint32_t nprocs;              // number of MPI ranks
  uint32_t num_columns;         // number of data columns (9 or 10)
  uint64_t total_events;        // sum of all ranks' events
  uint8_t  fingerprint[32];     // SHA-256 of trace anchor file
  uint64_t metadata_offset;     // file offset for variable-length metadata
  // Followed by variable-length arrays in the padded header:
  //   rank_event_counts[nprocs]: uint64
  //   column_offsets[num_columns]: uint64
  //   column_elem_sizes[num_columns]: uint32
};

// Opaque handle for mmap'd column-major cache file.
// Provides direct const pointers into mmap'd regions for each column,
// and per-rank offset/count for slicing.
struct ColumnMajorMmap {
  void *mmap_base = nullptr;
  size_t mmap_size = 0;
  int fd = -1;

  uint32_t nprocs = 0;
  uint64_t total_events = 0;
  uint32_t num_columns = 0;
  uint32_t flags = 0;

  // Per-rank cumulative event boundaries: rank_boundaries[i] = sum of events for ranks 0..i-1
  // rank_boundaries[nprocs] = total_events
  std::vector<uint64_t> rank_boundaries;

  // Column pointers into mmap'd region (read-only)
  const void *column_ptrs[COLMAJOR_MAX_COLUMNS] = {};
  uint32_t column_elem_sizes[COLMAJOR_MAX_COLUMNS] = {};

  // Metadata: per-rank comm_sets and coll_bytes (loaded into heap)
  struct RankMeta {
    std::vector<std::vector<uint64_t>> comm_sets;
    std::vector<uint64_t> coll_bytes_sent;
    std::vector<uint64_t> coll_bytes_received;
  };
  std::vector<RankMeta> rank_meta;

  // Column indices
  static constexpr int COL_EVENTS = 0;
  static constexpr int COL_TYPES = 1;
  static constexpr int COL_TIMESTAMPS = 2;
  static constexpr int COL_END_TIMESTAMPS = 3;
  static constexpr int COL_PIDS = 4;
  static constexpr int COL_SRCS = 5;
  static constexpr int COL_DSTS = 6;
  static constexpr int COL_TAGS = 7;
  static constexpr int COL_ROOTS = 8;
  static constexpr int COL_LEAVE_RECV_TS = 9;

  bool hasLeaveRecvTs() const { return (flags & COLMAJOR_FLAG_HAS_LEAVE_RECV_TS) != 0; }

  // Get typed const pointer to a rank's slice of a column
  template <typename T>
  const T *rankSlice(int col, int rank) const {
    return static_cast<const T *>(column_ptrs[col]) + rank_boundaries[rank];
  }

  // Get event count for a specific rank
  uint64_t rankEventCount(int rank) const {
    return rank_boundaries[rank + 1] - rank_boundaries[rank];
  }

  void close();
  ~ColumnMajorMmap() { close(); }

  // Non-copyable
  ColumnMajorMmap() = default;
  ColumnMajorMmap(const ColumnMajorMmap &) = delete;
  ColumnMajorMmap &operator=(const ColumnMajorMmap &) = delete;
  ColumnMajorMmap(ColumnMajorMmap &&o) noexcept;
  ColumnMajorMmap &operator=(ColumnMajorMmap &&o) noexcept;
};

// Get the column-major cache file path for a given trace path and nprocs.
std::string getColumnMajorCachePath(const std::string &trace_path, int nprocs);

// Check if a valid column-major cache file exists with matching fingerprint.
bool isColumnMajorCacheValid(const std::string &cache_path,
                             const uint8_t fingerprint[32], int nprocs);

// Write a column-major cache file. Called collectively by all MPI ranks.
// Each rank provides its own SoACacheData. Rank 0 gathers and writes.
bool writeColumnMajorCache(const std::string &cache_path,
                           const uint8_t fingerprint[32], int nprocs,
                           const SoACacheData &local_data, int rank);

// Open and mmap a column-major cache file. Returns a handle with direct
// pointers into the file for each column and per-rank metadata.
// All ranks can call this; each gets its own mmap mapping.
bool openColumnMajorCache(const std::string &cache_path,
                          const uint8_t fingerprint[32], int nprocs,
                          ColumnMajorMmap &out);

#endif // GPU_ANALYZER_COLUMN_MAJOR_CACHE_H
