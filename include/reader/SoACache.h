#ifndef GPU_ANALYZER_SOA_CACHE_H
#define GPU_ANALYZER_SOA_CACHE_H

#include "common/types.h"
#include <cstdint>
#include <string>
#include <vector>

// Binary SoA cache format for fast trace re-loading.
// Eliminates OTF2 callback-driven parsing on subsequent runs.
//
// File layout:
//   SoACacheHeader (fixed 100 bytes)
//   events[count]         : int32
//   types[count]          : int32
//   timestamps[count]     : uint64
//   end_timestamps[count] : uint64
//   pids[count]           : uint32
//   srcs[count]           : uint32
//   dsts[count]           : uint32
//   tags[count]           : uint32
//   roots[count]          : uint32
//   leave_recv_ts[count]  : uint64 (conditional on flags)
//   comm_sets (variable): [size:u64, members[size]:u64] ...
//   coll_bytes_sent[num_comm_sets]    : uint64
//   coll_bytes_received[num_comm_sets]: uint64

static constexpr uint64_t SOA_CACHE_MAGIC = 0x4548434143414F53ULL; // "SOACACHE"
static constexpr uint32_t SOA_CACHE_VERSION = 1;

// Flag bits
static constexpr uint32_t SOA_CACHE_FLAG_HAS_LEAVE_RECV_TS = 1;

struct SoACacheHeader {
  uint64_t magic;           // "SOACACHE"
  uint32_t version;         // format version
  uint32_t nprocs;          // number of MPI ranks in the cache
  uint32_t rank;            // this rank
  uint32_t flags;           // feature flags
  uint8_t  fingerprint[32]; // SHA-256 of trace anchor file
  uint64_t event_count;     // number of events
  uint64_t num_comm_sets;   // number of collective comm_sets
  uint8_t  reserved[24];    // must be zero
};

// Compute SHA-256 fingerprint of a file. Returns 32-byte hash.
// Uses a minimal built-in SHA-256 (no external dependency).
void computeFileFingerprint(const std::string &file_path, uint8_t out[32]);

// Get the cache file path for a given trace path, rank, and nprocs.
std::string getCachePath(const std::string &trace_path, int rank, int nprocs);

// Check if a valid cache file exists and its fingerprint matches.
// Returns true if cache is valid, fills header with the read header.
bool isCacheValid(const std::string &cache_path, const uint8_t fingerprint[32],
                  int rank, int nprocs, SoACacheHeader &header);

// Data bundle for cache read/write (per-rank data after redistribution)
struct SoACacheData {
  // SoA arrays (size = event_count)
  std::vector<int32_t> events;
  std::vector<int32_t> types;
  std::vector<uint64_t> timestamps;
  std::vector<uint64_t> end_timestamps;
  std::vector<uint32_t> pids;
  std::vector<uint32_t> srcs;
  std::vector<uint32_t> dsts;
  std::vector<uint32_t> tags;
  std::vector<uint32_t> roots;
  std::vector<uint64_t> leave_recv_ts; // empty if not applicable

  // Collective comm_sets
  std::vector<std::vector<uint64_t>> comm_sets;
  std::vector<uint64_t> coll_bytes_sent;
  std::vector<uint64_t> coll_bytes_received;
};

// Write cache file for a single rank.
bool writeSoACache(const std::string &cache_path, const uint8_t fingerprint[32],
                   int rank, int nprocs, const SoACacheData &data);

// Read cache file for a single rank.
// Returns true on success, fills data.
bool readSoACache(const std::string &cache_path, const uint8_t fingerprint[32],
                  int rank, int nprocs, SoACacheData &data);

// Read ONLY the comm_sets and collective bytes from a cache file (skips SoA arrays).
// Used to re-load comm_sets after they were freed to reduce peak memory.
bool readCacheCommSets(const std::string &cache_path,
                       const uint8_t fingerprint[32], int rank, int nprocs,
                       std::vector<std::vector<uint64_t>> &comm_sets,
                       std::vector<uint64_t> &coll_bytes_sent,
                       std::vector<uint64_t> &coll_bytes_received);

#endif // GPU_ANALYZER_SOA_CACHE_H
