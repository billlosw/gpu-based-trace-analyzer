#include "reader/SoACache.h"
#include "common/SHA256.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/stat.h>

// ============================================================
// Public API
// ============================================================

void computeFileFingerprint(const std::string &file_path, uint8_t out[32]) {
  sha256_file(file_path, out);
}

std::string getCachePath(const std::string &trace_path, int rank, int nprocs) {
  // trace_path is like "/path/to/traces.otf2"
  // Cache file: "/path/to/soa_cache/soa_cache_r<rank>_n<nprocs>.bin"
  std::string dir;
  size_t slash = trace_path.rfind('/');
  if (slash != std::string::npos)
    dir = trace_path.substr(0, slash + 1);
  else
    dir = "./";
  std::string cache_dir = dir + "soa_cache/";
  // Create soa_cache/ directory if it doesn't exist (rank 0 creates it;
  // other ranks may race harmlessly — mkdir returns EEXIST which we ignore)
  mkdir(cache_dir.c_str(), 0755);
  return cache_dir + "soa_cache_r" + std::to_string(rank) + "_n" +
         std::to_string(nprocs) + ".bin";
}

bool isCacheValid(const std::string &cache_path, const uint8_t fingerprint[32],
                  int rank, int nprocs, SoACacheHeader &header) {
  std::ifstream f(cache_path, std::ios::binary);
  if (!f.is_open())
    return false;

  f.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!f.good() || f.gcount() != sizeof(header))
    return false;

  if (header.magic != SOA_CACHE_MAGIC)
    return false;
  if (header.version != SOA_CACHE_VERSION)
    return false;
  if (header.nprocs != (uint32_t)nprocs)
    return false;
  if (header.rank != (uint32_t)rank)
    return false;
  if (memcmp(header.fingerprint, fingerprint, 32) != 0)
    return false;

  return true;
}

bool writeSoACache(const std::string &cache_path,
                   const uint8_t fingerprint[32], int rank, int nprocs,
                   const SoACacheData &data) {
  std::ofstream f(cache_path, std::ios::binary | std::ios::trunc);
  if (!f.is_open()) {
    std::cerr << "[SoACache] Warning: cannot write cache to " << cache_path
              << std::endl;
    return false;
  }

  SoACacheHeader hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.magic = SOA_CACHE_MAGIC;
  hdr.version = SOA_CACHE_VERSION;
  hdr.nprocs = nprocs;
  hdr.rank = rank;
  memcpy(hdr.fingerprint, fingerprint, 32);
  hdr.event_count = data.events.size();
  hdr.num_comm_sets = data.comm_sets.size();
  hdr.flags = 0;
  if (!data.leave_recv_ts.empty())
    hdr.flags |= SOA_CACHE_FLAG_HAS_LEAVE_RECV_TS;

  f.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));

  size_t n = hdr.event_count;
  if (n > 0) {
    f.write(reinterpret_cast<const char *>(data.events.data()),
            n * sizeof(int32_t));
    f.write(reinterpret_cast<const char *>(data.types.data()),
            n * sizeof(int32_t));
    f.write(reinterpret_cast<const char *>(data.timestamps.data()),
            n * sizeof(uint64_t));
    f.write(reinterpret_cast<const char *>(data.end_timestamps.data()),
            n * sizeof(uint64_t));
    f.write(reinterpret_cast<const char *>(data.pids.data()),
            n * sizeof(uint32_t));
    f.write(reinterpret_cast<const char *>(data.srcs.data()),
            n * sizeof(uint32_t));
    f.write(reinterpret_cast<const char *>(data.dsts.data()),
            n * sizeof(uint32_t));
    f.write(reinterpret_cast<const char *>(data.tags.data()),
            n * sizeof(uint32_t));
    f.write(reinterpret_cast<const char *>(data.roots.data()),
            n * sizeof(uint32_t));
    if (hdr.flags & SOA_CACHE_FLAG_HAS_LEAVE_RECV_TS) {
      f.write(reinterpret_cast<const char *>(data.leave_recv_ts.data()),
              n * sizeof(uint64_t));
    }
  }

  // Write comm_sets (variable length)
  for (const auto &cs : data.comm_sets) {
    uint64_t sz = cs.size();
    f.write(reinterpret_cast<const char *>(&sz), sizeof(sz));
    if (sz > 0)
      f.write(reinterpret_cast<const char *>(cs.data()),
              sz * sizeof(uint64_t));
  }

  // Write collective bytes
  if (!data.coll_bytes_sent.empty())
    f.write(reinterpret_cast<const char *>(data.coll_bytes_sent.data()),
            data.coll_bytes_sent.size() * sizeof(uint64_t));
  if (!data.coll_bytes_received.empty())
    f.write(reinterpret_cast<const char *>(data.coll_bytes_received.data()),
            data.coll_bytes_received.size() * sizeof(uint64_t));

  f.close();
  return f.good() || !f.fail();
}

bool readSoACache(const std::string &cache_path,
                  const uint8_t fingerprint[32], int rank, int nprocs,
                  SoACacheData &data) {
  SoACacheHeader hdr;
  if (!isCacheValid(cache_path, fingerprint, rank, nprocs, hdr))
    return false;

  std::ifstream f(cache_path, std::ios::binary);
  if (!f.is_open())
    return false;

  // Skip header
  f.seekg(sizeof(SoACacheHeader));

  size_t n = hdr.event_count;
  if (n > 0) {
    data.events.resize(n);
    data.types.resize(n);
    data.timestamps.resize(n);
    data.end_timestamps.resize(n);
    data.pids.resize(n);
    data.srcs.resize(n);
    data.dsts.resize(n);
    data.tags.resize(n);
    data.roots.resize(n);

    f.read(reinterpret_cast<char *>(data.events.data()), n * sizeof(int32_t));
    f.read(reinterpret_cast<char *>(data.types.data()), n * sizeof(int32_t));
    f.read(reinterpret_cast<char *>(data.timestamps.data()),
           n * sizeof(uint64_t));
    f.read(reinterpret_cast<char *>(data.end_timestamps.data()),
           n * sizeof(uint64_t));
    f.read(reinterpret_cast<char *>(data.pids.data()), n * sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(data.srcs.data()), n * sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(data.dsts.data()), n * sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(data.tags.data()), n * sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(data.roots.data()), n * sizeof(uint32_t));
    if (hdr.flags & SOA_CACHE_FLAG_HAS_LEAVE_RECV_TS) {
      data.leave_recv_ts.resize(n);
      f.read(reinterpret_cast<char *>(data.leave_recv_ts.data()),
             n * sizeof(uint64_t));
    }
  }

  // Read comm_sets
  size_t ncs = hdr.num_comm_sets;
  data.comm_sets.resize(ncs);
  for (size_t i = 0; i < ncs; i++) {
    uint64_t sz;
    f.read(reinterpret_cast<char *>(&sz), sizeof(sz));
    data.comm_sets[i].resize(sz);
    if (sz > 0)
      f.read(reinterpret_cast<char *>(data.comm_sets[i].data()),
             sz * sizeof(uint64_t));
  }

  // Read collective bytes
  if (ncs > 0) {
    data.coll_bytes_sent.resize(ncs);
    data.coll_bytes_received.resize(ncs);
    f.read(reinterpret_cast<char *>(data.coll_bytes_sent.data()),
           ncs * sizeof(uint64_t));
    f.read(reinterpret_cast<char *>(data.coll_bytes_received.data()),
           ncs * sizeof(uint64_t));
  }

  if (!f.good() && !f.eof()) {
    std::cerr << "[SoACache] Warning: read error in " << cache_path
              << std::endl;
    return false;
  }

  return true;
}
