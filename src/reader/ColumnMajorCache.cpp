#include "reader/ColumnMajorCache.h"
#include "common/SHA256.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <mpi.h>
#include <algorithm>
#include <climits>

// ============================================================
// Helpers
// ============================================================

static size_t align4K(size_t x) {
  return (x + 4095) & ~(size_t)4095;
}

std::string getColumnMajorCachePath(const std::string &trace_path, int nprocs) {
  std::string dir;
  size_t slash = trace_path.rfind('/');
  if (slash != std::string::npos)
    dir = trace_path.substr(0, slash + 1);
  else
    dir = "./";
  std::string cache_dir = dir + "soa_cache/";
  mkdir(cache_dir.c_str(), 0755);
  return cache_dir + "soa_cache_colmajor_n" + std::to_string(nprocs) + ".bin";
}

// Serialize metadata for one rank into a byte buffer
static std::vector<char> serializeRankMeta(
    const std::vector<std::vector<uint64_t>> &comm_sets,
    const std::vector<uint64_t> &cbs,
    const std::vector<uint64_t> &cbr) {
  // Calculate size
  size_t sz = sizeof(uint64_t); // ncs
  for (const auto &cs : comm_sets)
    sz += sizeof(uint64_t) + cs.size() * sizeof(uint64_t);
  sz += cbs.size() * sizeof(uint64_t);
  sz += cbr.size() * sizeof(uint64_t);

  std::vector<char> buf(sz);
  char *p = buf.data();

  uint64_t ncs = comm_sets.size();
  memcpy(p, &ncs, sizeof(ncs)); p += sizeof(ncs);
  for (const auto &cs : comm_sets) {
    uint64_t csz = cs.size();
    memcpy(p, &csz, sizeof(csz)); p += sizeof(csz);
    if (csz > 0) {
      memcpy(p, cs.data(), csz * sizeof(uint64_t));
      p += csz * sizeof(uint64_t);
    }
  }
  if (!cbs.empty()) {
    memcpy(p, cbs.data(), cbs.size() * sizeof(uint64_t));
    p += cbs.size() * sizeof(uint64_t);
  }
  if (!cbr.empty()) {
    memcpy(p, cbr.data(), cbr.size() * sizeof(uint64_t));
    p += cbr.size() * sizeof(uint64_t);
  }
  return buf;
}

// ============================================================
// Validation
// ============================================================

bool isColumnMajorCacheValid(const std::string &cache_path,
                             const uint8_t fingerprint[32], int nprocs) {
  int fd = ::open(cache_path.c_str(), O_RDONLY);
  if (fd < 0) return false;

  ColumnMajorHeader hdr;
  ssize_t nr = ::read(fd, &hdr, sizeof(hdr));
  ::close(fd);

  if (nr != (ssize_t)sizeof(hdr)) return false;
  if (hdr.magic != COLMAJOR_CACHE_MAGIC) return false;
  if (hdr.version != COLMAJOR_CACHE_VERSION) return false;
  if (hdr.nprocs != (uint32_t)nprocs) return false;
  if (memcmp(hdr.fingerprint, fingerprint, 32) != 0) return false;

  return true;
}

// ============================================================
// Writer — all ranks participate collectively
// ============================================================

bool writeColumnMajorCache(const std::string &cache_path,
                           const uint8_t fingerprint[32], int nprocs,
                           const SoACacheData &local_data, int rank) {
  // Step 1: Gather event counts to all ranks
  uint64_t local_count = local_data.events.size();
  std::vector<uint64_t> all_counts(nprocs);
  MPI_Allgather(&local_count, 1, MPI_UINT64_T, all_counts.data(), 1,
                MPI_UINT64_T, MPI_COMM_WORLD);

  uint64_t total_events = 0;
  for (int i = 0; i < nprocs; i++)
    total_events += all_counts[i];

  // Step 2: Determine leave_recv_ts presence
  int local_has_lrt = local_data.leave_recv_ts.empty() ? 0 : 1;
  int global_has_lrt = 0;
  MPI_Allreduce(&local_has_lrt, &global_has_lrt, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  bool has_leave_recv = (global_has_lrt != 0);
  uint32_t num_columns = has_leave_recv ? 10 : 9;

  uint32_t elem_sizes[COLMAJOR_MAX_COLUMNS] = {
    4, 4, 8, 8, 4, 4, 4, 4, 4, 8
  };

  // Step 3: Rank 0 opens file and writes header; broadcast success
  uint64_t column_offsets[COLMAJOR_MAX_COLUMNS] = {};
  {
    size_t off = COLMAJOR_HEADER_SIZE;
    for (uint32_t c = 0; c < num_columns; c++) {
      column_offsets[c] = off;
      off += align4K(total_events * elem_sizes[c]);
    }
  }
  uint64_t metadata_offset = column_offsets[num_columns - 1] +
                              align4K(total_events * elem_sizes[num_columns - 1]);

  // Check header fits in 4KB
  size_t header_var_size = sizeof(ColumnMajorHeader) +
                           nprocs * sizeof(uint64_t) +
                           num_columns * sizeof(uint64_t) +
                           num_columns * sizeof(uint32_t);
  int ok = (header_var_size <= COLMAJOR_HEADER_SIZE) ? 1 : 0;

  std::ofstream f;
  if (rank == 0) {
    if (ok) {
      f.open(cache_path, std::ios::binary | std::ios::trunc);
      if (!f.is_open()) ok = 0;
    }
    if (ok) {
      // Write 4KB header
      std::vector<char> hdr_block(COLMAJOR_HEADER_SIZE, 0);
      auto *hdr = reinterpret_cast<ColumnMajorHeader *>(hdr_block.data());
      hdr->magic = COLMAJOR_CACHE_MAGIC;
      hdr->version = COLMAJOR_CACHE_VERSION;
      hdr->flags = has_leave_recv ? COLMAJOR_FLAG_HAS_LEAVE_RECV_TS : 0;
      hdr->nprocs = nprocs;
      hdr->num_columns = num_columns;
      hdr->total_events = total_events;
      memcpy(hdr->fingerprint, fingerprint, 32);
      hdr->metadata_offset = metadata_offset;

      char *p = hdr_block.data() + sizeof(ColumnMajorHeader);
      memcpy(p, all_counts.data(), nprocs * sizeof(uint64_t));
      p += nprocs * sizeof(uint64_t);
      memcpy(p, column_offsets, num_columns * sizeof(uint64_t));
      p += num_columns * sizeof(uint64_t);
      memcpy(p, elem_sizes, num_columns * sizeof(uint32_t));

      f.write(hdr_block.data(), COLMAJOR_HEADER_SIZE);
    }
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!ok) return false;

  // Step 4: For each column, all ranks participate in gather, rank 0 writes.
  // We use a mode broadcast so all ranks agree on gatherv vs chunked send/recv.
  for (uint32_t c = 0; c < num_columns; c++) {
    uint32_t esize = elem_sizes[c];
    int64_t local_bytes = (int64_t)local_count * esize;
    int64_t total_bytes = (int64_t)total_events * esize;

    // Determine mode: 0 = MPI_Gatherv (fits in int), 1 = chunked send/recv
    int mode = 0;
    // Check if any displacement exceeds INT_MAX
    {
      int64_t max_displ = 0;
      for (int i = 0; i < nprocs; i++) {
        if (max_displ > INT_MAX || (int64_t)all_counts[i] * esize > INT_MAX) {
          mode = 1;
          break;
        }
        max_displ += (int64_t)all_counts[i] * esize;
      }
      if (max_displ > INT_MAX) mode = 1;
    }

    const void *local_ptr = nullptr;
    std::vector<uint64_t> lrt_zeros; // for leave_recv_ts fallback

    switch (c) {
      case 0: local_ptr = local_data.events.data(); break;
      case 1: local_ptr = local_data.types.data(); break;
      case 2: local_ptr = local_data.timestamps.data(); break;
      case 3: local_ptr = local_data.end_timestamps.data(); break;
      case 4: local_ptr = local_data.pids.data(); break;
      case 5: local_ptr = local_data.srcs.data(); break;
      case 6: local_ptr = local_data.dsts.data(); break;
      case 7: local_ptr = local_data.tags.data(); break;
      case 8: local_ptr = local_data.roots.data(); break;
      case 9:
        if (!local_data.leave_recv_ts.empty()) {
          local_ptr = local_data.leave_recv_ts.data();
        } else if (local_count > 0) {
          lrt_zeros.resize(local_count, 0);
          local_ptr = lrt_zeros.data();
        }
        break;
    }

    if (mode == 0) {
      // MPI_Gatherv — all ranks participate
      std::vector<int> recv_counts, displs;
      std::vector<char> col_buf;
      if (rank == 0) {
        recv_counts.resize(nprocs);
        displs.resize(nprocs);
        int64_t d = 0;
        for (int i = 0; i < nprocs; i++) {
          recv_counts[i] = (int)((int64_t)all_counts[i] * esize);
          displs[i] = (int)d;
          d += recv_counts[i];
        }
        col_buf.resize(total_bytes);
      }
      MPI_Gatherv(local_ptr, (int)local_bytes, MPI_BYTE,
                   rank == 0 ? col_buf.data() : nullptr,
                   rank == 0 ? recv_counts.data() : nullptr,
                   rank == 0 ? displs.data() : nullptr,
                   MPI_BYTE, 0, MPI_COMM_WORLD);

      if (rank == 0) {
        f.write(col_buf.data(), total_bytes);
        size_t aligned = align4K(total_bytes);
        if (aligned > (size_t)total_bytes) {
          std::vector<char> pad(aligned - total_bytes, 0);
          f.write(pad.data(), pad.size());
        }
      }
    } else {
      // Chunked: rank 0 receives from each rank individually
      if (rank == 0) {
        // Write rank 0's data first
        if (local_bytes > 0)
          f.write(static_cast<const char *>(local_ptr), local_bytes);

        // Receive from each other rank
        for (int r = 1; r < nprocs; r++) {
          int64_t r_bytes = (int64_t)all_counts[r] * esize;
          if (r_bytes == 0) continue;
          // Receive in chunks
          std::vector<char> chunk_buf(std::min(r_bytes, (int64_t)256 * 1024 * 1024));
          int64_t remaining = r_bytes;
          while (remaining > 0) {
            int chunk = (int)std::min(remaining, (int64_t)chunk_buf.size());
            MPI_Recv(chunk_buf.data(), chunk, MPI_BYTE, r,
                     c * 1000 + r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            f.write(chunk_buf.data(), chunk);
            remaining -= chunk;
          }
        }
        // Pad to 4KB
        size_t aligned = align4K(total_bytes);
        if (aligned > (size_t)total_bytes) {
          std::vector<char> pad(aligned - total_bytes, 0);
          f.write(pad.data(), pad.size());
        }
      } else {
        // Non-rank 0: send data to rank 0 in chunks
        if (local_bytes > 0) {
          int64_t remaining = local_bytes;
          int64_t sent = 0;
          int64_t chunk_max = 256LL * 1024 * 1024;
          while (remaining > 0) {
            int chunk = (int)std::min(remaining, chunk_max);
            MPI_Send(static_cast<const char *>(local_ptr) + sent, chunk,
                     MPI_BYTE, 0, c * 1000 + rank, MPI_COMM_WORLD);
            sent += chunk;
            remaining -= chunk;
          }
        }
      }
    }
  }

  // Step 5: Metadata — each non-rank-0 sends serialized blob to rank 0
  std::vector<char> my_meta = serializeRankMeta(
      local_data.comm_sets, local_data.coll_bytes_sent,
      local_data.coll_bytes_received);

  if (rank == 0) {
    // Write rank 0's metadata
    f.write(my_meta.data(), my_meta.size());

    // Receive and write metadata from each other rank
    for (int r = 1; r < nprocs; r++) {
      int64_t blob_size = 0;
      MPI_Recv(&blob_size, 1, MPI_INT64_T, r, 50000 + r,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (blob_size > 0) {
        std::vector<char> blob(blob_size);
        MPI_Recv(blob.data(), (int)blob_size, MPI_BYTE, r, 60000 + r,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        f.write(blob.data(), blob_size);
      }
    }

    f.close();
    bool write_ok = !f.fail();
    if (!write_ok)
      std::cerr << "[ColMajorCache] Write error" << std::endl;
    ok = write_ok ? 1 : 0;
  } else {
    // Send metadata blob to rank 0
    int64_t blob_size = (int64_t)my_meta.size();
    MPI_Send(&blob_size, 1, MPI_INT64_T, 0, 50000 + rank, MPI_COMM_WORLD);
    if (blob_size > 0)
      MPI_Send(my_meta.data(), (int)blob_size, MPI_BYTE, 0, 60000 + rank,
               MPI_COMM_WORLD);
  }

  // Broadcast final write status
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return (ok == 1);
}

// ============================================================
// Reader (mmap)
// ============================================================

ColumnMajorMmap::ColumnMajorMmap(ColumnMajorMmap &&o) noexcept
    : mmap_base(o.mmap_base), mmap_size(o.mmap_size), fd(o.fd),
      nprocs(o.nprocs), total_events(o.total_events),
      num_columns(o.num_columns), flags(o.flags),
      rank_boundaries(std::move(o.rank_boundaries)),
      rank_meta(std::move(o.rank_meta)) {
  memcpy(column_ptrs, o.column_ptrs, sizeof(column_ptrs));
  memcpy(column_elem_sizes, o.column_elem_sizes, sizeof(column_elem_sizes));
  o.mmap_base = nullptr;
  o.mmap_size = 0;
  o.fd = -1;
}

ColumnMajorMmap &ColumnMajorMmap::operator=(ColumnMajorMmap &&o) noexcept {
  if (this != &o) {
    close();
    mmap_base = o.mmap_base;
    mmap_size = o.mmap_size;
    fd = o.fd;
    nprocs = o.nprocs;
    total_events = o.total_events;
    num_columns = o.num_columns;
    flags = o.flags;
    rank_boundaries = std::move(o.rank_boundaries);
    rank_meta = std::move(o.rank_meta);
    memcpy(column_ptrs, o.column_ptrs, sizeof(column_ptrs));
    memcpy(column_elem_sizes, o.column_elem_sizes, sizeof(column_elem_sizes));
    o.mmap_base = nullptr;
    o.mmap_size = 0;
    o.fd = -1;
  }
  return *this;
}

void ColumnMajorMmap::close() {
  if (mmap_base && mmap_base != MAP_FAILED) {
    munmap(mmap_base, mmap_size);
    mmap_base = nullptr;
  }
  if (fd >= 0) {
    ::close(fd);
    fd = -1;
  }
  memset(column_ptrs, 0, sizeof(column_ptrs));
}

bool openColumnMajorCache(const std::string &cache_path,
                          const uint8_t fingerprint[32], int nprocs,
                          ColumnMajorMmap &out) {
  // Open read-only. mmap with MAP_PRIVATE + PROT_READ|PROT_WRITE to allow
  // cudaHostRegister (requires writable pages). MAP_PRIVATE gives COW semantics —
  // no writes actually happen to read-only columns, so no copy overhead.
  int fd = ::open(cache_path.c_str(), O_RDONLY);
  if (fd < 0) return false;

  struct stat st;
  if (fstat(fd, &st) != 0) { ::close(fd); return false; }
  size_t file_size = st.st_size;
  if (file_size < COLMAJOR_HEADER_SIZE) { ::close(fd); return false; }

  // MAP_PRIVATE allows PROT_WRITE even on O_RDONLY fd (COW pages).
  // cudaHostRegister requires PROT_WRITE to pin pages for DMA.
  void *base = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  if (base == MAP_FAILED) {
    std::cerr << "[ColMajorCache] mmap failed for " << cache_path << std::endl;
    ::close(fd);
    return false;
  }

  // Prefetch hints
  madvise(base, file_size, MADV_WILLNEED);

  // Validate header
  const auto *hdr = static_cast<const ColumnMajorHeader *>(base);
  if (hdr->magic != COLMAJOR_CACHE_MAGIC ||
      hdr->version != COLMAJOR_CACHE_VERSION ||
      hdr->nprocs != (uint32_t)nprocs ||
      memcmp(hdr->fingerprint, fingerprint, 32) != 0) {
    munmap(base, file_size);
    ::close(fd);
    return false;
  }

  out.mmap_base = base;
  out.mmap_size = file_size;
  out.fd = fd;
  out.nprocs = hdr->nprocs;
  out.total_events = hdr->total_events;
  out.num_columns = hdr->num_columns;
  out.flags = hdr->flags;

  // Parse variable-length header arrays
  const char *p = static_cast<const char *>(base) + sizeof(ColumnMajorHeader);
  const uint64_t *rank_counts = reinterpret_cast<const uint64_t *>(p);
  p += nprocs * sizeof(uint64_t);
  const uint64_t *col_offsets = reinterpret_cast<const uint64_t *>(p);
  p += hdr->num_columns * sizeof(uint64_t);
  const uint32_t *col_esizes = reinterpret_cast<const uint32_t *>(p);

  // Build rank boundaries (prefix sum)
  out.rank_boundaries.resize(nprocs + 1);
  out.rank_boundaries[0] = 0;
  for (int i = 0; i < nprocs; i++)
    out.rank_boundaries[i + 1] = out.rank_boundaries[i] + rank_counts[i];

  // Set column pointers directly into mmap'd region
  for (uint32_t c = 0; c < hdr->num_columns; c++) {
    out.column_ptrs[c] = static_cast<const char *>(base) + col_offsets[c];
    out.column_elem_sizes[c] = col_esizes[c];
  }

  // Parse metadata section (heap copy — small data)
  const char *meta = static_cast<const char *>(base) + hdr->metadata_offset;
  const char *file_end = static_cast<const char *>(base) + file_size;
  out.rank_meta.resize(nprocs);

  for (int r = 0; r < nprocs && meta + sizeof(uint64_t) <= file_end; r++) {
    uint64_t ncs = 0;
    memcpy(&ncs, meta, sizeof(ncs)); meta += sizeof(ncs);

    out.rank_meta[r].comm_sets.resize(ncs);
    for (uint64_t i = 0; i < ncs && meta + sizeof(uint64_t) <= file_end; i++) {
      uint64_t sz = 0;
      memcpy(&sz, meta, sizeof(sz)); meta += sizeof(sz);
      out.rank_meta[r].comm_sets[i].resize(sz);
      if (sz > 0 && meta + sz * sizeof(uint64_t) <= file_end) {
        memcpy(out.rank_meta[r].comm_sets[i].data(), meta, sz * sizeof(uint64_t));
        meta += sz * sizeof(uint64_t);
      }
    }
    if (ncs > 0 && meta + 2 * ncs * sizeof(uint64_t) <= file_end) {
      out.rank_meta[r].coll_bytes_sent.resize(ncs);
      memcpy(out.rank_meta[r].coll_bytes_sent.data(), meta, ncs * sizeof(uint64_t));
      meta += ncs * sizeof(uint64_t);
      out.rank_meta[r].coll_bytes_received.resize(ncs);
      memcpy(out.rank_meta[r].coll_bytes_received.data(), meta, ncs * sizeof(uint64_t));
      meta += ncs * sizeof(uint64_t);
    }
  }

  return true;
}
