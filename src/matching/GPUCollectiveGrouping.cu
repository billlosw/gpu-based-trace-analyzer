#include "matching/GPUCollectiveGrouping.h"
#include "matching/CollectiveGrouping.h"
#include "common/cuda_check.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

// ============================================================================
// GPU Collective Grouping: Sort + Segment Scan
//
// Key insight: for a fixed (event_type, communicator), all instances have
// the same group size P (= communicator size). So in a sorted run of K events,
// the first P events are group 0, next P are group 1, etc.
//
// Algorithm:
//   1. Filter collective events (GPU compaction)
//   2. Sort by (event_type, comm_set_hash, timestamp)
//   3. Mark segment boundaries where (type, hash) changes
//   4. Within each segment, group_id = position_within_segment / P
//   5. Build CSR on CPU from the sorted/grouped data (small data volume)
// ============================================================================

// FNV-1a hash matching the CPU implementation in CollectiveGrouping.cpp
static uint64_t hostHashCommSet(const std::vector<uint64_t> &sorted_cs) {
  uint64_t h = 14695981039346656037ULL; // FNV offset basis
  for (uint64_t v : sorted_cs) {
    for (int b = 0; b < 8; b++) {
      h ^= (v >> (b * 8)) & 0xFF;
      h *= 1099511628211ULL; // FNV prime
    }
  }
  return h;
}

// Kernel: classify events, emit (is_collective, comm_set_index_in_coll_array)
__global__ void kernelClassifyCollective(
    const event_t *__restrict__ events,
    size_t n,
    int *__restrict__ is_coll  // 1 if collective event, 0 otherwise
) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (idx >= n) return;

  event_t ev = events[idx];
  is_coll[idx] = (ev >= TT_MPI_Bcast && ev <= TT_MPI_AlltoAll) ? 1 : 0;
}

// Kernel: for compacted collective events, build sort keys and copy needed data
__global__ void kernelBuildCollSortKeys(
    const uint32_t *__restrict__ coll_soa_indices,  // original SoA indices
    const event_t *__restrict__ events,
    const uint64_t *__restrict__ timestamps,
    const uint32_t *__restrict__ pids,
    const uint32_t *__restrict__ roots,
    const uint64_t *__restrict__ comm_hashes,  // per-coll-event hash (uploaded)
    size_t num_coll,
    // outputs
    uint64_t *__restrict__ sort_key_primary,   // (event_type << 32) | hash_low32
    uint64_t *__restrict__ sort_key_secondary  // timestamp
) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (idx >= num_coll) return;

  uint32_t soa_idx = coll_soa_indices[idx];
  event_t ev = events[soa_idx];
  uint64_t ts = timestamps[soa_idx];
  uint64_t ch = comm_hashes[idx];

  // Primary key: combine event type and comm_set hash
  // event_type in high 32 bits, hash in low 32 bits
  sort_key_primary[idx] = ((uint64_t)(uint32_t)ev << 32) | (ch & 0xFFFFFFFF);
  sort_key_secondary[idx] = ts;
}

// Kernel: assign group IDs within sorted segments
// After sorting by primary key (type|hash), each segment has events from
// one (type, communicator) pair. Within each segment, the events are sorted
// by timestamp. Group i = position_within_segment / comm_size.
__global__ void kernelAssignCollGroupIds(
    const uint64_t *__restrict__ sorted_primary_keys,
    const uint32_t *__restrict__ comm_sizes,  // per-coll-event communicator size
    const uint32_t *__restrict__ seg_positions, // position within segment
    size_t num_coll,
    uint32_t *__restrict__ group_ids,          // output: unique group id
    uint32_t *__restrict__ within_group_pos    // output: position within group
) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (idx >= num_coll) return;

  uint32_t cs = comm_sizes[idx];
  uint32_t pos = seg_positions[idx];

  if (cs == 0) {
    group_ids[idx] = UINT32_MAX;
    within_group_pos[idx] = 0;
    return;
  }

  group_ids[idx] = pos / cs;
  within_group_pos[idx] = pos % cs;
}

void buildGPUCollectiveGroups(
    const TraceDataSoA &data,
    const std::vector<std::vector<uint64_t>> &comm_sets,
    const std::vector<uint64_t> &coll_bytes_sent,
    const std::vector<uint64_t> &coll_bytes_received,
    CollectiveGroupCSR &out_csr) {

  out_csr.deallocate();
  size_t n = data.count;
  if (n == 0) return;

  auto t_start = std::chrono::high_resolution_clock::now();

  // Check GPU availability
  int device;
  cudaError_t dev_err = cudaGetDevice(&device);
  if (dev_err != cudaSuccess) {
    std::cerr << "[GPU Coll] No GPU available, falling back to CPU" << std::endl;
    buildCollectiveGroups(data, comm_sets, coll_bytes_sent, coll_bytes_received, out_csr);
    return;
  }

  try {
    // Step 1: Count and extract collective event indices (CPU fast path)
    // Since collective events are <15% of total and the extraction is simple,
    // do the initial scan on CPU to build the index and comm_set mapping.
    std::vector<uint32_t> h_coll_soa_indices;
    std::vector<size_t> h_coll_to_commset; // maps compacted index -> comm_set index

    {
      size_t cs_idx = 0;
      for (size_t i = 0; i < n; i++) {
        event_t ev = data.events[i];
        if (ev >= TT_MPI_Bcast && ev <= TT_MPI_AlltoAll) {
          h_coll_soa_indices.push_back((uint32_t)i);
          if (cs_idx < comm_sets.size()) {
            h_coll_to_commset.push_back(cs_idx);
            cs_idx++;
          } else {
            h_coll_to_commset.push_back(SIZE_MAX);
          }
        }
      }
    }

    size_t num_coll = h_coll_soa_indices.size();
    if (num_coll == 0) {
      std::cout << "[GPU Coll] No collective events found" << std::endl;
      return;
    }

    auto t_prep = std::chrono::high_resolution_clock::now();
    double prep_ms = std::chrono::duration<double, std::milli>(t_prep - t_start).count();

    // Step 2: Compute comm_set hashes and sizes on CPU (tiny data)
    // Cache: comm_set_idx -> (sorted_cs, hash, size)
    struct CachedCS {
      std::vector<uint64_t> sorted_cs;
      uint64_t hash;
      uint32_t size;
    };
    std::unordered_map<size_t, CachedCS> cs_cache;

    std::vector<uint64_t> h_comm_hashes(num_coll);
    std::vector<uint32_t> h_comm_sizes(num_coll);

    for (size_t i = 0; i < num_coll; i++) {
      size_t cs_idx = h_coll_to_commset[i];
      if (cs_idx == SIZE_MAX || cs_idx >= comm_sets.size()) {
        h_comm_hashes[i] = 0;
        h_comm_sizes[i] = 0;
        continue;
      }

      auto it = cs_cache.find(cs_idx);
      if (it == cs_cache.end()) {
        CachedCS &entry = cs_cache[cs_idx];
        entry.sorted_cs = comm_sets[cs_idx];
        std::sort(entry.sorted_cs.begin(), entry.sorted_cs.end());
        entry.hash = hostHashCommSet(entry.sorted_cs);
        entry.size = (uint32_t)entry.sorted_cs.size();
        h_comm_hashes[i] = entry.hash;
        h_comm_sizes[i] = entry.size;
      } else {
        h_comm_hashes[i] = it->second.hash;
        h_comm_sizes[i] = it->second.size;
      }
    }

    auto t_hash = std::chrono::high_resolution_clock::now();
    double hash_ms = std::chrono::duration<double, std::milli>(t_hash - t_prep).count();

    // Step 3: Upload to GPU and sort
    thrust::device_vector<uint32_t> d_coll_soa_indices(h_coll_soa_indices);
    thrust::device_vector<uint64_t> d_comm_hashes(h_comm_hashes);
    thrust::device_vector<uint32_t> d_comm_sizes(h_comm_sizes);

    // Build sort keys on GPU
    thrust::device_vector<uint64_t> d_sort_primary(num_coll);
    thrust::device_vector<uint64_t> d_sort_secondary(num_coll);

    // Upload events and timestamps for sort key construction
    thrust::device_vector<event_t> d_events(data.events, data.events + n);
    thrust::device_vector<uint64_t> d_timestamps(data.timestamps, data.timestamps + n);
    thrust::device_vector<uint32_t> d_pids_all(data.pids, data.pids + n);
    thrust::device_vector<uint32_t> d_roots_all(data.roots, data.roots + n);

    int block_size = 256;
    int num_blocks = (int)((num_coll + block_size - 1) / block_size);

    kernelBuildCollSortKeys<<<num_blocks, block_size>>>(
        thrust::raw_pointer_cast(d_coll_soa_indices.data()),
        thrust::raw_pointer_cast(d_events.data()),
        thrust::raw_pointer_cast(d_timestamps.data()),
        thrust::raw_pointer_cast(d_pids_all.data()),
        thrust::raw_pointer_cast(d_roots_all.data()),
        thrust::raw_pointer_cast(d_comm_hashes.data()),
        num_coll,
        thrust::raw_pointer_cast(d_sort_primary.data()),
        thrust::raw_pointer_cast(d_sort_secondary.data()));
    CUDA_CHECK(cudaGetLastError());

    // Free large arrays
    d_events.clear(); d_events.shrink_to_fit();
    d_timestamps.clear(); d_timestamps.shrink_to_fit();
    d_pids_all.clear(); d_pids_all.shrink_to_fit();
    d_roots_all.clear(); d_roots_all.shrink_to_fit();

    // Sort: first by secondary (timestamp), then by primary (type|hash)
    // This gives us sorted by (type|hash, timestamp) with stable sort
    thrust::stable_sort_by_key(
        d_sort_secondary.begin(), d_sort_secondary.end(),
        thrust::make_zip_iterator(thrust::make_tuple(
            d_coll_soa_indices.begin(), d_sort_primary.begin(),
            d_comm_hashes.begin(), d_comm_sizes.begin())));

    thrust::stable_sort_by_key(
        d_sort_primary.begin(), d_sort_primary.end(),
        thrust::make_zip_iterator(thrust::make_tuple(
            d_coll_soa_indices.begin(), d_sort_secondary.begin(),
            d_comm_hashes.begin(), d_comm_sizes.begin())));

    auto t_sort = std::chrono::high_resolution_clock::now();
    double sort_ms = std::chrono::duration<double, std::milli>(t_sort - t_hash).count();

    // Step 4: Compute segment positions using exclusive_scan_by_key
    thrust::device_vector<uint32_t> d_seg_positions(num_coll);
    thrust::device_vector<uint32_t> d_ones(num_coll, 1);

    thrust::exclusive_scan_by_key(
        d_sort_primary.begin(), d_sort_primary.end(),
        d_ones.begin(), d_seg_positions.begin(),
        (uint32_t)0, thrust::equal_to<uint64_t>(),
        thrust::plus<uint32_t>());
    d_ones.clear(); d_ones.shrink_to_fit();

    // Step 5: Assign group IDs
    thrust::device_vector<uint32_t> d_group_ids(num_coll);
    thrust::device_vector<uint32_t> d_within_group(num_coll);

    kernelAssignCollGroupIds<<<num_blocks, block_size>>>(
        thrust::raw_pointer_cast(d_sort_primary.data()),
        thrust::raw_pointer_cast(d_comm_sizes.data()),
        thrust::raw_pointer_cast(d_seg_positions.data()),
        num_coll,
        thrust::raw_pointer_cast(d_group_ids.data()),
        thrust::raw_pointer_cast(d_within_group.data()));
    CUDA_CHECK(cudaGetLastError());

    auto t_group = std::chrono::high_resolution_clock::now();
    double group_ms = std::chrono::duration<double, std::milli>(t_group - t_sort).count();

    // Step 6: Download results to CPU for CSR construction
    // (Collective data is small — typically <200K events for n1024)
    std::vector<uint32_t> h_sorted_soa_indices(num_coll);
    std::vector<uint64_t> h_sorted_primary(num_coll);
    std::vector<uint32_t> h_sorted_group_ids(num_coll);
    std::vector<uint32_t> h_sorted_comm_sizes(num_coll);

    thrust::copy(d_coll_soa_indices.begin(), d_coll_soa_indices.end(),
                 h_sorted_soa_indices.begin());
    thrust::copy(d_sort_primary.begin(), d_sort_primary.end(),
                 h_sorted_primary.begin());
    thrust::copy(d_group_ids.begin(), d_group_ids.end(),
                 h_sorted_group_ids.begin());
    thrust::copy(d_comm_sizes.begin(), d_comm_sizes.end(),
                 h_sorted_comm_sizes.begin());

    // Free GPU memory
    d_coll_soa_indices.clear(); d_coll_soa_indices.shrink_to_fit();
    d_sort_primary.clear(); d_sort_primary.shrink_to_fit();
    d_sort_secondary.clear(); d_sort_secondary.shrink_to_fit();
    d_comm_hashes.clear(); d_comm_hashes.shrink_to_fit();
    d_comm_sizes.clear(); d_comm_sizes.shrink_to_fit();
    d_seg_positions.clear(); d_seg_positions.shrink_to_fit();
    d_group_ids.clear(); d_group_ids.shrink_to_fit();
    d_within_group.clear(); d_within_group.shrink_to_fit();

    auto t_download = std::chrono::high_resolution_clock::now();
    double download_ms = std::chrono::duration<double, std::milli>(t_download - t_group).count();

    // Step 7: Build CSR on CPU from sorted groups
    // Each unique (primary_key, group_id) pair defines a group.
    // Need to map soa_idx -> comm_set_idx for bytes lookup.
    // Rebuild this from the original h_coll_soa_indices mapping.
    std::unordered_map<uint32_t, size_t> soa_to_commset_map;
    {
      size_t cs_idx = 0;
      for (size_t i = 0; i < n; i++) {
        event_t ev = data.events[i];
        if (ev >= TT_MPI_Bcast && ev <= TT_MPI_AlltoAll) {
          if (cs_idx < comm_sets.size()) {
            soa_to_commset_map[(uint32_t)i] = cs_idx;
            cs_idx++;
          }
        }
      }
    }

    // Identify groups: consecutive elements with same (primary_key, group_id)
    struct GroupInfo {
      event_t type;
      id_t root;
      std::vector<uint32_t> member_soa_indices;
    };

    std::vector<GroupInfo> groups;
    size_t gi = 0;
    while (gi < num_coll) {
      uint64_t pkey = h_sorted_primary[gi];
      uint32_t gid = h_sorted_group_ids[gi];

      if (gid == UINT32_MAX) { gi++; continue; }

      // Find end of this group
      size_t gj = gi + 1;
      while (gj < num_coll &&
             h_sorted_primary[gj] == pkey &&
             h_sorted_group_ids[gj] == gid) {
        gj++;
      }

      if (gj - gi >= 2) { // only include groups with 2+ members
        GroupInfo g;
        uint32_t first_soa = h_sorted_soa_indices[gi];
        g.type = data.events[first_soa];
        g.root = data.roots[first_soa];
        for (size_t k = gi; k < gj; k++) {
          g.member_soa_indices.push_back(h_sorted_soa_indices[k]);
        }
        groups.push_back(std::move(g));
      }

      gi = gj;
    }

    // Build CSR
    size_t num_groups = groups.size();
    size_t total_members = 0;
    for (auto &g : groups) total_members += g.member_soa_indices.size();

    if (num_groups == 0) {
      std::cout << "[GPU Coll] No groups formed" << std::endl;
      return;
    }

    out_csr.num_groups = num_groups;
    out_csr.total_members = total_members;
    out_csr.offsets = (int32_t *)malloc((num_groups + 1) * sizeof(int32_t));
    out_csr.members = (int32_t *)malloc(total_members * sizeof(int32_t));
    out_csr.group_types = (event_t *)malloc(num_groups * sizeof(event_t));
    out_csr.group_roots = (id_t *)malloc(num_groups * sizeof(id_t));
    out_csr.member_bytes_sent = (uint64_t *)malloc(total_members * sizeof(uint64_t));
    out_csr.member_bytes_received = (uint64_t *)malloc(total_members * sizeof(uint64_t));

    size_t offset = 0;
    for (size_t g = 0; g < num_groups; g++) {
      out_csr.offsets[g] = (int32_t)offset;
      out_csr.group_types[g] = groups[g].type;
      out_csr.group_roots[g] = groups[g].root;
      for (auto midx : groups[g].member_soa_indices) {
        out_csr.members[offset] = (int32_t)midx;
        auto cs_it = soa_to_commset_map.find(midx);
        if (cs_it != soa_to_commset_map.end() && cs_it->second < coll_bytes_sent.size()) {
          out_csr.member_bytes_sent[offset] = coll_bytes_sent[cs_it->second];
          out_csr.member_bytes_received[offset] = coll_bytes_received[cs_it->second];
        } else {
          out_csr.member_bytes_sent[offset] = 0;
          out_csr.member_bytes_received[offset] = 0;
        }
        offset++;
      }
    }
    out_csr.offsets[num_groups] = (int32_t)offset;

    auto t_end = std::chrono::high_resolution_clock::now();
    double csr_ms = std::chrono::duration<double, std::milli>(t_end - t_download).count();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Count by type for diagnostics
    std::unordered_map<int, size_t> type_counts;
    for (auto &g : groups) type_counts[(int)g.type]++;

    std::cout << "[GPU Coll] Built " << num_groups
              << " groups with " << total_members << " total members" << std::endl;
    for (auto &[t, c] : type_counts) {
      if (t >= 0 && t < NUM_EVENT_T)
        std::cout << "  " << event_strings[t] << ": " << c << " groups" << std::endl;
    }
    std::cout << "[GPU Coll] Total: " << std::fixed << std::setprecision(1)
              << total_ms << " ms (prep=" << prep_ms
              << " hash=" << hash_ms
              << " sort=" << sort_ms
              << " group=" << group_ms
              << " download=" << download_ms
              << " csr=" << csr_ms << ")" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "[GPU Coll] GPU grouping failed (" << e.what()
              << "), falling back to CPU" << std::endl;
    cudaGetLastError();
    buildCollectiveGroups(data, comm_sets, coll_bytes_sent, coll_bytes_received, out_csr);
  }
}
