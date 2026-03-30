#include "matching/CollectiveGrouping.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

// FNV-1a hash for a sorted comm_set.
// Deterministic, fast, and has good distribution for integer sequences.
static uint64_t hashCommSet(const std::vector<uint64_t> &sorted_cs) {
  uint64_t h = 14695981039346656037ULL; // FNV offset basis
  for (uint64_t v : sorted_cs) {
    // Hash each byte of the 8-byte value
    for (int b = 0; b < 8; b++) {
      h ^= (v >> (b * 8)) & 0xFF;
      h *= 1099511628211ULL; // FNV prime
    }
  }
  return h;
}

// Combine event type and comm_set hash into a single lookup key.
static uint64_t makePendingKey(int event_type, uint64_t cs_hash) {
  // Mix event type into high bits to separate by type
  return cs_hash ^ ((uint64_t)event_type * 2654435761ULL);
}

// Build collective groups from SoA data on the CPU.
// Strategy: process collective events in timestamp order. For each event,
// look for an open (incomplete) group with the same type and comm_set that
// still needs this process's contribution. If found, add to it. Otherwise,
// create a new group.
//
// Optimizations over the original:
// 1. Hash-based lookup: O(1) amortized instead of linear scan over pending groups
// 2. Comm_set caching: normalize (sort) each unique comm_set only once
void buildCollectiveGroups(const TraceDataSoA &data,
                           const std::vector<std::vector<uint64_t>> &comm_sets,
                           CollectiveGroupCSR &out_csr) {
  out_csr.deallocate();

  size_t n = data.count;
  if (n == 0)
    return;

  // Collect indices of collective events and build soa_idx -> comm_set_idx map
  std::vector<size_t> coll_indices;
  std::unordered_map<size_t, size_t> soa_to_commset;
  {
    size_t cs_idx = 0;
    for (size_t i = 0; i < n; i++) {
      event_t ev = data.events[i];
      if (ev >= TT_MPI_Bcast && ev <= TT_MPI_AlltoAll) {
        coll_indices.push_back(i);
        if (cs_idx < comm_sets.size()) {
          soa_to_commset[i] = cs_idx;
          cs_idx++;
        }
      }
    }
  }

  if (coll_indices.empty())
    return;

  // Sort by timestamp
  std::sort(coll_indices.begin(), coll_indices.end(),
            [&](size_t a, size_t b) {
              return data.timestamps[a] < data.timestamps[b];
            });

  // Cache: comm_set_idx -> {sorted comm_set, hash}
  // MPI programs typically have 1-3 communicators, so this cache is tiny
  // but avoids millions of redundant sorts.
  struct CachedCommSet {
    std::vector<uint64_t> sorted_cs;
    uint64_t hash;
  };
  std::unordered_map<size_t, CachedCommSet> cs_cache;

  auto getCachedCommSet = [&](size_t cs_idx) -> const CachedCommSet & {
    auto it = cs_cache.find(cs_idx);
    if (it != cs_cache.end())
      return it->second;
    CachedCommSet &entry = cs_cache[cs_idx];
    entry.sorted_cs = comm_sets[cs_idx];
    std::sort(entry.sorted_cs.begin(), entry.sorted_cs.end());
    entry.hash = hashCommSet(entry.sorted_cs);
    return entry;
  };

  // Group tracking
  struct PendingGroup {
    event_t type;
    id_t root;
    std::vector<uint64_t> comm_set_key; // sorted comm_set for collision check
    uint64_t cs_hash;                   // pre-computed hash
    std::set<id_t> needed_pids;         // PIDs still expected
    std::vector<size_t> member_indices;  // SoA event indices
    size_t expected_size;
  };

  // Hash-based lookup: pending_key -> list of pending group indices
  // pending_key combines event_type and comm_set_hash
  std::unordered_map<uint64_t, std::vector<size_t>> pending_by_key;
  std::vector<PendingGroup> all_pending; // pool of all pending groups
  std::vector<PendingGroup> completed_groups;

  for (size_t ci = 0; ci < coll_indices.size(); ci++) {
    size_t ev_idx = coll_indices[ci];
    event_t etype = data.events[ev_idx];
    id_t pid = data.pids[ev_idx];
    id_t root = data.roots[ev_idx];

    // Get comm_set for this event (with caching)
    auto cs_it = soa_to_commset.find(ev_idx);
    if (cs_it == soa_to_commset.end() || cs_it->second >= comm_sets.size())
      continue;

    const CachedCommSet &cached = getCachedCommSet(cs_it->second);
    if (cached.sorted_cs.empty())
      continue;

    uint64_t pkey = makePendingKey((int)etype, cached.hash);
    bool matched = false;

    // O(1) amortized lookup: find pending groups with same type+hash
    auto pit = pending_by_key.find(pkey);
    if (pit != pending_by_key.end()) {
      auto &idx_list = pit->second;
      for (size_t li = 0; li < idx_list.size(); li++) {
        size_t pg_idx = idx_list[li];
        PendingGroup &pg = all_pending[pg_idx];

        // Hash collision check: verify exact comm_set match
        if (pg.cs_hash == cached.hash &&
            pg.comm_set_key == cached.sorted_cs &&
            pg.needed_pids.count(pid)) {
          pg.member_indices.push_back(ev_idx);
          pg.needed_pids.erase(pid);

          if (pg.needed_pids.empty()) {
            // Group complete — move to completed, remove from pending index
            completed_groups.push_back(std::move(pg));
            pg.expected_size = 0; // Mark as dead
            // Swap-remove from index list
            idx_list[li] = idx_list.back();
            idx_list.pop_back();
            if (idx_list.empty())
              pending_by_key.erase(pkey);
          }
          matched = true;
          break;
        }
      }
    }

    if (!matched) {
      // Create new group
      size_t new_idx = all_pending.size();
      all_pending.emplace_back();
      PendingGroup &pg = all_pending.back();
      pg.type = etype;
      pg.root = root;
      pg.comm_set_key = cached.sorted_cs;
      pg.cs_hash = cached.hash;
      pg.expected_size = cached.sorted_cs.size();
      pg.member_indices.push_back(ev_idx);
      // Build needed_pids from comm_set, excluding this pid
      for (auto member_pid : cached.sorted_cs) {
        if ((id_t)member_pid != pid) {
          pg.needed_pids.insert((id_t)member_pid);
        }
      }
      if (pg.needed_pids.empty()) {
        // Single-member group (shouldn't happen for real collectives)
        completed_groups.push_back(std::move(pg));
        pg.expected_size = 0;
      } else {
        pending_by_key[pkey].push_back(new_idx);
      }
    }
  }

  // Also include incomplete groups with at least 2 members
  for (auto &pg : all_pending) {
    if (pg.expected_size > 0 && pg.member_indices.size() >= 2) {
      completed_groups.push_back(std::move(pg));
    }
  }

  if (completed_groups.empty()) {
    std::cout << "[CollectiveGrouping] No groups formed" << std::endl;
    return;
  }

  // Build CSR arrays
  size_t num_groups = completed_groups.size();
  size_t total_members = 0;
  for (auto &g : completed_groups)
    total_members += g.member_indices.size();

  out_csr.num_groups = num_groups;
  out_csr.total_members = total_members;
  out_csr.offsets = (int32_t *)malloc((num_groups + 1) * sizeof(int32_t));
  out_csr.members = (int32_t *)malloc(total_members * sizeof(int32_t));
  out_csr.group_types = (event_t *)malloc(num_groups * sizeof(event_t));
  out_csr.group_roots = (id_t *)malloc(num_groups * sizeof(id_t));

  size_t offset = 0;
  for (size_t g = 0; g < num_groups; g++) {
    out_csr.offsets[g] = (int32_t)offset;
    out_csr.group_types[g] = completed_groups[g].type;
    out_csr.group_roots[g] = completed_groups[g].root;
    for (auto midx : completed_groups[g].member_indices) {
      out_csr.members[offset++] = (int32_t)midx;
    }
  }
  out_csr.offsets[num_groups] = (int32_t)offset;

  // Count by type for diagnostics
  std::unordered_map<int, size_t> type_counts;
  for (auto &g : completed_groups)
    type_counts[(int)g.type]++;

  std::cout << "[CollectiveGrouping] Built " << num_groups
            << " groups with " << total_members << " total members"
            << std::endl;
  for (auto &[t, c] : type_counts) {
    if (t >= 0 && t < NUM_EVENT_T)
      std::cout << "  " << event_strings[t] << ": " << c << " groups" << std::endl;
  }
}
