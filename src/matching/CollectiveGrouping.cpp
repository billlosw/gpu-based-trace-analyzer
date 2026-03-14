#include "matching/CollectiveGrouping.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

// Build collective groups from SoA data on the CPU.
// Strategy: process collective events in timestamp order. For each event,
// look for an open (incomplete) group with the same type and comm_set that
// still needs this process's contribution. If found, add to it. Otherwise,
// create a new group.
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

  // Represent a comm_set as a sorted vector for use as a map key
  auto normalizeCommSet = [](const std::vector<uint64_t> &cs) -> std::vector<uint64_t> {
    std::vector<uint64_t> sorted_cs(cs);
    std::sort(sorted_cs.begin(), sorted_cs.end());
    return sorted_cs;
  };

  // Group tracking
  struct PendingGroup {
    event_t type;
    id_t root;
    std::vector<uint64_t> comm_set_key; // sorted comm_set for identification
    std::set<id_t> needed_pids;         // PIDs still expected
    std::vector<size_t> member_indices;  // SoA event indices
    size_t expected_size;
  };

  // Key: event_type -> list of pending groups
  std::unordered_map<int, std::vector<PendingGroup>> pending_by_type;
  std::vector<PendingGroup> completed_groups;

  for (size_t ci = 0; ci < coll_indices.size(); ci++) {
    size_t ev_idx = coll_indices[ci];
    event_t etype = data.events[ev_idx];
    id_t pid = data.pids[ev_idx];
    id_t root = data.roots[ev_idx];

    // Get comm_set for this event
    std::vector<uint64_t> cs_key;
    size_t expected = 0;
    auto cs_it = soa_to_commset.find(ev_idx);
    if (cs_it != soa_to_commset.end() && cs_it->second < comm_sets.size()) {
      cs_key = normalizeCommSet(comm_sets[cs_it->second]);
      expected = cs_key.size();
    }

    if (cs_key.empty() || expected == 0)
      continue;

    int type_key = (int)etype;
    bool matched = false;

    // Try to find a pending group of the same type with matching comm_set
    // that still needs this pid
    auto &plist = pending_by_type[type_key];
    for (auto &pg : plist) {
      if (pg.comm_set_key == cs_key && pg.needed_pids.count(pid)) {
        pg.member_indices.push_back(ev_idx);
        pg.needed_pids.erase(pid);

        if (pg.needed_pids.empty()) {
          // Group complete
          completed_groups.push_back(std::move(pg));
          pg.expected_size = 0; // Mark for removal
        }
        matched = true;
        break;
      }
    }

    // Remove completed groups from pending
    plist.erase(
        std::remove_if(plist.begin(), plist.end(),
                        [](const PendingGroup &pg) { return pg.expected_size == 0; }),
        plist.end());

    if (!matched) {
      // Create new group
      PendingGroup pg;
      pg.type = etype;
      pg.root = root;
      pg.comm_set_key = cs_key;
      pg.expected_size = expected;
      pg.member_indices.push_back(ev_idx);
      // Build needed_pids from comm_set, excluding this pid
      for (auto member_pid : cs_key) {
        if ((id_t)member_pid != pid) {
          pg.needed_pids.insert((id_t)member_pid);
        }
      }
      if (pg.needed_pids.empty()) {
        // Single-member group (shouldn't happen for real collectives)
        completed_groups.push_back(std::move(pg));
      } else {
        plist.push_back(std::move(pg));
      }
    }
  }

  // Also include incomplete groups with at least 2 members
  for (auto &[type_key, plist] : pending_by_type) {
    for (auto &pg : plist) {
      if (pg.member_indices.size() >= 2) {
        completed_groups.push_back(std::move(pg));
      }
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
