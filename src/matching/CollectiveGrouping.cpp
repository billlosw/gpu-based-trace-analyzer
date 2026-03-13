#include "matching/CollectiveGrouping.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

// Build collective groups from SoA data on the CPU.
// Follows TileTrace's InteractionPattern logic for collective matching:
// - Root events create groups and collect members from their comm_set
// - Non-root events attach to an existing root group
// - Events are grouped by (type, root) in temporal order
void buildCollectiveGroups(const TraceDataSoA &data,
                           const std::vector<std::vector<uint64_t>> &comm_sets,
                           CollectiveGroupCSR &out_csr) {
  out_csr.deallocate();

  size_t n = data.count;
  if (n == 0)
    return;

  // Collect indices of collective events
  std::vector<size_t> coll_indices;
  for (size_t i = 0; i < n; i++) {
    event_t ev = data.events[i];
    if (ev >= TT_MPI_Bcast && ev <= TT_MPI_AlltoAll)
      coll_indices.push_back(i);
  }

  if (coll_indices.empty())
    return;

  // Sort collective indices by timestamp (should already be sorted from OTF2)
  std::sort(coll_indices.begin(), coll_indices.end(),
            [&](size_t a, size_t b) {
              return data.timestamps[a] < data.timestamps[b];
            });

  // Map from event SoA index to its comm_set index in the comm_sets vector
  // comm_sets is built by the reader and corresponds to collective events in order
  std::unordered_map<size_t, size_t> soa_to_commset;
  {
    size_t cs_idx = 0;
    for (size_t i = 0; i < n && cs_idx < comm_sets.size(); i++) {
      event_t ev = data.events[i];
      if (ev >= TT_MPI_Bcast && ev <= TT_MPI_AlltoAll) {
        soa_to_commset[i] = cs_idx;
        cs_idx++;
      }
    }
  }

  // Group tracking: each group has a root event index and member event indices.
  struct PendingGroup {
    size_t root_event_idx;
    std::vector<size_t> member_indices;
    size_t expected_size;
    bool complete;
  };

  // Key: (type_offset * 1000000 + root)
  // We use type_offset = event_type - TT_MPI_Bcast (same as TileTrace)
  auto makeGroupKey = [](event_t ev, id_t root) -> uint64_t {
    return (uint64_t)(ev - TT_MPI_Bcast) * 1000000ULL + root;
  };

  std::unordered_map<uint64_t, std::vector<PendingGroup>> pending;
  std::vector<PendingGroup> completed_groups;

  for (size_t ci_idx = 0; ci_idx < coll_indices.size(); ci_idx++) {
    size_t ev_idx = coll_indices[ci_idx];
    event_t etype = data.events[ev_idx];
    id_t pid = data.pids[ev_idx];
    id_t root = data.roots[ev_idx];

    uint64_t gkey = makeGroupKey(etype, root);

    size_t expected = 0;
    auto cs_it = soa_to_commset.find(ev_idx);
    if (cs_it != soa_to_commset.end() && cs_it->second < comm_sets.size()) {
      expected = comm_sets[cs_it->second].size();
    }

    if (pid == root) {
      // Root event: create new group
      PendingGroup pg;
      pg.root_event_idx = ev_idx;
      pg.member_indices.push_back(ev_idx);
      pg.expected_size = expected > 0 ? expected : 1;
      pg.complete = false;

      // Check if any pending non-root events match
      if (pending.count(gkey)) {
        auto &plist = pending[gkey];
        // Try to match pending members
        auto it = plist.begin();
        while (it != plist.end() &&
               pg.member_indices.size() < pg.expected_size) {
          if (it->root_event_idx == SIZE_MAX) {
            // This is a pending non-root event
            for (auto midx : it->member_indices) {
              pg.member_indices.push_back(midx);
            }
            it = plist.erase(it);
          } else {
            ++it;
          }
        }
      }

      if (pg.member_indices.size() >= pg.expected_size) {
        pg.complete = true;
        completed_groups.push_back(pg);
      } else {
        pending[gkey].push_back(pg);
      }
    } else {
      // Non-root event: try to attach to existing root group
      bool matched = false;

      if (pending.count(gkey)) {
        auto &plist = pending[gkey];
        for (auto &pg : plist) {
          if (pg.root_event_idx != SIZE_MAX &&
              pg.member_indices.size() < pg.expected_size) {
            pg.member_indices.push_back(ev_idx);
            if (pg.member_indices.size() >= pg.expected_size) {
              pg.complete = true;
              completed_groups.push_back(pg);
            }
            matched = true;
            break;
          }
        }
      }

      if (!matched) {
        // Store as pending non-root member
        PendingGroup pg;
        pg.root_event_idx = SIZE_MAX;
        pg.member_indices.push_back(ev_idx);
        pg.expected_size = expected > 0 ? expected : 0;
        pg.complete = false;
        pending[gkey].push_back(pg);
      }
    }
  }

  // Also include incomplete groups with at least 2 members (partial matches)
  for (auto &[key, plist] : pending) {
    for (auto &pg : plist) {
      if (!pg.complete && pg.member_indices.size() >= 2 &&
          pg.root_event_idx != SIZE_MAX) {
        completed_groups.push_back(pg);
      }
    }
  }

  if (completed_groups.empty())
    return;

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
    out_csr.group_types[g] = data.events[completed_groups[g].member_indices[0]];
    out_csr.group_roots[g] = data.roots[completed_groups[g].member_indices[0]];
    for (auto midx : completed_groups[g].member_indices) {
      out_csr.members[offset++] = (int32_t)midx;
    }
  }
  out_csr.offsets[num_groups] = (int32_t)offset;

  std::cout << "[CollectiveGrouping] Built " << num_groups
            << " groups with " << total_members << " total members"
            << std::endl;
}
