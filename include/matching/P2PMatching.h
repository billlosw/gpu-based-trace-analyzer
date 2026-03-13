#ifndef GPU_ANALYZER_P2P_MATCHING_H
#define GPU_ANALYZER_P2P_MATCHING_H

#include "data/TraceDataSoA.h"
#include <cstddef>

// Run GPU-based P2P send-recv matching.
// Fills data.match_partner[i] with the index of the matched partner event.
// For a Send event, match_partner[i] = index of the matching Recv.
// For a Recv event, match_partner[i] = index of the matching Send.
// Unmatched events keep match_partner[i] = -1.
void runP2PMatching(TraceDataSoA &data);

#endif // GPU_ANALYZER_P2P_MATCHING_H
