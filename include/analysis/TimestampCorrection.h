#pragma once

#include "data/TraceDataSoA.h"

#ifdef USE_SCALASCA_TIMESTAMPS

// Applies Controlled Logical Clock (CLC) timestamp correction to fix
// inter-node clock skew. This implements the forward + backward amortization
// algorithm from Scalasca (Geimer et al., PVM/MPI 2006) to match the results
// produced by "scalasca -analyze --time-correct".
//
// Must be called AFTER P2P matching (match_partner must be populated).
// Modifies timestamps[] and end_timestamps[] in-place.
// Has no effect on single-node traces (no clock violations to correct).
//
// Returns the number of clock condition violations detected.
size_t applyTimestampCorrection(TraceDataSoA &data);

#endif // USE_SCALASCA_TIMESTAMPS
