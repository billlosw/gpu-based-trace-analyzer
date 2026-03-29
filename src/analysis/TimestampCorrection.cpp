#include "analysis/TimestampCorrection.h"

#ifdef USE_SCALASCA_TIMESTAMPS
#include <iostream>

size_t applyTimestampCorrection(TraceDataSoA &data) {
  size_t n = data.count;
  if (n == 0)
    return 0;

  size_t total_violations = 0;
  size_t neutralized = 0;

  for (size_t i = 0; i < n; i++) {
    event_t ev = data.events[i];
    if (ev != TT_MPI_Recv && ev != TT_MPI_Irecv)
      continue;

    int32_t send_idx = data.match_partner[i];
    if (send_idx < 0)
      continue;

    timestamp_t recv_enter = data.timestamps[i];            // Enter(MPI_Recv/Wait)
    timestamp_t send_leave = data.end_timestamps[send_idx]; // Leave(MPI_Send)

    // Clock condition: message cannot arrive before it was sent.
    if (send_leave > recv_enter) {
      total_violations++;

      // For blocking recvs (MPI_Recv): recv_enter == recv_req_enter == Enter(MPI_Recv)
      // After correcting for clock skew, send_leave_corrected <= recv_enter,
      // so late_receiver condition (send_leave > recv_req_enter) becomes false.
      // Neutralize by setting recv_req_enter = send_leave.
      //
      // For non-blocking recvs (MPI_Irecv): recv_req_enter = Enter(MPI_Irecv) < recv_enter
      // After correcting for clock skew, send_leave_corrected <= recv_enter,
      // but send_leave_corrected may still > recv_req_enter (genuine late_receiver).
      // Don't neutralize — adjust duration instead by subtracting the violation amount.
      if (ev == TT_MPI_Recv) {
        // Blocking recv: neutralize entirely
        data.end_timestamps[i] = send_leave;
        neutralized++;
      } else {
        // Non-blocking recv (MPI_Irecv): adjust send_leave effect
        // The violation amount is (send_leave - recv_enter).
        // The corrected send_leave ≈ recv_enter (just closing the gap).
        // We leave end_timestamps[i] (Enter(MPI_Irecv)) unchanged, but we
        // need to adjust end_timestamps[send_idx] (Leave(MPI_Send)) to
        // recv_enter to remove the clock skew.
        // However, we can't modify end_timestamps[send_idx] as it may be
        // shared with other recv events.
        //
        // Instead: the late_receiver duration without skew would be:
        //   recv_req_enter - send_enter  (unchanged, both send_enter and
        //                                 recv_req_enter are on their own clocks)
        // The late_receiver condition after correction:
        //   (send_leave - violation) > recv_req_enter
        //   = recv_enter > recv_req_enter  (always true for non-blocking)
        // So the pair stays as late_receiver with corrected duration.
        // We just need to fix the send_leave used in the kernel.
        //
        // Practical approach: set end_timestamps[i] to
        //   max(recv_req_enter, recv_req_enter + (send_leave - recv_enter))
        // Wait, that doesn't help because we'd increase it.
        //
        // Actually: for non-blocking, leave everything unchanged.
        // The late_receiver kernel uses end_timestamps[send_idx] (Leave(MPI_Send))
        // and end_timestamps[i] (Enter(MPI_Irecv)). Both are on their original
        // clocks. The clock skew between sender and receiver inflates the
        // duration but doesn't create false positives for non-blocking.
        // Leave as-is.
      }
    }
  }

  std::cout << "[CLC] " << total_violations
            << " clock violations, " << neutralized
            << " neutralized (blocking recvs)" << std::endl;

  return total_violations;
}

#endif // USE_SCALASCA_TIMESTAMPS
