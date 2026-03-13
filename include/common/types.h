#ifndef GPU_ANALYZER_TYPES_H
#define GPU_ANALYZER_TYPES_H

#include <cstdint>

typedef uint64_t timestamp_t;
typedef uint32_t id_t;

enum event_type_t {
  ENTER = 0,
  LEAVE = 3,
};

enum event_t {
  TT_MPI_Init,
  TT_MPI_Comm_size,
  TT_MPI_Comm_rank,
  TT_MPI_Send,
  TT_MPI_Recv,
  TT_MPI_Isend,
  TT_MPI_Irecv,
  TT_MPI_Bcast,
  TT_MPI_Barrier,
  TT_MPI_Reduce,
  TT_MPI_Gather,
  TT_MPI_Gatherv,
  TT_MPI_Scatter,
  TT_MPI_Scatterv,
  TT_MPI_Reduce_Scatter,
  TT_MPI_Reduce_Scatter_Block,
  TT_MPI_All_Gather,
  TT_MPI_All_Gatherv,
  TT_MPI_All_Reduce,
  TT_MPI_AlltoAll,
  TT_MPI_Finalize,
};

#define NUM_EVENT_T 21

static const char *const event_strings[] = {
    "MPI_Init",       "MPI_Comm_size",   "MPI_Comm_rank",      "MPI_Send",
    "MPI_Recv",       "MPI_Isend",       "MPI_Irecv",          "MPI_Bcast",
    "MPI_Barrier",    "MPI_Reduce",      "MPI_Gather",         "MPI_Gatherv",
    "MPI_Scatter",    "MPI_Scatterv",    "MPI_Reduce_Scatter", "MPI_Reduce_Scatter_Block",
    "MPI_All_Gather", "MPI_All_Gatherv", "MPI_All_Reduce",     "MPI_AlltoAll",
    "MPI_Finalize"};

#endif // GPU_ANALYZER_TYPES_H
