#!/usr/bin/env bash
# Final paper data collection on the FUSE server.
# Run from /home/luosw22/claude/gpu-analyzer after syncing this repo.

set -Eeuo pipefail

TARGET_SET="${1:-all}"

GPU_TYPE="${GPU_TYPE:-5090}"
NODE="${NODE:-fuse2}"
PARTITION="${PARTITION:-Debug}"
MPI_RANKS="${MPI_RANKS:-64}"
ANALYZER="${ANALYZER:-./build/gpu_analyzer}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-$HOME/logs/final-data}"
RUN_STAMP="${RUN_STAMP:-$(date +%y%m%d-%H%M%S)}"

CG_TRACE_ROOT="${CG_TRACE_ROOT:-$HOME/claude/TileTraceClaude/exp/traces}"
LAMMPS_TRACE_ROOT="${LAMMPS_TRACE_ROOT:-$HOME/trace_data}"

case "$TARGET_SET" in
  all|cg|lammps) ;;
  *)
    echo "Usage: $0 [all|cg|lammps]" >&2
    exit 2
    ;;
esac

mkdir -p "$SERVER_LOG_DIR"

if [[ ! -x "$ANALYZER" ]]; then
  echo "Error: analyzer executable not found or not executable: $ANALYZER" >&2
  echo "Build on the server first, then rerun this script." >&2
  exit 1
fi

run_one() {
  local trace_set="$1"
  local label="$2"
  local trace_path="$3"
  local mode="$4"
  shift 4
  local -a analyzer_flags=("$@")

  if [[ "$mode" == "gpu-matching" ]]; then
    analyzer_flags+=("--gpu-matching")
  fi

  local log_name="${RUN_STAMP}-${trace_set}-${label}-g${GPU_TYPE}c${MPI_RANKS}-${mode}.log"
  local log_path="${SERVER_LOG_DIR}/${log_name}"
  local start_s
  local end_s
  local status=0

  {
    echo "=== final run start ==="
    echo "timestamp: $(date -Is)"
    echo "host: $(hostname)"
    echo "trace_set: ${trace_set}"
    echo "label: ${label}"
    echo "mode: ${mode}"
    echo "trace_path: ${trace_path}"
    echo "analyzer: ${ANALYZER}"
    echo "analyzer_flags: ${analyzer_flags[*]:-(none)}"
    echo "srun: -N 1 -n ${MPI_RANKS} -w ${NODE} -p ${PARTITION} --mpi=pmix --gres=gpu:${GPU_TYPE}:1"
    echo "server_log: ${log_path}"
    echo
  } | tee "$log_path"

  if [[ ! -f "$trace_path" ]]; then
    echo "Error: trace file not found: $trace_path" | tee -a "$log_path"
    return 1
  fi

  start_s="$(date +%s)"
  if ! srun -N 1 -n "$MPI_RANKS" -w "$NODE" -p "$PARTITION" \
      --mpi=pmix --gres="gpu:${GPU_TYPE}:1" \
      "$ANALYZER" "$trace_path" "${analyzer_flags[@]}" 2>&1 | tee -a "$log_path"; then
    status=1
  fi
  end_s="$(date +%s)"

  {
    echo
    echo "=== final run end ==="
    echo "timestamp: $(date -Is)"
    echo "exit_status: ${status}"
    echo "elapsed_seconds: $((end_s - start_s))"
  } | tee -a "$log_path"

  return "$status"
}

run_trace_twice() {
  local trace_set="$1"
  local label="$2"
  local trace_path="$3"
  shift 3
  local -a base_flags=("$@")
  local mode
  local failures=0

  for mode in normal gpu-matching; do
    echo "Processing ${trace_set}/${label} (${mode})"
    if ! run_one "$trace_set" "$label" "$trace_path" "$mode" "${base_flags[@]}"; then
      failures=$((failures + 1))
    fi
  done

  return "$failures"
}

run_cg_set() {
  local -a trace_dirs=("cg.B" "cg.C" "cg.D")
  local -a labels=("cgB" "cgC" "cgD")
  local failures=0
  local i

  for i in "${!trace_dirs[@]}"; do
    if ! run_trace_twice "cg" "${labels[$i]}" \
        "${CG_TRACE_ROOT}/${trace_dirs[$i]}/traces.otf2"; then
      failures=$((failures + 1))
    fi
  done

  return "$failures"
}

run_lammps_set() {
  local -a trace_dirs=(
    "__64_qtraces"
    "__128_qtraces"
    "__256_qtraces"
    "__512_qtraces"
    "lammps_n1024"
    "lammps_n2048"
  )
  local -a labels=("lammps64" "lammps128" "lammps256" "lammps512" "lammps1024" "lammps2048")
  local failures=0
  local i

  for i in "${!trace_dirs[@]}"; do
    if ! run_trace_twice "lammps" "${labels[$i]}" \
        "${LAMMPS_TRACE_ROOT}/${trace_dirs[$i]}/traces.otf2" --time-correct; then
      failures=$((failures + 1))
    fi
  done

  return "$failures"
}

main() {
  local failures=0

  if [[ "$TARGET_SET" == "all" || "$TARGET_SET" == "cg" ]]; then
    if ! run_cg_set; then
      failures=$((failures + 1))
    fi
  fi

  if [[ "$TARGET_SET" == "all" || "$TARGET_SET" == "lammps" ]]; then
    if ! run_lammps_set; then
      failures=$((failures + 1))
    fi
  fi

  echo "Final data collection complete. Server logs: ${SERVER_LOG_DIR}"
  if [[ "$failures" -ne 0 ]]; then
    echo "Completed with ${failures} failed trace group(s)." >&2
    return 1
  fi
}

main
