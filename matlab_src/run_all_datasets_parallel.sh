#!/usr/bin/env bash
# GNU Parallel runner for all TEST_DATASET trajectories.
# Runs run_all_experiments per trajectory and computes stats outside that function.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

detect_cores() {
    if command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.logicalcpu 2>/dev/null && return 0
    fi
    if command -v nproc >/dev/null 2>&1; then
        nproc && return 0
    fi
    if command -v getconf >/dev/null 2>&1; then
        getconf _NPROCESSORS_ONLN 2>/dev/null && return 0
    fi
    echo 1
}

run_matlab_batch() {
    local code="$1"
    (cd "$SCRIPT_DIR" && matlab -batch "$code")
}

run_stats_for_target() {
    local label="$1"
    local logfile="$2"
    shift 2
    local targets=("$@")

    if [ ${#targets[@]} -eq 0 ]; then
        echo "[$label] no targets provided for stats; skipping." | tee -a "$logfile"
        return 0
    fi

    local joined=""
    local t
    for t in "${targets[@]}"; do
        if [ -n "$joined" ]; then
            joined+="', '"
        fi
        joined+="$t"
    done

    echo "[$label] running stats_experiment_results on ${#targets[@]} target(s)..." | tee -a "$logfile"
    run_matlab_batch "stats_experiment_results('$joined', 'plot', false);" >>"$logfile" 2>&1
}

DATA_ROOT="$SCRIPT_DIR/data/TEST_DATASET"
RESULTS_ROOT_BASE_REL="results/TEST_DATASET_RUNS"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT_REL="$RESULTS_ROOT_BASE_REL/$RUN_ID"
RESULTS_ROOT="$SCRIPT_DIR/$RESULTS_ROOT_REL"
LOG_ROOT="$RESULTS_ROOT/logs"
STATS_ROOT="$RESULTS_ROOT/stats"
JOB_LIST="$RESULTS_ROOT/job_list.txt"

CORES="$(detect_cores)"
JOBS="${JOBS:-$CORES}"
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -lt 1 ]; then
    JOBS=1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: TEST_DATASET root not found: $DATA_ROOT"
    exit 1
fi

if ! command -v parallel >/dev/null 2>&1; then
    echo "ERROR: GNU parallel is required but not found."
    echo "Install with: brew install parallel"
    exit 1
fi

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$STATS_ROOT"

echo "== mWidar TEST_DATASET GNU parallel runner =="
echo "Script dir     : $SCRIPT_DIR"
echo "Data root      : $DATA_ROOT"
echo "Results root   : $RESULTS_ROOT"
echo "Detected cores : $CORES"
echo "Parallel jobs  : $JOBS"

echo "Building job list at $JOB_LIST ..."
: > "$JOB_LIST"

declare -a all_dataset_dirs=()

for behavior_dir in "$DATA_ROOT"/*; do
    [ -d "$behavior_dir" ] || continue
    behavior="$(basename "$behavior_dir")"

    shopt -s nullglob
    traj_files=("$behavior_dir"/*.mat)
    shopt -u nullglob

    for src_mat in "${traj_files[@]}"; do
        traj_base="$(basename "${src_mat%.mat}")"
        dataset_dir_rel="$RESULTS_ROOT_BASE_REL/$RUN_ID/$behavior/$traj_base"
        echo "$behavior|$traj_base|$src_mat|$dataset_dir_rel" >> "$JOB_LIST"
        all_dataset_dirs+=("$dataset_dir_rel")
    done
done

if [ ! -s "$JOB_LIST" ]; then
    echo "No jobs found under $DATA_ROOT."
    exit 1
fi

run_job() {
    local behavior="$1"
    local traj_base="$2"
    local src_mat="$3"
    local dataset_dir_rel="$4"
    local dataset_dir_abs="$SCRIPT_DIR/$dataset_dir_rel"
    local diary_dir_rel="$dataset_dir_rel/diary_logs"

    mkdir -p "$dataset_dir_abs/diary_logs"
    cp -f "$src_mat" "$dataset_dir_abs/data.mat"

    local run_log="$LOG_ROOT/run_${behavior}_${traj_base}.log"
    local stats_log="$LOG_ROOT/stats_${behavior}_${traj_base}.log"

    echo "[RUN] $behavior/$traj_base" | tee "$run_log"
    run_matlab_batch "run_all_experiments('dataset_dir', '$dataset_dir_rel', 'diary_dir', '$diary_dir_rel');" >>"$run_log" 2>&1
    run_stats_for_target "trajectory:$behavior/$traj_base" "$stats_log" "$dataset_dir_rel"
}

export SCRIPT_DIR LOG_ROOT
export -f run_matlab_batch run_stats_for_target run_job

echo "Running trajectory jobs with GNU parallel..."
# NOTE: GNU parallel --colsep takes a regex. A bare '|' means alternation and
# matches empty strings, which splits every character. Use a character class
# to split on a literal pipe.
parallel -j "$JOBS" --colsep '[|]' run_job {1} {2} {3} {4} :::: "$JOB_LIST"

echo "Computing behavior-level and global stats..."
for behavior_dir in "$DATA_ROOT"/*; do
    [ -d "$behavior_dir" ] || continue
    behavior="$(basename "$behavior_dir")"

    declare -a behavior_targets=()
    while IFS='|' read -r b traj_base src_mat dataset_dir_rel; do
        if [ "$b" = "$behavior" ]; then
            behavior_targets+=("$dataset_dir_rel")
        fi
    done < "$JOB_LIST"

    if [ ${#behavior_targets[@]} -gt 0 ]; then
        behavior_stats_log="$LOG_ROOT/stats_${behavior}_aggregate.log"
        run_stats_for_target "behavior:$behavior" "$behavior_stats_log" "${behavior_targets[@]}"
    fi
done

if [ ${#all_dataset_dirs[@]} -gt 0 ]; then
    global_stats_log="$LOG_ROOT/stats_ALL_aggregate.log"
    run_stats_for_target "global" "$global_stats_log" "${all_dataset_dirs[@]}"
fi

echo "All TEST_DATASET simulations completed."
echo "Results: $RESULTS_ROOT"
echo "Logs   : $LOG_ROOT"
