#!/usr/bin/env bash
# Run all trajectories in data/TEST_DATASET using run_all_experiments,
# and compute stats at trajectory, behavior-type, and global levels.

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
    matlab -batch "$code"
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

DATA_ROOT="data/TEST_DATASET"
RESULTS_ROOT_BASE="results/TEST_DATASET_RUNS"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="$RESULTS_ROOT_BASE/$RUN_ID"
LOG_ROOT="$RESULTS_ROOT/logs"
STATS_ROOT="$RESULTS_ROOT/stats"

CORES="$(detect_cores)"
MAX_JOBS="${MAX_JOBS:-$CORES}"
if ! [[ "$MAX_JOBS" =~ ^[0-9]+$ ]] || [ "$MAX_JOBS" -lt 1 ]; then
    MAX_JOBS=1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: TEST_DATASET root not found: $DATA_ROOT"
    exit 1
fi

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$STATS_ROOT"

echo "== mWidar TEST_DATASET batch runner =="
echo "Script dir     : $SCRIPT_DIR"
echo "Data root      : $DATA_ROOT"
echo "Results root   : $RESULTS_ROOT"
echo "Detected cores : $CORES"
echo "MAX_JOBS       : $MAX_JOBS"

wait_for_jobs() {
    while [ "$(jobs -r | wc -l | tr -d ' ')" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
}

declare -a all_dataset_dirs=()
declare -a behavior_names=()

for behavior_dir in "$DATA_ROOT"/*; do
    [ -d "$behavior_dir" ] || continue
    behavior="$(basename "$behavior_dir")"
    behavior_names+=("$behavior")

    echo ""
    echo "-- Behavior: $behavior --"

    declare -a behavior_dataset_dirs=()

    shopt -s nullglob
    traj_files=("$behavior_dir"/*.mat)
    shopt -u nullglob

    if [ ${#traj_files[@]} -eq 0 ]; then
        echo "No trajectory .mat files found in $behavior_dir (skipping)."
        continue
    fi

    for src_mat in "${traj_files[@]}"; do
        traj_base="$(basename "${src_mat%.mat}")"
        dataset_dir="$RESULTS_ROOT/$behavior/$traj_base"
        dataset_dir_rel="results/TEST_DATASET_RUNS/$RUN_ID/$behavior/$traj_base"
        diary_dir_rel="$dataset_dir_rel/diary_logs"

        mkdir -p "$dataset_dir" "$dataset_dir/diary_logs"
        cp -f "$src_mat" "$dataset_dir/data.mat"

        run_log="$LOG_ROOT/run_${behavior}_${traj_base}.log"
        stats_log="$LOG_ROOT/stats_${behavior}_${traj_base}.log"

        wait_for_jobs
        (
            set -euo pipefail
            echo "[RUN] $behavior/$traj_base" | tee "$run_log"
            run_matlab_batch "run_all_experiments('dataset_dir', '$dataset_dir_rel', 'diary_dir', '$diary_dir_rel');" >>"$run_log" 2>&1
            run_stats_for_target "trajectory:$behavior/$traj_base" "$stats_log" "$dataset_dir_rel"
        ) &

        behavior_dataset_dirs+=("$dataset_dir_rel")
        all_dataset_dirs+=("$dataset_dir_rel")
    done

    echo "Waiting for trajectory jobs to finish for behavior $behavior..."
    wait

    if [ ${#behavior_dataset_dirs[@]} -gt 0 ]; then
        behavior_stats_log="$LOG_ROOT/stats_${behavior}_aggregate.log"
        run_stats_for_target "behavior:$behavior" "$behavior_stats_log" "${behavior_dataset_dirs[@]}"
    fi

done

echo ""
echo "Waiting for any remaining jobs to finish..."
wait

if [ ${#all_dataset_dirs[@]} -gt 0 ]; then
    global_stats_log="$LOG_ROOT/stats_ALL_aggregate.log"
    run_stats_for_target "global" "$global_stats_log" "${all_dataset_dirs[@]}"
fi

echo ""
echo "All TEST_DATASET simulations completed."
echo "Results: $RESULTS_ROOT"
echo "Logs   : $LOG_ROOT"
