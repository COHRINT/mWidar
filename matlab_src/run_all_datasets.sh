#!/bin/bash
# save as run_all_datasets.sh

cd "/Users/anthonylabarca/Library/CloudStorage/OneDrive-UCB-O365/Research/mWidar/matlab_src"

# Maximum number of parallel jobs
MAX_JOBS=10

# Function to wait for jobs to complete if we hit the limit
wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
}

echo "Starting parallel execution of all datasets..."

# Run all combinations of datasets, filters, and data association algorithms in parallel
for dataset in "T1_near" "T2_far" "T3_border" "T4_parab" "T5_parab_noise"
do
    for filter in "HybridPF" "KF"
    do
        for DA in "PDA" "GNN"
        do
            wait_for_jobs  # Wait if we have too many jobs running
            echo "Starting $dataset with $filter and $DA..."
            matlab -batch "main('$dataset', '$filter', 'DA', '$DA', 'Debug', false, 'FinalPlot', 'animation', 'DynamicPlot', false, 'InitializeTrue', true)" > "output_${filter}_${DA}_${dataset}.log" 2>&1 &
        done
    done
done

# Wait for all background jobs to complete
echo "Waiting for all jobs to complete..."
wait

echo "All simulations completed!"
echo "Check the output_*.log files for results."
