#!/bin/bash
# Advanced parallel execution using GNU parallel
# Install with: brew install parallel (on macOS)

cd "/Users/anthonylabarca/Library/CloudStorage/OneDrive-UCB-O365/Research/mWidar/matlab_src"

# Create logs directory if it doesn't exist
mkdir -p logs

# Create a list of all combinations to run
echo "Creating job list..."
{
    for dataset in "T1_near" "T2_far" "T3_border" "T4_parab" "T5_parab_noise"; do
        # for filter in "HybridPF" "KF" "HMM"; do
            # for DA in "PDA" "GNN"; do
        for filter in "HybridPF"; do
            for DA in "PDA"; do
                echo "$dataset $filter $DA"
            done
        done
    done
} > job_list.txt

# Function to run a single job
run_job() {
    dataset=$1
    filter=$2
    DA=$3
    echo "Starting $dataset with $filter and $DA..."
    matlab -batch "main('$dataset', '$filter', 'DA', '$DA', 'Debug', false, 'FinalPlot', 'animation', 'DynamicPlot', true)" > "logs/output_${filter}_${DA}_${dataset}.log" 2>&1
    echo "Completed $dataset with $filter and $DA"
}

# Export the function so parallel can use it
export -f run_job

# Run jobs in parallel (adjust -j for number of parallel jobs)
# -j 0 uses all available CPU cores, or specify a number like -j 4
parallel -j 0 --colsep ' ' run_job {1} {2} {3} :::: job_list.txt

echo "All simulations completed!"
echo "Check the logs/output_*.log files for results."

# Clean up
rm job_list.txt
