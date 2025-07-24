#!/bin/bash
# Advanced parallel execution using GNU parallel
# Install with: brew install parallel (on macOS)

cd "/Users/anthonylabarca/Library/CloudStorage/OneDrive-UCB-O365/Research/mWidar/matlab_src"

# Create a list of all combinations to run
echo "Creating job list..."
{
    for dataset in "T1_near" "T2_far" "T3_border" "T4_parab" "T5_parab_noise"; do
        for filter in "HybridPF" "KF"; do
            echo "$dataset $filter"
        done
    done
} > job_list.txt

# Function to run a single job
run_job() {
    dataset=$1
    filter=$2
    echo "Starting $dataset with $filter..."
    matlab -batch "main('$dataset', '$filter')" > "output_${filter}_${dataset}.log" 2>&1
    echo "Completed $dataset with $filter"
}

# Export the function so parallel can use it
export -f run_job

# Run jobs in parallel (adjust -j for number of parallel jobs)
parallel -j 4 --colsep ' ' run_job :::: job_list.txt

echo "All simulations completed!"
echo "Check the output_*.log files for results."

# Clean up
rm job_list.txt
