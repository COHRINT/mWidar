#!/bin/bash
# Simple script to launch individual jobs manually
# Run each line in a separate terminal window

echo "Copy and paste these commands into separate terminal windows:"
echo ""

datasets=("T1_near" "T2_far" "T3_border" "T4_parab" "T5_parab_noise")
filters=("HybridPF" "KF")

for dataset in "${datasets[@]}"; do
    for filter in "${filters[@]}"; do
        echo "cd '/Users/anthonylabarca/Library/CloudStorage/OneDrive-UCB-O365/Research/mWidar/matlab_src' && matlab -batch \"main('$dataset', '$filter')\" > output_${filter}_${dataset}.log 2>&1"
    done
done
