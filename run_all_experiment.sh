#!/bin/bash

# this script runs inference for all prompt types

echo "Running all prompt type experiments..."

# Baseline experiments
bash run_inference.sh baseline 16
bash run_inference.sh role_based 16

# Few-shot experiments  
bash run_inference.sh few_shot_base 8    # Smaller batch for longer prompts
bash run_inference.sh few_shot_role_based 8

# CoT experiments
bash run_inference.sh baseline 16 True
bash run_inference.sh role_based 16 True
bash run_inference.sh few_shot_base 8 True
bash run_inference.sh few_shot_role_based 8 True

echo "All experiments completed!"