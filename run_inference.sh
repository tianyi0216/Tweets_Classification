#!/bin/bash 

OPTS=""
OPTS="$OPTS --input_file data/Q2_20230202_majority.csv"
OPTS="$OPTS --output_file data/Q2_20230202_majority_predictions.csv"
OPTS="$OPTS --prompt_type baseline"
OPTS="$OPTS --batch_size 16"
OPTS="$OPTS --model_name google/flan-t5-large"

python src/predict.py $OPTS