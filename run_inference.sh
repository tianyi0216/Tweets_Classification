#!/bin/bash 

PROMPT_TYPE=${1:-"baseline"}
BATCH_SIZE=${2:-16}
COT=${3:-False}
MODEL_NAME=${4:-"model/full_finetuned"}

OPTS=""
OPTS="$OPTS --input_file data/Q2_20230202_majority.csv"
OPTS="$OPTS --output_file output/full_nonlora/Q2_20230202_majority_predictions_${PROMPT_TYPE}_${COT}.csv"
OPTS="$OPTS --prompt_type $PROMPT_TYPE"
OPTS="$OPTS --batch_size $BATCH_SIZE"
OPTS="$OPTS --model_name $MODEL_NAME"
OPTS="$OPTS --cot $COT" # add cot to the prompt
OPTS="$OPTS --max_length 512"
OPTS="$OPTS --max_new_tokens 10"
# OPTS="$OPTS --adapter_path model/checkpoint-1725" # uncomment if finetuned with LoRA

echo "Running inference with prompt_type: $PROMPT_TYPE, batch_size: $BATCH_SIZE"

python src/predict.py $OPTS