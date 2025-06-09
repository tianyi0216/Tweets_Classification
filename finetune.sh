#!/bin/bash

# Fine-tuning script for tweet stance classification
# Usage: ./finetune.sh [prompt_type] [use_lora] [epochs] [batch_size]

PROMPT_TYPE=${1:-"baseline"}
EPOCHS=${2:-5}
BATCH_SIZE=${3:-16}
LEARNING_RATE=${4:-1e-4}
OUTPUT_DIR=${5:-"./checkpoints"}
USE_LORA=${6:-"true"}

OPTS=""
OPTS="$OPTS --data_file data/Q2_20230202_majority.csv"
OPTS="$OPTS --output_dir $OUTPUT_DIR"
OPTS="$OPTS --prompt_type $PROMPT_TYPE"
OPTS="$OPTS --epochs $EPOCHS"
OPTS="$OPTS --batch_size $BATCH_SIZE"
OPTS="$OPTS --learning_rate $LEARNING_RATE"
OPTS="$OPTS --model_name google/flan-t5-large"
OPTS="$OPTS --wandb"

if [ "$USE_LORA" = "true" ]; then
    OPTS="$OPTS --lora"
    OPTS="$OPTS --lora_r 16"
    OPTS="$OPTS --lora_alpha 32"
    OPTS="$OPTS --lora_dropout 0.1"
fi

python src/train.py $OPTS
