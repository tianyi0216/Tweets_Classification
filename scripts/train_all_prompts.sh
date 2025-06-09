#!/bin/bash

# Train models for all prompt types with LoRA
# Usage: ./scripts/train_all_prompts.sh

echo "🚀 Training all prompt types with LoRA..."

# Baseline prompt
echo "📝 Training baseline prompt..."
./finetune.sh baseline true 5 16

# Role-based prompt  
echo "📝 Training role-based prompt..."
./finetune.sh role_based true 5 16

echo "✅ All models trained!"
echo "📁 Check ./models/ directory for results" 