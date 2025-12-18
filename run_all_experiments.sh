#!/bin/bash

# Script to run all experiments described in the paper:
# 1. Baseline (frozen SBERT) with VLM
# 2. Baseline (frozen SBERT) without VLM
# 3. Fine-tuned SBERT with VLM
# 4. Fine-tuned SBERT without VLM

# Configuration
TRAIN_DATA="data/train.json"
VAL_DATA="data/val.json"
TEST_DATA="data/test.json"
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR="outputs"
EPOCHS=10
BATCH_SIZE=32
SAMPLES_PER_CLUSTER=4
DEVICE="cuda"

echo "======================================"
echo "Running All Experiments"
echo "======================================"
echo ""

# Check if data files exist
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA"
    echo "Please ensure your data files are in the correct location."
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "Error: Validation data not found at $VAL_DATA"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "Error: Test data not found at $TEST_DATA"
    exit 1
fi

echo "Using data files:"
echo "  Train: $TRAIN_DATA"
echo "  Val:   $VAL_DATA"
echo "  Test:  $TEST_DATA"
echo ""

# ====================================
# Experiment 1: Baseline with VLM
# ====================================
echo "======================================"
echo "Experiment 1: Baseline (Frozen) with VLM"
echo "======================================"
echo ""

python run_baseline.py \
    --model_name $MODEL_NAME \
    --test_data $TEST_DATA \
    --use_vlm \
    --batch_size 64 \
    --output_dir ${OUTPUT_DIR}/baseline_with_vlm \
    --save_embeddings \
    --device $DEVICE

echo ""
echo "Experiment 1 completed!"
echo ""

# ====================================
# Experiment 2: Baseline without VLM
# ====================================
echo "======================================"
echo "Experiment 2: Baseline (Frozen) without VLM"
echo "======================================"
echo ""

python run_baseline.py \
    --model_name $MODEL_NAME \
    --test_data $TEST_DATA \
    --batch_size 64 \
    --output_dir ${OUTPUT_DIR}/baseline_without_vlm \
    --save_embeddings \
    --device $DEVICE

echo ""
echo "Experiment 2 completed!"
echo ""

# ====================================
# Experiment 3: Fine-tuned with VLM
# ====================================
echo "======================================"
echo "Experiment 3: Fine-tuned with VLM"
echo "======================================"
echo ""

python train.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --use_vlm \
    --model_name $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --samples_per_cluster $SAMPLES_PER_CLUSTER \
    --epochs $EPOCHS \
    --lr 2e-5 \
    --temperature 0.07 \
    --output_dir $OUTPUT_DIR \
    --experiment_name finetuned_with_vlm \
    --save_best_only \
    --device $DEVICE

echo ""
echo "Training completed! Now evaluating..."
echo ""

python evaluate.py \
    --model_path ${OUTPUT_DIR}/finetuned_with_vlm/best_model \
    --test_data $TEST_DATA \
    --use_vlm \
    --batch_size 64 \
    --output_dir ${OUTPUT_DIR}/finetuned_with_vlm/evaluation_results \
    --save_embeddings \
    --device $DEVICE

echo ""
echo "Experiment 3 completed!"
echo ""

# ====================================
# Experiment 4: Fine-tuned without VLM
# ====================================
echo "======================================"
echo "Experiment 4: Fine-tuned without VLM"
echo "======================================"
echo ""

python train.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --model_name $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --samples_per_cluster $SAMPLES_PER_CLUSTER \
    --epochs $EPOCHS \
    --lr 2e-5 \
    --temperature 0.07 \
    --output_dir $OUTPUT_DIR \
    --experiment_name finetuned_without_vlm \
    --save_best_only \
    --device $DEVICE

echo ""
echo "Training completed! Now evaluating..."
echo ""

python evaluate.py \
    --model_path ${OUTPUT_DIR}/finetuned_without_vlm/best_model \
    --test_data $TEST_DATA \
    --batch_size 64 \
    --output_dir ${OUTPUT_DIR}/finetuned_without_vlm/evaluation_results \
    --save_embeddings \
    --device $DEVICE

echo ""
echo "Experiment 4 completed!"
echo ""

# ====================================
# Summary
# ====================================
echo "======================================"
echo "All Experiments Completed!"
echo "======================================"
echo ""
echo "Results saved in:"
echo "  1. ${OUTPUT_DIR}/baseline_with_vlm/"
echo "  2. ${OUTPUT_DIR}/baseline_without_vlm/"
echo "  3. ${OUTPUT_DIR}/finetuned_with_vlm/"
echo "  4. ${OUTPUT_DIR}/finetuned_without_vlm/"
echo ""
echo "You can now compare the metrics across all four experiments."
echo ""
