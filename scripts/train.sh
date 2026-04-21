#!/bin/bash
# Training script for CrystalICL

set -e

# Default parameters
MODEL_NAME="Qwen/Qwen3-8B"
OUTPUT_DIR="./output/crystalicl_qwen3_8b"
DATA_PATH=""
NUM_EPOCHS=3
BATCH_SIZE=1
LEARNING_RATE=5e-4
K_SHOT=3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --k-shot)
            K_SHOT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash scripts/train.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL_NAME       Base model (default: Qwen/Qwen3-8B)"
            echo "  --output OUTPUT_DIR      Output directory"
            echo "  --data DATA_PATH         Training data path"
            echo "  --epochs NUM             Number of epochs (default: 3)"
            echo "  --batch-size SIZE        Batch size (default: 1)"
            echo "  --lr RATE                Learning rate (default: 5e-4)"
            echo "  --k-shot K               K-shot learning (default: 3)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=================================="
echo "CrystalICL Training"
echo "=================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  Data: ${DATA_PATH:-'sample data'}"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  K-shot: $K_SHOT"
echo ""

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Build command
CMD="python scripts/run_crystalicl.py \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --k_shot $K_SHOT \
    --do_train"

# Add data path if provided
if [ -n "$DATA_PATH" ]; then
    CMD="$CMD --data_path $DATA_PATH --data_format json"
else
    CMD="$CMD --use_sample_data --num_samples 100"
fi

echo "Starting training..."
echo ""

# Run training
eval $CMD

echo ""
echo "=================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=================================="
