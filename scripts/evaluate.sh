#!/bin/bash
# Evaluation script for CrystalICL

set -e

# Default parameters
MODEL_PATH="./output/crystalicl_qwen3_8b"
TEST_DATA="./data/test.json"
OUTPUT_FILE="./output/evaluation_results.json"
NUM_SAMPLES=1000
NUM_UNCONDITIONAL=10000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --test-data)
            TEST_DATA="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --num-unconditional)
            NUM_UNCONDITIONAL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash scripts/evaluate.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model PATH             Model path"
            echo "  --test-data PATH         Test data path"
            echo "  --output FILE            Output file"
            echo "  --num-samples N          Number of conditional samples (default: 1000)"
            echo "  --num-unconditional N    Number of unconditional samples (default: 10000)"
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
echo "CrystalICL Evaluation"
echo "=================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Test data: $TEST_DATA"
echo "  Output: $OUTPUT_FILE"
echo "  Conditional samples: $NUM_SAMPLES"
echo "  Unconditional samples: $NUM_UNCONDITIONAL"
echo ""

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please train a model first using: bash scripts/train.sh"
    exit 1
fi

# Check if test data exists
if [ ! -f "$TEST_DATA" ]; then
    echo "Warning: Test data not found at $TEST_DATA"
    echo "Creating sample test data..."
    python -c "
from src.data import CrystalDataLoader
loader = CrystalDataLoader()
data = loader.create_sample_dataset(100)
splits = loader.split_dataset(data)
loader.save_to_json(splits['test'], '$TEST_DATA')
"
fi

echo "Starting evaluation..."
echo ""

# Run evaluation
python src/evaluation/evaluate_complete.py \
    --model_path "$MODEL_PATH" \
    --test_data "$TEST_DATA" \
    --output "$OUTPUT_FILE" \
    --num_samples "$NUM_SAMPLES" \
    --num_unconditional "$NUM_UNCONDITIONAL"

echo ""
echo "=================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "=================================="
