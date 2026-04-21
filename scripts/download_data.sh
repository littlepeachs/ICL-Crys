#!/bin/bash
# Download Materials Project datasets

set -e

# Default parameters
DATASET="all"
OUTPUT_DIR="./data"
API_KEY="${MP_API_KEY}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash scripts/download_data.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset NAME           Dataset to download (mp20|mp30|p5|c24|all)"
            echo "  --output DIR             Output directory (default: ./data)"
            echo "  --api-key KEY            Materials Project API key"
            echo "  --help                   Show this help message"
            echo ""
            echo "Note: You can also set MP_API_KEY environment variable"
            echo "Get your API key from: https://materialsproject.org/api"
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
echo "Download Materials Project Data"
echo "=================================="
echo ""

# Check API key
if [ -z "$API_KEY" ]; then
    echo "Error: Materials Project API key not provided"
    echo ""
    echo "Please provide API key using one of:"
    echo "  1. --api-key option"
    echo "  2. MP_API_KEY environment variable"
    echo ""
    echo "Get your API key from: https://materialsproject.org/api"
    exit 1
fi

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Set API key
export MP_API_KEY="$API_KEY"

echo "Starting download..."
echo ""

# Run download
python src/data/mp_dataset_loader.py \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --api_key "$API_KEY"

echo ""
echo "=================================="
echo "Download completed!"
echo "Data saved to: $OUTPUT_DIR"
echo "=================================="
