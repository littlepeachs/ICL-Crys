#!/bin/bash
# Test script for CrystalICL

set -e

echo "=================================="
echo "CrystalICL Module Tests"
echo "=================================="

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo ""
echo "Running module tests..."
python scripts/test_modules.py

echo ""
echo "=================================="
echo "All tests completed!"
echo "=================================="
