#!/bin/bash
# Main entry point for CrystalICL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              CrystalICL - Crystal Generation                 ║"
echo "║           In-Context Learning with Qwen3-8B                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Show usage
show_usage() {
    echo "Usage: bash crystalicl.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup              Setup environment and install dependencies"
    echo "  test               Run module tests"
    echo "  train              Train the model"
    echo "  evaluate           Evaluate the model"
    echo "  download           Download Materials Project datasets"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  bash crystalicl.sh setup"
    echo "  bash crystalicl.sh train --data ./data/mp20.json"
    echo "  bash crystalicl.sh evaluate --model ./output/my_model"
    echo ""
    echo "For command-specific help:"
    echo "  bash crystalicl.sh train --help"
    echo "  bash crystalicl.sh evaluate --help"
    echo ""
}

# Check if command is provided
if [ $# -eq 0 ]; then
    show_usage
    exit 0
fi

COMMAND=$1
shift

# Execute command
case $COMMAND in
    setup)
        echo -e "${GREEN}Running setup...${NC}"
        bash "$SCRIPT_DIR/setup.sh" "$@"
        ;;
    test)
        echo -e "${GREEN}Running tests...${NC}"
        bash "$SCRIPT_DIR/test.sh" "$@"
        ;;
    train)
        echo -e "${GREEN}Starting training...${NC}"
        bash "$SCRIPT_DIR/train.sh" "$@"
        ;;
    evaluate|eval)
        echo -e "${GREEN}Starting evaluation...${NC}"
        bash "$SCRIPT_DIR/evaluate.sh" "$@"
        ;;
    download)
        echo -e "${GREEN}Downloading datasets...${NC}"
        bash "$SCRIPT_DIR/download_data.sh" "$@"
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$COMMAND'${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac
