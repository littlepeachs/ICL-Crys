# Project Structure

## 📁 Directory Layout

```
ICL-Crys/
├── crystalicl.sh                 # Main entry point script
├── README.md                     # Project overview
├── requirements.txt              # Python dependencies
├── config.json                   # Configuration file
├── .gitignore                    # Git ignore rules
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── models/                   # Model implementations
│   │   ├── __init__.py
│   │   ├── crystal_tokenization.py    # SGS format tokenization
│   │   ├── sgs_parser.py              # SGS format parser
│   │   ├── instruction_builder.py     # Instruction construction
│   │   └── train_crystalicl.py        # Training with Qwen3-8B
│   │
│   ├── data/                     # Data loading and processing
│   │   ├── __init__.py
│   │   ├── data_loader.py             # Generic data loader
│   │   └── mp_dataset_loader.py       # Materials Project loader
│   │
│   ├── evaluation/               # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── evaluate.py                # Basic evaluation
│   │   ├── evaluate_complete.py       # Complete evaluation
│   │   ├── metrics_calculator.py      # Metrics calculator
│   │   ├── compute_paper_metrics.py   # Paper metrics (87.5%)
│   │   └── complete_metrics_with_dft.py  # With DFT (100%)
│   │
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── example_selector.py        # Example selection strategies
│       ├── structure_validator.py     # Structure validation
│       └── dft_calculator.py          # DFT calculation interface
│
├── scripts/                      # Bash scripts
│   ├── setup.sh                  # Environment setup
│   ├── train.sh                  # Training script
│   ├── evaluate.sh               # Evaluation script
│   ├── download_data.sh          # Download datasets
│   ├── test.sh                   # Run tests
│   ├── run_crystalicl.py         # Python runner
│   └── test_modules.py           # Module tests
│
├── docs/                         # Documentation
│   ├── README_QWEN3.md           # Qwen3-8B documentation
│   ├── USAGE_GUIDE.md            # Detailed usage guide
│   ├── METRICS_COMPLETION_EXPLAINED.md  # Metrics explanation
│   ├── PROJECT_SUMMARY.md        # Technical summary
│   ├── MISSING_FEATURES_REPORT.md  # Missing features report
│   └── ...                       # Other documentation
│
├── examples/                     # Usage examples
│   └── examples.py               # Python examples
│
├── tests/                        # Test files
│   └── (test files)
│
├── data/                         # Data directory (created at runtime)
│   ├── mp20.json
│   ├── mp30.json
│   └── ...
│
├── output/                       # Output directory (created at runtime)
│   └── crystalicl_qwen3_8b/
│
└── venv/                         # Virtual environment (created by setup)
```

## 🔧 Key Components

### Source Code (`src/`)

#### Models (`src/models/`)
- **crystal_tokenization.py**: Converts crystal structures to SGS format
- **sgs_parser.py**: Parses SGS format back to structures
- **instruction_builder.py**: Builds zero-shot and few-shot instructions
- **train_crystalicl.py**: Training pipeline with Qwen3-8B and LoRA

#### Data (`src/data/`)
- **data_loader.py**: Generic data loading (JSON, CIF)
- **mp_dataset_loader.py**: Materials Project API integration

#### Evaluation (`src/evaluation/`)
- **metrics_calculator.py**: Basic metrics calculation
- **compute_paper_metrics.py**: Paper Table 1 & 2 metrics
- **complete_metrics_with_dft.py**: 100% metrics with DFT

#### Utils (`src/utils/`)
- **example_selector.py**: Three selection strategies (C, F, CF)
- **structure_validator.py**: Structure validation and fixing
- **dft_calculator.py**: DFT interface (VASP, QE, ML)

### Scripts (`scripts/`)

All scripts are executable bash files:

- **setup.sh**: One-command environment setup
- **train.sh**: Training with customizable parameters
- **evaluate.sh**: Evaluation with all metrics
- **download_data.sh**: Download MP20/MP30/P5/C24
- **test.sh**: Run all module tests

### Main Entry Point

**crystalicl.sh**: Unified interface for all operations

```bash
bash crystalicl.sh setup      # Setup environment
bash crystalicl.sh train      # Train model
bash crystalicl.sh evaluate   # Evaluate model
bash crystalicl.sh download   # Download data
bash crystalicl.sh test       # Run tests
```

## 📊 Data Flow

```
Input Data (CIF/JSON)
    ↓
CrystalDataLoader
    ↓
CrystalTokenizer (SGS format)
    ↓
InstructionBuilder
    ↓
CrystalICLTrainer (Qwen3-8B + LoRA)
    ↓
Generated Text
    ↓
SGSParser
    ↓
Structure Validator
    ↓
Metrics Calculator
    ↓
Evaluation Results
```

## 🚀 Quick Commands

```bash
# Setup
bash crystalicl.sh setup

# Train with sample data
bash crystalicl.sh train

# Train with real data
bash crystalicl.sh train --data ./data/mp20.json --epochs 5

# Evaluate
bash crystalicl.sh evaluate --model ./output/crystalicl_qwen3_8b

# Download datasets
export MP_API_KEY="your_key"
bash crystalicl.sh download --dataset mp20

# Run tests
bash crystalicl.sh test
```

## 📝 Configuration

Edit `config.json` to customize:
- Model parameters (name, LoRA settings)
- Training hyperparameters (epochs, batch size, learning rate)
- Evaluation settings (metrics, sample counts)
- Data paths and formats

## 🔍 Finding Files

Use these patterns to locate specific functionality:

- **Tokenization**: `src/models/crystal_tokenization.py`
- **Training**: `src/models/train_crystalicl.py`
- **Evaluation**: `src/evaluation/*.py`
- **Data loading**: `src/data/*.py`
- **Utilities**: `src/utils/*.py`
- **Scripts**: `scripts/*.sh`
- **Documentation**: `docs/*.md`
- **Examples**: `examples/*.py`

## 📚 Documentation

- **README.md**: Project overview and quick start
- **docs/USAGE_GUIDE.md**: Comprehensive usage guide
- **docs/README_QWEN3.md**: Qwen3-8B specific documentation
- **docs/METRICS_COMPLETION_EXPLAINED.md**: Evaluation metrics details

---

**Last Updated**: 2026-04-21
