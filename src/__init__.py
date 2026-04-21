"""
CrystalICL - Crystal Generation with In-Context Learning
"""

__version__ = "1.0.0"
__author__ = "CrystalICL Team"
__description__ = "Crystal generation using Qwen3-8B with in-context learning"

from src.models import CrystalTokenizer, SGSParser, InstructionBuilder, CrystalICLTrainer
from src.data import CrystalDataLoader, MaterialsProjectLoader
from src.evaluation import CrystalMetricsCalculator, PaperMetricsComputer
from src.utils import ExampleSelector, StructureValidator, DFTCalculator

__all__ = [
    'CrystalTokenizer',
    'SGSParser',
    'InstructionBuilder',
    'CrystalICLTrainer',
    'CrystalDataLoader',
    'MaterialsProjectLoader',
    'CrystalMetricsCalculator',
    'PaperMetricsComputer',
    'ExampleSelector',
    'StructureValidator',
    'DFTCalculator',
]
