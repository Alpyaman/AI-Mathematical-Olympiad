"""Data module for mathematical reasoning datasets."""

from .data_schema import (
    MathProblem,
    MathSolution,
    ReasoningStep,
    DifficultyLevel,
    ProblemType,
    DatasetStatistics,
)
from .data_formatter import (
    ChainOfThoughtFormatter,
    ProblemFormatter,
    BatchFormatter,
    format_dataset_for_finetuning,
)
from .data_loader import MathDatasetLoader, create_sample_problems
from .dataset import (
    MathReasoningDataset,
    DataCollator,
    create_dataloaders,
    split_dataset,
    stratified_split,
)
from .preprocessing import (
    DataQualityFilter,
    TextNormalizer,
    DataPreprocessor,
    validate_problem,
    compute_problem_score,
)
from .pretraining_dataset import (
    TextStreamDataset,
    MixedDomainDataset,
    PreTrainingDataCollator,
    prepare_pretraining_data,
    create_sample_pretraining_data,
)

__all__ = [
    # Schema
    "MathProblem",
    "MathSolution",
    "ReasoningStep",
    "DifficultyLevel",
    "ProblemType",
    "DatasetStatistics",
    # Formatting
    "ChainOfThoughtFormatter",
    "ProblemFormatter",
    "BatchFormatter",
    "format_dataset_for_finetuning",
    # Loading
    "MathDatasetLoader",
    "create_sample_problems",
    # Dataset
    "MathReasoningDataset",
    "DataCollator",
    "create_dataloaders",
    "split_dataset",
    "stratified_split",
    # Preprocessing
    "DataQualityFilter",
    "TextNormalizer",
    "DataPreprocessor",
    "validate_problem",
    "compute_problem_score",
    # Pre-training (Phase 2.1)
    "TextStreamDataset",
    "MixedDomainDataset",
    "PreTrainingDataCollator",
    "prepare_pretraining_data",
    "create_sample_pretraining_data",
]