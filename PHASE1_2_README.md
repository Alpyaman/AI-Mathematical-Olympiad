# Phase 1.2: Data Curation and Preprocessing

## Overview

Phase 1.2 implements a comprehensive data pipeline for mathematical reasoning, with support for olympiad-level problems and Chain-of-Thought (CoT) formatting.

## Key Features

### 🎯 Core Components

1. **Data Schema** - Structured representation of mathematical problems
2. **CoT Formatting** - Multiple formatting styles for reasoning
3. **Data Loading** - Support for MATH, AoPS, IMO, and custom datasets
4. **Quality Filtering** - Automated filtering and validation
5. **PyTorch Integration** - Ready-to-train dataset classes
6. **Tokenizer Integration** - Seamless integration with 200+ math symbols

## Architecture

### Data Schema

```python
from src.data import MathProblem, MathSolution, ReasoningStep

# Create a solution with steps
solution = MathSolution(steps=[], final_answer="x = 3")
solution.add_step(
    "Start with the equation",
    mathematical_expression="2x + 4 = 10"
)

# Create a problem
problem = MathProblem(
    problem_id="PROB_001",
    problem_statement="Solve for x: 2x + 4 = 10",
    solution=solution,
    difficulty=DifficultyLevel.EASY,
    problem_type=ProblemType.ALGEBRA
)
```

### Chain-of-Thought Formatting

```python
from src.data import ChainOfThoughtFormatter

formatter = ChainOfThoughtFormatter(use_special_tokens=True)

# Format for training
formatted = formatter.format_for_training(problem)
print(formatted['input'])   # Problem statement
print(formatted['target'])  # Step-by-step solution
```

### Data Loading

```python
from src.data import MathDatasetLoader

loader = MathDatasetLoader(cache_dir="./data/cache")

# Load MATH dataset
math_problems = loader.load_math_dataset(split="train")

# Load custom dataset
custom_problems = loader.load_custom_dataset("my_problems.json")

# Compute statistics
stats = loader.compute_statistics(math_problems)
print(stats)
```

### Quality Filtering

```python
from src.data import DataPreprocessor, DataQualityFilter

# Create filter with custom thresholds
quality_filter = DataQualityFilter(
    min_problem_length=20,
    min_solution_steps=2,
    require_mathematical_symbols=True
)

# Preprocess dataset
preprocessor = DataPreprocessor(quality_filter=quality_filter)
processed = preprocessor.preprocess(
    problems,
    filter_quality=True,
    normalize_text=True
)
```

### PyTorch Dataset

```python
from src.data import MathReasoningDataset, create_dataloaders
from src.tokenizer import MathTokenizer

# Create dataset
tokenizer = MathTokenizer()
dataset = MathReasoningDataset(
    problems=processed,
    tokenizer=tokenizer,
    max_length=2048
)

# Split into train/val/test
from src.data import split_dataset
train, val, test = split_dataset(
    processed,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Create train/val datasets
train_dataset = MathReasoningDataset(train, tokenizer)
val_dataset = MathReasoningDataset(val, tokenizer)

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size=8
)
```

## Directory Structure

```
src/data/
├── __init__.py              # Module exports
├── data_schema.py          # Problem/solution data structures
├── data_formatter.py       # CoT and other formatters
├── data_loader.py          # Dataset loading utilities
├── dataset.py              # PyTorch dataset class
└── preprocessing.py        # Quality filtering and preprocessing
```

## Data Schema Details

### MathProblem

```python
@dataclass
class MathProblem:
    problem_id: str
    problem_statement: str
    solution: MathSolution
    difficulty: DifficultyLevel     # EASY, MEDIUM, HARD, OLYMPIAD, IMO
    problem_type: ProblemType       # ALGEBRA, GEOMETRY, NUMBER_THEORY, etc.
    topics: List[str]
    source: str
    year: Optional[int]
```

### MathSolution

```python
@dataclass
class MathSolution:
    steps: List[ReasoningStep]
    final_answer: str
    answer_type: str  # "exact", "approximate", "symbolic", "proof"
    verification: Optional[str]
```

### ReasoningStep

```python
@dataclass
class ReasoningStep:
    step_number: int
    description: str
    mathematical_expression: Optional[str]
    justification: Optional[str]
```

## Formatting Styles

### 1. Chain-of-Thought (CoT)

Structured format with special tokens for step-by-step reasoning:

```
<bos>
Problem: [problem statement]

<solution>
<step>
Step 1: [description]
<math>[expression]</math>
  Reasoning: [justification]
</step>
...
<answer>[final answer]</answer>
</solution>
<eos>
```

### 2. Minimal Format

Simple problem + answer format:

```
[problem statement]

Answer: [final answer]
```

### 3. Conversational Format

Instruction-following dialogue format:

```
User: Can you solve this problem?
[problem statement]