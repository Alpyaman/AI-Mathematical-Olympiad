"""
Phase 1.2 Demo: Data Curation and Preprocessing

This script demonstrates the complete data pipeline for mathematical reasoning:
1. Data schema and problem representation
2. Chain-of-Thought formatting
3. Data loading and collection
4. Quality filtering and preprocessing
5. Dataset creation for training
6. Integration with the mathematical tokenizer
"""

from src.data import (
    create_sample_problems,
    ChainOfThoughtFormatter,
    ProblemFormatter,
    DataQualityFilter,
    DataPreprocessor,
    MathDatasetLoader,
    split_dataset,
    validate_problem,
    compute_problem_score,
)
from src.tokenizer import MathTokenizer


def demo_data_schema():
    """Demonstrate the data schema and problem structure."""
    print("\n" + "="*70)
    print("PHASE 1.2 - DATA SCHEMA DEMONSTRATION")
    print("="*70)

    # Create sample problems
    problems = create_sample_problems()

    print(f"\nCreated {len(problems)} sample problems\n")

    # Display first problem in detail
    problem = problems[0]
    print(problem)
    print()

    # Show problem metadata
    print("Metadata:")
    print(f"  Difficulty: {problem.difficulty.value}")
    print(f"  Type: {problem.problem_type.value}")
    print(f"  Topics: {', '.join(problem.topics)}")
    print(f"  Solution Steps: {len(problem.solution.steps)}")
    print()

    return problems


def demo_cot_formatting(problems):
    """Demonstrate Chain-of-Thought formatting."""
    print("\n" + "="*70)
    print("PHASE 1.2 - CHAIN-OF-THOUGHT FORMATTING")
    print("="*70)

    # Create formatters
    cot_formatter = ChainOfThoughtFormatter(use_special_tokens=True)
    problem = problems[1]  # Use the number theory problem

    print("\n1. Full CoT Format (with special tokens):")
    print("-" * 70)
    formatted = cot_formatter.format_problem(problem)
    print(formatted)

    print("\n\n2. Training Format (input/target separation):")
    print("-" * 70)
    train_format = cot_formatter.format_for_training(problem)
    print("INPUT:")
    print(train_format['input'])
    print("\nTARGET:")
    print(train_format['target'])

    print("\n\n3. Minimal Format (answer only):")
    print("-" * 70)
    minimal = ProblemFormatter.format_minimal(problem)
    print(minimal)

    print("\n\n4. Conversational Format:")
    print("-" * 70)
    conversational = ProblemFormatter.format_conversational(problem)
    print(conversational)


def demo_quality_filtering(problems):
    """Demonstrate quality filtering."""
    print("\n\n" + "="*70)
    print("PHASE 1.2 - QUALITY FILTERING")
    print("="*70)

    # Create quality filter
    quality_filter = DataQualityFilter(
        min_problem_length=20,
        min_solution_steps=2,
        require_mathematical_symbols=True
    )

    print("\nQuality Filter Configuration:")
    print(f"  Min Problem Length: {quality_filter.min_problem_length}")
    print(f"  Min Solution Steps: {quality_filter.min_solution_steps}")
    print(f"  Require Math Symbols: {quality_filter.require_mathematical_symbols}")

    print(f"\n\nValidating {len(problems)} problems...")
    for problem in problems:
        passes = quality_filter.filter_problem(problem)
        score = compute_problem_score(problem)
        issues = validate_problem(problem)

        status = "✓ PASS" if passes else "✗ FAIL"
        print(f"\n{status} - {problem.problem_id}")
        print(f"  Quality Score: {score:.2f}")
        print(f"  Steps: {len(problem.solution.steps)}")
        print(f"  Problem Length: {len(problem.problem_statement)} chars")
        if issues:
            print(f"  Issues: {', '.join(issues)}")


def demo_preprocessing_pipeline(problems):
    """Demonstrate complete preprocessing pipeline."""
    print("\n\n" + "="*70)
    print("PHASE 1.2 - PREPROCESSING PIPELINE")
    print("="*70)

    # Create preprocessor
    preprocessor = DataPreprocessor()

    print("\nPreprocessing pipeline includes:")
    print("  1. Quality filtering")
    print("  2. Text normalization")
    print("  3. Validation")

    # Preprocess
    print(f"\n\nPreprocessing {len(problems)} problems...")
    processed = preprocessor.preprocess(
        problems,
        filter_quality=True,
        normalize_text=True,
        verbose=True
    )

    print(f"\n\nResult: {len(processed)} high-quality problems ready for training")

    return processed


def demo_tokenizer_integration(problems):
    """Demonstrate tokenizer integration with data pipeline."""
    print("\n\n" + "="*70)
    print("PHASE 1.2 - TOKENIZER INTEGRATION")
    print("="*70)

    # Initialize tokenizer
    tokenizer = MathTokenizer()

    # Format and tokenize a problem
    formatter = ChainOfThoughtFormatter(use_special_tokens=True)
    problem = problems[1]

    print("\nProblem:")
    print(problem.problem_statement[:100] + "...")

    # Format for training
    formatted = formatter.format_problem(problem)

    print(f"\n\nFormatted Length: {len(formatted)} characters")

    # Tokenize
    encoded = tokenizer.encode(formatted, max_length=512, padding=True, truncation=True)

    print(f"Tokenized Length: {len(encoded['input_ids'])} tokens")
    print(f"Attention Mask Length: {len(encoded['attention_mask'])}")

    # Show first few tokens
    print("\nFirst 20 tokens:")
    print(encoded['input_ids'][:20])

    # Decode to verify
    decoded = tokenizer.decode(encoded['input_ids'][:100])
    print("\nDecoded (first 100 tokens):")
    print(decoded[:200] + "...")


def demo_dataset_splitting(problems):
    """Demonstrate train/val/test splitting."""
    print("\n\n" + "="*70)
    print("PHASE 1.2 - DATASET SPLITTING")
    print("="*70)

    print(f"\nTotal problems: {len(problems)}")

    # Standard split
    train, val, test = split_dataset(
        problems,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=True
    )

    print("\nStandard Split:")
    print(f"  Training:   {len(train)} problems (70%)")
    print(f"  Validation: {len(val)} problems (15%)")
    print(f"  Test:       {len(test)} problems (15%)")

    return train, val, test


def demo_dataset_statistics():
    """Demonstrate dataset statistics."""
    print("\n\n" + "="*70)
    print("PHASE 1.2 - DATASET STATISTICS")
    print("="*70)

    # Create sample problems
    problems = create_sample_problems()

    # Compute statistics
    loader = MathDatasetLoader()
    stats = loader.compute_statistics(problems)

    print()
    print(stats)


def demo_pytorch_dataset():
    """Demonstrate PyTorch dataset creation."""
    print("\n\n" + "="*70)
    print("PHASE 1.2 - PYTORCH DATASET")
    print("="*70)

    try:
        import torch
        from src.data import MathReasoningDataset, DataCollator

        # Create problems and tokenizer
        problems = create_sample_problems()
        tokenizer = MathTokenizer()

        # Create dataset
        dataset = MathReasoningDataset(
            problems=problems,
            tokenizer=tokenizer,
            max_length=512,
        )

        print(f"\nDataset created with {len(dataset)} examples")

        # Get a sample
        sample = dataset[0]

        print("\nSample batch item:")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Attention Mask shape: {sample['attention_mask'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")
        print(f"  Problem ID: {sample['problem_id']}")

        # Create collator and test batching
        collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([dataset[0], dataset[1]])

        print("\nBatched (2 examples):")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention Mask shape: {batch['attention_mask'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")

        print("\n✓ PyTorch dataset ready for training!")

    except ImportError:
        print("\nPyTorch not available, skipping PyTorch dataset demo")
        print("Install PyTorch to use the full training pipeline")


def print_phase_summary():
    """Print Phase 1.2 summary."""
    print("\n\n" + "="*70)
    print("PHASE 1.2 IMPLEMENTATION SUMMARY")
    print("="*70)

    summary = """
Phase 1.2: Data Curation and Preprocessing - COMPLETED ✓

Key Components Implemented:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Data Schema
   • MathProblem: Complete problem representation
   • MathSolution: Step-by-step solution structure
   • ReasoningStep: Individual reasoning steps
   • Difficulty levels: Easy, Medium, Hard, Olympiad, IMO
   • Problem types: Algebra, Geometry, Number Theory, etc.

2. Chain-of-Thought (CoT) Formatting
   • ChainOfThoughtFormatter: Structures for step-by-step reasoning
   • Multiple format styles: CoT, minimal, conversational
   • Special tokens: <step>, <solution>, <proof>, etc.
   • Input/target separation for training

3. Data Loading
   • MathDatasetLoader: Unified interface for multiple sources
   • Support for MATH dataset (Hendrycks et al.)
   • Support for AoPS dataset
   • Custom dataset format support (JSON, JSONL)
   • Automatic solution parsing

4. Quality Filtering & Preprocessing
   • DataQualityFilter: Length, complexity, content checks
   • TextNormalizer: Consistent formatting
   • Validation: Problem structure and content validation
   • Quality scoring: Automated quality assessment

5. PyTorch Dataset Integration
   • MathReasoningDataset: PyTorch-compatible dataset
   • DataCollator: Efficient batching with padding
   • Train/val/test splitting (standard and stratified)
   • Full tokenizer integration

6. Tokenizer Integration
   • Seamless integration with mathematical tokenizer
   • Preserves all mathematical symbols (200+)
   • Proper handling of special tokens
   • Efficient encoding/decoding

Data Pipeline Flow:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Raw Data → Load → Filter → Normalize → Format (CoT) → Tokenize → Dataset → Training

Supported Datasets:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ MATH (12,500 problems, competition level)
✓ AoPS (Art of Problem Solving)
✓ IMO (International Mathematical Olympiad)
✓ Custom JSON/JSONL formats

Features:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Step-by-step reasoning preservation
✓ Mathematical notation preservation (200+ symbols)
✓ Quality filtering and validation
✓ Multiple formatting styles
✓ Stratified splitting by difficulty/type
✓ PyTorch DataLoader integration
✓ Efficient batching and padding

Next Steps (Phase 2: Training):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Implement training loop
• Set up learning rate scheduling
• Define evaluation metrics
• Implement checkpointing
• Add distributed training support
"""
    print(summary)


def main():
    """Run all Phase 1.2 demonstrations."""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "AI MATHEMATICAL OLYMPIAD" + " "*29 + "║")
    print("║" + " "*10 + "Phase 1.2: Data Curation & Preprocessing" + " "*18 + "║")
    print("╚" + "═"*68 + "╝")

    # 1. Data schema
    problems = demo_data_schema()

    # 2. CoT formatting
    demo_cot_formatting(problems)

    # 3. Quality filtering
    demo_quality_filtering(problems)

    # 4. Preprocessing pipeline
    processed = demo_preprocessing_pipeline(problems)

    # 5. Tokenizer integration
    demo_tokenizer_integration(processed)

    # 6. Dataset splitting
    train, val, test = demo_dataset_splitting(processed)

    # 7. Dataset statistics
    demo_dataset_statistics()

    # 8. PyTorch dataset
    demo_pytorch_dataset()

    # 9. Summary
    print_phase_summary()

    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70 + "\n")

    print("You can now:")
    print("  1. Download MATH dataset and load with MathDatasetLoader")
    print("  2. Create custom datasets in JSON format")
    print("  3. Train the model with the preprocessed data")
    print("  4. Proceed to Phase 2: Training Infrastructure\n")


if __name__ == "__main__":
    main()