"""
Data Preprocessing and Quality Filtering

This module provides utilities for preprocessing and filtering
mathematical reasoning data for optimal training quality.
"""

import re
from typing import List, Optional
from .data_schema import MathProblem


class DataQualityFilter:
    """
    Filters mathematical problems based on quality metrics.
    """

    def __init__(
        self,
        min_problem_length: int = 20,
        max_problem_length: int = 2000,
        min_solution_steps: int = 1,
        max_solution_steps: int = 20,
        min_answer_length: int = 1,
        require_mathematical_symbols: bool = False,
    ):
        """
        Initialize quality filter.

        Args:
            min_problem_length: Minimum problem statement length
            max_problem_length: Maximum problem statement length
            min_solution_steps: Minimum number of solution steps
            max_solution_steps: Maximum number of solution steps
            min_answer_length: Minimum final answer length
            require_mathematical_symbols: Require mathematical notation
        """
        self.min_problem_length = min_problem_length
        self.max_problem_length = max_problem_length
        self.min_solution_steps = min_solution_steps
        self.max_solution_steps = max_solution_steps
        self.min_answer_length = min_answer_length
        self.require_mathematical_symbols = require_mathematical_symbols

    def filter_problem(self, problem: MathProblem) -> bool:
        """
        Check if a problem meets quality criteria.

        Args:
            problem: Problem to check

        Returns:
            True if problem passes all filters
        """
        # Length checks
        problem_len = len(problem.problem_statement)
        if problem_len < self.min_problem_length or problem_len > self.max_problem_length:
            return False

        # Solution steps check
        num_steps = len(problem.solution.steps)
        if num_steps < self.min_solution_steps or num_steps > self.max_solution_steps:
            return False

        # Answer check
        if len(problem.solution.final_answer) < self.min_answer_length:
            return False

        # Mathematical symbols check
        if self.require_mathematical_symbols:
            if not self._contains_mathematical_content(problem.problem_statement):
                return False

        # Check for malformed content
        if self._is_malformed(problem):
            return False

        return True

    def filter_dataset(self, problems: List[MathProblem]) -> List[MathProblem]:
        """
        Filter a dataset of problems.

        Args:
            problems: List of problems to filter

        Returns:
            Filtered list of problems
        """
        return [p for p in problems if self.filter_problem(p)]

    def _contains_mathematical_content(self, text: str) -> bool:
        """Check if text contains mathematical symbols or notation."""
        math_indicators = [
            r'\d+',  # Numbers
            r'[+\-*/=<>≤≥≠]',  # Operations
            r'[∀∃∈⊂∪∩]',  # Set theory
            r'[∫∑∏∂]',  # Calculus
            r'[α-ωΑ-Ω]',  # Greek letters
            r'\\[a-zA-Z]+',  # LaTeX commands
        ]

        for pattern in math_indicators:
            if re.search(pattern, text):
                return True

        return False

    def _is_malformed(self, problem: MathProblem) -> bool:
        """Check if problem has malformed content."""
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', problem.problem_statement)) / \
                            max(len(problem.problem_statement), 1)

        if special_char_ratio > 0.5:  # More than 50% special characters
            return True

        # Check for empty steps
        for step in problem.solution.steps:
            if not step.description.strip():
                return True

        # Check for placeholder text
        placeholders = ['TODO', 'TBD', '[...]', '???']
        full_text = problem.problem_statement + problem.solution.final_answer
        if any(placeholder in full_text for placeholder in placeholders):
            return True

        return False


class TextNormalizer:
    """
    Normalizes mathematical text for consistent formatting.
    """

    def __init__(self, preserve_latex: bool = True):
        """
        Initialize text normalizer.

        Args:
            preserve_latex: Whether to preserve LaTeX commands
        """
        self.preserve_latex = preserve_latex

    def normalize_problem(self, problem: MathProblem) -> MathProblem:
        """
        Normalize all text in a problem.

        Args:
            problem: Problem to normalize

        Returns:
            Normalized problem
        """
        problem.problem_statement = self.normalize_text(problem.problem_statement)

        for step in problem.solution.steps:
            step.description = self.normalize_text(step.description)
            if step.mathematical_expression:
                step.mathematical_expression = self.normalize_text(step.mathematical_expression)
            if step.justification:
                step.justification = self.normalize_text(step.justification)

        problem.solution.final_answer = self.normalize_text(problem.solution.final_answer)

        if problem.solution.verification:
            problem.solution.verification = self.normalize_text(problem.solution.verification)

        return problem

    def normalize_text(self, text: str) -> str:
        """
        Normalize a single text string.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # Normalize dashes
        text = text.replace('—', '-').replace('–', '-')

        # Normalize ellipsis
        text = text.replace('...', '…')

        # Normalize mathematical notation (optional)
        text = text.replace('×', '*')
        if not self.preserve_latex:
            # Convert LaTeX to Unicode (basic conversions)
            replacements = {
                r'\\leq': '≤',
                r'\\geq': '≥',
                r'\\neq': '≠',
                r'\\in': '∈',
                r'\\subset': '⊂',
                r'\\cup': '∪',
                r'\\cap': '∩',
                r'\\forall': '∀',
                r'\\exists': '∃',
                r'\\multiply': '*',
                r'\\times': '*',
                r'\\div': '÷',
                r'\\pm': '±',
                r'\\sqrt': '√',
            }
            for latex, unicode in replacements.items():
                text = text.replace(latex, unicode)

        return text


class DataAugmenter:
    """
    Augments mathematical problems for data diversity.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize data augmenter.

        Args:
            seed: Random seed
        """
        import random
        self.random = random.Random(seed)

    def augment_problem(self, problem: MathProblem) -> MathProblem:
        """
        Create an augmented version of a problem.

        Args:
            problem: Problem to augment

        Returns:
            Augmented problem
        """
        # For now, return a copy (placeholder for future augmentation strategies)
        # Future augmentations could include:
        # - Rephrasing problem statement
        # - Changing variable names
        # - Reordering solution steps (when order doesn't matter)
        # - Adding alternative solution methods

        return problem


class DataPreprocessor:
    """
    Complete preprocessing pipeline for mathematical reasoning data.
    """

    def __init__(
        self,
        quality_filter: Optional[DataQualityFilter] = None,
        normalizer: Optional[TextNormalizer] = None,
        augmenter: Optional[DataAugmenter] = None,
    ):
        """
        Initialize preprocessor.

        Args:
            quality_filter: Quality filter (creates default if None)
            normalizer: Text normalizer (creates default if None)
            augmenter: Data augmenter (optional)
        """
        self.quality_filter = quality_filter or DataQualityFilter()
        self.normalizer = normalizer or TextNormalizer()
        self.augmenter = augmenter

    def preprocess(
        self,
        problems: List[MathProblem],
        filter_quality: bool = True,
        normalize_text: bool = True,
        augment_data: bool = False,
        verbose: bool = True,
    ) -> List[MathProblem]:
        """
        Preprocess a dataset of problems.

        Args:
            problems: List of problems to preprocess
            filter_quality: Whether to apply quality filtering
            normalize_text: Whether to normalize text
            augment_data: Whether to augment data
            verbose: Whether to print progress

        Returns:
            Preprocessed list of problems
        """
        if verbose:
            print(f"Starting preprocessing with {len(problems)} problems...")

        # Quality filtering
        if filter_quality:
            before = len(problems)
            problems = self.quality_filter.filter_dataset(problems)
            after = len(problems)
            if verbose:
                print(f"Quality filtering: {before} -> {after} problems ({before-after} filtered)")

        # Text normalization
        if normalize_text:
            problems = [self.normalizer.normalize_problem(p) for p in problems]
            if verbose:
                print("Text normalization: complete")

        # Data augmentation
        if augment_data and self.augmenter:
            augmented = []
            for problem in problems:
                augmented.append(problem)
                augmented.append(self.augmenter.augment_problem(problem))
            problems = augmented
            if verbose:
                print(f"Data augmentation: {len(problems)} total problems")

        if verbose:
            print(f"Preprocessing complete: {len(problems)} problems ready for training")

        return problems


def validate_problem(problem: MathProblem) -> List[str]:
    """
    Validate a problem and return list of issues.

    Args:
        problem: Problem to validate

    Returns:
        List of validation issues (empty if valid)
    """
    issues = []

    # Check required fields
    if not problem.problem_id:
        issues.append("Missing problem_id")

    if not problem.problem_statement or not problem.problem_statement.strip():
        issues.append("Empty problem statement")

    if not problem.solution.steps:
        issues.append("No solution steps")

    if not problem.solution.final_answer or not problem.solution.final_answer.strip():
        issues.append("Empty final answer")

    # Check step numbering
    for i, step in enumerate(problem.solution.steps, 1):
        if step.step_number != i:
            issues.append(f"Step numbering mismatch: expected {i}, got {step.step_number}")

    # Check for reasonable content
    if len(problem.problem_statement) < 10:
        issues.append("Problem statement too short")

    if len(problem.solution.final_answer) > 500:
        issues.append("Final answer suspiciously long")

    return issues


def compute_problem_score(problem: MathProblem) -> float:
    """
    Compute a quality score for a problem (0-1).

    Args:
        problem: Problem to score

    Returns:
        Quality score between 0 and 1
    """
    score = 1.0

    # Penalize for validation issues
    issues = validate_problem(problem)
    score -= len(issues) * 0.1

    # Reward for detailed solutions
    num_steps = len(problem.solution.steps)
    if num_steps >= 3:
        score += 0.1
    if num_steps >= 5:
        score += 0.1

    # Reward for justifications
    justifications = sum(1 for step in problem.solution.steps if step.justification)
    score += min(justifications * 0.05, 0.2)

    # Reward for mathematical expressions
    expressions = sum(1 for step in problem.solution.steps if step.mathematical_expression)
    score += min(expressions * 0.05, 0.2)

    # Penalize for very short or very long problems
    problem_len = len(problem.problem_statement)
    if problem_len < 30:
        score -= 0.2
    elif problem_len > 1500:
        score -= 0.1

    return max(0.0, min(1.0, score))