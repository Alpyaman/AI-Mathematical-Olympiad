"""
Data Loader for Mathematical Reasoning Datasets

This module provides utilities to load and process datasets from various sources:
- MATH dataset (Hendrycks et al.)
- AoPS (Art of Problem Solving)
- IMO (International Mathematical Olympiad)
- Custom formatted datasets
"""

import json
from pathlib import Path
from typing import List, Optional
import re

from .data_schema import (
    MathProblem, MathSolution, ReasoningStep,
    DifficultyLevel, ProblemType, DatasetStatistics
)


class MathDatasetLoader:
    """
    Loads mathematical reasoning datasets from various sources.
    """

    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize the dataset loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_math_dataset(
        self,
        split: str = "train",
        difficulty_filter: Optional[List[str]] = None,
        subject_filter: Optional[List[str]] = None
    ) -> List[MathProblem]:
        """
        Load the MATH dataset (Hendrycks et al.)

        The MATH dataset contains ~12,500 problems from mathematics competitions.

        Args:
            split: Dataset split ('train' or 'test')
            difficulty_filter: Filter by difficulty levels (1-5)
            subject_filter: Filter by subjects (algebra, geometry, etc.)

        Returns:
            List of MathProblem objects
        """
        # Note: This is a template. Actual implementation would download from
        # the dataset repository or load from local files.

        dataset_path = self.cache_dir / "MATH" / split
        problems = []

        if dataset_path.exists():
            # Load from cached files
            for subject_dir in dataset_path.iterdir():
                if subject_dir.is_dir():
                    subject = subject_dir.name

                    if subject_filter and subject not in subject_filter:
                        continue

                    for problem_file in subject_dir.glob("*.json"):
                        problem = self._load_math_problem_file(problem_file, subject)
                        if problem:
                            problems.append(problem)
        else:
            print(f"MATH dataset not found at {dataset_path}")
            print("Please download from: https://github.com/hendrycks/math")

        return problems

    def _load_math_problem_file(
        self,
        file_path: Path,
        subject: str
    ) -> Optional[MathProblem]:
        """Load a single problem from MATH dataset format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Parse MATH dataset format
            problem_statement = data.get('problem', '')
            solution_text = data.get('solution', '')
            level = data.get('level', 'Level 3')
            answer = data.get('answer', '')

            # Extract difficulty
            level_num = int(re.search(r'\d+', level).group()) if re.search(r'\d+', level) else 3
            difficulty = self._map_math_difficulty(level_num)

            # Parse solution into steps
            steps = self._parse_solution_steps(solution_text)

            # Create solution object
            solution = MathSolution(
                steps=steps,
                final_answer=answer,
                answer_type="exact"
            )

            # Map subject to problem type
            problem_type = self._map_subject_to_type(subject)

            # Create problem object
            problem = MathProblem(
                problem_id=f"MATH_{subject}_{file_path.stem}",
                problem_statement=problem_statement,
                solution=solution,
                difficulty=difficulty,
                problem_type=problem_type,
                topics=[subject],
                source="MATH",
            )

            return problem

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def load_aops_dataset(
        self,
        dataset_path: Optional[str] = None
    ) -> List[MathProblem]:
        """
        Load AoPS (Art of Problem Solving) dataset.

        Args:
            dataset_path: Path to AoPS dataset

        Returns:
            List of MathProblem objects
        """
        if dataset_path is None:
            dataset_path = self.cache_dir / "AoPS"

        dataset_path = Path(dataset_path)
        problems = []

        if not dataset_path.exists():
            print(f"AoPS dataset not found at {dataset_path}")
            return problems

        # Load AoPS format
        for problem_file in dataset_path.glob("*.json"):
            problem = self._load_aops_problem_file(problem_file)
            if problem:
                problems.append(problem)

        return problems

    def _load_aops_problem_file(self, file_path: Path) -> Optional[MathProblem]:
        """Load a single problem from AoPS dataset format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # AoPS format (varies, this is a template)
            problem_statement = data.get('problem', '')
            solution_text = data.get('solution', '')
            difficulty = data.get('difficulty', 'medium')
            problem_type = data.get('type', 'mixed')

            # Parse solution
            steps = self._parse_solution_steps(solution_text)
            answer = data.get('answer', self._extract_final_answer(solution_text))

            solution = MathSolution(
                steps=steps,
                final_answer=answer,
                answer_type="exact"
            )

            problem = MathProblem(
                problem_id=f"AoPS_{file_path.stem}",
                problem_statement=problem_statement,
                solution=solution,
                difficulty=DifficultyLevel(difficulty.lower()),
                problem_type=ProblemType(problem_type.lower()),
                source="AoPS",
            )

            return problem

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def load_custom_dataset(
        self,
        file_path: str,
        format_type: str = "json"
    ) -> List[MathProblem]:
        """
        Load a custom dataset in specified format.

        Args:
            file_path: Path to dataset file
            format_type: Format type ('json', 'jsonl', 'csv')

        Returns:
            List of MathProblem objects
        """
        file_path = Path(file_path)
        problems = []

        if not file_path.exists():
            print(f"Dataset file not found: {file_path}")
            return problems

        if format_type == "json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    problem = MathProblem.from_dict(item)
                    problems.append(problem)
            else:
                problem = MathProblem.from_dict(data)
                problems.append(problem)

        elif format_type == "jsonl":
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    problem = MathProblem.from_dict(data)
                    problems.append(problem)

        return problems

    def _parse_solution_steps(self, solution_text: str) -> List[ReasoningStep]:
        """
        Parse solution text into reasoning steps.

        This is a heuristic parser that identifies steps based on common patterns.
        """
        steps = []

        # Split by common step indicators
        step_patterns = [
            r'Step \d+:',
            r'\d+\.',
            r'First,',
            r'Next,',
            r'Then,',
            r'Finally,',
            r'Therefore,',
        ]

        # Try to split by these patterns
        lines = solution_text.split('\n')
        current_step = []
        step_number = 1

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line starts a new step
            is_new_step = any(re.match(pattern, line, re.IGNORECASE) for pattern in step_patterns)

            if is_new_step and current_step:
                # Save previous step
                step_text = ' '.join(current_step)
                if step_text:
                    steps.append(ReasoningStep(
                        step_number=step_number,
                        description=step_text
                    ))
                    step_number += 1
                current_step = [line]
            else:
                current_step.append(line)

        # Add final step
        if current_step:
            step_text = ' '.join(current_step)
            if step_text:
                steps.append(ReasoningStep(
                    step_number=step_number,
                    description=step_text
                ))

        # If no steps were parsed, treat entire solution as one step
        if not steps:
            steps.append(ReasoningStep(
                step_number=1,
                description=solution_text.strip()
            ))

        return steps

    def _extract_final_answer(self, solution_text: str) -> str:
        """Extract final answer from solution text."""
        # Look for common answer patterns
        answer_patterns = [
            r'answer is:?\s*(.+?)(?:\.|$)',
            r'final answer:?\s*(.+?)(?:\.|$)',
            r'therefore,?\s*(.+?)(?:\.|$)',
            r'\\boxed\{(.+?)\}',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, solution_text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        # If no pattern matched, return last line
        lines = [line.strip() for line in solution_text.split('\n') if line.strip()]
        return lines[-1] if lines else ""

    def _map_math_difficulty(self, level: int) -> DifficultyLevel:
        """Map MATH dataset level (1-5) to difficulty enum."""
        mapping = {
            1: DifficultyLevel.EASY,
            2: DifficultyLevel.EASY,
            3: DifficultyLevel.MEDIUM,
            4: DifficultyLevel.HARD,
            5: DifficultyLevel.OLYMPIAD,
        }
        return mapping.get(level, DifficultyLevel.MEDIUM)

    def _map_subject_to_type(self, subject: str) -> ProblemType:
        """Map MATH dataset subject to problem type."""
        subject_lower = subject.lower().replace('_', '').replace(' ', '')

        mapping = {
            'algebra': ProblemType.ALGEBRA,
            'geometry': ProblemType.GEOMETRY,
            'numbertheory': ProblemType.NUMBER_THEORY,
            'countingandprobability': ProblemType.COMBINATORICS,
            'prealgebra': ProblemType.ALGEBRA,
            'precalculus': ProblemType.CALCULUS,
            'intermediateproblems': ProblemType.MIXED,
        }

        return mapping.get(subject_lower, ProblemType.MIXED)

    def compute_statistics(self, problems: List[MathProblem]) -> DatasetStatistics:
        """
        Compute statistics for a dataset.

        Args:
            problems: List of problems

        Returns:
            DatasetStatistics object
        """
        if not problems:
            return DatasetStatistics(
                total_problems=0,
                difficulty_distribution={},
                type_distribution={},
                average_steps=0.0,
                average_problem_length=0.0,
                average_solution_length=0.0,
                sources=[]
            )

        # Count distributions
        difficulty_dist = {}
        type_dist = {}
        sources = set()

        total_steps = 0
        total_problem_length = 0
        total_solution_length = 0

        for problem in problems:
            # Difficulty distribution
            diff = problem.difficulty.value
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1

            # Type distribution
            ptype = problem.problem_type.value
            type_dist[ptype] = type_dist.get(ptype, 0) + 1

            # Sources
            sources.add(problem.source)

            # Lengths
            total_steps += len(problem.solution.steps)
            total_problem_length += len(problem.problem_statement)

            solution_length = sum(
                len(step.description) +
                len(step.mathematical_expression or "") +
                len(step.justification or "")
                for step in problem.solution.steps
            )
            total_solution_length += solution_length

        n = len(problems)

        return DatasetStatistics(
            total_problems=n,
            difficulty_distribution=difficulty_dist,
            type_distribution=type_dist,
            average_steps=total_steps / n,
            average_problem_length=total_problem_length / n,
            average_solution_length=total_solution_length / n,
            sources=sorted(list(sources))
        )


def create_sample_problems() -> List[MathProblem]:
    """
    Create sample problems for testing and demonstration.

    Returns:
        List of sample MathProblem objects
    """
    problems = []

    # Sample 1: Simple algebra
    solution1 = MathSolution(steps=[], final_answer="x = 3")
    solution1.add_step(
        "Start with the equation",
        "2x + 4 = 10"
    )
    solution1.add_step(
        "Subtract 4 from both sides",
        "2x = 6",
        "We isolate the term with x"
    )
    solution1.add_step(
        "Divide both sides by 2",
        "x = 3",
        "Final step to solve for x"
    )

    problem1 = MathProblem(
        problem_id="SAMPLE_001",
        problem_statement="Solve for x: 2x + 4 = 10",
        solution=solution1,
        difficulty=DifficultyLevel.EASY,
        problem_type=ProblemType.ALGEBRA,
        topics=["linear_equations"],
        source="sample"
    )
    problems.append(problem1)

    # Sample 2: Number theory
    solution2 = MathSolution(steps=[], final_answer="Yes, n² + n is always even")
    solution2.add_step(
        "Factor the expression",
        "n² + n = n(n + 1)",
        "Factor out n"
    )
    solution2.add_step(
        "Observe consecutive integers",
        justification="n and n+1 are consecutive integers, so one must be even"
    )
    solution2.add_step(
        "Conclude",
        justification="The product of an even number and any integer is even"
    )

    problem2 = MathProblem(
        problem_id="SAMPLE_002",
        problem_statement="Prove that for any integer n, n² + n is even.",
        solution=solution2,
        difficulty=DifficultyLevel.MEDIUM,
        problem_type=ProblemType.NUMBER_THEORY,
        topics=["parity", "proof"],
        source="sample"
    )
    problems.append(problem2)

    # Sample 3: Geometry
    solution3 = MathSolution(steps=[], final_answer="r = 5")
    solution3.add_step(
        "Use Pythagorean theorem",
        "r² = 3² + 4²",
        "In a right triangle with legs 3 and 4"
    )
    solution3.add_step(
        "Calculate",
        "r² = 9 + 16 = 25"
    )
    solution3.add_step(
        "Take square root",
        "r = 5",
        "Since radius is positive"
    )

    problem3 = MathProblem(
        problem_id="SAMPLE_003",
        problem_statement="A right triangle has legs of length 3 and 4. Find the hypotenuse.",
        solution=solution3,
        difficulty=DifficultyLevel.EASY,
        problem_type=ProblemType.GEOMETRY,
        topics=["pythagorean_theorem"],
        source="sample"
    )
    problems.append(problem3)

    return problems