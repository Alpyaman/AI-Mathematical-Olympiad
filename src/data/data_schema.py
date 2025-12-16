"""
Data schema for Mathematical Reasoning Problems

This module defines the structure for olympiad-level mathematical problems and their solutions, optimized for Chain-of-Thought reasoning.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum



class DifficultyLevel(Enum):
    """Difficulty levels for mathematical problems."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    OLYMPIAD = "olympiad"
    IMO = "imo"


class ProblemType(Enum):
    """Types of mathematical problems."""
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    CALCULUS = "calculus"
    PROBABILITY = "probability"
    ANALYSIS = "analysis"
    DISCRETE_MATH = "discrete_math"
    MIXED = "mixed"


@dataclass
class ReasoningStep:
    """
    A single step in a mathematical solution.
    
    Each step represents one logical deduction or calculation in the problem-solving process.
    """
    step_number: int
    description: str
    mathematical_expression: Optional[str] = None
    justification: Optional[str] = None

    def __str__(self) -> str:
        """Format step for display."""
        text = f"Step {self.step_number}: {self.description}"
        if self.mathematical_expression:
            text += f"\nExpression: {self.mathematical_expression}"
        if self.justification:
            text += f"\nBecause: {self.justification}"
        return text


@dataclass
class MathSolution:
    """
    Complete solution to a mathematical problem.
    
    Contains the full reasoning chain from problem statement to final answer.
    """
    steps: List[ReasoningStep]
    final_answer: str
    answer_type: str = "exact"  # e.g., exact, approximate, symbolic, proof
    verification: Optional[str] = None
    alternative_solutions: List[str] = field(default_factory=list)

    def add_step(self, description: str, mathematical_expression: Optional[str] = None, justification: Optional[str] = None) -> None:
        """Add a reasoning step to the solution."""
        step_number = len(self.steps) + 1
        step = ReasoningStep(step_number=step_number, description=description, mathematical_expression=mathematical_expression, justification=justification)
        self.steps.append(step)
    
    def get_step_count(self) -> int:
        """Get the number of reasoning steps."""
        return len(self.steps)
    
    def __str__(self) -> str:
        """Format solution for display."""
        text = "Solution:\n"
        for step in self.steps:
            text += f"\n{step}\n"
        text += f"\nFinal Answer: {self.final_answer}"
        if self.verification:
            text += f"\nVerification: {self.verification}"
        return text


@dataclass
class MathProblem:
    """
    A mathematical problem with metadata and solution.

    This is the main data structure for training examples.
    """
    # Core content
    problem_id: str
    problem_statement: str
    solution: MathSolution

    # Metadata
    difficulty: DifficultyLevel
    problem_type: ProblemType
    topics: List[str] = field(default_factory=list)
    source: str = "unknown"
    year: Optional[int] = None

    # Additional context
    problem_number: Optional[int] = None
    contest_name: Optional[str] = None
    time_limit: Optional[int] = None  # in minutes

    # Processing metadata
    language: str = "en"
    has_diagram: bool = False
    requires_computation: bool = False

    # Quality metrics
    solution_quality_score: Optional[float] = None  # e.g., rating from 0 to 1
    clarity_score: Optional[float] = None  # e.g., rating from 0 to 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert problem to dictionary format."""
        return {
            "problem_id": self.problem_id,
            "problem_statement": self.problem_statement,
            "solution": {
                "steps": [
                    {
                        "step_number": step.step_number,
                        "description": step.description,
                        "mathematical_expression": step.mathematical_expression,
                        "justification": step.justification,
                    }
                    for step in self.solution.steps
                ],
                "final_answer": self.solution.final_answer,
                "answer_type": self.solution.answer_type,
                "verification": self.solution.verification,
            },
            "difficulty": self.difficulty.value,
            "problem_type": self.problem_type.value,
            "topics": self.topics,
            "source": self.source,
            "year": self.year,
            "metadata": {
                "problem_number": self.problem_number,
                "contest_name": self.contest_name,
                "time_limit": self.time_limit,
                "language": self.language,
                "has_diagram": self.has_diagram,
                "requires_computation": self.requires_computation,
            },
            "quality": {
                "solution_quality_score": self.solution_quality_score,
                "clarity_score": self.clarity_score,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MathProblem":
        """Create problem for dictionary format."""
        # Parse solution
        steps = [
            ReasoningStep(
                step_number=step["step_number"],
                description=step["description"],
                mathematical_expression=step.get("mathematical_expression"),
                justification=step.get("justification"),
            )
            for step in data["solution"]["steps"]
        ]

        solution = MathSolution(
            steps=steps,
            final_answer=data["solution"]["final_answer"],
            answer_type=data["solution"].get("answer_type", "exact"),
            verification=data["solution"].get("verification"),
        )

        # Parse metadata
        metadata = data.get("metadata", {})
        quality = data.get("quality", {})

        return cls(
            problem_id=data["problem_id"],
            problem_statement=data["problem_statement"],
            solution=solution,
            difficulty=DifficultyLevel(data["difficulty"]),
            problem_type=ProblemType(data["problem_type"]),
            topics=data.get("topics", []),
            source=data.get("source", "unknown"),
            year=data.get("year"),
            problem_number=metadata.get("problem_number"),
            contest_name=metadata.get("contest_name"),
            time_limit=metadata.get("time_limit"),
            language=metadata.get("language", "en"),
            has_diagram=metadata.get("has_diagram", False),
            requires_computation=metadata.get("requires_computation", False),
            solution_quality_score=quality.get("solution_quality_score"),
            clarity_score=quality.get("clarity_score"),
        )
    
    def __str__(self) -> str:
        """Format problem for display."""
        text = f"Problem {self.problem_id}"
        if self.contest_name:
            text += f" from {self.contest_name}"
            if self.year:
                text += f" ({self.year})"
        text += f"\nDifficulty: {self.difficulty.value.capitalize()}"
        text += f"\nType: {self.problem_type.value.replace('_', ' ').title()}"
        if self.topics:
            text += f"\nTopics: {', '.join(self.topics)}"
        text += f"\n\n{self.problem_statement}\n"
        text += f"\n{self.solution}\n"
        return text


@dataclass
class DatasetStatistics:
    """Statistics about a dataset of mathematical problems."""
    total_problems: int
    difficulty_distribution: Dict[str, int]
    type_distribution: Dict[str, int]
    average_steps: float
    average_problem_length: float
    average_solution_length: float
    sources: List[str]

    def __str__(self) -> str:
        """Format statistics for display."""
        text = "Dataset Statistics:\n"
        text += f"Total Problems: {self.total_problems}\n\n"

        text += "Difficulty Distribution:\n"
        for difficulty, count in sorted(self.difficulty_distribution.items()):
            percentage = (count / self.total_problems) * 100
            text += f"  {difficulty:12s}: {count:4d} ({percentage:5.1f}%)\n"
        text += "\nProblem Type Distribution:\n"
        for ptype, count in sorted(self.type_distribution.items()):
            percentage = (count / self.total_problems) * 100
            text += f"  {ptype:20s}: {count:4d} ({percentage:5.1f}%)\n"
        
        text += f"\nAverage Reasoning Steps: {self.average_steps:.1f}\n"
        text += f"Average Problem Length (chars): {self.average_problem_length:.1f}\n"
        text += f"Average Solution Length (chars): {self.average_solution_length:.1f}\n"
        text += f"Sources: {', '.join(self.sources)}\n"
        return text