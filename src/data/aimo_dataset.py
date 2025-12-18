"""
AIMO Dataset Loader for Fine-Tuning

This module provides data loading utilities for the AIMO Competition dataset, formatting problems for supervised fine-tuning on mathematical reasoning.
"""

import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AIMOProblem:
    """Represents an AIMO competition problem."""
    id: str
    problem: str
    answer: str

    def __post_init__(self):
        """Normalize the problem and answer text."""
        self.problem = " ".join(self.problem.split())
        self.answer = str(self.answer).strip()


class AIMODatasetLoader:
    """
    Loader for AIMO competition dataset.

    The dataset contains mathematical olympiad problems with:
    - id: Problem identifier
    - problem: Problem statement (often with LaTeX formatting)
    - answer: Numerical answer
    """

    def __init__(self, csv_path: str):
        """
        Initialize AIMO Dataset Loader.

        Args:
            csv_path (str): Path to the CSV file containing the dataset.
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    def load(self) -> List[AIMOProblem]:
        """
        Load all the problems from the CSV file.

        Returns:
            List of AIMOProblem objects.
        """
        problems = []

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                problem = AIMOProblem(
                    id=row['id'],
                    problem=row['problem'],
                    answer=row['answer']
                )
                problems.append(problem)
        
        return problems
    
    def load_split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> Tuple[List[AIMOProblem], List[AIMOProblem], List[AIMOProblem]]:
        """
        Load and split the dataset into training, validation, and test sets.
        
        Args:
            train_ratio (float): Proportion of data to use for training.
            val_ratio (float): Proportion of data to use for validation.
            test_ratio (float): Proportion of data to use for testing.
            seed (int): Random seed for shuffling.
        
        Returns:
            Tuple containing lists of AIMOProblem for train, val, and test sets.
        """
        import random

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

        # Load all problems
        problems = self.load()

        # Shuffle
        random.seed(seed)
        random.shuffle(problems)

        # Split
        n = len(problems)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = problems[:train_end]
        val = problems[train_end:val_end]
        test = problems[val_end:]

        return train, val, test
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary containing dataset statistics.
        """
        problems = self.load()

        # Calculate statistics
        problem_lengths = [len(p.problem) for p in problems]
        answer_lengths = [len(p.answer) for p in problems]

        stats = {
            "total_problems": len(problems),
            "avg_problem_length": sum(problem_lengths) / len(problems) if problems else 0,
            "max_problem_length": max(problem_lengths) if problems else 0,
            "min_problem_length": min(problem_lengths) if problems else 0,
            "avg_answer_length": sum(answer_lengths) / len(problems) if problems else 0,
            "answer_types": self._analyze_answer_types(problems),
        }

        return stats
    
    def _analyze_answer_types(self, problems: List[AIMOProblem]) -> Dict[str, int]:
        """Analyze the types of answers in the dataset."""
        types = {
            "integer": 0,
            "float": 0,
            "other": 0
        }

        for problem in problems:
            answer = problem.answer.strip()

            # Try to parse as number
            try:
                if '.' in answer:
                    float(answer)
                    types["float"] += 1
                else:
                    int(answer)
                    types["integer"] += 1
            except ValueError:
                types["other"] += 1
        
        return types

class AIMOFormatter:
    """
    Format AIMO problems for supervised fine-tuning.

    Converts problems into training format with special tokens and prompts for chain-of-thought reasoning.
    """

    def __init__(self, use_special_tokens: bool = True, include_reasoning: bool = True):
        """
        Initialize AIMO Formatter.

        Args:
            use_special_tokens (bool): Whether to use special tokens for problem and answer.
            include_reasoning (bool): Whether to include a prompt for chain-of-thought reasoning.
        """
        self.use_special_tokens = use_special_tokens
        self.include_reasoning = include_reasoning
    
    def format_for_training(self, problem: AIMOProblem) -> str:
        """
        Format a problem for training (include answers).

        Args:
            problem (AIMOProblem): The problem to format.
        
        Returns:
            Formatted string for training.
        """
        if self.use_special_tokens:
            parts = ["<bos>"]

            # Problem
            parts.append("<problem>")
            parts.append(problem.problem)
            parts.append("</problem>")

            # Reasoning prompt
            if self.include_reasoning:
                parts.append("<solution>")
                parts.append("Let's think step by step.")
                parts.append("</solution>")
            
            # Answer
            parts.append("<answer>")
            parts.append(str(problem.answer))
            parts.append("</answer>")

            parts.append("<eos>")

            return "\n".join(parts)
        else:
            # Plain text format
            if self.include_reasoning:
                return (f"Problem: {problem.problem}\n\n",
                        "Solution: Let me solve this step by step.\n\n",
                        f"Answer: {problem.answer}")
            else:
                return (f"Problem: {problem.problem}\n\n",f"Answer: {problem.answer}")
    
    def format_for_inference(self, problem: AIMOProblem) -> str:
        """
        Format a problem for inference (no answer).

        Args:
            problem (AIMOProblem): The problem to format.
        
        Returns:
            Formatted string for inference.
        """
        if self.use_special_tokens:
            parts = ["<bos>"]

            # Problem
            parts.append("<problem>")
            parts.append(problem.problem)
            parts.append("</problem>")

            # Reasoning prompt
            if self.include_reasoning:
                parts.append("<solution>")
                parts.append("Let's think step by step.")
                parts.append("</solution>")
            
            # Answer prompt
            parts.append("<answer>")

            return "\n".join(parts)
        else:
            # Plain text format
            if self.include_reasoning:
                return (f"Problem: {problem.problem}\n\n",
                        "Solution: Let me solve this step by step.\n\n")
            else:
                return (f"Problem: {problem.problem}\n\n")
    
    def extract_answer(self, generated_text: str) -> Optional[str]:
        """
        Extract the answer from generated text.

        Args:
            generated_text (str): The text generated by the model.
        
        Returns:
            Extracted answer as a string, or None if not found.
        """
        if self.use_special_tokens:
            # Look for <answer> ... </answer>
            match = re.search(r"<answer>\s*(.*?)\s*</answer>", generated_text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Fallback: Look for "Answer: " or "The answer is "
        patterns = [
            r'(?:Answer|answer):\s*([^\n]+)',
            r'(?:The answer is|the answer is)\s*([^\n]+)',
            r'(?:Therefore|therefore),?\s*(?:the answer is)?\s*([^\n]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, generated_text)
            if match:
                answer = match.group(1).strip()
                # Clean up common endings
                answer = re.sub(r'\.$', '', answer) # Remove trailing period
                return answer
        
        return None

def create_aimo_jsonl(csv_path: str, output_path: str, formatter: Optional[AIMOFormatter] = None):
    """
    Convert AIMO CSV to JSONL format for training.

    Args:
        csv_path: Path to CSV file
        output_path: Path to output JSONL file
        formatter: Optional custom formatter
    """
    if formatter is None:
        formatter = AIMOFormatter()

    loader = AIMODatasetLoader(csv_path)
    problems = loader.load()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for problem in problems:
            text = formatter.format_for_training(problem)
            entry = {
                "id": problem.id,
                "text": text,
                "answer": problem.answer,
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Created JSONL with {len(problems)} problems: {output_path}")

# Convenience function
def load_aimo_dataset(csv_path: str = "./data/reference.csv") -> List[AIMOProblem]:
    """
    Quick load AIMO dataset.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of AIMO problems
    """
    loader = AIMODatasetLoader(csv_path)
    return loader.load()
            