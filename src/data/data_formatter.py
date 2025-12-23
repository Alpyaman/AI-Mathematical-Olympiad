"""
Data Formatting for Mathematical Reasoning

This module provides formatters for converting mathematical problems into
training formats optimized for Chain-of-Thought (CoT) reasoning.
"""

from typing import List, Optional, Dict
from .data_schema import MathProblem, ReasoningStep, MathSolution


class ChainOfThoughtFormatter:
    """
    Formats mathematical problems for Chain-of-Thought reasoning.

    The formatter converts structured problems into text sequences that
    encourage step-by-step reasoning in the model.
    """

    def __init__(
        self,
        use_special_tokens: bool = True,
        include_step_numbers: bool = True,
        include_justifications: bool = True,
        add_verification: bool = True,
        add_eos_token: bool = True
    ):
        """
        Initialize the CoT formatter.

        Args:
            use_special_tokens: Use <step>, <proof>, etc. tokens
            include_step_numbers: Include "Step N:" in output
            include_justifications: Include reasoning justifications
            add_verification: Include solution verification when available
        """
        self.use_special_tokens = use_special_tokens
        self.include_step_numbers = include_step_numbers
        self.include_justifications = include_justifications
        self.add_verification = add_verification
        self.add_eos_token = add_eos_token

    def format_problem(self, problem: MathProblem, include_solution: bool = True) -> str:
        """
        Format a complete problem with optional solution.

        Args:
            problem: The mathematical problem to format
            include_solution: Whether to include the solution

        Returns:
            Formatted text string ready for tokenization
        """
        parts = []

        # Problem statement
        if self.use_special_tokens:
            parts.append("<bos>")

        parts.append(f"Problem: {problem.problem_statement}")

        if include_solution:
            parts.append(self.format_solution(problem.solution))

        # Add EOS token based on configuration
        if self.use_special_tokens:
            parts.append("<eos>")
        elif self.add_eos_token:
            # Add EOS token even when special tokens are disabled
            parts.append("<eos>")

        return "\n\n".join(parts)

    def format_solution(self, solution: MathSolution) -> str:
        """
        Format a solution with step-by-step reasoning.

        Args:
            solution: The solution to format

        Returns:
            Formatted solution text
        """
        parts = []

        # Solution header
        if self.use_special_tokens:
            parts.append("<solution>")
        else:
            parts.append("Solution:")

        # Format each reasoning step
        for step in solution.steps:
            step_text = self.format_step(step)
            parts.append(step_text)

        # Final answer
        if self.use_special_tokens:
            parts.append(f"<answer>{solution.final_answer}</answer>")
        else:
            parts.append(f"\nFinal Answer: {solution.final_answer}")

        # Verification if available
        if self.add_verification and solution.verification:
            if self.use_special_tokens:
                parts.append(f"<verification>{solution.verification}</verification>")
            else:
                parts.append(f"\nVerification: {solution.verification}")

        if self.use_special_tokens:
            parts.append("</solution>")

        return "\n\n".join(parts)

    def format_step(self, step: ReasoningStep) -> str:
        """
        Format a single reasoning step.

        Args:
            step: The reasoning step to format

        Returns:
            Formatted step text
        """
        parts = []

        # Step marker
        if self.use_special_tokens:
            parts.append("<step>")

        # Step number and description
        if self.include_step_numbers:
            header = f"Step {step.step_number}: {step.description}"
        else:
            header = step.description

        parts.append(header)

        # Mathematical expression
        if step.mathematical_expression:
            if self.use_special_tokens:
                parts.append(f"<math>{step.mathematical_expression}</math>")
            else:
                parts.append(f"  {step.mathematical_expression}")

        # Justification
        if self.include_justifications and step.justification:
            parts.append(f"  Reasoning: {step.justification}")

        if self.use_special_tokens:
            parts.append("</step>")

        return "\n".join(parts)

    def format_for_training(
        self,
        problem: MathProblem,
        max_length: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Format problem for training with input/output separation.

        Args:
            problem: The problem to format
            max_length: Optional maximum length (for truncation)

        Returns:
            Dictionary with 'input' and 'target' keys
        """
        # Input: problem statement
        input_text = f"Problem: {problem.problem_statement}\n\nSolution:"

        # Target: complete solution
        target_text = self.format_solution(problem.solution)

        # Truncate if needed
        if max_length:
            input_text = input_text[:max_length // 2]
            target_text = target_text[:max_length // 2]

        return {
            "input": input_text,
            "target": target_text,
            "full": f"{input_text}\n\n{target_text}"
        }


class ProblemFormatter:
    """
    Formats problems in various styles for different training objectives.
    """

    @staticmethod
    def format_minimal(problem: MathProblem) -> str:
        """
        Minimal format: just problem and final answer.

        Useful for answer-only training or evaluation.
        """
        return f"{problem.problem_statement}\n\nAnswer: {problem.solution.final_answer}"

    @staticmethod
    def format_compact(problem: MathProblem) -> str:
        """
        Compact format: problem and solution without detailed steps.

        Useful for models that need to generate their own reasoning.
        """
        parts = [problem.problem_statement, "\n\nSolution:"]

        # Combine all steps into a flowing paragraph
        reasoning = " ".join(
            step.description for step in problem.solution.steps
        )
        parts.append(reasoning)

        parts.append(f"\n\nTherefore, the answer is {problem.solution.final_answer}.")

        return " ".join(parts)

    @staticmethod
    def format_detailed(problem: MathProblem) -> str:
        """
        Detailed format: problem with full step-by-step solution.

        Includes all reasoning steps, expressions, and justifications.
        """
        formatter = ChainOfThoughtFormatter(
            use_special_tokens=False,
            include_step_numbers=True,
            include_justifications=True,
            add_verification=True,
            add_eos_token=False
        )
        return formatter.format_problem(problem, include_solution=True)

    @staticmethod
    def format_conversational(problem: MathProblem) -> str:
        """
        Conversational format: problem as Q&A dialogue.

        Useful for instruction-following training.
        """
        parts = [
            "User: Can you solve this problem?",
            f"\n{problem.problem_statement}",
            "\n\nAssistant: I'll solve this step by step.\n"
        ]

        for step in problem.solution.steps:
            parts.append(f"\n{step.description}")
            if step.mathematical_expression:
                parts.append(f"\n{step.mathematical_expression}")

        parts.append(f"\n\nThe final answer is: {problem.solution.final_answer}")

        return "".join(parts)

    @staticmethod
    def format_for_evaluation(problem: MathProblem) -> Dict[str, str]:
        """
        Format problem for evaluation/testing.

        Returns:
            Dictionary with 'problem', 'expected_answer', and metadata
        """
        return {
            "problem": problem.problem_statement,
            "expected_answer": problem.solution.final_answer,
            "answer_type": problem.solution.answer_type,
            "difficulty": problem.difficulty.value,
            "problem_type": problem.problem_type.value,
            "num_steps": len(problem.solution.steps),
            "problem_id": problem.problem_id,
        }


class BatchFormatter:
    """
    Formats batches of problems for efficient training.
    """

    def __init__(self, formatter: ChainOfThoughtFormatter):
        """
        Initialize batch formatter.

        Args:
            formatter: The CoT formatter to use for individual problems
        """
        self.formatter = formatter

    def format_batch(
        self,
        problems: List[MathProblem],
        max_length: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Format a batch of problems.

        Args:
            problems: List of problems to format
            max_length: Optional maximum length per problem

        Returns:
            List of formatted problem dictionaries
        """
        return [
            self.formatter.format_for_training(problem, max_length)
            for problem in problems
        ]

    def format_with_padding(
        self,
        problems: List[MathProblem],
        target_length: int,
        pad_token: str = "<pad>"
    ) -> List[str]:
        """
        Format problems with padding to target length.

        Args:
            problems: List of problems to format
            target_length: Target sequence length
            pad_token: Token to use for padding

        Returns:
            List of padded formatted strings
        """
        formatted = []

        for problem in problems:
            text = self.formatter.format_problem(problem)

            # Pad to target length
            if len(text) < target_length:
                padding = pad_token * (target_length - len(text))
                text += padding

            # Truncate if too long
            elif len(text) > target_length:
                text = text[:target_length]

            formatted.append(text)

        return formatted


def format_dataset_for_finetuning(
    problems: List[MathProblem],
    format_style: str = "cot",
    output_file: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Format entire dataset for fine-tuning.

    Args:
        problems: List of problems to format
        format_style: Format style ('cot', 'minimal', 'conversational')
        output_file: Optional file to save formatted data

    Returns:
        List of formatted examples
    """
    if format_style == "cot":
        formatter = ChainOfThoughtFormatter(use_special_tokens=True)
        formatted = [formatter.format_for_training(p) for p in problems]

    elif format_style == "minimal":
        formatted = [
            {"text": ProblemFormatter.format_minimal(p)}
            for p in problems
        ]

    elif format_style == "conversational":
        formatted = [
            {"text": ProblemFormatter.format_conversational(p)}
            for p in problems
        ]

    else:
        raise ValueError(f"Unknown format style: {format_style}")

    # Save if output file specified
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted, f, indent=2, ensure_ascii=False)

    return formatted