"""
Evaluation Metrics for Mathematical Reasoning

This module provides metrics for evaluating mathematical problem-solving
performance, including accuracy, exact match, and solution quality.
"""

from typing import List, Dict
import warnings

from .answer_extraction import (
    extract_answer,
    compare_answers,
    normalize_answer,
)


class MathEvaluator:
    """
    Evaluator for mathematical reasoning tasks.

    Computes various metrics for model performance on math problems.
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize evaluator.

        Args:
            tolerance: Tolerance for numerical answer comparison
        """
        self.tolerance = tolerance

    def evaluate_single(
        self,
        generated_text: str,
        ground_truth_answer: str,
    ) -> Dict[str, any]:
        """
        Evaluate a single generated solution.

        Args:
            generated_text: Generated solution text
            ground_truth_answer: Ground truth answer

        Returns:
            Dictionary with evaluation results
        """
        # Extract answer
        extracted_answer = extract_answer(generated_text)

        # Check correctness
        is_correct = False
        if extracted_answer is not None:
            is_correct = compare_answers(
                extracted_answer,
                ground_truth_answer,
                self.tolerance
            )

        return {
            "extracted_answer": extracted_answer,
            "ground_truth": ground_truth_answer,
            "is_correct": is_correct,
            "answer_found": extracted_answer is not None,
        }

    def evaluate_batch(self, generated_texts: List[str], ground_truth_answers: List[str]) -> Dict[str, any]:
        """
        Evaluate a batch of generated solutions.

        Args:
            generated_texts: List of generated solution texts
            ground_truth_answers: List of ground truth answers

        Returns:
            Dictionary with aggregated metrics
        """
        if len(generated_texts) != len(ground_truth_answers):
            raise ValueError(
                f"Length mismatch: {len(generated_texts)} generated vs "
                f"{len(ground_truth_answers)} ground truth"
            )

        results = []
        for generated, ground_truth in zip(generated_texts, ground_truth_answers):
            result = self.evaluate_single(generated, ground_truth)
            results.append(result)

        # Compute aggregated metrics
        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        answer_found = sum(1 for r in results if r["answer_found"])

        metrics = {
            "accuracy": correct / total if total > 0 else 0.0,
            "answer_extraction_rate": answer_found / total if total > 0 else 0.0,
            "total_samples": total,
            "correct": correct,
            "answer_found": answer_found,
            "results": results,
        }

        return metrics


def compute_accuracy(generated_texts: List[str], ground_truth_answers: List[str], tolerance: float = 1e-6) -> float:
    """
    Compute accuracy on a set of problems.

    Args:
        generated_texts: List of generated solution texts
        ground_truth_answers: List of ground truth answers
        tolerance: Tolerance for numerical comparison

    Returns:
        Accuracy (fraction correct)
    """
    evaluator = MathEvaluator(tolerance=tolerance)
    metrics = evaluator.evaluate_batch(generated_texts, ground_truth_answers)
    return metrics["accuracy"]


def compute_exact_match(generated_texts: List[str], ground_truth_answers: List[str]) -> float:
    """
    Compute exact match accuracy (no tolerance).

    Args:
        generated_texts: List of generated solution texts
        ground_truth_answers: List of ground truth answers

    Returns:
        Exact match rate
    """
    if len(generated_texts) != len(ground_truth_answers):
        raise ValueError("Length mismatch")

    matches = 0
    for generated, ground_truth in zip(generated_texts, ground_truth_answers):
        extracted = extract_answer(generated)
        if extracted is not None:
            extracted_norm = normalize_answer(extracted)
            gt_norm = normalize_answer(ground_truth)
            if extracted_norm == gt_norm:
                matches += 1

    return matches / len(generated_texts) if generated_texts else 0.0


def evaluate_predictions(predictions: Dict[str, str], ground_truths: Dict[str, str], tolerance: float = 1e-6) -> Dict[str, any]:
    """
    Evaluate predictions for a set of problems.

    Args:
        predictions: Dictionary mapping problem_id -> generated_text
        ground_truths: Dictionary mapping problem_id -> ground_truth_answer
        tolerance: Tolerance for numerical comparison

    Returns:
        Dictionary with detailed evaluation metrics
    """
    # Find common problem IDs
    common_ids = set(predictions.keys()) & set(ground_truths.keys())

    if not common_ids:
        warnings.warn("No common problem IDs found")
        return {
            "accuracy": 0.0,
            "total_samples": 0,
            "correct": 0,
        }

    # Extract texts and answers for common IDs
    generated_texts = [predictions[pid] for pid in common_ids]
    ground_truth_answers = [ground_truths[pid] for pid in common_ids]

    # Evaluate
    evaluator = MathEvaluator(tolerance=tolerance)
    metrics = evaluator.evaluate_batch(generated_texts, ground_truth_answers)

    # Add problem-level details
    problem_results = {}
    for pid, result in zip(common_ids, metrics["results"]):
        problem_results[pid] = result

    metrics["problem_results"] = problem_results
    metrics["problem_ids"] = list(common_ids)

    return metrics


def compute_per_difficulty_metrics(predictions: Dict[str, str], ground_truths: Dict[str, str], difficulties: Dict[str, str], tolerance: float = 1e-6) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics broken down by difficulty level.

    Args:
        predictions: Dictionary mapping problem_id -> generated_text
        ground_truths: Dictionary mapping problem_id -> ground_truth_answer
        difficulties: Dictionary mapping problem_id -> difficulty_level
        tolerance: Tolerance for numerical comparison

    Returns:
        Dictionary mapping difficulty_level -> metrics
    """
    # Group by difficulty
    difficulty_groups = {}
    for pid in predictions.keys():
        if pid in ground_truths and pid in difficulties:
            diff = difficulties[pid]
            if diff not in difficulty_groups:
                difficulty_groups[diff] = {"predictions": {}, "ground_truths": {}}

            difficulty_groups[diff]["predictions"][pid] = predictions[pid]
            difficulty_groups[diff]["ground_truths"][pid] = ground_truths[pid]

    # Evaluate each difficulty level
    results = {}
    for diff, data in difficulty_groups.items():
        metrics = evaluate_predictions(
            data["predictions"],
            data["ground_truths"],
            tolerance
        )

        results[diff] = {
            "accuracy": metrics["accuracy"],
            "total_samples": metrics["total_samples"],
            "correct": metrics["correct"],
        }

    return results