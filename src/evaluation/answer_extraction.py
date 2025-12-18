"""
Answer Extraction and Validation

Utilities for extracting and comparing answers from model-generated solutions.
"""

import re
from typing import Optional, List, Union


class AnswerExtractor:
    """
    Extract answers from generated mathematical solutions.

    Handles various answer formats including:
    - Boxed answers: \\boxed{42}
    - Tagged answers: <answer>42</answer>
    - Natural language: "The answer is 42", "Therefore, 42"
    - Final line answers
    """

    def __init__(self):
        """Initialize answer extractor with patterns."""
        self.patterns = [
            # LaTeX boxed
            (r'\\boxed\{([^}]+)\}', 'boxed'),
            # XML/HTML tags
            (r'<answer>\s*(.*?)\s*</answer>', 'tagged'),
            (r'\[ANSWER\]\s*(.*?)\s*\[/ANSWER\]', 'tagged'),
            # Natural language
            (r'(?:the\s+)?answer\s+is\s*:?\s*([^\n.]+)', 'natural'),
            (r'therefore\s*,?\s*(?:the\s+answer\s+is)?\s*:?\s*([^\n.]+)', 'natural'),
            (r'thus\s*,?\s*(?:the\s+answer\s+is)?\s*:?\s*([^\n.]+)', 'natural'),
            (r'final\s+answer\s*:?\s*([^\n.]+)', 'natural'),
            # Direct statement
            (r'=\s*([0-9,]+)$', 'direct'),
        ]

    def extract(self, text: str) -> Optional[str]:
        """
        Extract answer from text using multiple strategies.

        Args:
            text: Generated solution text

        Returns:
            Extracted answer or None
        """
        if not text:
            return None

        # Try each pattern
        for pattern, pattern_type in self.patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                return self._clean_answer(answer)

        # Fallback: Try to find last number in text
        numbers = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', text)
        if numbers:
            return numbers[-1]

        return None

    def _clean_answer(self, answer: str) -> str:
        """Clean up extracted answer."""
        # Remove trailing punctuation
        answer = re.sub(r'[.,;!?]+$', '', answer)

        # Remove LaTeX formatting
        answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
        answer = re.sub(r'\\[a-zA-Z]+\s*', '', answer)

        # Remove extra whitespace
        answer = ' '.join(answer.split())

        return answer.strip()

    def extract_all(self, text: str) -> List[str]:
        """
        Extract all potential answers from text.

        Args:
            text: Generated solution text

        Returns:
            List of extracted answers
        """
        answers = []

        for pattern, _ in self.patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                answer = self._clean_answer(match.group(1))
                if answer and answer not in answers:
                    answers.append(answer)

        return answers


def extract_answer(text: str) -> Optional[str]:
    """
    Quick function to extract answer from text.

    Args:
        text: Generated solution text

    Returns:
        Extracted answer or None
    """
    extractor = AnswerExtractor()
    return extractor.extract(text)


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.

    Args:
        answer: Answer string

    Returns:
        Normalized answer
    """
    if not answer:
        return ""

    # Convert to lowercase
    answer = answer.lower()

    # Remove commas from numbers
    answer = answer.replace(',', '')

    # Remove extra whitespace
    answer = ' '.join(answer.split())

    # Remove common prefixes/suffixes
    answer = re.sub(r'^(?:approximately|about)\s+', '', answer)
    answer = re.sub(r'\s+(?:dollars|units|degrees)$', '', answer)

    # Try to parse as number and standardize
    try:
        # Try integer first
        num = int(answer)
        return str(num)
    except ValueError:
        try:
            # Try float
            num = float(answer)
            # Remove trailing zeros
            if num == int(num):
                return str(int(num))
            return f"{num:.10g}"  # Remove trailing zeros
        except ValueError:
            pass

    return answer.strip()


def compare_answers(predicted: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """
    Compare two answers for equality.

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        tolerance: Tolerance for numerical comparison

    Returns:
        True if answers match, False otherwise
    """
    if not predicted or not ground_truth:
        return False

    # Normalize both
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Exact string match
    if pred_norm == gt_norm:
        return True

    # Try numerical comparison
    try:
        pred_num = float(pred_norm)
        gt_num = float(gt_norm)

        # Check if within tolerance
        if abs(pred_num - gt_num) <= tolerance:
            return True

        # Check relative error for large numbers
        if gt_num != 0:
            rel_error = abs(pred_num - gt_num) / abs(gt_num)
            if rel_error <= tolerance:
                return True

    except (ValueError, TypeError):
        pass

    return False


def extract_and_compare(generated_text: str, ground_truth: str, tolerance: float = 1e-6) -> tuple:
    """
    Extract answer from generated text and compare with ground truth.

    Args:
        generated_text: Generated solution text
        ground_truth: Ground truth answer
        tolerance: Tolerance for numerical comparison

    Returns:
        Tuple of (extracted_answer, is_correct)
    """
    extractor = AnswerExtractor()
    extracted = extractor.extract(generated_text)

    if extracted is None:
        return None, False

    is_correct = compare_answers(extracted, ground_truth, tolerance)

    return extracted, is_correct


def validate_answer_format(answer: str) -> bool:
    """
    Validate that answer is in a reasonable format.

    Args:
        answer: Answer string

    Returns:
        True if valid format, False otherwise
    """
    if not answer or not answer.strip():
        return False

    # Check length (answers shouldn't be too long)
    if len(answer) > 100:
        return False

    # Check if it contains actual content
    if not re.search(r'[a-zA-Z0-9]', answer):
        return False

    return True


def parse_numerical_answer(answer: str) -> Optional[Union[int, float]]:
    """
    Parse answer as a number.

    Args:
        answer: Answer string

    Returns:
        Parsed number or None
    """
    if not answer:
        return None

    # Remove commas
    answer = answer.replace(',', '')

    try:
        # Try integer first
        return int(answer)
    except ValueError:
        try:
            # Try float
            return float(answer)
        except ValueError:
            return None