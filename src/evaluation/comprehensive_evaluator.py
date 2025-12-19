"""
Comprehensive Evaluation System for Mathematical Reasoning

Provides end-to-end evaluation pipeline for trained models.
"""

import torch
import json
import os
from typing import List, Dict, Optional
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from .answer_extraction import AnswerExtractor, compare_answers
from .metrics import MathEvaluator


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation system for mathematical reasoning models.
    
    Features:
    - Batch evaluation with progress tracking
    - Per-difficulty and per-topic metrics
    - Answer extraction and validation
    - Detailed error analysis
    - Report generation (JSON, CSV, HTML)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tolerance: float = 1e-6,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize comprehensive evaluator.
        
        Args:
            model: Trained mathematical reasoning model
            tokenizer: Mathematical tokenizer
            device: Device to run inference on
            tolerance: Numerical comparison tolerance
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.tolerance = tolerance
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        self.model.eval()
        self.answer_extractor = AnswerExtractor()
        self.math_evaluator = MathEvaluator(tolerance=tolerance)
        
    def generate_solution(self, problem: str) -> str:
        """
        Generate solution for a single problem.
        
        Args:
            problem: Problem statement
            
        Returns:
            Generated solution text
        """
        # Format input
        prompt = f"Problem: {problem}\n\nSolution:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def evaluate_dataset(
        self,
        problems: List[Dict],
        batch_size: int = 8,
        save_predictions: bool = True,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate model on a dataset of problems.
        
        Args:
            problems: List of problem dictionaries with 'problem_statement', 'solution', etc.
            batch_size: Batch size for inference
            save_predictions: Whether to save predictions
            output_dir: Directory to save results
            
        Returns:
            Dictionary with comprehensive metrics
        """
        print(f"\n🔬 Evaluating on {len(problems)} problems...")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {batch_size}")
        
        results = []
        predictions = {}
        ground_truths = {}
        difficulties = {}
        topics = {}
        
        # Process in batches
        for i in tqdm(range(0, len(problems), batch_size), desc="Evaluating"):
            batch = problems[i:i + batch_size]
            
            for problem in batch:
                problem_id = problem.get('problem_id', f'problem_{i}')
                problem_stmt = problem['problem_statement']
                ground_truth = problem['solution'].final_answer
                
                # Generate solution
                try:
                    generated = self.generate_solution(problem_stmt)
                except Exception as e:
                    print(f"\n⚠️ Error generating for {problem_id}: {e}")
                    generated = ""
                
                # Extract answer
                extracted = self.answer_extractor.extract(generated)
                
                # Check correctness
                is_correct = False
                if extracted:
                    is_correct = compare_answers(extracted, ground_truth, self.tolerance)
                
                # Store results
                result = {
                    'problem_id': problem_id,
                    'problem': problem_stmt,
                    'generated': generated,
                    'extracted_answer': extracted,
                    'ground_truth': ground_truth,
                    'is_correct': is_correct,
                    'difficulty': problem.get('difficulty', 'unknown'),
                    'topics': problem.get('topics', []),
                    'problem_type': problem.get('problem_type', 'unknown'),
                }
                results.append(result)
                
                # For metric computation
                predictions[problem_id] = generated
                ground_truths[problem_id] = ground_truth
                difficulties[problem_id] = str(problem.get('difficulty', 'unknown'))
                topics[problem_id] = problem.get('topics', [])
        
        # Compute overall metrics
        print("\n📊 Computing metrics...")
        overall_metrics = self._compute_overall_metrics(results)
        
        # Compute per-difficulty metrics
        difficulty_metrics = self._compute_per_difficulty_metrics(results)
        
        # Compute per-topic metrics
        topic_metrics = self._compute_per_topic_metrics(results)
        
        # Error analysis
        error_analysis = self._analyze_errors(results)
        
        # Combine all metrics
        evaluation_results = {
            'overall': overall_metrics,
            'by_difficulty': difficulty_metrics,
            'by_topic': topic_metrics,
            'error_analysis': error_analysis,
            'detailed_results': results,
            'metadata': {
                'total_problems': len(problems),
                'model_config': {
                    'max_length': self.max_length,
                    'temperature': self.temperature,
                    'top_p': self.top_p,
                    'device': self.device,
                },
                'timestamp': datetime.now().isoformat(),
            }
        }
        
        # Save results
        if save_predictions and output_dir:
            self._save_results(evaluation_results, output_dir)
        
        return evaluation_results
    
    def _compute_overall_metrics(self, results: List[Dict]) -> Dict:
        """Compute overall metrics."""
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        answer_found = sum(1 for r in results if r['extracted_answer'] is not None)
        
        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'total': total,
            'correct': correct,
            'incorrect': total - correct,
            'answer_extraction_rate': answer_found / total if total > 0 else 0.0,
            'answer_found': answer_found,
            'answer_not_found': total - answer_found,
        }
    
    def _compute_per_difficulty_metrics(self, results: List[Dict]) -> Dict:
        """Compute metrics per difficulty level."""
        difficulty_groups = {}
        
        for result in results:
            diff = str(result['difficulty'])
            if diff not in difficulty_groups:
                difficulty_groups[diff] = []
            difficulty_groups[diff].append(result)
        
        difficulty_metrics = {}
        for diff, group_results in difficulty_groups.items():
            total = len(group_results)
            correct = sum(1 for r in group_results if r['is_correct'])
            
            difficulty_metrics[diff] = {
                'accuracy': correct / total if total > 0 else 0.0,
                'total': total,
                'correct': correct,
                'incorrect': total - correct,
            }
        
        return difficulty_metrics
    
    def _compute_per_topic_metrics(self, results: List[Dict]) -> Dict:
        """Compute metrics per topic."""
        topic_groups = {}
        
        for result in results:
            topics = result['topics'] if isinstance(result['topics'], list) else [result['topics']]
            for topic in topics:
                if topic not in topic_groups:
                    topic_groups[topic] = []
                topic_groups[topic].append(result)
        
        topic_metrics = {}
        for topic, group_results in topic_groups.items():
            total = len(group_results)
            correct = sum(1 for r in group_results if r['is_correct'])
            
            topic_metrics[topic] = {
                'accuracy': correct / total if total > 0 else 0.0,
                'total': total,
                'correct': correct,
                'incorrect': total - correct,
            }
        
        return topic_metrics
    
    def _analyze_errors(self, results: List[Dict]) -> Dict:
        """Analyze common error patterns."""
        errors = []
        
        for result in results:
            if not result['is_correct']:
                error_type = 'no_answer_extracted' if result['extracted_answer'] is None else 'wrong_answer'
                errors.append({
                    'problem_id': result['problem_id'],
                    'error_type': error_type,
                    'extracted': result['extracted_answer'],
                    'ground_truth': result['ground_truth'],
                    'difficulty': result['difficulty'],
                })
        
        # Count error types
        error_type_counts = {}
        for error in errors:
            et = error['error_type']
            error_type_counts[et] = error_type_counts.get(et, 0) + 1
        
        return {
            'total_errors': len(errors),
            'error_type_counts': error_type_counts,
            'error_samples': errors[:20],  # First 20 errors for inspection
        }
    
    def _save_results(self, results: Dict, output_dir: str):
        """Save evaluation results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON
        json_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved JSON results to: {json_path}")
        
        # Save detailed results as CSV
        df = pd.DataFrame(results['detailed_results'])
        csv_path = os.path.join(output_dir, 'detailed_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved detailed CSV to: {csv_path}")
        
        # Generate and save summary report
        self._generate_summary_report(results, output_dir)
    
    def _generate_summary_report(self, results: Dict, output_dir: str):
        """Generate human-readable summary report."""
        report_path = os.path.join(output_dir, 'evaluation_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MATHEMATICAL REASONING MODEL - EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall metrics
            overall = results['overall']
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy:              {overall['accuracy']:.2%}\n")
            f.write(f"Total Problems:        {overall['total']}\n")
            f.write(f"Correct:               {overall['correct']}\n")
            f.write(f"Incorrect:             {overall['incorrect']}\n")
            f.write(f"Answer Extraction Rate: {overall['answer_extraction_rate']:.2%}\n")
            f.write(f"Answers Found:         {overall['answer_found']}\n")
            f.write(f"Answers Not Found:     {overall['answer_not_found']}\n\n")
            
            # Per-difficulty metrics
            f.write("PERFORMANCE BY DIFFICULTY\n")
            f.write("-" * 40 + "\n")
            for diff, metrics in results['by_difficulty'].items():
                f.write(f"{diff}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})\n")
            f.write("\n")
            
            # Per-topic metrics
            f.write("PERFORMANCE BY TOPIC\n")
            f.write("-" * 40 + "\n")
            for topic, metrics in results['by_topic'].items():
                f.write(f"{topic}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})\n")
            f.write("\n")
            
            # Error analysis
            error_analysis = results['error_analysis']
            f.write("ERROR ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Errors:          {error_analysis['total_errors']}\n")
            f.write("Error Types:\n")
            for error_type, count in error_analysis['error_type_counts'].items():
                f.write(f"  {error_type}: {count}\n")
            f.write("\n")
            
            # Metadata
            f.write("METADATA\n")
            f.write("-" * 40 + "\n")
            f.write(f"Timestamp: {results['metadata']['timestamp']}\n")
            f.write(f"Device: {results['metadata']['model_config']['device']}\n")
            f.write(f"Max Length: {results['metadata']['model_config']['max_length']}\n")
            f.write(f"Temperature: {results['metadata']['model_config']['temperature']}\n")
            f.write(f"Top-p: {results['metadata']['model_config']['top_p']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"💾 Saved summary report to: {report_path}")
    
    def print_summary(self, results: Dict):
        """Print evaluation summary to console."""
        print("\n" + "=" * 80)
        print("📊 EVALUATION SUMMARY")
        print("=" * 80)
        
        overall = results['overall']
        print(f"\n✅ Overall Accuracy: {overall['accuracy']:.2%}")
        print(f"   Correct: {overall['correct']}/{overall['total']}")
        print(f"   Answer Extraction Rate: {overall['answer_extraction_rate']:.2%}")
        
        print("\n📈 Performance by Difficulty:")
        for diff, metrics in sorted(results['by_difficulty'].items()):
            print(f"   {diff:15s}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
        
        print("\n📚 Top Topics by Accuracy:")
        topic_items = list(results['by_topic'].items())
        topic_items.sort(key=lambda x: x[1]['accuracy'], reverse=True)
        for topic, metrics in topic_items[:5]:
            print(f"   {topic:20s}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
        
        print("\n⚠️ Error Analysis:")
        error_analysis = results['error_analysis']
        print(f"   Total Errors: {error_analysis['total_errors']}")
        for error_type, count in error_analysis['error_type_counts'].items():
            print(f"   {error_type}: {count}")
        
        print("\n" + "=" * 80)
