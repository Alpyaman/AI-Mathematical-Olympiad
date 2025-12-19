"""
Kaggle Competition Submission Generator

Creates submission files for AIMO competition.
"""

import torch
import pandas as pd
from tqdm import tqdm

from .answer_extraction import AnswerExtractor


class KaggleSubmissionGenerator:
    """
    Generate Kaggle competition submissions.
    
    Handles:
    - Loading test data
    - Batch inference
    - Answer extraction
    - Submission file formatting
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize submission generator.
        
        Args:
            model: Trained model
            tokenizer: Mathematical tokenizer
            device: Device for inference
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        self.model.eval()
        self.answer_extractor = AnswerExtractor()
    
    def generate_solution(self, problem: str) -> str:
        """Generate solution for a single problem."""
        prompt = f"Problem: {problem}\n\nSolution:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
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
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def generate_submission(
        self,
        test_csv_path: str,
        output_path: str = "submission.csv",
        batch_size: int = 8,
        problem_column: str = "problem",
        id_column: str = "id",
    ) -> pd.DataFrame:
        """
        Generate Kaggle submission file.
        
        Args:
            test_csv_path: Path to test.csv
            output_path: Path to save submission.csv
            batch_size: Batch size for inference
            problem_column: Name of problem column in CSV
            id_column: Name of ID column in CSV
            
        Returns:
            Submission DataFrame
        """
        print("\n🎯 Generating Kaggle submission...")
        print(f"   Test file: {test_csv_path}")
        print(f"   Device: {self.device}")
        
        # Load test data
        test_df = pd.read_csv(test_csv_path)
        print(f"   Loaded {len(test_df)} test problems")
        
        # Prepare results
        predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(test_df), batch_size), desc="Generating solutions"):
            batch = test_df.iloc[i:i + batch_size]
            
            for _, row in batch.iterrows():
                problem_id = row[id_column]
                problem = row[problem_column]
                
                try:
                    # Generate solution
                    solution = self.generate_solution(problem)
                    
                    # Extract answer
                    answer = self.answer_extractor.extract(solution)
                    
                    # If no answer extracted, try backup strategies
                    if answer is None:
                        # Try to find any number in the solution
                        import re
                        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', solution)
                        answer = numbers[-1] if numbers else "0"
                    
                    predictions.append({
                        id_column: problem_id,
                        'answer': answer,
                    })
                    
                except Exception as e:
                    print(f"\n⚠️ Error processing {problem_id}: {e}")
                    predictions.append({
                        id_column: problem_id,
                        'answer': "0",  # Default fallback
                    })
        
        # Create submission DataFrame
        submission_df = pd.DataFrame(predictions)
        
        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        print(f"\n✅ Submission saved to: {output_path}")
        print(f"   Total predictions: {len(submission_df)}")
        
        return submission_df
    
    def generate_with_multiple_attempts(
        self,
        test_csv_path: str,
        output_path: str = "submission.csv",
        num_attempts: int = 3,
        batch_size: int = 8,
    ) -> pd.DataFrame:
        """
        Generate submission with multiple attempts per problem (ensemble).
        
        Takes the most common answer across multiple generation attempts.
        
        Args:
            test_csv_path: Path to test.csv
            output_path: Path to save submission.csv
            num_attempts: Number of generation attempts per problem
            batch_size: Batch size for inference
            
        Returns:
            Submission DataFrame
        """
        print(f"\n🎯 Generating Kaggle submission with {num_attempts} attempts per problem...")
        
        test_df = pd.read_csv(test_csv_path)
        print(f"   Loaded {len(test_df)} test problems")
        
        predictions = []
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating"):
            problem_id = row['id']
            problem = row['problem']
            
            # Generate multiple solutions
            answers = []
            for attempt in range(num_attempts):
                try:
                    solution = self.generate_solution(problem)
                    answer = self.answer_extractor.extract(solution)
                    if answer:
                        answers.append(answer)
                except Exception as e:
                    print(f"\n Error during generation attempt: {e}")
                    pass
            
            # Take most common answer (or first if all different)
            if answers:
                from collections import Counter
                most_common = Counter(answers).most_common(1)[0][0]
                final_answer = most_common
            else:
                final_answer = "0"
            
            predictions.append({
                'id': problem_id,
                'answer': final_answer,
            })
        
        submission_df = pd.DataFrame(predictions)
        submission_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Ensemble submission saved to: {output_path}")
        return submission_df
