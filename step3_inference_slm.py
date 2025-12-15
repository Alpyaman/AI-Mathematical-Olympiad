"""
Step 3: Inference with Trained SLM
Use the trained model to solve problems and generate submissions
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from pathlib import Path

class MathSLMInference:
    """Inference engine for trained mathematical SLM"""
    
    def __init__(self, model_path='models/math_slm', device=None):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading model from {model_path}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def format_prompt(self, problem):
        """Format problem as prompt for the model"""
        prompt = f"""### Instruction:
Solve this mathematical problem:

{problem}

### Response:
Let me solve this step by step:

"""
        return prompt
    
    def generate_solution(self, problem, max_length=512, temperature=0.7):
        """Generate solution for a problem"""
        
        prompt = self.format_prompt(problem)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        response = full_response[len(prompt):]
        
        return response
    
    def extract_answer(self, solution_text):
        """Extract numerical answer from solution text"""
        
        # Look for explicit answer markers
        patterns = [
            r'[Ff]inal [Aa]nswer[:\s]+(\d+)',
            r'[Tt]he answer is[:\s]+(\d+)',
            r'[Aa]nswer[:\s]+(\d+)',
            r'=\s*(\d+)\s*$',
            r'^\s*(\d+)\s*$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution_text, re.MULTILINE)
            if match:
                return int(match.group(1))
        
        # If no explicit marker, find the last number mentioned
        numbers = re.findall(r'\b(\d+)\b', solution_text)
        if numbers:
            return int(numbers[-1])
        
        # Default fallback
        return 0
    
    def solve_problem(self, problem, verbose=False):
        """
        Solve a single problem
        
        Returns:
            answer: numerical answer
            solution: full solution text
        """
        solution = self.generate_solution(problem)
        answer = self.extract_answer(solution)
        
        if verbose:
            print(f"\nProblem: {problem[:100]}...")
            print(f"\nSolution:\n{solution}")
            print(f"\nExtracted Answer: {answer}")
        
        return answer, solution
    
    def solve_batch(self, problems, batch_size=4, verbose=False):
        """Solve multiple problems efficiently"""
        results = []
        
        for i in range(0, len(problems), batch_size):
            batch = problems[i:i+batch_size]
            
            if verbose:
                print(f"\nProcessing batch {i//batch_size + 1}/{(len(problems)-1)//batch_size + 1}")
            
            for problem in batch:
                answer, solution = self.solve_problem(problem, verbose=False)
                results.append({
                    'answer': answer,
                    'solution': solution
                })
        
        return results

def generate_submission(model_path='models/math_slm',
                       test_path='data/test.csv',
                       output_path='submission_slm.csv'):
    """Generate submission file using trained SLM"""
    
    print("="*80)
    print("Generating Predictions with Trained SLM")
    print("="*80)
    
    # Load test data
    test_data = pd.read_csv(test_path)
    print(f"\nLoaded {len(test_data)} test problems")
    
    # Initialize inference
    print("\nInitializing inference engine...")
    solver = MathSLMInference(model_path)
    
    # Solve problems
    print("\nSolving problems...")
    problems = test_data['problem'].tolist()
    results = solver.solve_batch(problems, batch_size=1, verbose=True)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_data['id'],
        'answer': [r['answer'] for r in results]
    })
    
    # Display sample results
    print("\n" + "="*80)
    print("Sample Predictions:")
    print("="*80)
    for i in range(min(3, len(submission))):
        print(f"\nProblem {i+1}: {test_data.iloc[i]['id']}")
        print(f"Question: {problems[i]}")
        print(f"Answer: {submission.iloc[i]['answer']}")
        print(f"\nSolution: {results[i]['solution'][:200]}...")
        print("-"*80)
    
    # Save submission
    submission.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved to {output_path}")
    print("="*80)
    
    return submission

def test_model_interactively(model_path='models/math_slm'):
    """Test the model interactively"""
    
    print("="*80)
    print("Interactive Math SLM Testing")
    print("="*80)
    print("\nLoading model...")
    
    solver = MathSLMInference(model_path)
    
    print("\nModel ready! Enter 'quit' to exit.")
    print("="*80)
    
    while True:
        print("\n")
        problem = input("Enter a math problem: ")
        
        if problem.lower() in ['quit', 'exit', 'q']:
            break
        
        if not problem.strip():
            continue
        
        print("\nSolving...")
        answer, solution = solver.solve_problem(problem, verbose=False)
        
        print("\n" + "-"*80)
        print("Solution:")
        print(solution)
        print(f"\nFinal Answer: {answer}")
        print("-"*80)
    
    print("\nGoodbye!")

if __name__ == "__main__":
    import sys
    
    if '--interactive' in sys.argv:
        # Interactive testing mode
        test_model_interactively()
    else:
        # Generate submission
        generate_submission()
