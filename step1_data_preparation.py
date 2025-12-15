"""
Step 1: Data Preparation for SLM Training
Prepares mathematical olympiad problems for training a custom language model
"""

import pandas as pd
import json
import re
from pathlib import Path

class MathDataPreparator:
    """Prepare mathematical olympiad data for SLM training"""
    
    def __init__(self):
        self.training_examples = []
    
    def create_instruction_format(self, problem, answer, include_reasoning=True):
        """
        Create instruction-following format for training
        Format: System prompt + Problem + Step-by-step reasoning + Answer
        """
        
        # System message
        system_prompt = "You are an expert mathematician solving olympiad-level problems. Think step by step and provide clear reasoning."
        
        # User instruction
        user_message = f"Solve this mathematical problem:\n\n{problem}"
        
        # Assistant response (what we want the model to learn)
        if include_reasoning:
            # For training, we'd ideally have step-by-step solutions
            # Since we only have final answers, we'll create structured outputs
            assistant_response = f"""Let me solve this step by step:

Problem Analysis:
{self._analyze_problem(problem)}

Solution Process:
{self._create_solution_sketch(problem, answer)}

Final Answer: {answer}"""
        else:
            assistant_response = f"The answer is: {answer}"
        
        return {
            'system': system_prompt,
            'user': user_message,
            'assistant': assistant_response,
            'problem': problem,
            'answer': answer
        }
    
    def _analyze_problem(self, problem):
        """Create problem analysis (heuristic-based)"""
        analysis = []
        
        problem_lower = problem.lower()
        
        # Identify problem type
        if 'triangle' in problem_lower or 'circle' in problem_lower:
            analysis.append("- This is a geometry problem")
        if 'remainder' in problem_lower or 'divides' in problem_lower:
            analysis.append("- This involves modular arithmetic")
        if 'sequence' in problem_lower or 'sum' in problem_lower:
            analysis.append("- This involves sequences or series")
        if 'function' in problem_lower:
            analysis.append("- This is a functional equation problem")
        if 'permutation' in problem_lower or 'combination' in problem_lower:
            analysis.append("- This is a combinatorics problem")
        
        # Identify key elements
        numbers = re.findall(r'\b\d+\b', problem)
        if numbers:
            analysis.append(f"- Key numbers: {', '.join(numbers[:5])}")
        
        if not analysis:
            analysis.append("- This is an advanced olympiad problem requiring careful analysis")
        
        return '\n'.join(analysis)
    
    def _create_solution_sketch(self, problem, answer):
        """Create a solution sketch (template-based)"""
        # This is a simplified sketch - in practice, you'd want actual solutions
        return f"""1. Identify the key mathematical concepts involved
2. Set up equations or relationships based on the problem constraints
3. Apply appropriate mathematical techniques
4. Simplify and calculate the final result
5. Verify the answer satisfies all conditions

Through systematic calculation, we arrive at: {answer}"""
    
    def load_reference_data(self, path='data/reference.csv'):
        """Load reference problems and create training examples"""
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} reference problems")
        
        for idx, row in df.iterrows():
            example = self.create_instruction_format(
                row['problem'], 
                row['answer'],
                include_reasoning=True
            )
            self.training_examples.append(example)
        
        return self.training_examples
    
    def augment_data(self, augmentation_factor=3):
        """
        Augment training data by:
        1. Paraphrasing questions
        2. Adding different reasoning paths
        3. Creating similar problems with different numbers
        """
        print(f"\nAugmenting data (factor: {augmentation_factor})...")
        original_count = len(self.training_examples)
        augmented = []
        
        for example in self.training_examples:
            augmented.append(example)
            
            # Create variations (simplified augmentation)
            for i in range(augmentation_factor - 1):
                # Variation 1: Different phrasing
                varied_example = example.copy()
                varied_example['user'] = f"Problem {i+1}:\n{example['problem']}"
                augmented.append(varied_example)
        
        self.training_examples = augmented
        print(f"Augmented from {original_count} to {len(self.training_examples)} examples")
        
        return self.training_examples
    
    def create_simple_problems(self, num_examples=100):
        """
        Generate simple arithmetic/algebra problems for foundational training
        This helps the model learn basic mathematical reasoning
        """
        simple_problems = []
        
        import random
        
        for i in range(num_examples):
            # Simple arithmetic
            if i % 4 == 0:
                a, b = random.randint(1, 100), random.randint(1, 100)
                op = random.choice(['+', '-', '*'])
                if op == '+':
                    answer = a + b
                elif op == '-':
                    answer = a - b
                else:
                    answer = a * b
                problem = f"What is ${a}{op}{b}$?"
            
            # Simple equations
            elif i % 4 == 1:
                a = random.randint(1, 20)
                b = random.randint(1, 20)
                answer = a
                problem = f"Solve ${b}+x={a+b}$ for $x$."
            
            # Modular arithmetic
            elif i % 4 == 2:
                a = random.randint(10, 100)
                m = random.randint(3, 20)
                answer = a % m
                problem = f"What is the remainder when ${a}$ is divided by ${m}$?"
            
            # Factorials/combinations
            else:
                n = random.randint(3, 8)
                k = random.randint(1, n-1)
                from math import comb
                answer = comb(n, k)
                problem = f"Calculate $\\binom{{{n}}}{{{k}}}$."
            
            example = self.create_instruction_format(problem, answer, include_reasoning=False)
            simple_problems.append(example)
        
        print(f"Generated {len(simple_problems)} simple training problems")
        return simple_problems
    
    def format_for_training(self, output_format='jsonl'):
        """
        Format data for different training frameworks
        - jsonl: For custom training loops
        - huggingface: For Hugging Face Trainer
        - alpaca: For Alpaca-style instruction tuning
        """
        
        if output_format == 'jsonl':
            # One JSON object per line
            formatted = []
            for example in self.training_examples:
                formatted.append({
                    'instruction': example['user'],
                    'output': example['assistant'],
                    'system': example['system']
                })
            return formatted
        
        elif output_format == 'alpaca':
            # Alpaca instruction format
            formatted = []
            for example in self.training_examples:
                formatted.append({
                    'instruction': example['user'],
                    'input': '',
                    'output': example['assistant']
                })
            return formatted
        
        elif output_format == 'conversational':
            # Conversational format for chat models
            formatted = []
            for example in self.training_examples:
                formatted.append({
                    'messages': [
                        {'role': 'system', 'content': example['system']},
                        {'role': 'user', 'content': example['user']},
                        {'role': 'assistant', 'content': example['assistant']}
                    ]
                })
            return formatted
        
        return self.training_examples
    
    def save_training_data(self, output_path='data/training_data.jsonl', format='jsonl'):
        """Save prepared training data"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        formatted_data = self.format_for_training(output_format=format)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in formatted_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"\nSaved {len(formatted_data)} training examples to {output_path}")
        return output_path
    
    def create_validation_split(self, val_ratio=0.1):
        """Split data into train/validation sets"""
        import random
        random.shuffle(self.training_examples)
        
        split_idx = int(len(self.training_examples) * (1 - val_ratio))
        train_data = self.training_examples[:split_idx]
        val_data = self.training_examples[split_idx:]
        
        print(f"\nTrain: {len(train_data)} examples")
        print(f"Validation: {len(val_data)} examples")
        
        return train_data, val_data

def prepare_data_pipeline():
    """Complete data preparation pipeline"""
    print("="*80)
    print("Mathematical Olympiad - Data Preparation for SLM Training")
    print("="*80)
    
    preparator = MathDataPreparator()
    
    # Step 1: Load reference problems
    print("\nStep 1: Loading reference problems...")
    preparator.load_reference_data('data/reference.csv')
    
    # Step 2: Generate simple problems for foundational learning
    print("\nStep 2: Generating simple training problems...")
    simple_problems = preparator.create_simple_problems(num_examples=200)
    preparator.training_examples.extend(simple_problems)
    
    # Step 3: Augment data
    print("\nStep 3: Augmenting training data...")
    preparator.augment_data(augmentation_factor=2)
    
    # Step 4: Create train/val split
    print("\nStep 4: Creating train/validation split...")
    train_data, val_data = preparator.create_validation_split(val_ratio=0.15)
    
    # Step 5: Save formatted data
    print("\nStep 5: Saving training data...")
    
    # Save in multiple formats
    preparator.training_examples = train_data
    preparator.save_training_data('data/train.jsonl', format='jsonl')
    
    preparator.training_examples = val_data
    preparator.save_training_data('data/val.jsonl', format='jsonl')
    
    # Also save in Alpaca format
    preparator.training_examples = train_data
    preparator.save_training_data('data/train_alpaca.jsonl', format='alpaca')
    
    print("\n" + "="*80)
    print("Data Preparation Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the generated training data")
    print("2. Optionally add more augmentation")
    print("3. Proceed to model training (train_slm.py)")
    print("="*80)

if __name__ == "__main__":
    prepare_data_pipeline()
