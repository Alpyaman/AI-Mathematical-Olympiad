"""
Training script for AI Mathematical Olympiad Problem Solver
This script implements a custom ML model to solve olympiad math problems
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

class MathProblemDataset:
    """Dataset handler for mathematical olympiad problems"""
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.problems = self.data['problem'].tolist()
        self.answers = self.data.get('answer', [None] * len(self.problems)).tolist()
        self.ids = self.data['id'].tolist()
    
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'problem': self.problems[idx],
            'answer': self.answers[idx]
        }
    
    def format_prompt(self, problem, answer=None):
        """Format problem as a training prompt"""
        prompt = f"""Problem: {problem}

Let me solve this step by step:
"""
        if answer is not None:
            prompt += f"""
Solution:
The answer is: {answer}"""
        
        return prompt

class FeatureExtractor:
    """Extract features from mathematical problems"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 3))
        self.scaler = StandardScaler()
    
    def extract_text_features(self, problems):
        """Extract TF-IDF features from problem text"""
        return self.tfidf.fit_transform(problems)
    
    def extract_numerical_features(self, problems):
        """Extract numerical and structural features"""
        features = []
        
        for problem in problems:
            feat = {}
            
            # Basic text features
            feat['length'] = len(problem)
            feat['word_count'] = len(problem.split())
            feat['digit_count'] = sum(c.isdigit() for c in problem)
            feat['uppercase_count'] = sum(c.isupper() for c in problem)
            
            # LaTeX/Math features
            feat['dollar_signs'] = problem.count('$')
            feat['equation_count'] = problem.count('equation')
            feat['frac_count'] = problem.count('\\frac')
            feat['sum_count'] = problem.count('\\sum')
            feat['integral_count'] = problem.count('\\int')
            feat['sqrt_count'] = problem.count('\\sqrt')
            feat['cdot_count'] = problem.count('\\cdot')
            feat['times_count'] = problem.count('\\times')
            
            # Mathematical operations
            feat['plus_count'] = problem.count('+')
            feat['minus_count'] = problem.count('-')
            feat['multiply_count'] = problem.count('*')
            feat['divide_count'] = problem.count('/')
            feat['equals_count'] = problem.count('=')
            
            # Numbers in text
            numbers = re.findall(r'\d+', problem)
            feat['number_count'] = len(numbers)
            if numbers:
                nums = [int(n) for n in numbers]
                feat['max_number'] = max(nums)
                feat['min_number'] = min(nums)
                feat['avg_number'] = np.mean(nums)
                feat['sum_numbers'] = sum(nums)
            else:
                feat['max_number'] = 0
                feat['min_number'] = 0
                feat['avg_number'] = 0
                feat['sum_numbers'] = 0
            
            # Problem type indicators
            feat['has_triangle'] = int('triangle' in problem.lower())
            feat['has_circle'] = int('circle' in problem.lower())
            feat['has_sequence'] = int('sequence' in problem.lower())
            feat['has_function'] = int('function' in problem.lower())
            feat['has_probability'] = int('probability' in problem.lower())
            feat['has_remainder'] = int('remainder' in problem.lower())
            feat['has_divisibility'] = int('divid' in problem.lower())
            feat['has_modulo'] = int('mod' in problem.lower() or '\\pmod' in problem)
            
            # Complexity indicators
            feat['has_summation'] = int('\\sum' in problem)
            feat['has_product'] = int('\\prod' in problem)
            feat['has_limit'] = int('\\lim' in problem)
            feat['has_derivative'] = int('derivative' in problem.lower())
            
            features.append(feat)
        
        return pd.DataFrame(features)
    
    def fit_transform(self, problems):
        """Combine all features"""
        # Text features
        tfidf_features = self.tfidf.fit_transform(problems).toarray()
        
        # Numerical features
        numerical_features = self.extract_numerical_features(problems).values
        numerical_features = self.scaler.fit_transform(numerical_features)
        
        # Combine
        combined_features = np.hstack([tfidf_features, numerical_features])
        return combined_features
    
    def transform(self, problems):
        """Transform new problems"""
        tfidf_features = self.tfidf.transform(problems).toarray()
        numerical_features = self.extract_numerical_features(problems).values
        numerical_features = self.scaler.transform(numerical_features)
        return np.hstack([tfidf_features, numerical_features])

class MathSolverModel:
    """Custom ML model for mathematical problem solving"""
    
    def __init__(self, model_type='neural_network'):
        self.model_type = model_type
        self.feature_extractor = FeatureExtractor()
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'neural_network':
            self.model = MLPRegressor(
                hidden_layer_sizes=(512, 256, 128, 64),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Initialized {model_type} model")
    
    def train(self, problems, answers):
        """Train the model"""
        print("Extracting features from problems...")
        X = self.feature_extractor.fit_transform(problems)
        y = np.array(answers)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        print(f"Feature dimension: {X.shape[1]}")
        
        # Train model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        print("\nTraining Results:")
        print(f"  Train MSE: {train_mse:.2f}, MAE: {train_mae:.2f}")
        print(f"  Val MSE: {val_mse:.2f}, MAE: {val_mae:.2f}")
        
        return {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_mae': train_mae,
            'val_mae': val_mae
        }
    
    def predict(self, problems):
        """Predict answers for problems"""
        X = self.feature_extractor.transform(problems)
        predictions = self.model.predict(X)
        # Round to integers as answers are typically integers
        return np.round(predictions).astype(int)
    
    def save(self, path):
        """Save model to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_extractor': self.feature_extractor,
                'model_type': self.model_type
            }, f)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_extractor = data['feature_extractor']
            self.model_type = data['model_type']
        print(f"Model loaded from {path}")

def train_model(model_type='neural_network'):
    """Train the mathematical problem solver"""
    print("="*80)
    print("AI Mathematical Olympiad - Custom ML Training Pipeline")
    print("="*80)
    
    # Load reference data
    print("\n1. Loading training data...")
    train_dataset = MathProblemDataset('data/reference.csv')
    print(f"Loaded {len(train_dataset)} training examples")
    
    # Initialize model
    print(f"\n2. Initializing {model_type} model...")
    solver = MathSolverModel(model_type=model_type)
    
    # Train the model
    print("\n3. Training model...")
    problems = [item['problem'] for item in train_dataset]
    answers = [item['answer'] for item in train_dataset]
    
    metrics = solver.train(problems, answers)
    
    # Save model
    print("\n4. Saving model...")
    solver.save(f'models/math_solver_{model_type}.pkl')
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    
    return solver, metrics

def generate_predictions(solver, test_path='data/test.csv', output_path='submission.csv'):
    """Generate predictions for test set"""
    print("\n5. Generating predictions for test set...")
    
    test_data = pd.read_csv(test_path)
    problems = test_data['problem'].tolist()
    
    print(f"Predicting {len(problems)} problems...")
    predictions = solver.predict(problems)
    
    # Create submission file
    submission_df = pd.DataFrame({
        'id': test_data['id'],
        'answer': predictions
    })
    
    # Display sample predictions
    print("\nSample predictions:")
    for idx in range(min(5, len(submission_df))):
        print(f"  Problem {idx+1} ({submission_df.iloc[idx]['id']}):")
        print(f"    {test_data.iloc[idx]['problem'][:80]}...")
        print(f"    Predicted answer: {submission_df.iloc[idx]['answer']}")
    
    # Save predictions
    submission_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    return submission_df

def compare_models():
    """Compare different model types"""
    print("="*80)
    print("Model Comparison")
    print("="*80)
    
    model_types = ['random_forest', 'gradient_boosting', 'neural_network']
    results = {}
    
    for model_type in model_types:
        print(f"\n\nTraining {model_type}...")
        print("-"*80)
        solver, metrics = train_model(model_type)
        results[model_type] = metrics
    
    # Print comparison
    print("\n\n" + "="*80)
    print("Model Comparison Results")
    print("="*80)
    print(f"{'Model':<25} {'Train MAE':<15} {'Val MAE':<15} {'Val MSE':<15}")
    print("-"*80)
    for model_type, metrics in results.items():
        print(f"{model_type:<25} {metrics['train_mae']:<15.2f} {metrics['val_mae']:<15.2f} {metrics['val_mse']:<15.2f}")
    
    return results

def main():
    """Main training and inference pipeline"""
    # Train model (you can choose: 'random_forest', 'gradient_boosting', or 'neural_network')
    solver, metrics = train_model(model_type='neural_network')
    
    # Generate predictions
    generate_predictions(solver)
    
    print("\n" + "="*80)
    print("Training and prediction complete!")
    print("="*80)

if __name__ == "__main__":
    main()

