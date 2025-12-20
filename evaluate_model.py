"""
Main Evaluation Script

Run comprehensive evaluation on trained models.

Usage:
    python evaluate_model.py --checkpoint checkpoints/best_model.pt --test-data data/test.csv
"""

import argparse
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.model.decoder import MathTransformerDecoder
from src.tokenizer.math_tokenizer import MathTokenizer
from src.config.model_config import MathTransformerConfig
from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from src.evaluation.kaggle_submission import KaggleSubmissionGenerator

def load_model(checkpoint_path: str, config_name: str = "small"):
    """Load trained model from checkpoint."""
    print(f"\n📦 Loading model from {checkpoint_path}...")
    
    # Load checkpoint first
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load config from checkpoint if available, otherwise use config_name
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("   Using config from checkpoint")
    else:
        from src.config.model_config import get_small_config, get_base_config
        if config_name == "small":
            config = get_small_config()
        elif config_name == "base":
            config = get_base_config()
        else:
            config = MathTransformerConfig()
        print(f"   Using {config_name} config")
    
    # Initialize tokenizer
    tokenizer = MathTokenizer()
    
    # Update vocab size if needed
    if config.vocab_size != len(tokenizer):
        print(f"   Warning: Config vocab_size ({config.vocab_size}) != tokenizer vocab_size ({len(tokenizer)})")
        print(f"   Using checkpoint vocab_size: {config.vocab_size}")
    
    # Initialize model with config from checkpoint
    model = MathTransformerDecoder(config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("✅ Model loaded successfully")
    print(f"   Layers: {config.num_hidden_layers}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, config


def evaluate_on_test_set(args):
    """Evaluate model on test set."""
    # Load model
    model, tokenizer, config = load_model(args.checkpoint, args.config)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    # Load test data
    print(f"\n📚 Loading test data from {args.test_data}...")
    import pandas as pd
    df = pd.read_csv(args.test_data)
    
    # Convert to list of dictionaries
    test_problems = []
    for _, row in df.iterrows():
        test_problems.append({
            'problem_id': row['id'] if 'id' in df.columns else f"problem_{_}",
            'problem_statement': row['problem'] if 'problem' in df.columns else row.get('problem_statement', ''),
            'solution': type('obj', (object,), {'final_answer': None})()  # Dummy solution for test data
        })
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        problems=test_problems,
        batch_size=args.batch_size,
        save_predictions=True,
        output_dir=args.output_dir,
    )
    
    # Print summary
    evaluator.print_summary(results)
    
    return results


def generate_kaggle_submission(args):
    """Generate Kaggle submission file."""
    # Load model
    model, tokenizer, config = load_model(args.checkpoint, args.config)
    
    # Initialize submission generator
    generator = KaggleSubmissionGenerator(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    # Generate submission
    if args.ensemble:
        submission_df = generator.generate_with_multiple_attempts(
            test_csv_path=args.test_csv,
            output_path=args.submission_path,
            num_attempts=args.num_attempts,
            batch_size=args.batch_size,
        )
    else:
        submission_df = generator.generate_submission(
            test_csv_path=args.test_csv,
            output_path=args.submission_path,
            batch_size=args.batch_size,
        )
    
    print("\n🎉 Submission ready!")
    print(f"   File: {args.submission_path}")
    print("   Upload to Kaggle: https://www.kaggle.com/competitions/aimo/submit")
    
    return submission_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate mathematical reasoning model")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="small",
                       choices=["small", "base", "large"],
                       help="Model configuration")
    
    # Mode
    parser.add_argument("--mode", type=str, default="evaluate",
                       choices=["evaluate", "kaggle", "both"],
                       help="Evaluation mode")
    
    # Data arguments
    parser.add_argument("--test-data", type=str,
                       help="Path to test dataset (for evaluation mode)")
    parser.add_argument("--test-csv", type=str, default="data/test.csv",
                       help="Path to test.csv (for Kaggle mode)")
    
    # Generation arguments
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Nucleus sampling parameter")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for inference")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--submission-path", type=str, default="submission.csv",
                       help="Path to save Kaggle submission")
    
    # Ensemble arguments
    parser.add_argument("--ensemble", action="store_true",
                       help="Use ensemble (multiple attempts per problem)")
    parser.add_argument("--num-attempts", type=int, default=3,
                       help="Number of attempts for ensemble")
    
    # Device
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on")
    
    args = parser.parse_args()
    
    # Run requested mode
    if args.mode == "evaluate":
        if not args.test_data:
            parser.error("--test-data is required for evaluate mode")
        evaluate_on_test_set(args)
    
    elif args.mode == "kaggle":
        generate_kaggle_submission(args)
    
    elif args.mode == "both":
        if not args.test_data:
            parser.error("--test-data is required for evaluate mode")
        evaluate_on_test_set(args)
        generate_kaggle_submission(args)


if __name__ == "__main__":
    main()
