"""
Quick Start: Complete SLM Training Pipeline
Run this script to execute all steps automatically
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display output"""
    print("\n" + "="*80)
    print(f"📍 {description}")
    print("="*80)
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  AI Mathematical Olympiad - SLM Training Pipeline                   ║
║  Complete automated training workflow                                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Check if data directory exists
    if not Path('data').exists():
        print("❌ 'data' directory not found!")
        print("Please ensure data/reference.csv and data/test.csv exist")
        return
    
    # Step 1: Data Preparation
    if not run_command(
        f'{sys.executable} step1_data_preparation.py',
        "Step 1: Data Preparation"
    ):
        return
    
    # Check if training data was created
    if not Path('data/train.jsonl').exists():
        print("❌ Training data not created!")
        return
    
    # Step 2: Model Training
    print("\n" + "="*80)
    print("⚠️  IMPORTANT: Model Training")
    print("="*80)
    print("Training requires:")
    print("  - GPU with 8-16GB VRAM (recommended)")
    print("  - Or ~4-24 hours on CPU")
    print("  - ~10GB disk space for model")
    print("\nIf you don't have GPU, consider:")
    print("  - Google Colab (free GPU)")
    print("  - Kaggle Notebooks (free GPU)")
    
    response = input("\nDo you want to start training now? (y/n): ")
    
    if response.lower() == 'y':
        if not run_command(
            f'{sys.executable} step2_train_slm.py',
            "Step 2: Model Training"
        ):
            return
    else:
        print("\n⏭️  Skipping training step")
        print("You can train later with: python step2_train_slm.py")
        return
    
    # Step 3: Generate Predictions
    if Path('models/math_slm').exists():
        if not run_command(
            f'{sys.executable} step3_inference_slm.py',
            "Step 3: Generate Predictions"
        ):
            return
    else:
        print("\n⚠️  Trained model not found!")
        print("Train the model first with: python step2_train_slm.py")
        return
    
    # Summary
    print("\n" + "="*80)
    print("🎉 Pipeline Complete!")
    print("="*80)
    
    if Path('submission_slm.csv').exists():
        print("\n✅ Submission file created: submission_slm.csv")
        print("\nNext steps:")
        print("  1. Review the submission file")
        print("  2. Test interactively: python step3_inference_slm.py --interactive")
        print("  3. Upload to Kaggle competition")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
