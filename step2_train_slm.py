"""
Step 2: SLM Training Pipeline
Fine-tune a small language model for mathematical reasoning
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import json
from pathlib import Path
import numpy as np

class MathDataset(Dataset):
    """Custom dataset for mathematical problems"""
    
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as instruction-following
        if 'instruction' in item:
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        elif 'messages' in item:
            # Conversational format
            text = ""
            for msg in item['messages']:
                text += f"{msg['role']}: {msg['content']}\n"
        else:
            text = str(item)
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }

class SLMTrainer:
    """Trainer for Small Language Model"""
    
    def __init__(self, 
                 model_name='microsoft/phi-2',  # or 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
                 output_dir='models/math_slm',
                 use_lora=True):
        """
        Initialize SLM trainer
        
        Args:
            model_name: Base model to fine-tune
            output_dir: Where to save the trained model
            use_lora: Use LoRA for parameter-efficient fine-tuning
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_lora = use_lora
        
        print(f"Initializing SLM Trainer")
        print(f"  Base model: {model_name}")
        print(f"  Output directory: {output_dir}")
        print(f"  LoRA: {use_lora}")
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map='auto' if torch.cuda.is_available() else None
        )
        
        # Configure for training
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Apply LoRA if requested
        if use_lora:
            self._apply_lora()
    
    def _apply_lora(self):
        """Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning"""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            print("\nApplying LoRA for parameter-efficient fine-tuning...")
            
            # LoRA configuration
            lora_config = LoraConfig(
                r=16,  # Rank
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],  # Which layers to adapt
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Prepare model
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
            
        except ImportError:
            print("\n⚠ PEFT library not installed. Install with: pip install peft")
            print("Training without LoRA (more memory intensive)")
            self.use_lora = False
    
    def train(self, 
              train_data_path='data/train.jsonl',
              val_data_path='data/val.jsonl',
              num_epochs=3,
              batch_size=4,
              learning_rate=2e-5,
              save_steps=100):
        """Train the model"""
        
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        
        # Load datasets
        print("\nLoading datasets...")
        train_dataset = MathDataset(train_data_path, self.tokenizer)
        val_dataset = MathDataset(val_data_path, self.tokenizer)
        
        print(f"Train examples: {len(train_dataset)}")
        print(f"Validation examples: {len(val_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=save_steps,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("\nStarting training...")
        trainer.train()
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print("\n" + "="*80)
        print("Training Complete!")
        print(f"Model saved to: {self.output_dir}")
        print("="*80)
        
        return trainer
    
    def evaluate(self, test_data_path='data/val.jsonl'):
        """Evaluate the model"""
        print("\nEvaluating model...")
        
        test_dataset = MathDataset(test_data_path, self.tokenizer)
        
        # Simple evaluation
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for i in range(min(10, len(test_dataset))):
                sample = test_dataset[i]
                inputs = {k: v.unsqueeze(0).to(self.model.device) for k, v in sample.items()}
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                total_loss += loss.item()
        
        avg_loss = total_loss / min(10, len(test_dataset))
        print(f"Average loss: {avg_loss:.4f}")
        
        return avg_loss

def train_slm_pipeline():
    """Complete SLM training pipeline"""
    
    print("="*80)
    print("Mathematical Olympiad - SLM Training Pipeline")
    print("="*80)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    if device == "cpu":
        print("⚠ WARNING: Training on CPU will be very slow")
        print("  Consider using Google Colab or Kaggle notebooks with GPU")
    
    # Initialize trainer
    print("\n" + "="*80)
    print("Step 1: Initialize Trainer")
    print("="*80)
    
    # Choose a small model that can run in Kaggle
    # Options:
    # - 'microsoft/phi-2' (2.7B) - Good balance
    # - 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' (1.1B) - Smaller
    # - 'EleutherAI/pythia-410m' (410M) - Very small
    
    trainer = SLMTrainer(
        model_name='microsoft/phi-2',  # Change this to your preferred model
        output_dir='models/math_slm',
        use_lora=True  # Use LoRA for efficiency
    )
    
    # Train
    print("\n" + "="*80)
    print("Step 2: Training")
    print("="*80)
    
    trainer.train(
        train_data_path='data/train.jsonl',
        val_data_path='data/val.jsonl',
        num_epochs=3,
        batch_size=2,  # Small batch size for memory efficiency
        learning_rate=2e-5,
        save_steps=100
    )
    
    # Evaluate
    print("\n" + "="*80)
    print("Step 3: Evaluation")
    print("="*80)
    
    trainer.evaluate('data/val.jsonl')
    
    print("\n" + "="*80)
    print("Training Pipeline Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Test the model with inference_slm.py")
    print("2. Generate predictions for submission")
    print("3. Package for Kaggle deployment")
    print("="*80)

if __name__ == "__main__":
    train_slm_pipeline()
