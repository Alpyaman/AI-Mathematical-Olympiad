"""
Fine-Tuner for Phase 2.2: Supervised Fine-Tuning

Extends PreTrainer with evaluation and generation capabilities specific
to mathematical reasoning tasks.
"""

from pathlib import Path
from typing import Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError(
        "PyTorch is required for the FineTuner class. "
        "Please install torch: pip install torch"
    )

from .pretrainer import PreTrainer
from .finetuning_config import FineTuningConfig
from ..evaluation import MathEvaluator, extract_answer
from ..data.aimo_dataset import AIMOFormatter


class FineTuner(PreTrainer):
    """
    Fine-tuning trainer for mathematical reasoning.

    Extends PreTrainer with:
    - Mathematical accuracy evaluation
    - Solution generation
    - Answer extraction
    - Problem-specific logging
    """

    def __init__(
        self,
        model: nn.Module,
        config: FineTuningConfig,
        train_dataloader,
        val_dataloader: Optional[Any] = None,
        test_dataloader: Optional[Any] = None,
        tokenizer=None,
        formatter: Optional[AIMOFormatter] = None,
    ):
        """
        Initialize fine-tuner.

        Args:
            model: Model to fine-tune
            config: Fine-tuning configuration
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            test_dataloader: Optional test data loader
            tokenizer: Tokenizer for generation
            formatter: Problem formatter for inference
        """
        # Initialize parent PreTrainer
        # Convert FineTuningConfig to PreTrainingConfig-compatible dict
        pretrain_config = self._convert_config(config)

        super().__init__(
            model=model,
            config=pretrain_config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )

        # Store fine-tuning specific attributes
        self.ft_config = config
        self.test_dataloader = test_dataloader
        self.tokenizer = tokenizer
        self.formatter = formatter or AIMOFormatter()
        self.evaluator = MathEvaluator()

        # Freeze layers if requested
        if config.freeze_embeddings:
            self._freeze_embeddings()

        if config.freeze_layers > 0:
            self._freeze_initial_layers(config.freeze_layers)

        # Load pretrained checkpoint if provided
        if config.pretrained_checkpoint:
            self.load_pretrained_checkpoint(config.pretrained_checkpoint)

        # Metrics tracking
        self.best_accuracy = 0.0
        self.accuracies = []

    def _convert_config(self, ft_config: FineTuningConfig):
        """Convert FineTuningConfig to PreTrainingConfig format."""
        from .config import PreTrainingConfig

        return PreTrainingConfig(
            model_config_name=ft_config.model_config_name,
            vocab_size=ft_config.vocab_size,
            max_seq_length=ft_config.max_seq_length,
            micro_batch_size=ft_config.micro_batch_size,
            gradient_accumulation_steps=ft_config.gradient_accumulation_steps,
            max_steps=ft_config.max_steps or 10000,
            warmup_steps=ft_config.warmup_steps,
            learning_rate=ft_config.learning_rate,
            min_learning_rate=ft_config.min_learning_rate,
            weight_decay=ft_config.weight_decay,
            max_grad_norm=ft_config.max_grad_norm,
            mixed_precision=ft_config.mixed_precision,
            gradient_checkpointing=ft_config.gradient_checkpointing,
            checkpoint_dir=ft_config.checkpoint_dir,
            save_interval=ft_config.save_interval,
            log_interval=ft_config.log_interval,
            eval_interval=ft_config.eval_interval,
            use_wandb=ft_config.use_wandb,
            wandb_project=ft_config.wandb_project,
            use_tensorboard=ft_config.use_tensorboard,
            num_workers=ft_config.num_workers,
            seed=ft_config.seed,
        )

    def _freeze_embeddings(self):
        """Freeze embedding layer."""
        print("Freezing embedding layer...")

        for name, param in self.raw_model.named_parameters():
            if 'embed' in name.lower():
                param.requires_grad = False
                print(f"  Frozen: {name}")

    def _freeze_initial_layers(self, num_layers: int):
        """Freeze initial transformer layers."""
        print(f"Freezing first {num_layers} layers...")

        frozen_count = 0
        for name, param in self.raw_model.named_parameters():
            if 'layers' in name:
                # Extract layer number
                import re
                match = re.search(r'layers\.(\d+)', name)
                if match:
                    layer_num = int(match.group(1))
                    if layer_num < num_layers:
                        param.requires_grad = False
                        frozen_count += 1

        print(f"  Frozen {frozen_count} parameters in first {num_layers} layers")

    def load_pretrained_checkpoint(self, checkpoint_path: str):
        """Load pretrained model weights."""
        print(f"\nLoading pretrained checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        if 'model' in checkpoint:
            self.raw_model.load_state_dict(checkpoint['model'])
        else:
            # Assume checkpoint is just state dict
            self.raw_model.load_state_dict(checkpoint)

        print("✓ Pretrained weights loaded successfully")

    @torch.no_grad()
    def generate_solution(self, problem: str, max_length: int = 512, temperature: float = 0.8, top_p: float = 0.95) -> str:
        """
        Generate solution for a problem.

        Args:
            problem: Problem statement
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Generated solution text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for generation")

        self.model.eval()

        # Format problem for inference
        prompt = self.formatter.format_for_inference(
            type('Problem', (), {'problem': problem})()
        )

        # Tokenize
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded['input_ids']], dtype=torch.long).to(self.device)

        # Generate
        for _ in range(max_length):
            # Forward pass
            outputs = self.model(input_ids)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature

            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Stop if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Decode
        generated_ids = input_ids[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        self.model.train()
        return generated_text

    @torch.no_grad()
    def evaluate_accuracy(self) -> Dict[str, float]:
        """
        Evaluate mathematical accuracy on validation set.

        Returns:
            Dictionary with accuracy metrics
        """
        if self.val_dataloader is None or self.tokenizer is None:
            return {"accuracy": 0.0}

        self.model.eval()

        all_predictions = []
        all_ground_truths = []
        num_evaluated = 0

        print("\nEvaluating mathematical accuracy...")

        for batch in self.val_dataloader:
            if num_evaluated >= self.ft_config.eval_steps:
                break

            # Get problem and answer from batch
            if "answer" not in batch:
                continue

            batch_size = batch["input_ids"].size(0)

            for i in range(batch_size):
                if num_evaluated >= self.ft_config.eval_steps:
                    break

                # Get input (problem only, no solution)
                input_ids = batch["input_ids"][i]
                attention_mask = batch["attention_mask"][i]

                # Find where solution starts (after <solution> token)
                # For now, use first half as input
                problem_len = attention_mask.sum().item() // 2
                input_ids = input_ids[:problem_len].unsqueeze(0).to(self.device)

                # Generate solution
                outputs = self.model(input_ids)

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

                # Greedy decode for fast evaluation
                predicted_tokens = torch.argmax(logits[0], dim=-1).tolist()
                generated_text = self.tokenizer.decode(predicted_tokens)

                # Extract answer
                predicted_answer = extract_answer(generated_text)

                # Get ground truth
                if isinstance(batch["answer"], list):
                    gt_answer = batch["answer"][i]
                else:
                    gt_answer = batch["answer"][i].item() if hasattr(batch["answer"][i], 'item') else str(batch["answer"][i])

                all_predictions.append(predicted_answer or "")
                all_ground_truths.append(str(gt_answer))

                num_evaluated += 1

        # Compute metrics
        metrics = self.evaluator.evaluate_batch(all_predictions, all_ground_truths)

        self.model.train()

        return {
            "accuracy": metrics["accuracy"],
            "answer_extraction_rate": metrics["answer_extraction_rate"],
            "num_evaluated": num_evaluated,
        }

    def train(self):
        """
        Main training loop with periodic accuracy evaluation.

        Extends parent train() with mathematical accuracy tracking.
        """
        print(f"\n{'='*70}")
        print("STARTING SUPERVISED FINE-TUNING")
        print(f"{'='*70}")
        print(f"Pretrained checkpoint: {self.ft_config.pretrained_checkpoint or 'None (training from scratch)'}")
        print(f"Frozen layers: {self.ft_config.freeze_layers}")
        print(f"Frozen embeddings: {self.ft_config.freeze_embeddings}")
        print(f"Learning rate: {self.ft_config.learning_rate} (lower than pre-training)")
        print(f"{'='*70}\n")

        # Call parent train method but override evaluation
        original_validate = self.validate
        self.validate = self._validate_with_accuracy

        try:
            super().train()
        finally:
            self.validate = original_validate

        # Final evaluation
        if self.test_dataloader is not None:
            print("\n" + "="*70)
            print("FINAL TEST SET EVALUATION")
            print("="*70)
            self._evaluate_test_set()

    def _validate_with_accuracy(self) -> float:
        """
        Validation with mathematical accuracy evaluation.

        Returns:
            Validation loss
        """
        # Standard loss-based validation
        val_loss = self.validate.__wrapped__(self)  # Call original

        # Mathematical accuracy evaluation
        if self.ft_config.generate_samples and self.tokenizer is not None:
            accuracy_metrics = self.evaluate_accuracy()

            accuracy = accuracy_metrics["accuracy"]
            self.accuracies.append(accuracy)

            # Update best accuracy
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                if self.is_main_process:
                    self.save_checkpoint("best_accuracy.pt")

            # Log metrics
            if self.is_main_process:
                metrics = {
                    "val/accuracy": accuracy,
                    "val/answer_extraction_rate": accuracy_metrics["answer_extraction_rate"],
                    "val/best_accuracy": self.best_accuracy,
                }
                self._log_metrics(metrics, self.global_step)

                print(f"  Mathematical Accuracy: {accuracy:.4f} (best: {self.best_accuracy:.4f})")

        return val_loss

    def _evaluate_test_set(self):
        """Evaluate on test set."""
        # TODO: Implement test set evaluation
        print("Test set evaluation not yet implemented")

    def save_checkpoint(self, filename: str):
        """Save checkpoint with additional fine-tuning metrics."""
        checkpoint = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "best_accuracy": self.best_accuracy,
            "tokens_seen": self.tokens_seen,
            "config": self.ft_config.to_dict(),
        }

        if self.scaler is not None:
            checkpoint["scaler"] = self.scaler.state_dict()

        checkpoint_path = Path(self.config.checkpoint_dir) / filename

        if self.is_main_process:
            torch.save(checkpoint, str(checkpoint_path))
            print(f"Checkpoint saved: {checkpoint_path}")