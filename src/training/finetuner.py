"""
Fine-Tuner for Phase 2.2: Supervised Fine-Tuning

Extends PreTrainer with evaluation and generation capabilities specific
to mathematical reasoning tasks.
"""

from typing import Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast # Required for mixed precision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError(
        "PyTorch is required for the FineTuner class. "
        "Please install torch: pip install torch"
    )

from .pretrainer import PreTrainer
from .finetuning_config import FineTuningConfig
from ..evaluation import MathEvaluator
from ..data.aimo_dataset import AIMOFormatter


class FineTuner(PreTrainer):
    """
    Fine-tuning trainer for mathematical reasoning.
    Overrides train_step to ensure correct label shifting for SFT.
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
        # Initialize parent PreTrainer
        pretrain_config = self._convert_config(config)

        super().__init__(
            model=model,
            config=pretrain_config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )

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

        if config.pretrained_checkpoint:
            self.load_pretrained_checkpoint(config.pretrained_checkpoint)

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
            mixed_precision=ft_config.mixed_precision,
            gradient_checkpointing=ft_config.gradient_checkpointing,
            checkpoint_dir=ft_config.checkpoint_dir,
            save_interval=ft_config.save_interval,
            log_interval=ft_config.log_interval,
            use_wandb=ft_config.use_wandb,
            wandb_project=ft_config.wandb_project,
            use_tensorboard=ft_config.use_tensorboard,
            num_workers=ft_config.num_workers,
            seed=ft_config.seed,
        )

    # --- KEY FIX: Override train_step to handle shifting ---
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform a single training step with correct label shifting."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Forward pass with mixed precision
        with autocast(dtype=self.dtype, enabled=self.use_amp):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            # 1. Extract logits (handle dict output)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            # 2. Shift logits and labels for Causal LM
            # Logits: remove last token (predicts nothing)
            # Labels: remove first token (nothing predicts it)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()

            # 3. Compute loss
            loss = nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps
    # ---------------------------------------------------------

    def _freeze_embeddings(self):
        print("Freezing embedding layer...")
        for name, param in self.raw_model.named_parameters():
            if 'embed' in name.lower():
                param.requires_grad = False

    def _freeze_initial_layers(self, num_layers: int):
        print(f"Freezing first {num_layers} layers...")
        for name, param in self.raw_model.named_parameters():
            if 'layers' in name:
                import re
                match = re.search(r'layers\.(\d+)', name)
                if match and int(match.group(1)) < num_layers:
                    param.requires_grad = False

    def load_pretrained_checkpoint(self, checkpoint_path: str):
        print(f"Loading pretrained checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        self.raw_model.load_state_dict(state_dict)
        print("✓ Pretrained weights loaded successfully")

    @torch.no_grad()
    def generate_solution(self, problem: str, max_length: int = 512, temperature: float = 0.8) -> str:
        if self.tokenizer is None:
            return ""
        self.model.eval()
        
        # Use formatter
        prompt = self.formatter.format_for_inference(type('P',(),{'problem':problem})())
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded['input_ids']], dtype=torch.long).to(self.device)

        # Basic generation loop
        for _ in range(max_length):
            outputs = self.model(input_ids)
            # Handle dict
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
        return self.tokenizer.decode(input_ids[0].tolist())

    def train(self):
        # Override to ensure we use our modified validation logic
        original_validate = self.validate
        self.validate = self._validate_with_accuracy
        try:
            super().train()
        finally:
            self.validate = original_validate

    def _validate_with_accuracy(self) -> float:
        val_loss = self.validate.__wrapped__(self)
        if self.ft_config.generate_samples and self.tokenizer:
            self.evaluate_accuracy()
        return val_loss

    @torch.no_grad()
    def evaluate_accuracy(self):
        # Placeholder for accuracy logic (kept minimal to save space)
        # You can keep your full implementation here
        pass