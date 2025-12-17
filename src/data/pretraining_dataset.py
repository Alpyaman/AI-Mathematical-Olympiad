"""
Pre-training Dataset for Large-Scale Corpus

This module provides streaming datasets for Phase 2.1: Base Pre-training,
handling massive text corpora that don't fit in memory.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Union

try:
    import torch
    from torch.utils.data import IterableDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class IterableDataset:
        pass

from ..tokenizer.math_tokenizer import MathTokenizer


class TextStreamDataset(IterableDataset):
    """
    Streaming dataset for pre-training on large text corpora.

    Reads data line-by-line from JSONL files to avoid loading entire
    datasets into memory.
    """

    def __init__(
        self,
        data_paths: Union[str, List[str]],
        tokenizer: MathTokenizer,
        max_seq_length: int = 2048,
        buffer_size: int = 10000,
        shuffle_buffer: bool = True,
        seed: int = 42,
    ):
        """
        Initialize streaming dataset.

        Args:
            data_paths: Path(s) to JSONL file(s) containing text data
            tokenizer: Tokenizer for encoding text
            max_seq_length: Maximum sequence length
            buffer_size: Size of shuffle buffer
            shuffle_buffer: Whether to shuffle using buffer
            seed: Random seed
        """
        super().__init__()

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        self.data_paths = [Path(p) for p in data_paths]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.buffer_size = buffer_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        # Verify files exist
        for path in self.data_paths:
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")

    def _read_jsonl(self, file_path: Path) -> Iterator[Dict]:
        """Read JSONL file line by line."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def _shuffle_buffer_iterator(self, iterator: Iterator, buffer_size: int) -> Iterator:
        """Shuffle iterator using a buffer."""
        buffer = []

        for item in iterator:
            buffer.append(item)
            if len(buffer) >= buffer_size:
                random.shuffle(buffer)
                for buffered_item in buffer:
                    yield buffered_item
                buffer = []

        # Yield remaining items
        if buffer:
            random.shuffle(buffer)
            for item in buffer:
                yield item

    def _tokenize_example(self, text: str) -> Dict:
        """Tokenize a text example."""
        # Tokenize with truncation
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding=False,
            truncation=True
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # For causal LM, labels are input_ids shifted by 1
        # We'll handle this in the training loop
        if TORCH_AVAILABLE:
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

    def __iter__(self) -> Iterator[Dict]:
        """Iterate through the dataset."""
        # Set random seed for this worker
        worker_info = torch.utils.data.get_worker_info() if TORCH_AVAILABLE else None
        if worker_info is not None:
            # Split data across workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Each worker gets a subset of files
            worker_data_paths = [
                p for i, p in enumerate(self.data_paths)
                if i % num_workers == worker_id
            ]

            seed = self.seed + worker_id
        else:
            worker_data_paths = self.data_paths
            seed = self.seed

        random.seed(seed)

        # Create iterator over all files
        def data_iterator():
            for data_path in worker_data_paths:
                for item in self._read_jsonl(data_path):
                    # Extract text from various formats
                    text = item.get('text') or item.get('content') or item.get('passage', '')
                    if text:
                        yield text

        # Apply shuffle buffer if requested
        iterator = data_iterator()
        if self.shuffle_buffer:
            iterator = self._shuffle_buffer_iterator(iterator, self.buffer_size)

        # Tokenize and yield examples
        for text in iterator:
            yield self._tokenize_example(text)


class MixedDomainDataset(IterableDataset):
    """
    Mixed dataset that samples from multiple domains (e.g., mathematical and general text).

    This is key for Phase 2.1: combining ArXiv papers with general text.
    """

    def __init__(
        self,
        datasets: Dict[str, TextStreamDataset],
        sampling_weights: Dict[str, float],
        seed: int = 42,
    ):
        """
        Initialize mixed domain dataset.

        Args:
            datasets: Dictionary mapping domain names to datasets
            sampling_weights: Dictionary mapping domain names to sampling probabilities
            seed: Random seed
        """
        super().__init__()

        self.datasets = datasets
        self.sampling_weights = sampling_weights
        self.seed = seed

        # Normalize weights
        total_weight = sum(sampling_weights.values())
        self.sampling_probs = {
            k: v / total_weight for k, v in sampling_weights.items()
        }

        # Verify all datasets are present
        for domain in sampling_weights.keys():
            if domain not in datasets:
                raise ValueError(f"Dataset for domain '{domain}' not provided")

    def __iter__(self) -> Iterator[Dict]:
        """Iterate through mixed datasets."""
        # Create iterators for each domain
        iterators = {
            domain: iter(dataset) for domain, dataset in self.datasets.items()
        }

        # Set random seed
        worker_info = torch.utils.data.get_worker_info() if TORCH_AVAILABLE else None
        if worker_info is not None:
            seed = self.seed + worker_info.id
        else:
            seed = self.seed

        random.seed(seed)

        # Sample from domains according to weights
        domains = list(self.sampling_probs.keys())
        probs = [self.sampling_probs[d] for d in domains]

        while True:
            try:
                # Sample a domain
                domain = random.choices(domains, weights=probs, k=1)[0]

                # Get next item from that domain
                item = next(iterators[domain])

                # Add domain information
                item['domain'] = domain

                yield item

            except StopIteration:
                # If any iterator is exhausted, we're done
                break


class PreTrainingDataCollator:
    """
    Data collator for pre-training with causal language modeling.
    """

    def __init__(self, pad_token_id: int = 0):
        """
        Initialize the collator.

        Args:
            pad_token_id: Token ID to use for padding
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.

        Args:
            batch: List of examples

        Returns:
            Batched tensors
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for batching")

        # Find max length in batch
        max_length = max(example["input_ids"].shape[0] for example in batch)

        # Prepare batched tensors
        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)

        # Fill in the tensors
        for i, example in enumerate(batch):
            seq_len = example["input_ids"].shape[0]
            input_ids[i, :seq_len] = example["input_ids"]
            attention_mask[i, :seq_len] = example["attention_mask"]

        # Create labels for causal LM (predict next token)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Mask padding tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def prepare_pretraining_data(
    data_dir: str,
    sources: List[str],
    tokenizer: MathTokenizer,
    max_seq_length: int = 2048,
    mix_weights: Optional[List[float]] = None,
) -> MixedDomainDataset:
    """
    Prepare mixed domain pre-training dataset.

    Args:
        data_dir: Root directory containing data
        sources: List of data sources (e.g., ['arxiv', 'general'])
        tokenizer: Tokenizer
        max_seq_length: Maximum sequence length
        mix_weights: Optional mixing weights (defaults to uniform)

    Returns:
        Mixed domain dataset
    """
    data_dir = Path(data_dir)

    if mix_weights is None:
        mix_weights = [1.0] * len(sources)

    # Create dataset for each source
    datasets = {}
    sampling_weights = {}

    for source, weight in zip(sources, mix_weights):
        source_dir = data_dir / source

        # Find all JSONL files in source directory
        jsonl_files = list(source_dir.glob("*.jsonl"))

        if not jsonl_files:
            print(f"Warning: No JSONL files found in {source_dir}")
            continue

        # Create streaming dataset
        datasets[source] = TextStreamDataset(
            data_paths=[str(f) for f in jsonl_files],
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        sampling_weights[source] = weight

    if not datasets:
        raise ValueError(f"No valid data sources found in {data_dir}")

    # Create mixed dataset
    return MixedDomainDataset(
        datasets=datasets,
        sampling_weights=sampling_weights,
    )


def create_sample_pretraining_data(output_dir: str = "./data/pretraining"):
    """
    Create sample pre-training data for testing.

    This creates small JSONL files with mathematical and general text.
    """
    output_dir = Path(output_dir)

    # Create directories
    arxiv_dir = output_dir / "arxiv"
    general_dir = output_dir / "general"
    arxiv_dir.mkdir(parents=True, exist_ok=True)
    general_dir.mkdir(parents=True, exist_ok=True)

    # Sample ArXiv-style mathematical texts
    arxiv_samples = [
        "Let X be a topological space and f: X → ℝ a continuous function. Then f is bounded on compact subsets.",
        "Theorem: For any prime p and integer a not divisible by p, we have a^(p-1) ≡ 1 (mod p).",
        "Consider the differential equation dy/dx = f(x,y). A solution exists in a neighborhood of (x₀,y₀) if f is continuous.",
        "The Riemann hypothesis states that all non-trivial zeros of ζ(s) lie on the critical line Re(s) = 1/2.",
        "Let G be a finite group and H a subgroup. Then |G| = |H| · [G:H] by Lagrange's theorem.",
    ]

    # Sample general texts
    general_samples = [
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
        "The quick brown fox jumps over the lazy dog. This is a common pangram used in typing practice.",
        "Climate change refers to long-term shifts in temperatures and weather patterns on Earth.",
        "Programming languages like Python and Java are widely used in software development today.",
        "The history of mathematics spans thousands of years across many cultures and civilizations.",
    ]

    # Write ArXiv samples
    with open(arxiv_dir / "sample.jsonl", 'w', encoding='utf-8') as f:
        for text in arxiv_samples:
            f.write(json.dumps({"text": text}) + "\n")

    # Write general samples
    with open(general_dir / "sample.jsonl", 'w', encoding='utf-8') as f:
        for text in general_samples:
            f.write(json.dumps({"text": text}) + "\n")

    print(f"Sample pre-training data created in {output_dir}")
    print(f"  - ArXiv: {len(arxiv_samples)} samples")
    print(f"  - General: {len(general_samples)} samples")