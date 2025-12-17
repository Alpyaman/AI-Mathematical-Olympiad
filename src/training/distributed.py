"""
Distributed Training Utilities

Utilities for setting up distributed training with PyTorch DDP/FSDP.
"""

import os

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError(
        "PyTorch is required for distributed training utilities. "
        "Please install torch: pip install torch"
    )


def setup_distributed(backend: str = "nccl", timeout_minutes: int = 30) -> dict:
    """
    Initialize distributed training.

    This function sets up distributed training using PyTorch's distributed package.
    It detects whether running in a distributed environment and initializes accordingly.

    Args:
        backend: Backend to use ('nccl' for GPU, 'gloo' for CPU)
        timeout_minutes: Timeout for distributed operations

    Returns:
        Dictionary with distributed info: {
            'world_size': int,
            'rank': int,
            'local_rank': int,
            'is_distributed': bool,
            'is_main_process': bool,
        }
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for distributed training")

    # Check if running in distributed mode
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Distributed mode
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=torch.distributed.timedelta(minutes=timeout_minutes),
        )

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"

        is_distributed = True
        is_main_process = rank == 0

    else:
        # Single process mode
        rank = 0
        world_size = 1
        local_rank = 0
        is_distributed = False
        is_main_process = True

        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "device": device,
        "is_distributed": is_distributed,
        "is_main_process": is_main_process,
    }


def cleanup_distributed():
    """Clean up distributed training."""
    if TORCH_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return TORCH_AVAILABLE and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduce dictionary of tensors across all processes.

    Args:
        input_dict: Dictionary of tensors to reduce
        average: Whether to average (True) or sum (False)

    Returns:
        Reduced dictionary
    """
    if not is_distributed():
        return input_dict

    world_size = get_world_size()
    names = sorted(input_dict.keys())
    values = [input_dict[k] for k in names]

    # Stack all values
    values = torch.stack(values)

    # Reduce across all processes
    dist.all_reduce(values)

    if average:
        values /= world_size

    # Convert back to dict
    return {k: v.item() for k, v in zip(names, values)}


def gather_object(obj, dst: int = 0):
    """
    Gather objects from all processes to destination rank.

    Args:
        obj: Object to gather
        dst: Destination rank

    Returns:
        List of objects if rank==dst, None otherwise
    """
    if not is_distributed():
        return [obj]

    world_size = get_world_size()
    rank = get_rank()

    if rank == dst:
        gathered = [None for _ in range(world_size)]
        dist.gather_object(obj, gathered, dst=dst)
        return gathered
    else:
        dist.gather_object(obj, dst=dst)
        return None


def all_gather_object(obj):
    """
    Gather objects from all processes to all processes.

    Args:
        obj: Object to gather

    Returns:
        List of objects from all processes
    """
    if not is_distributed():
        return [obj]

    world_size = get_world_size()
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered


def save_on_main_process(state_dict: dict, path: str):
    """
    Save checkpoint only on main process.

    Args:
        state_dict: State dictionary to save
        path: Path to save to
    """
    if is_main_process():
        torch.save(state_dict, path)

    # Wait for main process to finish saving
    barrier()


def load_checkpoint(path: str, device: str = "cpu"):
    """
    Load checkpoint on all processes.

    Args:
        path: Path to checkpoint
        device: Device to load to

    Returns:
        Loaded state dictionary
    """
    # Wait for all processes to reach this point
    barrier()

    return torch.load(path, map_location=device)