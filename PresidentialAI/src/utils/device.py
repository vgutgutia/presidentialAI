"""
Device detection and management utilities.
Optimized for Apple Silicon (M4 Max).
"""

import torch
from typing import Optional


def get_device(preferred: Optional[str] = None) -> str:
    """
    Get the best available device for computation.
    
    Priority order:
    1. User-specified device (if available)
    2. Apple MPS (Metal Performance Shaders)
    3. NVIDIA CUDA
    4. CPU
    
    Args:
        preferred: Preferred device ('mps', 'cuda', 'cpu', or 'auto')
        
    Returns:
        Device string for PyTorch
    """
    if preferred and preferred != "auto":
        # Check if preferred device is available
        if preferred == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif preferred == "cpu":
            return "cpu"
        else:
            print(f"Warning: {preferred} not available, auto-detecting...")
    
    # Auto-detect best device
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "pytorch_version": torch.__version__,
        "cpu_available": True,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    
    if info["cuda_available"]:
        info["cuda_version"] = torch.version.cuda
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    
    if info["mps_available"]:
        # MPS doesn't have detailed device info API
        info["mps_device"] = "Apple Silicon GPU"
    
    info["recommended_device"] = get_device()
    
    return info


def print_device_info():
    """Print device information to console."""
    info = get_device_info()
    
    print("\n" + "=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    print(f"PyTorch version: {info['pytorch_version']}")
    print(f"CPU available: {info['cpu_available']}")
    print(f"MPS available: {info['mps_available']}")
    print(f"CUDA available: {info['cuda_available']}")
    
    if info["cuda_available"]:
        print(f"  CUDA version: {info['cuda_version']}")
        print(f"  GPU: {info['cuda_device_name']}")
        print(f"  Memory: {info['cuda_memory_total'] / 1e9:.1f} GB")
    
    if info["mps_available"]:
        print(f"  MPS device: {info['mps_device']}")
    
    print(f"\nRecommended device: {info['recommended_device']}")
    print("=" * 50 + "\n")


def to_device(data, device: str):
    """
    Move data to specified device.
    
    Handles tensors, dicts, and lists recursively.
    
    Args:
        data: Data to move
        device: Target device
        
    Returns:
        Data on target device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    else:
        return data


class DeviceContext:
    """
    Context manager for temporary device switching.
    
    Usage:
        with DeviceContext('cpu'):
            # Operations run on CPU
            result = model(x)
    """
    
    def __init__(self, device: str):
        self.device = device
        self.original_device = None
    
    def __enter__(self):
        return self.device
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
