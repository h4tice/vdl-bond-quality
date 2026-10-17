"""
Shared utilities for multi-well pipeline.
"""

import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Reproducibility için tüm RNG'leri sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Kullanılabilir en iyi cihazı döndür (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------- Pretty printing ----------

def print_header(msg: str, width: int = 80) -> None:
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def print_section(msg: str, width: int = 80) -> None:
    print("\n" + "-" * width)
    print(f"  {msg}")
    print("-" * width)


def print_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def print_success(msg: str) -> None:
    print(f"[OK]   {msg}")


def print_warning(msg: str) -> None:
    print(f"[WARN] {msg}")


def print_error(msg: str) -> None:
    print(f"[ERR]  {msg}")


# ---------- File helpers ----------

def find_files(root: Path, pattern: str) -> list:
    """root altında pattern'e uyan tüm dosyaları recursive bul."""
    return sorted(Path(root).rglob(pattern))


def human_readable_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"
