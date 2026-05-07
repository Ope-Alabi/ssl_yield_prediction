"""
ClimateDataset
==============
PyTorch Dataset for SSL pretraining and downstream fine-tuning.

Each sample is a (window_size, n_features) climate tensor.
The dataset supports two modes:
  - 'pretrain' : returns (anchor, positive) augmented pair + raw window
  - 'finetune' : returns (window, yield_label)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# ── augmentation helpers ────────────────────────────────────────────────────

def _jitter(x: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """Add Gaussian noise."""
    return x + np.random.normal(0, sigma, x.shape).astype(np.float32)


def _time_mask(x: np.ndarray, max_mask_ratio: float = 0.15) -> np.ndarray:
    """Zero out a contiguous random time segment."""
    x = x.copy()
    T = x.shape[0]
    mask_len = np.random.randint(1, max(2, int(T * max_mask_ratio)))
    start = np.random.randint(0, T - mask_len)
    x[start: start + mask_len] = 0.0
    return x


def _scale(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Random amplitude scaling per feature."""
    scale = np.random.normal(1.0, sigma, (1, x.shape[1])).astype(np.float32)
    return x * scale


def _permute(x: np.ndarray, n_segments: int = 4) -> np.ndarray:
    """Permute equal-sized time segments."""
    x = x.copy()
    T = x.shape[0]
    seg = T // n_segments
    idx = np.random.permutation(n_segments)
    pieces = [x[i * seg: (i + 1) * seg] for i in idx]
    remainder = x[n_segments * seg:]
    return np.concatenate(pieces + [remainder], axis=0)


AUGMENTATIONS = [_jitter, _time_mask, _scale, _permute]


def augment(x: np.ndarray) -> np.ndarray:
    """Apply 2 random augmentations from the pool."""
    chosen = np.random.choice(len(AUGMENTATIONS), size=2, replace=False)
    for i in chosen:
        x = AUGMENTATIONS[i](x)
    return x


# ── Dataset ─────────────────────────────────────────────────────────────────

class ClimateSSLDataset(Dataset):
    """
    Parameters
    ----------
    X : np.ndarray  shape (N, T, F)  — windowed climate sequences
    y : np.ndarray  shape (N,)       — yield labels (used only in finetune mode)
    mode : str      'pretrain' | 'finetune'
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None, mode: str = "pretrain"):
        assert mode in ("pretrain", "finetune"), "mode must be 'pretrain' or 'finetune'"
        self.X    = X.astype(np.float32)
        self.y    = y.astype(np.float32) if y is not None else None
        self.mode = mode

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]          # (T, F)

        if self.mode == "pretrain":
            anchor   = torch.from_numpy(x)
            positive = torch.from_numpy(augment(x.copy()))
            return anchor, positive

        else:  # finetune
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return torch.from_numpy(x), label


def load_split(split_dir: str, split: str, mode: str) -> ClimateSSLDataset:
    """
    Convenience loader.

    split : 'pretrain' | 'finetune' | 'test'
    mode  : 'pretrain' | 'finetune'
    """
    import os
    X = np.load(os.path.join(split_dir, f"X_{split}.npy"))
    y = np.load(os.path.join(split_dir, f"y_{split}.npy"))
    return ClimateSSLDataset(X, y, mode=mode)