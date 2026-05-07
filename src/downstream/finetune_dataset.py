"""
downstream/finetune_dataset.py
==============================
PyTorch Dataset for yield prediction fine-tuning.

Loads windowed climate arrays (X) and yield labels (y).
Optionally loads and encodes tabular side features:
  - zone        (seedling / north / south)  → one-hot (3 dims)
  - fert_recipe (0–9)                       → one-hot (10 dims)
  - age_of_crop (0–31 days)                 → min-max scaled (1 dim)

Side features are concatenated → 14-dim vector per window.
Pass use_side=False to ignore them (simpler baseline).
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ZONE_MAP   = {"seedling": 0, "north": 1, "south": 2}
N_ZONES    = 3
N_RECIPES  = 10      # fert_recipe values 0–9
N_SIDE     = N_ZONES + N_RECIPES + 1   # 14 total


def _encode_side(meta_slice: pd.DataFrame) -> np.ndarray:
    """
    Encode side features for a slice of metadata rows.
    Returns float32 array of shape (len(meta_slice), N_SIDE).
    """
    n = len(meta_slice)
    out = np.zeros((n, N_SIDE), dtype=np.float32)

    # Zone one-hot
    for i, z in enumerate(meta_slice["zone"].values):
        idx = ZONE_MAP.get(str(z), 0)
        out[i, idx] = 1.0

    # Fert recipe one-hot
    for i, r in enumerate(meta_slice["fert_recipe"].values):
        try:
            ridx = int(r)
            if 0 <= ridx < N_RECIPES:
                out[i, N_ZONES + ridx] = 1.0
        except (ValueError, TypeError):
            pass

    # Age of crop — min-max scaled to [0, 1]
    ages = meta_slice["age_of_crop"].values.astype(np.float32)
    max_age = 31.0
    out[:, N_ZONES + N_RECIPES] = np.clip(ages / max_age, 0.0, 1.0)

    return out


class YieldDataset(Dataset):
    """
    Parameters
    ----------
    X        : (N, T, F)   climate windows
    y        : (N,)        yield labels
    side     : (N, 14) | None   encoded side features
    y_mean   : float       training set mean (for normalization)
    y_std    : float       training set std
    normalize_y : bool     if True, standardize yield labels
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        side: np.ndarray | None = None,
        y_mean: float = 0.0,
        y_std: float = 1.0,
        normalize_y: bool = True,
    ):
        self.X    = torch.from_numpy(X.astype(np.float32))
        self.side = torch.from_numpy(side) if side is not None else None

        y = y.astype(np.float32)
        if normalize_y:
            y = (y - y_mean) / (y_std + 1e-8)
        self.y = torch.from_numpy(y)

        self.y_mean = y_mean
        self.y_std  = y_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.side is not None:
            return self.X[idx], self.side[idx], self.y[idx]
        return self.X[idx], self.y[idx]

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to raw yield units."""
        return y_norm * self.y_std + self.y_mean


def load_finetune_datasets(
    splits_dir: str,
    metadata_path: str,
    use_side: bool = True,
    normalize_y: bool = True,
):
    """
    Load finetune and test splits as YieldDataset objects.

    Computes y_mean / y_std from finetune set only (no test leakage).

    Returns
    -------
    ds_train : YieldDataset
    ds_test  : YieldDataset
    y_mean   : float
    y_std    : float
    n_side   : int   (14 if use_side else 0)
    """
    X_ft   = np.load(os.path.join(splits_dir, "X_finetune.npy"))
    y_ft   = np.load(os.path.join(splits_dir, "y_finetune.npy"))
    X_te   = np.load(os.path.join(splits_dir, "X_test.npy"))
    y_te   = np.load(os.path.join(splits_dir, "y_test.npy"))

    # Yield stats from train only
    # Filter out zero-yield rows (seedling zone before harvest)
    mask_ft = y_ft > 0
    y_mean  = float(y_ft[mask_ft].mean())
    y_std   = float(y_ft[mask_ft].std())

    # Side features
    side_ft = side_te = None
    n_side  = 0

    if use_side:
        meta = pd.read_csv(metadata_path)
        meta["timestamp"] = pd.to_datetime(meta["timestamp"])
        n      = len(meta)
        i1     = int(n * 0.70)
        i2     = int(n * 0.85)

        # Match metadata rows to windows using stride=12, window=24
        # Window i ends at row: 24 + i*12 - 1  (same logic as prepare_data.py)
        WINDOW, STRIDE = 24, 12

        def window_end_indices(n_windows, offset):
            return [offset + WINDOW - 1 + i * STRIDE for i in range(n_windows)]

        ft_ends  = window_end_indices(len(X_ft), i1)
        te_ends  = window_end_indices(len(X_te), i2)

        ft_ends  = [min(e, n - 1) for e in ft_ends]
        te_ends  = [min(e, n - 1) for e in te_ends]

        side_ft  = _encode_side(meta.iloc[ft_ends].reset_index(drop=True))
        side_te  = _encode_side(meta.iloc[te_ends].reset_index(drop=True))
        n_side   = N_SIDE

    ds_train = YieldDataset(X_ft, y_ft, side_ft, y_mean, y_std, normalize_y)
    ds_test  = YieldDataset(X_te, y_te, side_te, y_mean, y_std, normalize_y)

    return ds_train, ds_test, y_mean, y_std, n_side