"""
Step 1: Data Preparation for SSL Pretraining
=============================================
- Loads raw CSV
- Cleans & imputes the 5 climate features
- Normalizes using RobustScaler (handles outliers well)
- Segments data into fixed-length sliding windows per crop lifecycle
- Splits into pretrain / finetune / test sets
- Saves processed arrays to data/processed/ and data/splits/
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import pickle

# ── paths ──────────────────────────────────────────────────────────────────
RAW_PATH    = "data/raw/crop_lifecycle_with_recipes_and_channels.csv"
PROC_DIR    = "data/processed"
SPLIT_DIR   = "data/splits"

# ── climate features used for SSL ──────────────────────────────────────────
CLIMATE_COLS = [
    "climate_temperature",
    "climate_relative_humidity",
    "climate_par",
    "climate_co2",
    "climate_vpd",
]

# ── windowing config ───────────────────────────────────────────────────────
WINDOW_SIZE = 24      # 24 hourly steps = 1 day
STRIDE      = 12      # 50% overlap between windows

# ── split ratios ───────────────────────────────────────────────────────────
PRETRAIN_RATIO  = 0.70   # SSL pretraining  (no labels needed)
FINETUNE_RATIO  = 0.15   # downstream fine-tuning (labels used)
TEST_RATIO      = 0.15   # held-out evaluation


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, parse timestamps, sort, and impute climate columns."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Forward-fill then backward-fill short gaps in climate sensors
    df[CLIMATE_COLS] = (
        df[CLIMATE_COLS]
        .ffill(limit=6)   # fill up to 6 consecutive missing hours
        .bfill(limit=6)
    )

    # Drop any remaining rows where climate data is still missing
    before = len(df)
    df = df.dropna(subset=CLIMATE_COLS).reset_index(drop=True)
    after = len(df)
    print(f"[clean] Dropped {before - after} rows with unresolvable NaNs "
          f"({100*(before-after)/before:.2f}%)")

    return df


def normalize(df: pd.DataFrame):
    """Fit RobustScaler on climate cols, return scaled array + scaler."""
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df[CLIMATE_COLS].values)
    return scaled, scaler


def make_windows(
    scaled: np.ndarray,
    labels: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
):
    """
    Slide a window across the time series.

    Returns
    -------
    X : (N, window_size, n_features)   climate windows
    y : (N,)                           yield label for each window
                                       (taken from the last row of the window)
    """
    X, y = [], []
    n = len(scaled)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        X.append(scaled[start:end])
        y.append(labels[end - 1])   # yield at the final timestep of the window
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def split_windows(X, y, pretrain=PRETRAIN_RATIO, finetune=FINETUNE_RATIO):
    """Chronological (non-shuffled) train / finetune / test split."""
    n = len(X)
    i1 = int(n * pretrain)
    i2 = int(n * (pretrain + finetune))

    return (
        X[:i1],  y[:i1],   # pretrain  (SSL — labels not used in training)
        X[i1:i2], y[i1:i2], # finetune
        X[i2:],  y[i2:],   # test
    )


def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # 1. Load & clean
    print("[1/5] Loading and cleaning data …")
    df = load_and_clean(RAW_PATH)
    print(f"      {len(df):,} rows retained | {len(CLIMATE_COLS)} climate features")

    # 2. Normalize
    print("[2/5] Normalizing climate features …")
    scaled, scaler = normalize(df)

    # Save scaler for later inference
    with open(os.path.join(PROC_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # 3. Save cleaned/scaled climate array
    print("[3/5] Saving processed arrays …")
    np.save(os.path.join(PROC_DIR, "climate_scaled.npy"), scaled.astype(np.float32))
    df[["timestamp", "yield", "harvest_date", "zone",
        "age_of_crop", "fert_recipe"]].to_csv(
        os.path.join(PROC_DIR, "metadata.csv"), index=False
    )

    # 4. Build windows
    print("[4/5] Building sliding windows …")
    labels = df["yield"].values
    X, y = make_windows(scaled, labels)
    print(f"      Windows shape: X={X.shape}, y={y.shape}")

    # 5. Split
    print("[5/5] Splitting into pretrain / finetune / test …")
    X_pre, y_pre, X_ft, y_ft, X_test, y_test = split_windows(X, y)

    for name, arr in [
        ("X_pretrain", X_pre), ("y_pretrain", y_pre),
        ("X_finetune", X_ft),  ("y_finetune", y_ft),
        ("X_test",     X_test),("y_test",     y_test),
    ]:
        np.save(os.path.join(SPLIT_DIR, f"{name}.npy"), arr)

    print("\n✅ Data preparation complete.")
    print(f"   Pretrain  : {len(X_pre):>7,} windows")
    print(f"   Finetune  : {len(X_ft):>7,} windows")
    print(f"   Test      : {len(X_test):>7,} windows")
    print(f"\n   Files saved to '{PROC_DIR}/' and '{SPLIT_DIR}/'")


if __name__ == "__main__":
    main()