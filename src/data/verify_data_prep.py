"""
verify_data_prep.py
====================
Quick sanity-checks after running prepare_data.py.
Run from the project root:  python src/data/verify_data_prep.py
"""

import os
import numpy as np
import pickle

SPLIT_DIR = "data/splits"
PROC_DIR  = "data/processed"

SPLITS = ["pretrain", "finetune", "test"]


def check():
    print("=" * 50)
    print("  Data Preparation Verification")
    print("=" * 50)

    # 1. Check all expected files exist
    expected_files = (
        [os.path.join(SPLIT_DIR, f"{p}_{s}.npy")
         for s in SPLITS for p in ("X", "y")]
        + [os.path.join(PROC_DIR, "climate_scaled.npy"),
           os.path.join(PROC_DIR, "scaler.pkl"),
           os.path.join(PROC_DIR, "metadata.csv")]
    )

    all_ok = True
    for f in expected_files:
        exists = os.path.isfile(f)
        status = "✅" if exists else "❌ MISSING"
        print(f"  {status}  {f}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n⚠️  Some files are missing. Run prepare_data.py first.")
        return

    # 2. Shape checks
    print("\n── Split shapes ─────────────────────────────")
    total_windows = 0
    for s in SPLITS:
        X = np.load(os.path.join(SPLIT_DIR, f"X_{s}.npy"))
        y = np.load(os.path.join(SPLIT_DIR, f"y_{s}.npy"))
        print(f"  {s:<10} X={X.shape}  y={y.shape}  "
              f"dtype={X.dtype}")
        assert X.ndim == 3, f"X_{s} should be 3-D (N, T, F)"
        assert y.ndim == 1, f"y_{s} should be 1-D (N,)"
        assert len(X) == len(y), "X and y length mismatch"
        total_windows += len(X)
    print(f"  {'TOTAL':<10} {total_windows:,} windows")

    # 3. Value range after scaling (should be roughly -3 to 3)
    print("\n── Scaled value ranges ──────────────────────")
    X_pre = np.load(os.path.join(SPLIT_DIR, "X_pretrain.npy"))
    print(f"  min={X_pre.min():.3f}  max={X_pre.max():.3f}  "
          f"mean={X_pre.mean():.3f}  std={X_pre.std():.3f}")

    # 4. Scaler check
    print("\n── Scaler ───────────────────────────────────")
    with open(os.path.join(PROC_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    print(f"  Type    : {type(scaler).__name__}")
    print(f"  Center  : {scaler.center_}")
    print(f"  Scale   : {scaler.scale_}")

    print("\n✅ All checks passed. Ready for Step 2 (SSL Architecture).")


if __name__ == "__main__":
    check()