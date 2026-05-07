"""
utils/plot_losses.py
====================
Reads the JSON loss history saved by ssl_trainer.py and plots
all four loss curves (total, contrastive, masked, generative).

Usage
-----
    python src/utils/plot_losses.py
    python src/utils/plot_losses.py --path outputs/ssl_loss_history.json
"""

import argparse
import json
import os


def plot(history_path: str = "outputs/ssl_loss_history.json",
         save_path:    str = "outputs/ssl_loss_curves.png"):

    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not installed. Run:  pip install matplotlib")
        return

    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}")
        print("Run ssl_trainer.py first.")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs         = [r["epoch"]         for r in history]
    total          = [r["total"]         for r in history]
    loss_contrast  = [r["loss_contrast"] for r in history]
    loss_masked    = [r["loss_masked"]   for r in history]
    loss_gen       = [r["loss_gen"]      for r in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("SSL Pretraining — Loss Curves", fontsize=14, fontweight="bold")

    pairs = [
        (axes[0, 0], total,         "Total Loss",              "#2563eb"),
        (axes[0, 1], loss_contrast, "Contrastive (NT-Xent)",   "#16a34a"),
        (axes[1, 0], loss_masked,   "Masked Prediction (MSE)", "#dc2626"),
        (axes[1, 1], loss_gen,      "Generative (MSE)",        "#9333ea"),
    ]

    for ax, values, title, color in pairs:
        ax.plot(epochs, values, color=color, linewidth=1.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Annotate final value
        ax.annotate(
            f"final: {values[-1]:.4f}",
            xy=(epochs[-1], values[-1]),
            xytext=(-60, 10), textcoords="offset points",
            fontsize=9, color=color,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Loss curves saved → {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="outputs/ssl_loss_history.json")
    parser.add_argument("--out",  default="outputs/ssl_loss_curves.png")
    args = parser.parse_args()
    plot(args.path, args.out)