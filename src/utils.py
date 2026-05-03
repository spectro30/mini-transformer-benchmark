import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def plot_training_curves(results, save_path="training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for r in results:
        epochs = range(1, len(r["history"]["train_loss"]) + 1)
        ax1.plot(epochs, r["history"]["train_loss"], label=f"Model {r['model']}")
        ax2.plot(epochs, r["history"]["val_acc"],    label=f"Model {r['model']}")

    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"saved to {save_path}")