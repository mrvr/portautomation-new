"""Plot training curves and confusion matrices."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.callbacks import History

from portautomation.config import FIGURES_DIR


def plot_training_history(
    history: History,
    title: str,
    output_path: Path | str | None = None,
) -> Path:
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(epochs_range, acc, label="Training Accuracy")
    axes[0].plot(epochs_range, val_acc, label="Validation Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Training and Validation Accuracy")

    axes[1].plot(epochs_range, loss, label="Training Loss")
    axes[1].plot(epochs_range, val_loss, label="Validation Loss")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Training and Validation Loss")
    fig.suptitle(title)

    output_path = Path(output_path) if output_path else FIGURES_DIR / f"{_slug(title)}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    output_path: Path | str | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(title)

    output_path = Path(output_path) if output_path else FIGURES_DIR / f"{_slug(title)}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
