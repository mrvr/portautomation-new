"""Plot training curves and confusion matrices."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.callbacks import History

from portautomation.config import FIGURES_DIR
from portautomation.validation import validate_class_names, validate_confusion_matrix, validate_path


REQUIRED_HISTORY_KEYS = ("accuracy", "val_accuracy", "loss", "val_loss")


def _validate_history(history: History) -> History:
    if not isinstance(history, History):
        raise TypeError(f"history must be a History object, got {type(history).__name__}")
    missing = [key for key in REQUIRED_HISTORY_KEYS if key not in history.history]
    if missing:
        raise ValueError(f"history is missing keys: {missing}")
    if not history.history["accuracy"]:
        warnings.warn("history contains no recorded epochs.", UserWarning, stacklevel=2)
    return history


def plot_training_history(
    history: History,
    title: str,
    output_path: Path | str | None = None,
) -> Path:
    history = _validate_history(history)
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")

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
    output_path = validate_path(output_path, "output_path")
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
    cm = validate_confusion_matrix(cm)
    if class_names is not None:
        class_names = validate_class_names(class_names)
        if len(class_names) != cm.shape[0]:
            warnings.warn(
                "class_names length does not match confusion matrix size.",
                UserWarning,
                stacklevel=2,
            )
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")

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
    output_path = validate_path(output_path, "output_path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
