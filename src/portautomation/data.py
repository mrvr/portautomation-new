"""Load boat images, split them, and build TensorFlow datasets."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from portautomation.config import (
    BATCH_SIZE,
    DATA_DIR,
    IMAGE_SCALING,
    IMAGE_SIZE,
)

logger = logging.getLogger(__name__)


def load_images_to_dataframe(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Build a dataframe of image paths and class labels.

    The original notebook stored decoded pixel arrays in the dataframe.
    Paths are used here instead so the full dataset stays in memory-light form.
    """
    data_dir = Path(data_dir)
    rows: list[dict[str, str]] = []

    for label_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for image_path in sorted(label_dir.glob("*.jpg")):
            rows.append(
                {
                    "file_name": image_path.name,
                    "label": label_dir.name,
                    "file_path": str(image_path),
                }
            )

    df = pd.DataFrame(rows, columns=["file_name", "label", "file_path"])
    logger.info("Loaded %s images from %s", len(df), data_dir)
    return df


def summarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return class counts and percentages."""
    counts = df["label"].value_counts().rename("count")
    percents = df["label"].value_counts(normalize=True).mul(100).rename("percent")
    return pd.concat([counts, percents], axis=1)


def save_images_to_directory(df_split: pd.DataFrame, target_dir: Path | str) -> Path:
    """Copy split images into class subfolders for image_dataset_from_directory."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df_split.iterrows():
        class_dir = target_dir / row["label"]
        class_dir.mkdir(parents=True, exist_ok=True)
        destination = class_dir / row["file_name"]
        if not destination.exists():
            shutil.copy2(row["file_path"], destination)

    return target_dir


def split_dataframe(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    val_size: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle and split a labeled image dataframe."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    if val_size is None:
        return train_df, test_df

    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=random_state, shuffle=True
    )
    return train_df, val_df, test_df


def _dataset_from_directory(
    directory: Path | str,
    *,
    shuffle: bool,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    image_scaling: float = IMAGE_SCALING,
) -> tuple[tf.data.Dataset, list[str]]:
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        interpolation="nearest",
        batch_size=batch_size,
        shuffle=shuffle,
    )
    class_names = list(dataset.class_names)
    normalization_layer = tf.keras.layers.Rescaling(image_scaling)
    return dataset.map(lambda x, y: (normalization_layer(x), y)), class_names


def build_split_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_root: Path | str,
    *,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    image_scaling: float = IMAGE_SCALING,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Write splits to disk and return normalized Keras datasets."""
    split_root = Path(split_root)
    train_dir = split_root / "train"
    val_dir = split_root / "val"
    test_dir = split_root / "test"

    save_images_to_directory(train_df, train_dir)
    save_images_to_directory(val_df, val_dir)
    save_images_to_directory(test_df, test_dir)

    train_ds, class_names = _dataset_from_directory(
        train_dir,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size,
        image_scaling=image_scaling,
    )
    val_ds, _ = _dataset_from_directory(
        val_dir,
        shuffle=False,
        image_size=image_size,
        batch_size=batch_size,
        image_scaling=image_scaling,
    )
    test_ds, _ = _dataset_from_directory(
        test_dir,
        shuffle=False,
        image_size=image_size,
        batch_size=batch_size,
        image_scaling=image_scaling,
    )
    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(label_counts: pd.Series, class_names: list[str]) -> dict[int, float]:
    """Give higher weight to underrepresented classes."""
    max_samples = float(label_counts.max())
    weights: dict[int, float] = {}
    for index, name in enumerate(class_names):
        count = float(label_counts.get(name, 0))
        weights[index] = 1.0 if count == 0 else max_samples / count
    return weights


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
