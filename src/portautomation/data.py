"""Load boat images, split them, and build TensorFlow datasets."""

from __future__ import annotations

import logging
import warnings
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
from portautomation.device import optimize_dataset
from portautomation.validation import (
    validate_directory,
    validate_image_dataframe,
    validate_image_scaling,
    validate_image_size,
    validate_positive_int,
    validate_ratio,
    validate_class_names,
    warn_class_imbalance,
    warn_small_split,
)

logger = logging.getLogger(__name__)


def load_images_to_dataframe(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Build a dataframe of image paths and class labels."""
    data_dir = validate_directory(data_dir, "data_dir")
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
    if df.empty:
        logger.warning("No .jpg images found under %s", data_dir)
    logger.info("Loaded %s images from %s", len(df), data_dir)
    return df


def summarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return class counts and percentages."""
    validate_image_dataframe(df)
    counts = df["label"].value_counts().rename("count")
    percents = df["label"].value_counts(normalize=True).mul(100).rename("percent")
    summary = pd.concat([counts, percents], axis=1)
    warn_class_imbalance(counts)
    return summary


def save_images_to_directory(df_split: pd.DataFrame, target_dir: Path | str) -> Path:
    """Copy split images into class subfolders for image_dataset_from_directory."""
    validate_image_dataframe(df_split, "df_split")
    target_dir = validate_path_writable(target_dir)

    for _, row in df_split.iterrows():
        class_dir = target_dir / row["label"]
        class_dir.mkdir(parents=True, exist_ok=True)
        destination = class_dir / row["file_name"]
        source = Path(row["file_path"])
        if not source.exists():
            logger.warning("Skipping missing image: %s", source)
            continue
        if not destination.exists():
            shutil.copy2(source, destination)

    return target_dir


def validate_path_writable(path: Path | str) -> Path:
    from portautomation.validation import validate_path

    resolved = validate_path(path, "target_dir")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def split_dataframe(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    val_size: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle and split a labeled image dataframe."""
    validate_image_dataframe(df)
    test_size = validate_ratio(test_size, "test_size")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an int")
    if val_size is not None:
        val_size = validate_ratio(val_size, "val_size")

    if len(df) < 2:
        raise ValueError("dataframe must contain at least 2 rows to split")

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    warn_small_split(test_df, "test split")

    if val_size is None:
        warn_small_split(train_df, "train split")
        return train_df, test_df

    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=random_state, shuffle=True
    )
    warn_small_split(train_df, "train split")
    warn_small_split(val_df, "validation split")
    return train_df, val_df, test_df


def _dataset_from_directory(
    directory: Path | str,
    *,
    shuffle: bool,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    image_scaling: float = IMAGE_SCALING,
) -> tuple[tf.data.Dataset, list[str]]:
    directory = validate_directory(directory, "directory")
    image_size = validate_image_size(image_size)
    batch_size = validate_positive_int(batch_size, "batch_size")
    image_scaling = validate_image_scaling(image_scaling)

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
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    return optimize_dataset(dataset, for_training=shuffle), class_names


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
    validate_image_dataframe(train_df, "train_df")
    validate_image_dataframe(val_df, "val_df")
    validate_image_dataframe(test_df, "test_df")
    split_root = validate_path_writable(split_root)

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
    if not isinstance(label_counts, pd.Series):
        raise TypeError("label_counts must be a pandas Series")
    class_names = validate_class_names(class_names)

    max_samples = float(label_counts.max()) if not label_counts.empty else 0.0
    if max_samples == 0:
        logger.warning("label_counts is empty; using uniform class weights")
        return {index: 1.0 for index in range(len(class_names))}

    weights: dict[int, float] = {}
    for index, name in enumerate(class_names):
        count = float(label_counts.get(name, 0))
        if count == 0:
            warnings.warn(
                f"Class '{name}' has zero training samples; weight set to 1.0.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning("Class '%s' has zero training samples; weight set to 1.0", name)
            weights[index] = 1.0
        else:
            weights[index] = max_samples / count
    return weights


def seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    np.random.seed(seed)
    tf.random.set_seed(seed)
