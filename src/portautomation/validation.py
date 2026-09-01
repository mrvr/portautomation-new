"""Input validation helpers with warnings for soft issues."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

IMAGE_DF_COLUMNS = ("file_name", "label", "file_path")


def validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def validate_ratio(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    ratio = float(value)
    if not 0 < ratio < 1:
        raise ValueError(f"{name} must be between 0 and 1 (exclusive), got {ratio}")
    return ratio


def validate_image_size(image_size: Any) -> tuple[int, int]:
    if not isinstance(image_size, tuple) or len(image_size) != 2:
        raise TypeError("image_size must be a tuple of (width, height)")
    width = validate_positive_int(image_size[0], "image_width")
    height = validate_positive_int(image_size[1], "image_height")
    return width, height


def validate_image_scaling(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"image_scaling must be a number, got {type(value).__name__}")
    scaling = float(value)
    if scaling <= 0:
        raise ValueError(f"image_scaling must be positive, got {scaling}")
    if scaling > 1:
        warnings.warn(
            f"image_scaling is {scaling}; expected a fraction such as 1/255 for normalization.",
            UserWarning,
            stacklevel=2,
        )
    return scaling


def validate_path(path: Any, name: str, must_exist: bool = False) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError(f"{name} must be a path, got {type(path).__name__}")
    resolved = Path(path)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def validate_directory(path: Any, name: str, must_exist: bool = True) -> Path:
    resolved = validate_path(path, name, must_exist=must_exist)
    if must_exist and not resolved.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {resolved}")
    return resolved


def validate_image_dataframe(df: Any, name: str = "dataframe") -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(df).__name__}")

    missing = [column for column in IMAGE_DF_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

    if df.empty:
        warnings.warn(f"{name} is empty; downstream steps may fail.", UserWarning, stacklevel=2)
        return df

    if df["label"].isna().any():
        warnings.warn(f"{name} contains missing labels.", UserWarning, stacklevel=2)

    missing_files = df[~df["file_path"].map(lambda p: Path(p).exists())]
    if not missing_files.empty:
        warnings.warn(
            f"{name} references {len(missing_files)} image file(s) that do not exist.",
            UserWarning,
            stacklevel=2,
        )

    return df


def validate_class_names(class_names: Any) -> list[str]:
    if not isinstance(class_names, list):
        raise TypeError(f"class_names must be a list, got {type(class_names).__name__}")
    if not class_names:
        raise ValueError("class_names must not be empty")
    if not all(isinstance(name, str) and name for name in class_names):
        raise ValueError("class_names must be a list of non-empty strings")
    if len(set(class_names)) != len(class_names):
        warnings.warn("class_names contains duplicate entries.", UserWarning, stacklevel=2)
    return class_names


def validate_label_array(values: Any, name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(values).__name__}")
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {values.shape}")
    if values.size == 0:
        warnings.warn(f"{name} is empty.", UserWarning, stacklevel=2)
    return values


def validate_confusion_matrix(cm: Any) -> np.ndarray:
    if not isinstance(cm, np.ndarray):
        raise TypeError(f"cm must be a numpy.ndarray, got {type(cm).__name__}")
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be a square 2-D array, got shape {cm.shape}")
    if cm.size == 0:
        warnings.warn("Confusion matrix is empty.", UserWarning, stacklevel=2)
    return cm


def warn_class_imbalance(label_counts: pd.Series, threshold: float = 0.1) -> None:
    if label_counts.empty:
        return
    proportions = label_counts / label_counts.sum()
    minority = proportions.min()
    if minority <= threshold:
        warnings.warn(
            f"Class imbalance detected: smallest class is {minority:.1%} of samples.",
            UserWarning,
            stacklevel=2,
        )


def warn_small_split(df: pd.DataFrame, split_name: str, min_rows: int = 2) -> None:
    if len(df) < min_rows:
        warnings.warn(
            f"{split_name} has only {len(df)} sample(s); training may be unstable.",
            UserWarning,
            stacklevel=2,
        )
