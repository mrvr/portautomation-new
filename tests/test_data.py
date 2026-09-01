import warnings
from pathlib import Path

import pandas as pd
import pytest

from portautomation.data import (
    compute_class_weights,
    load_images_to_dataframe,
    save_images_to_directory,
    split_dataframe,
    summarize_labels,
)


def test_load_images_to_dataframe(sample_image_tree: Path):
    df = load_images_to_dataframe(sample_image_tree)
    assert len(df) == 6
    assert set(df.columns) == {"file_name", "label", "file_path"}


def test_load_images_to_dataframe_rejects_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_images_to_dataframe(tmp_path / "missing")


def test_summarize_labels(sample_dataframe: pd.DataFrame):
    summary = summarize_labels(sample_dataframe)
    assert "count" in summary.columns
    assert summary["count"].sum() == len(sample_dataframe)


def test_split_dataframe_returns_three_splits(sample_dataframe: pd.DataFrame):
    train_df, val_df, test_df = split_dataframe(
        sample_dataframe, test_size=0.34, random_state=1, val_size=0.34
    )
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_dataframe)


def test_split_dataframe_rejects_invalid_ratio(sample_dataframe: pd.DataFrame):
    with pytest.raises(ValueError):
        split_dataframe(sample_dataframe, test_size=1.5, random_state=1)


def test_split_dataframe_rejects_too_small_dataframe():
    tiny = pd.DataFrame(
        {
            "file_name": ["0.jpg"],
            "label": ["class_a"],
            "file_path": ["/tmp/0.jpg"],
        }
    )
    with pytest.raises(ValueError):
        split_dataframe(tiny, test_size=0.5, random_state=1)


def test_save_images_to_directory(sample_dataframe: pd.DataFrame, tmp_path: Path):
    target = tmp_path / "split"
    result = save_images_to_directory(sample_dataframe, target)
    assert result.exists()
    assert len(list(result.rglob("*.jpg"))) == len(sample_dataframe)


def test_compute_class_weights_warns_on_missing_class():
    counts = pd.Series({"class_a": 4})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        weights = compute_class_weights(counts, ["class_a", "class_b"])
    assert weights[1] == 1.0
    assert any("zero training samples" in str(w.message).lower() for w in caught)
