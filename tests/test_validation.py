import warnings

import numpy as np
import pandas as pd
import pytest

from portautomation.validation import (
    validate_class_names,
    validate_confusion_matrix,
    validate_image_dataframe,
    validate_image_scaling,
    validate_image_size,
    validate_label_array,
    validate_positive_int,
    validate_ratio,
    warn_class_imbalance,
)


def test_validate_positive_int_accepts_valid_value():
    assert validate_positive_int(5, "epochs") == 5


def test_validate_positive_int_rejects_bool():
    with pytest.raises(TypeError):
        validate_positive_int(True, "epochs")


def test_validate_ratio_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        validate_ratio(1.0, "test_size")
    with pytest.raises(ValueError):
        validate_ratio(0.0, "test_size")


def test_validate_image_size_requires_tuple():
    with pytest.raises(TypeError):
        validate_image_size([224, 224])


def test_validate_image_scaling_warns_above_one():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = validate_image_scaling(255)
    assert value == 255.0
    assert any("image_scaling" in str(w.message) for w in caught)


def test_validate_image_dataframe_warns_on_empty():
    empty = pd.DataFrame(columns=["file_name", "label", "file_path"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_image_dataframe(empty)
    assert any("empty" in str(w.message).lower() for w in caught)


def test_validate_image_dataframe_rejects_missing_columns():
    df = pd.DataFrame({"label": ["a"]})
    with pytest.raises(ValueError):
        validate_image_dataframe(df)


def test_validate_class_names_warns_on_duplicates():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        names = validate_class_names(["a", "a", "b"])
    assert names == ["a", "a", "b"]
    assert any("duplicate" in str(w.message).lower() for w in caught)


def test_validate_label_array_and_confusion_matrix():
    labels = validate_label_array(np.array([0, 1, 2]), "labels")
    assert labels.shape == (3,)
    cm = validate_confusion_matrix(np.array([[2, 0], [1, 3]]))
    assert cm.shape == (2, 2)


def test_warn_class_imbalance_emits_warning():
    counts = pd.Series({"a": 90, "b": 10})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_class_imbalance(counts)
    assert any("imbalance" in str(w.message).lower() for w in caught)
