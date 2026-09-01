import warnings

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import layers

from portautomation.evaluate import (
    build_confusion_matrix,
    collect_predictions,
    evaluate_model,
    print_classification_report,
)
from portautomation.models import build_cnn, compile_classifier


def _build_toy_model(num_classes: int = 2) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return compile_classifier(model)


def _toy_dataset(num_classes: int = 2, batches: int = 2) -> tf.data.Dataset:
    images = tf.random.uniform((batches * 4, 32, 32, 3))
    labels = tf.one_hot(tf.random.uniform((batches * 4,), maxval=num_classes, dtype=tf.int32), num_classes)
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(4)


def test_build_cnn_rejects_invalid_image_size():
    with pytest.raises(TypeError):
        build_cnn(image_size=(224))


def test_evaluate_model_returns_metrics():
    model = _build_toy_model()
    metrics = evaluate_model(model, _toy_dataset())
    assert set(metrics) == {"loss", "accuracy", "precision", "recall"}


def test_collect_predictions_shapes_match():
    model = _build_toy_model()
    true_classes, predicted_classes = collect_predictions(model, _toy_dataset())
    assert true_classes.shape == predicted_classes.shape


def test_build_confusion_matrix_rejects_mismatched_lengths(sample_label_arrays):
    true_classes, predicted_classes = sample_label_arrays
    with pytest.raises(ValueError):
        build_confusion_matrix(true_classes, np.array([0, 1]))


def test_print_classification_report_warns_on_short_class_names(sample_label_arrays):
    true_classes, predicted_classes = sample_label_arrays
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = print_classification_report(true_classes, predicted_classes, ["a", "b"])
    assert "precision" in report
    assert any("class_names length" in str(w.message) for w in caught)
