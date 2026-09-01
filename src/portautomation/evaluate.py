"""Evaluation helpers: test metrics, confusion matrix, classification report."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from portautomation.validation import (
    validate_class_names,
    validate_confusion_matrix,
    validate_label_array,
)

logger = logging.getLogger(__name__)


def _validate_model(model: tf.keras.Model) -> tf.keras.Model:
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model must be a tf.keras.Model, got {type(model).__name__}")
    if not model.built:
        warnings.warn("Model has not been built yet; evaluation may fail.", UserWarning, stacklevel=2)
    return model


def _validate_dataset(dataset: tf.data.Dataset, name: str) -> tf.data.Dataset:
    if not isinstance(dataset, tf.data.Dataset):
        raise TypeError(f"{name} must be a tf.data.Dataset, got {type(dataset).__name__}")
    return dataset


def evaluate_model(model: tf.keras.Model, test_ds: tf.data.Dataset) -> dict[str, float]:
    model = _validate_model(model)
    test_ds = _validate_dataset(test_ds, "test_ds")

    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_ds, verbose=0)
    metrics = {
        "loss": float(test_loss),
        "accuracy": float(test_acc),
        "precision": float(test_precision),
        "recall": float(test_recall),
    }
    logger.info(
        "Test loss=%.4f accuracy=%.4f precision=%.4f recall=%.4f",
        metrics["loss"],
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
    )
    return metrics


def collect_predictions(
    model: tf.keras.Model, test_ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    model = _validate_model(model)
    test_ds = _validate_dataset(test_ds, "test_ds")

    predictions = model.predict(test_ds, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes: list[int] = []
    for _, labels in test_ds:
        true_classes.extend(np.argmax(labels.numpy(), axis=1))
    return np.array(true_classes), predicted_classes


def build_confusion_matrix(true_classes: np.ndarray, predicted_classes: np.ndarray) -> np.ndarray:
    true_classes = validate_label_array(true_classes, "true_classes")
    predicted_classes = validate_label_array(predicted_classes, "predicted_classes")
    if true_classes.shape != predicted_classes.shape:
        raise ValueError(
            "true_classes and predicted_classes must have the same length: "
            f"{true_classes.shape[0]} vs {predicted_classes.shape[0]}"
        )
    return confusion_matrix(true_classes, predicted_classes)


def print_classification_report(
    true_classes: np.ndarray,
    predicted_classes: np.ndarray,
    class_names: list[str] | None = None,
) -> str:
    true_classes = validate_label_array(true_classes, "true_classes")
    predicted_classes = validate_label_array(predicted_classes, "predicted_classes")
    if class_names is not None:
        class_names = validate_class_names(class_names)
        observed = sorted(set(true_classes.tolist()) | set(predicted_classes.tolist()))
        if len(class_names) != len(observed):
            warnings.warn(
                "class_names length does not match the number of observed labels; "
                "falling back to numeric labels in the report.",
                UserWarning,
                stacklevel=2,
            )
            class_names = None

    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        zero_division=0,
    )
    logger.info("\n%s", report)
    return report
