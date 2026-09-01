"""Evaluation helpers: test metrics, confusion matrix, classification report."""

from __future__ import annotations

import logging

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


def evaluate_model(model: tf.keras.Model, test_ds: tf.data.Dataset) -> dict[str, float]:
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_ds)
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


def collect_predictions(model: tf.keras.Model, test_ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(test_ds)
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes: list[int] = []
    for _, labels in test_ds:
        true_classes.extend(np.argmax(labels.numpy(), axis=1))
    return np.array(true_classes), predicted_classes


def build_confusion_matrix(true_classes: np.ndarray, predicted_classes: np.ndarray) -> np.ndarray:
    return confusion_matrix(true_classes, predicted_classes)


def print_classification_report(
    true_classes: np.ndarray,
    predicted_classes: np.ndarray,
    class_names: list[str] | None = None,
) -> str:
    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        zero_division=0,
    )
    logger.info("\n%s", report)
    return report
