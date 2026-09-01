"""Training helpers for the CNN and MobileNet classifiers."""

from __future__ import annotations

import warnings
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, History

from portautomation.config import CNN_EPOCHS, MOBILE_EARLY_STOPPING_PATIENCE, MOBILE_EPOCHS
from portautomation.validation import validate_non_negative_int, validate_positive_int, validate_path


def _validate_training_inputs(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
) -> None:
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model must be a tf.keras.Model, got {type(model).__name__}")
    if not isinstance(train_ds, tf.data.Dataset):
        raise TypeError("train_ds must be a tf.data.Dataset")
    if not isinstance(val_ds, tf.data.Dataset):
        raise TypeError("val_ds must be a tf.data.Dataset")
    if not getattr(model, "optimizer", None):
        warnings.warn(
            "Model is not compiled; call compile_classifier() before training.",
            UserWarning,
            stacklevel=2,
        )


def _validate_class_weights(class_weight: dict[int, float] | None) -> dict[int, float] | None:
    if class_weight is None:
        return None
    if not isinstance(class_weight, dict):
        raise TypeError("class_weight must be a dict mapping class index to weight")
    for key, value in class_weight.items():
        if isinstance(key, bool) or not isinstance(key, int):
            raise TypeError("class_weight keys must be integers")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("class_weight values must be numbers")
        if float(value) <= 0:
            raise ValueError(f"class_weight for class {key} must be positive")
    return class_weight


def train_cnn(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weight: dict[int, float] | None = None,
    epochs: int = CNN_EPOCHS,
) -> History:
    _validate_training_inputs(model, train_ds, val_ds)
    epochs = validate_positive_int(epochs, "epochs")
    class_weight = _validate_class_weights(class_weight)

    return model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        class_weight=class_weight,
        verbose=0,
    )


def train_mobilenet(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weight: dict[int, float] | None = None,
    epochs: int = MOBILE_EPOCHS,
    patience: int = MOBILE_EARLY_STOPPING_PATIENCE,
) -> History:
    _validate_training_inputs(model, train_ds, val_ds)
    epochs = validate_positive_int(epochs, "epochs")
    patience = validate_non_negative_int(patience, "patience")
    class_weight = _validate_class_weights(class_weight)

    if patience == 0:
        warnings.warn("patience is 0; early stopping will not be effective.", UserWarning)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
    return model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=0,
    )


def save_model(model: tf.keras.Model, path: Path | str) -> Path:
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model must be a tf.keras.Model, got {type(model).__name__}")
    path = validate_path(path, "path")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path
