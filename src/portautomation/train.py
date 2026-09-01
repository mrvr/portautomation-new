"""Training helpers for the CNN and MobileNet classifiers."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, History

from portautomation.config import CNN_EPOCHS, MOBILE_EARLY_STOPPING_PATIENCE, MOBILE_EPOCHS


def train_cnn(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weight: dict[int, float] | None = None,
    epochs: int = CNN_EPOCHS,
) -> History:
    return model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        class_weight=class_weight,
    )


def train_mobilenet(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weight: dict[int, float] | None = None,
    epochs: int = MOBILE_EPOCHS,
    patience: int = MOBILE_EARLY_STOPPING_PATIENCE,
) -> History:
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
    )


def save_model(model: tf.keras.Model, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path
