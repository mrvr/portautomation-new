"""CNN from scratch and MobileNetV2 transfer-learning classifiers."""

from __future__ import annotations

import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)

from portautomation.config import IMAGE_SIZE, NUM_CLASSES
from portautomation.metrics import precision_m, recall_m
from portautomation.validation import validate_image_size, validate_positive_int


def build_cnn(
    image_size: tuple[int, int] = IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """Scratch CNN used in notebook section 1.4."""
    width, height = validate_image_size(image_size)
    num_classes = validate_positive_int(num_classes, "num_classes")

    model = keras.Sequential(
        [
            Input(shape=(width, height, 3)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(),
            GlobalAveragePooling2D(),
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def build_mobilenet(
    image_size: tuple[int, int] = IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
    weights: str | None = "imagenet",
) -> keras.Model:
    """MobileNetV2 transfer-learning model used in notebook section 2.4."""
    width, height = validate_image_size(image_size)
    num_classes = validate_positive_int(num_classes, "num_classes")
    if weights is not None and not isinstance(weights, str):
        raise TypeError("weights must be a string or None")

    if weights is None:
        warnings.warn(
            "Building MobileNetV2 without pretrained weights.",
            UserWarning,
            stacklevel=2,
        )

    pretrained = tf.keras.applications.MobileNetV2(
        input_shape=(width, height, 3),
        include_top=False,
        weights=weights,
    )
    pretrained.trainable = False

    model = keras.Sequential(
        [
            pretrained,
            GlobalAveragePooling2D(),
            Dropout(0.2),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.1),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.1),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def compile_classifier(model: keras.Model, optimizer: str = "adam") -> keras.Model:
    """Compile with Adam, categorical crossentropy, accuracy, precision, and recall."""
    if not isinstance(model, keras.Model):
        raise TypeError(f"model must be a keras.Model, got {type(model).__name__}")
    if not isinstance(optimizer, str) or not optimizer.strip():
        raise ValueError("optimizer must be a non-empty string")

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", precision_m, recall_m],
    )
    return model
