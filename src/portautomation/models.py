"""CNN from scratch and MobileNetV2 transfer-learning classifiers."""

from __future__ import annotations

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


def build_cnn(
    image_size: tuple[int, int] = IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """Scratch CNN used in notebook section 1.4."""
    width, height = image_size
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
) -> keras.Model:
    """MobileNetV2 transfer-learning model used in notebook section 2.4."""
    width, height = image_size
    pretrained = tf.keras.applications.MobileNetV2(
        input_shape=(width, height, 3),
        include_top=False,
        weights="imagenet",
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
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", precision_m, recall_m],
    )
    return model
