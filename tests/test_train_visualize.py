import warnings
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.callbacks import History

from portautomation.train import save_model, train_cnn
from portautomation.visualize import plot_confusion_matrix, plot_training_history


def _compiled_model(num_classes: int = 2) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _dataset(num_classes: int = 2) -> tf.data.Dataset:
    images = tf.random.uniform((8, 32, 32, 3))
    labels = tf.one_hot(tf.random.uniform((8,), maxval=num_classes, dtype=tf.int32), num_classes)
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(4)


def test_train_cnn_warns_when_model_not_compiled():
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(32, 32, 3)), tf.keras.layers.Dense(2)]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(Exception):
            train_cnn(model, _dataset(), _dataset(), epochs=1)
    assert any("not compiled" in str(w.message).lower() for w in caught)


def test_save_model_writes_file(tmp_path: Path):
    model = _compiled_model()
    path = save_model(model, tmp_path / "model.keras")
    assert path.exists()


def test_plot_training_history_writes_file(tmp_path: Path):
    history = History()
    history.history = {
        "accuracy": [0.5, 0.6],
        "val_accuracy": [0.4, 0.5],
        "loss": [1.0, 0.8],
        "val_loss": [1.1, 0.9],
    }
    output = plot_training_history(history, "Test History", tmp_path / "history.png")
    assert output.exists()


def test_plot_confusion_matrix_warns_on_mismatched_class_names(tmp_path: Path):
    cm = np.array([[3, 1], [0, 2]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = plot_confusion_matrix(cm, class_names=["only"], output_path=tmp_path / "cm.png")
    assert output.exists()
    assert any("class_names length" in str(w.message) for w in caught)
