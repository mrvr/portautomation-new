import warnings

import pytest
import tensorflow as tf

from portautomation.models import build_cnn, build_mobilenet, compile_classifier


def test_build_cnn_has_expected_output_shape():
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = build_cnn(image_size=(32, 32), num_classes=3)
    import tensorflow as tf

    with tf.device("/CPU:0"):
        output = model(tf.zeros((1, 32, 32, 3)))
    assert output.shape == (1, 3)


def test_build_mobilenet_without_weights_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = build_mobilenet(image_size=(32, 32), num_classes=3, weights=None)
    assert model.output_shape == (None, 3)
    assert any("pretrained weights" in str(w.message).lower() for w in caught)


def test_compile_classifier_rejects_empty_optimizer():
    model = build_cnn(image_size=(32, 32), num_classes=2)
    with pytest.raises(ValueError):
        compile_classifier(model, optimizer="  ")


def test_compile_classifier_rejects_invalid_model():
    with pytest.raises(TypeError):
        compile_classifier("not-a-model")
