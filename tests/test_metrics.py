import numpy as np
import tensorflow as tf

from portautomation.metrics import precision_m, recall_m


def test_precision_and_recall_with_perfect_prediction():
    y_true = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    y_pred = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    assert float(precision_m(y_true, y_pred)) == 1.0
    assert float(recall_m(y_true, y_pred)) == 1.0


def test_precision_with_no_positive_predictions():
    y_true = tf.constant([[1.0, 0.0]])
    y_pred = tf.constant([[0.0, 1.0]])
    precision = float(precision_m(y_true, y_pred))
    assert 0.0 <= precision <= 1.0
