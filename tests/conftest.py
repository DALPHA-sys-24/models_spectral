"""Réglages communs des tests : TensorFlow silencieux, sur CPU, session Keras vidée après chaque test."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # tests rapides et reproductibles sur CPU
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402


@pytest.fixture(autouse=True)
def clear_keras():
    yield
    tf.keras.backend.clear_session()
