"""
Couches et modèles.

    pytest tests/test_models.py                    # équivalences et quelques modèles (rapide)
    pytest tests/test_models.py -m slow            # tous les modèles, toutes les formes

Couches : avec lambda_in = 1, lambda_out = 0, alpha = 1 figés, SpecCnn1D/2D redonnent Conv1D,
Conv2D et DepthwiseConv2D. Modèles : construits avec configs/classique.json et
configs/all_spectral.json, sur des images (32x32x3, 28x28x1) et des signaux (128x3, 100x1).
"""
import os

import numpy as np
import pytest
import tensorflow as tf

from specprune import CONFIGS_DIR
from specprune.layers import SpecCnn1D, SpecCnn2D
from specprune.models import AgentNetwork, available_models, configure_spectral

FROZEN = dict(use_bias=False, trainable_lambda_in=False, trainable_lambda_out=False,
              trainable_alpha=False)
TOL = 1e-4


@pytest.mark.parametrize("groups,filters,stride",
                         [(1, 4, 1), (2, 4, 1), (3, 6, 1), (6, 6, 1), (6, 6, 2), (6, 12, 1)])
def test_speccnn2d_equals_conv2d(groups, filters, stride):
    x = tf.random.normal((2, 9, 9, 6))
    sc = SpecCnn2D(filters, 3, strides=stride, padding="same", groups=groups, **FROZEN)
    y = sc(x)
    c = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", groups=groups, use_bias=False)
    c(x)
    c.set_weights([np.transpose(sc.phi_weights.numpy(), (2, 3, 1, 0))])
    assert float(tf.reduce_max(tf.abs(y - c(x)))) < TOL


def test_speccnn2d_depthwise():
    x = tf.random.normal((2, 9, 9, 6))
    sc = SpecCnn2D(6, 3, padding="same", groups=6, **FROZEN)
    y = sc(x)
    dw = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)
    dw(x)
    dw.set_weights([np.transpose(sc.phi_weights.numpy(), (2, 3, 0, 1))])
    assert float(tf.reduce_max(tf.abs(y - dw(x)))) < TOL


@pytest.mark.parametrize("groups,filters,stride", [(1, 4, 1), (3, 6, 1), (6, 6, 1), (6, 6, 2), (1, 4, 2)])
def test_speccnn1d_equals_conv1d(groups, filters, stride):
    # "same" des couches spectrales = padding symétrique (K-1)/2 (convention PyTorch) : on compare
    # à Conv1D "valid" précédée du même padding (Keras "same" met tout à droite avec stride 2).
    x = tf.random.normal((2, 16, 6))
    sc = SpecCnn1D(filters, 3, strides=stride, padding="same", groups=groups, **FROZEN)
    y = sc(x)
    c = tf.keras.layers.Conv1D(filters, 3, strides=stride, padding="valid", groups=groups, use_bias=False)
    xp = tf.pad(x, [[0, 0], [1, 1], [0, 0]])
    c(xp)
    c.set_weights([np.transpose(sc.phi_weights.numpy(), (2, 1, 0))])
    assert float(tf.reduce_max(tf.abs(y - c(xp)))) < TOL


def test_layer_config_roundtrip():
    x = tf.random.normal((2, 9, 9, 6))
    cfg = SpecCnn2D(6, 3, groups=3).get_config()
    assert SpecCnn2D.from_config(cfg)(x).shape == (2, 7, 7, 6)


def _check_model(mode, name, shape):
    configure_spectral(os.path.join(CONFIGS_DIR, f"{mode}.json"))
    agent = AgentNetwork(arch=name, num_classes=10)
    out = agent(tf.random.normal((2,) + shape))
    assert out.shape == (2, 10)
    assert np.allclose(out.numpy().sum(1), 1, atol=1e-4)


@pytest.mark.parametrize("mode", ["classique", "all_spectral"])
@pytest.mark.parametrize("shape", [(32, 32, 3), (100, 1)])
def test_resnet20(mode, shape):
    _check_model(mode, "spectral_resnet20", shape)


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["classique", "all_spectral"])
@pytest.mark.parametrize("name", available_models())
@pytest.mark.parametrize("shape", [(32, 32, 3), (28, 28, 1), (128, 3), (100, 1)])
def test_all_models(mode, name, shape):
    _check_model(mode, name, shape)


@pytest.mark.parametrize("shape,arch", [((16, 16, 3), "spectral_resnet20"), ((40, 2), "spectral_resnet20"),
                                        ((16, 16, 3), "spectral_mobilenetv2")])
def test_fast_path_same_as_spectral(shape, arch):
    """Couches aux lambda figés à (1, 0, 1) : la convolution classique donne la même sortie et le même gradient."""
    from specprune.layers import set_fast_path
    from specprune.runs import spectral_layers

    def build(fast):
        set_fast_path(fast)
        configure_spectral(os.path.join(CONFIGS_DIR, "spectral_config.json"))
        agent = AgentNetwork(arch=arch, num_classes=10)
        agent(tf.zeros((1,) + shape))
        return agent

    try:
        slow, fast = build(False), build(True)
    finally:
        set_fast_path(True)
    fast.set_weights(slow.get_weights())
    assert not any(l.classic for l in spectral_layers(slow))
    assert any(l.classic for l in spectral_layers(fast)) and not spectral_layers(fast)[0].classic
    x, y = tf.random.normal((4,) + shape), tf.constant([0, 1, 2, 3])
    grads = []
    for agent in (slow, fast):
        with tf.GradientTape() as tape:
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, agent(x, training=True))
        grads.append(tape.gradient(loss, agent.trainable_variables))
    assert float(tf.reduce_max(tf.abs(slow(x) - fast(x)))) < TOL
    for a, b in zip(*grads):
        assert float(tf.reduce_max(tf.abs(a - b))) < 1e-3


def test_fast_path_refuses_lambda_writes():
    from specprune.prune import apply_masks
    from specprune.runs import spectral_layers
    configure_spectral(os.path.join(CONFIGS_DIR, "spectral_config.json"))
    agent = AgentNetwork(arch="spectral_resnet20", num_classes=10)
    agent(tf.zeros((1, 16, 16, 3)))
    layer = spectral_layers(agent)[1]
    with pytest.raises(ValueError, match="convolution classique"):
        apply_masks(agent, {layer.name: np.zeros(layer.lambda_in.shape, bool)})
