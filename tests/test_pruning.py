"""
Pruning (specprune/prune.py) : méthodes gradient / element / mixte, critères percentile et rank,
padding exclu, gel des lambda pour le post-training (seuls les phi bougent, les zéros restent à
zéro), réassignation de valeurs.

    pytest tests/test_pruning.py
"""
import numpy as np
import pytest
import tensorflow as tf

from specprune.models import AgentNetwork, configure_spectral
from specprune.prune import (apply_masks, assign_spectral_weights, compute_masks, freeze_spectral,
                             get_spectral_values, prune_spectral, select_layers)
from specprune.runs import spectral_layers, valid_positions

SHAPES = [(16, 16, 3), (64, 2)]   # image (SpecCnn2D) et signal (SpecCnn1D)


def build(shape):
    # lambda_in entraîné dans les couches 0 et 1, valeurs aléatoires pour un classement net
    configure_spectral({"default": {"trainable_lambda_in": False, "trainable_lambda_out": False,
                                    "trainable_alpha": False},
                        "layers": {"0-1": {"trainable_lambda_in": True, "lambda_in_l1": 0.0,
                                           "lambda_in_initializer": "random_normal"}}})
    agent = AgentNetwork(arch="spectral_resnet20", num_classes=3)
    agent(tf.zeros((1,) + shape))
    return agent


def setup(shape):
    agent = build(shape)
    l0, l1 = select_layers(agent)
    assert [l.name for l in (l0, l1)] == [l.name for l in spectral_layers(agent)[:2]]
    original = get_spectral_values(agent, "lambda_in", "0-1")
    w0, v0 = np.abs(original[l0.name]), valid_positions(l0)
    pix = v0[0, 0]                    # positions hors padding
    n_pix = int(pix.sum())
    return agent, l0, l1, original, w0, v0, pix, n_pix


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("crit", ["percentile", "rank"])
def test_methods(shape, crit):
    agent, l0, l1, original, w0, v0, pix, n_pix = setup(shape)
    for p in (0.0, 1.0):          # p = 0 : rien ; p = 1 : tout
        for method in ("gradient", "element", "mixte"):
            rep = compute_masks(agent, p, method=method, criterion=crit, values=original)
            assert rep.pruned == (0 if p == 0 else rep.total), (crit, p, method)
            for layer in (l0, l1):  # positions de padding toujours gardées
                assert rep.masks[layer.name][~valid_positions(layer)].all()

    p = 0.4
    k = round(p * n_pix)
    # gradient : dans chaque filtre, la fraction p des |w| (tous canaux) est retirée
    m = compute_masks(agent, p, method="gradient", criterion=crit, values=original).masks[l0.name]
    for f in range(m.shape[0]):
        vals = w0[f][v0[f]]
        gone, kept = vals[~m[f][v0[f]]], vals[m[f][v0[f]]]
        assert abs(gone.size - p * vals.size) <= 1 and gone.max() <= kept.min()
    assert not (m == m[:, :1]).all()   # un pixel peut garder une partie de ses canaux

    # element : par filtre, pixel retiré pour tous les canaux, selon la somme des |w| sur C
    m = compute_masks(agent, p, method="element", criterion=crit, values=original).masks[l0.name]
    assert (m == m[:, :1]).all()
    for f in range(m.shape[0]):
        score, keep = w0[f].sum(axis=0)[pix], m[f, 0][pix]
        assert abs((~keep).sum() - k) <= 1 and score[~keep].max() <= score[keep].min()
    assert not (m == m[:1]).all()      # les filtres ne retirent pas les mêmes pixels

    # mixte : mêmes pixels retirés pour tous les filtres, score sommé sur F et C
    m = compute_masks(agent, p, method="mixte", criterion=crit, values=original).masks[l0.name]
    assert (m == m[:1, :1]).all()
    score, keep = w0.sum(axis=(0, 1))[pix], m[0, 0][pix]
    assert abs((~keep).sum() - k) <= 1 and score[~keep].max() <= score[keep].min()


@pytest.mark.parametrize("shape", SHAPES)
def test_freeze_and_assign(shape):
    agent, l0, l1, original, w0, v0, pix, n_pix = setup(shape)
    # pruning en place : poids hors masque à zéro
    rep = prune_spectral(agent, 0.3, layers="0")
    assert (l0.lambda_in.numpy()[~rep.masks[l0.name]] == 0).all()

    # post-training : seuls les phi sont entraînés, les zéros restent à zéro
    apply_masks(agent, rep.masks, values=original)
    counts = freeze_spectral(agent, "phi")
    assert all(v.name == "phi_weights" for v in agent.trainable_variables)
    phi_before = l0.phi_weights.numpy().copy()
    agent.compile(optimizer=tf.keras.optimizers.SGD(0.1), loss="sparse_categorical_crossentropy")
    x = np.random.randn(16, *shape).astype("float32")
    agent.fit(x, np.random.randint(0, 3, 16), epochs=1, verbose=0)
    assert np.allclose(l0.lambda_in.numpy(), np.where(rep.masks[l0.name], original[l0.name], 0))
    assert not np.allclose(l0.phi_weights.numpy(), phi_before)

    # réassignation (ex. médiane de plusieurs runs)
    assert assign_spectral_weights(agent, original, verbose=False) == 2
    assert np.allclose(l0.lambda_in.numpy(), original[l0.name])
    assert counts["trainable"] > 0
