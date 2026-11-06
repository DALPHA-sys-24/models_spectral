"""
Chemin rapide des couches spectrales : une couche dont lambda_in, lambda_out et alpha sont figés à
leurs valeurs neutres (1, 0, 1) calcule exactement une convolution classique (W = phi). On la
calcule alors avec tf.nn.conv1d / conv2d / depthwise_conv2d au lieu de la formulation spectrale,
qui garde en mémoire les fenêtres de l'entrée (K fois l'entrée) et fait un einsum par position.

La décision est prise une fois, à la construction de la couche (layer.classic), d'après les valeurs
des poids. Elle est revue par refresh_classic() quand des poids sont rechargés (load_run), et une
écriture dans les lambda d'une couche classique est refusée (prune.apply_masks), car le graphe déjà
tracé ne la verrait pas.

    from specprune.layers import set_fast_path
    set_fast_path(False)   # tout calculer par la formulation spectrale (comparaison, débogage)
"""
import numpy as np
import tensorflow as tf

_ENABLED = True


def set_fast_path(enabled: bool) -> None:
    """Active ou non le chemin rapide pour les couches construites ensuite."""
    global _ENABLED
    _ENABLED = bool(enabled)


def fast_path_enabled() -> bool:
    return _ENABLED


def _is(var, value) -> bool:
    try:
        return bool(np.all(var.numpy() == value))
    except Exception:  # noqa: BLE001  (construction dans un graphe : valeurs inaccessibles)
        return False


def is_neutral(layer) -> bool:
    """lambda_in = 1, lambda_out = 0, alpha = 1, tous figés."""
    if layer.trainable_lambda_in or layer.trainable_lambda_out or layer.trainable_alpha:
        return False
    return _is(layer.lambda_in, 1.0) and _is(layer.lambda_out, 0.0) and _is(layer.alpha, 1.0)


def _supported(layer) -> bool:
    return layer.groups == 1 or layer._C == 1   # convolution standard, ou depthwise (groups = C)


def refresh_classic(layer) -> bool:
    """(Re)décide le chemin rapide d'après les valeurs actuelles des poids."""
    layer.classic = _ENABLED and is_neutral(layer) and _supported(layer)
    return layer.classic


def classic_conv_2d(X, phi, stride_h, stride_w, pad_h, pad_w, groups):
    """X (B, H, W, C), phi (F, Cg, Kh, Kw) -> (B, Oh, Ow, F), padding symétrique comme la couche."""
    if pad_h or pad_w:
        X = tf.pad(X, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
    strides = [1, stride_h, stride_w, 1]
    if groups == 1:
        kernel = tf.transpose(phi, (2, 3, 1, 0))                     # (Kh, Kw, C, F)
        return tf.nn.conv2d(X, kernel, strides, "VALID")
    # depthwise : groups = C, Cg = 1, F = C * m ; filtre f = c * m + d voit le canal c
    F, _, Kh, Kw = phi.shape
    C = groups
    kernel = tf.reshape(tf.transpose(phi[:, 0], (1, 2, 0)), (Kh, Kw, C, F // C))
    return tf.nn.depthwise_conv2d(X, kernel, strides, "VALID")


def classic_conv_1d(X, phi, stride, pad, groups):
    """X (B, L, C), phi (F, Cg, K) -> (B, O, F)."""
    if pad:
        X = tf.pad(X, [[0, 0], [pad, pad], [0, 0]])
    if groups == 1:
        return tf.nn.conv1d(X, tf.transpose(phi, (2, 1, 0)), stride, "VALID")  # (K, C, F)
    y = classic_conv_2d(X[:, tf.newaxis], phi[:, :, tf.newaxis, :], 1, stride, 0, 0, groups)
    return y[:, 0]
