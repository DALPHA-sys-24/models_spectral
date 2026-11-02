"""
Blocs de couches partagés par les architectures. Toutes les convolutions sont spectrales :
SpecCnn2D pour des images (batch, H, W, C), SpecCnn1D pour des signaux (batch, T, C). Le choix
se fait d'après la forme de l'entrée : les mêmes architectures servent aux deux.
"""
import copy
import json
import math
from typing import Any, Dict, List, Union

import tensorflow as tf

from spectralcnn import SpecCnn1D, SpecCnn2D

L = tf.keras.layers

# --- Configuration spectrale couche par couche ------------------------------------
#
# Toutes les convolutions des modèles sont des SpecCnn2D (ou SpecCnn1D), numérotées dans l'ordre de
# construction (0 = première couche / stem, puis chaque conv 3x3, projection 1x1 ou
# depthwise). La configuration dit, pour chaque indice, quels poids la couche entraîne :
#
#   {
#     "default": {"trainable_lambda_in": false, "trainable_lambda_out": false,
#                 "trainable_alpha": false},
#     "layers": {
#       "0": {"trainable_lambda_in": true, "lambda_in_l1": 1e-6}
#     }
#   }
#
# Clés d'une règle :
#   trainable_phi, trainable_lambda_in, trainable_lambda_out, trainable_alpha
#       un poids non entraîné reste à sa valeur initiale : lambda_in = 1, lambda_out = 0,
#       alpha = 1 (et phi = 1). Avec lambda_in, lambda_out et alpha figés, la couche est
#       exactement une Conv2D (DepthwiseConv2D pour les depthwise), ou une Conv1D pour des
#       signaux : modèle classique. Seule différence : "same" pad symétriquement (convention
#       PyTorch) ; avec stride 2 sur une taille paire, Keras mettrait tout le padding à droite.
#   phi_l1, lambda_in_l1, lambda_out_l1, alpha_l1     coefficient L1 (0 = aucune pénalité)
#   tout autre argument de SpecCnn2D / SpecCnn1D (initialiseurs...) est transmis tel quel.
# Clés de "layers" : un indice ("0"), un intervalle ("1-5", bornes incluses) ou "*".
# Les règles s'appliquent dans l'ordre : default, puis "*", puis chaque clé qui contient
# l'indice. Sans configuration : valeurs par défaut de SpecCnn2D (tout est entraîné).
# Une même configuration donne le même montage en 1D et en 2D : les valeurs par défaut de
# SpecCnn2D (initialiseurs, L1) sont aussi données à SpecCnn1D (voir _defaults_like_2d).

_DEFAULT_CONFIG: Dict[str, Any] = {"default": {}, "layers": {}}
_CONFIG: Dict[str, Any] = copy.deepcopy(_DEFAULT_CONFIG)
_L1_KEYS = {"phi_l1": "phi_regularizer", "lambda_in_l1": "lambda_in_regularizer",
            "lambda_out_l1": "lambda_out_regularizer", "alpha_l1": "alpha_regularizer"}
_conv_index = 0
_plan: List[Dict[str, Any]] = []


def configure_spectral(config: Union[None, str, Dict[str, Any]] = None, **default_kwargs) -> None:
    """
    Fixe la configuration spectrale des modèles (avant leur construction).

        configure_spectral("configs/spectral_config.json")   # fichier JSON
        configure_spectral({"default": {...}, "layers": {...}})
        configure_spectral(trainable_alpha=False)            # règle commune à toutes les couches
        configure_spectral()                                 # valeurs par défaut de SpecCnn2D
    """
    if isinstance(config, str):
        with open(config) as f:
            config = json.load(f)
    config = copy.deepcopy(config) if config else copy.deepcopy(_DEFAULT_CONFIG)
    config.setdefault("default", {})
    config.setdefault("layers", {})
    config["default"].update(default_kwargs)
    for key, rule in [("default", config["default"])] + list(config["layers"].items()):
        if key != "default":
            _parse_key(key)  # erreur immédiate si une clé est mal écrite
        if "spectral" in rule:
            raise ValueError(f"Règle '{key}' : la clé 'spectral' n'existe plus, toutes les couches "
                             "sont spectrales. Pour une couche classique, mettre trainable_lambda_in, "
                             "trainable_lambda_out et trainable_alpha à false.")
    _CONFIG.clear()
    _CONFIG.update(config)


def get_spectral_config() -> Dict[str, Any]:
    """Configuration spectrale active (copie)."""
    return copy.deepcopy(_CONFIG)


def start_backbone() -> None:
    """Remet la numérotation des convolutions à 0 (appelé avant chaque construction)."""
    global _conv_index
    _conv_index = 0
    _plan.clear()


def spectral_plan() -> List[Dict[str, Any]]:
    """Réglages appliqués à chaque convolution lors de la dernière construction."""
    return copy.deepcopy(_plan)


def _parse_key(key: str):
    if key == "*":
        return None
    if "-" in key:
        lo, hi = key.split("-")
        return int(lo), int(hi)
    return int(key), int(key)


def _settings_for(index: int) -> Dict[str, Any]:
    settings = dict(_CONFIG["default"])
    rules = _CONFIG["layers"]
    if "*" in rules:
        settings.update(rules["*"])
    for key, rule in rules.items():
        bounds = _parse_key(key)
        if bounds and bounds[0] <= index <= bounds[1]:
            settings.update(rule)
    return settings


def _next_conv(kind: str, **info) -> Dict[str, Any]:
    """Arguments de la couche spectrale suivante, selon la configuration."""
    global _conv_index
    index, _conv_index = _conv_index, _conv_index + 1
    settings = _settings_for(index)
    _plan.append({"index": index, "kind": kind, **info, **settings})
    kwargs = {}
    for key, value in settings.items():
        if key in _L1_KEYS:
            # L1(0.0) et non None : None réactiverait la L1(1e-2) par défaut de SpecCnn2D
            kwargs[_L1_KEYS[key]] = tf.keras.regularizers.L1(float(value))
        else:
            kwargs[key] = value
    return kwargs


def _defaults_like_2d(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complète les arguments d'une SpecCnn1D avec les valeurs par défaut de SpecCnn2D.

    SpecCnn1D initialise un lambda_in entraîné en glorot_uniform (valeurs proches de 0) et n'a
    pas de L1 par défaut ; SpecCnn2D part de 1 et met une L1(1e-2). Sans cet alignement, une
    même configuration ne donnerait pas le même modèle sur des signaux et sur des images.
    Ce que la configuration fixe explicitement n'est pas modifié.
    """
    init = tf.keras.initializers
    for weight, trained, frozen in [("lambda_in", 1.0, init.Ones), ("lambda_out", 0.0, init.Zeros),
                                    ("alpha", 1.0, init.Ones)]:
        if kwargs.get(f"trainable_{weight}", True):
            kwargs.setdefault(f"{weight}_initializer", init.TruncatedNormal(mean=trained, stddev=0.05))
        else:
            kwargs.setdefault(f"{weight}_initializer", frozen())
    for weight in ("lambda_in", "lambda_out"):
        if kwargs.get(f"trainable_{weight}", True):
            kwargs.setdefault(f"{weight}_regularizer", tf.keras.regularizers.L1(1e-2))
    return kwargs


def _spectral_conv(x, filters, kernel, stride, groups, phi_initializer, settings):
    """SpecCnn2D pour une image (batch, H, W, C), SpecCnn1D pour un signal (batch, T, C)."""
    kwargs = {"phi_initializer": phi_initializer}
    kwargs.update(settings)
    layer = SpecCnn2D
    if len(x.shape) == 3:
        layer = SpecCnn1D
        kwargs = _defaults_like_2d(kwargs)
    return layer(filters, kernel, strides=stride, padding="same", use_bias=False,
                 groups=groups, **kwargs)(x)


# --- Blocs ------------------------------------------------------------------------

def _phi_he_normal(fan_in: int):
    """
    he_normal calculé sur le vrai fan_in (C * Kh * Kw).

    phi_weights est rangé en (F, C, Kh, Kw) et non (Kh, Kw, C, F) : l'initialiseur Keras
    "he_normal" y calculerait un fan_in faux (Kh * F * C). En 1D : (F, C, K), fan_in = C * K.
    """
    stddev = math.sqrt(2.0 / fan_in) / 0.87962566103423978  # correction de la troncature
    return tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=stddev)


def _activate(x, activation):
    if activation == "relu":
        x = L.ReLU()(x)
    elif activation == "relu6":
        x = L.ReLU(max_value=6.0)(x)
    return x


def _kernel_size(x, kernel):
    """Nombre de positions du noyau : kernel² pour une image, kernel pour un signal."""
    return kernel ** (len(x.shape) - 2)


def spec_conv_bn(x, filters, kernel=3, stride=1, activation="relu"):
    """Conv spectrale sans biais + BatchNorm + activation optionnelle ("relu", "relu6" ou None)."""
    settings = _next_conv("conv", filters=filters, kernel=kernel, stride=stride,
                          input_shape=list(x.shape[1:]))
    x = _spectral_conv(x, filters, kernel, stride, 1,
                       _phi_he_normal(x.shape[-1] * _kernel_size(x, kernel)), settings)
    x = L.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    return _activate(x, activation)


def spec_depthwise_bn(x, stride=1, activation="relu"):
    """Conv spectrale depthwise 3x3 (ou 3 en 1D ; groups = filters = C) + BatchNorm + activation."""
    channels = x.shape[-1]
    settings = _next_conv("depthwise", filters=channels, kernel=3, stride=stride,
                          input_shape=list(x.shape[1:]))
    # fan_in d'une depthwise : Kh * Kw (K en 1D)
    x = _spectral_conv(x, channels, 3, stride, channels, _phi_he_normal(_kernel_size(x, 3)), settings)
    x = L.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    return L.ReLU(max_value=6.0 if activation == "relu6" else None)(x)
