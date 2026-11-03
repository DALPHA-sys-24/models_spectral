"""
ResNet (He et al. 2016) pour petites images (28x28, 32x32) ou signaux (T x C).

- spectral_resnet20/32/44/56/110 : version CIFAR, 3 étages de 16, 32, 64 filtres.
- spectral_resnet10/18/34 : version ImageNet avec un stem 3x3 de stride 1 et sans max-pool.

Toutes les convolutions (3x3 et projections 1x1) sont des SpecCnn2D, ou des SpecCnn1D (noyaux 3
et 1) quand l'entrée est un signal. Les poids qu'elles
entraînent se règlent couche par couche (spectral_config.json, voir layers.configure_spectral) ;
avec lambda_in = 1, lambda_out = 0 et alpha = 1 figés, on retrouve le ResNet classique (1D ou 2D).
"""
from typing import List

import tensorflow as tf

from .blocks import spec_conv_bn
from .registry import register_model

L = tf.keras.layers


def basic_block(x, filters, stride, conv=spec_conv_bn):
    """Bloc résiduel basique : deux conv 3x3, projection 1x1 si la forme change."""
    shortcut = x
    out = conv(x, filters, 3, stride)
    out = conv(out, filters, 3, 1, activation=None)
    if stride != 1 or x.shape[-1] != filters:
        shortcut = conv(x, filters, 1, stride, activation=None)
    return L.ReLU()(L.Add()([out, shortcut]))


def resnet(inputs, stem_filters: int, stage_filters: List[int], blocks_per_stage: List[int],
           width: float = 1.0, conv=spec_conv_bn):
    """
    ResNet générique : stem, puis un étage par entrée de stage_filters (stride 2 dès le 2e).
    width multiplie le nombre de filtres de chaque couche (ResNet « large »).
    conv construit chaque bloc conv + BN + activation.
    """
    def scaled(filters):
        return max(1, int(round(filters * width)))

    x = conv(inputs, scaled(stem_filters), 3, 1)
    for stage, (filters, n_blocks) in enumerate(zip(map(scaled, stage_filters), blocks_per_stage)):
        for block in range(n_blocks):
            stride = 2 if stage > 0 and block == 0 else 1
            x = basic_block(x, filters, stride, conv)
    return x


def resnet_cifar(inputs, depth: int, width: float = 1.0, conv=spec_conv_bn):
    """ResNet CIFAR de profondeur 6n + 2."""
    n = (depth - 2) // 6
    return resnet(inputs, 16, [16, 32, 64], [n, n, n], width, conv)


def resnet_imagenet(inputs, depth: int, width: float = 1.0, conv=spec_conv_bn):
    """ResNet ImageNet (stem 3x3 de stride 1, sans max-pool) de profondeur 10, 18 ou 34."""
    blocks = {10: [1, 1, 1, 1], 18: [2, 2, 2, 2], 34: [3, 4, 6, 3]}[depth]
    return resnet(inputs, 64, [64, 128, 256, 512], blocks, width, conv)


@register_model("spectral_resnet20")
def spectral_resnet20(inputs, width=1.0):
    return resnet_cifar(inputs, 20, width)


@register_model("spectral_resnet32")
def spectral_resnet32(inputs, width=1.0):
    return resnet_cifar(inputs, 32, width)


@register_model("spectral_resnet44")
def spectral_resnet44(inputs, width=1.0):
    return resnet_cifar(inputs, 44, width)


@register_model("spectral_resnet56")
def spectral_resnet56(inputs, width=1.0):
    return resnet_cifar(inputs, 56, width)


@register_model("spectral_resnet110")
def spectral_resnet110(inputs, width=1.0):
    return resnet_cifar(inputs, 110, width)


@register_model("spectral_resnet10")
def spectral_resnet10(inputs, width=1.0):
    return resnet_imagenet(inputs, 10, width)


@register_model("spectral_resnet18")
def spectral_resnet18(inputs, width=1.0):
    return resnet_imagenet(inputs, 18, width)


@register_model("spectral_resnet34")
def spectral_resnet34(inputs, width=1.0):
    return resnet_imagenet(inputs, 34, width)

