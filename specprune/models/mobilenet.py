"""
MobileNetV1 (Howard et al. 2017) et MobileNetV2 (Sandler et al. 2018) pour petites images
ou signaux.

Les premiers strides sont ramenés à 1 : avec les strides ImageNet, une image 32x32
serait réduite à 1x1 avant les derniers blocs.

Toutes les convolutions sont des SpecCnn2D (stem 3x3, 1x1, et depthwise 3x3 avec groups=C), ou
des SpecCnn1D de mêmes noyaux (3, 1, depthwise 3) quand l'entrée est un signal.
Avec lambda_in = 1, lambda_out = 0 et alpha = 1 figés, on retrouve le MobileNet classique.
"""
import tensorflow as tf

from .blocks import spec_conv_bn, spec_depthwise_bn
from .registry import register_model

L = tf.keras.layers

# (filtres de sortie, stride) de chaque bloc depthwise-separable
V1_BLOCKS = [
    (64, 1), (128, 1), (128, 1), (256, 2), (256, 1), (512, 2),
    (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2), (1024, 1),
]

# (expansion, filtres de sortie, répétitions, stride du premier bloc)
V2_BLOCKS = [
    (1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
    (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1),
]


def scaled(filters: int, width: float) -> int:
    """Nombre de filtres multiplié par width, arrondi au multiple de 8."""
    return max(8, int(filters * width + 4) // 8 * 8)


def mobilenet_v1(inputs, width: float = 1.0, conv=spec_conv_bn, depthwise=spec_depthwise_bn):
    x = conv(inputs, scaled(32, width), 3, 1)
    for filters, stride in V1_BLOCKS:
        x = depthwise(x, stride)
        x = conv(x, scaled(filters, width), 1, 1)
    return x


def inverted_residual(x, expansion, filters, stride, conv=spec_conv_bn, depthwise=spec_depthwise_bn):
    """Bloc MobileNetV2 : expansion 1x1, depthwise 3x3, projection 1x1 linéaire."""
    out = x
    if expansion != 1:
        out = conv(out, x.shape[-1] * expansion, 1, 1, activation="relu6")
    out = depthwise(out, stride, activation="relu6")
    out = conv(out, filters, 1, 1, activation=None)
    if stride == 1 and x.shape[-1] == filters:
        out = L.Add()([x, out])
    return out


def mobilenet_v2(inputs, width: float = 1.0, conv=spec_conv_bn, depthwise=spec_depthwise_bn):
    x = conv(inputs, scaled(32, width), 3, 1, activation="relu6")
    for expansion, filters, repeats, stride in V2_BLOCKS:
        for i in range(repeats):
            x = inverted_residual(x, expansion, scaled(filters, width), stride if i == 0 else 1,
                                  conv, depthwise)
    return conv(x, scaled(1280, max(1.0, width)), 1, 1, activation="relu6")


# width multiplie la largeur nominale du modèle (AgentNetwork passe toujours width=...).
@register_model("spectral_mobilenetv1")
def spectral_mobilenetv1(inputs, width=1.0):
    return mobilenet_v1(inputs, width=width)


@register_model("spectral_mobilenetv2")
def spectral_mobilenetv2(inputs, width=1.0):
    return mobilenet_v2(inputs, width=width)


@register_model("spectral_mobilenetv2_0.5")
def spectral_mobilenetv2_half(inputs, width=1.0):
    return mobilenet_v2(inputs, width=0.5 * width)

