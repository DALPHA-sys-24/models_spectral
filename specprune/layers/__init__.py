"""Couches de convolution spectrale (SpecCnn1D, SpecCnn2D) pour TensorFlow/Keras.

Ancien paquet spectralcnn ; le code des couches garde ses docstrings d'origine (en anglais).

Public API
----------
    from specprune.layers import SpecCnn1D, SpecCnn2D
    from specprune.layers.spectral_conv_op import spectral_kernel_1d, spectral_kernel_2d
"""

from .speccnn1d import SpecCnn1D
from .speccnn2d import SpecCnn2D

__all__ = ["SpecCnn1D", "SpecCnn2D"]
__version__ = "0.2.0"  # 0.2.0: groups argument (grouped / depthwise convolution)
