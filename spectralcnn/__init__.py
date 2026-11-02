"""spectralcnn — Spectral Convolutional Neural Network layers for TensorFlow/Keras.

Public API
----------
    from spectralcnn import SpecCnn1D, SpecCnn2D
    from spectralcnn.spectral_conv_op import spectral_kernel_1d, spectral_kernel_2d
"""

from .speccnn1d import SpecCnn1D
from .speccnn2d import SpecCnn2D

__all__ = ["SpecCnn1D", "SpecCnn2D"]
__version__ = "0.2.0"  # 0.2.0: groups argument (grouped / depthwise convolution)
