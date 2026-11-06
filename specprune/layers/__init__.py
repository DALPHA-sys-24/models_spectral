"""Couches de convolution spectrale (SpecCnn1D, SpecCnn2D) pour TensorFlow/Keras.

Ancien paquet spectralcnn ; le code des couches garde ses docstrings d'origine (en anglais).

Public API
----------
    from specprune.layers import SpecCnn1D, SpecCnn2D
    from specprune.layers import set_fast_path   # False : formulation spectrale partout

Une couche dont lambda_in, lambda_out et alpha sont figés à (1, 0, 1) est calculée comme une
convolution classique (fast.py) : mêmes résultats, beaucoup moins de mémoire et de temps.
"""

from .speccnn1d import SpecCnn1D
from .speccnn2d import SpecCnn2D
from .fast import fast_path_enabled, refresh_classic, set_fast_path

__all__ = ["SpecCnn1D", "SpecCnn2D", "fast_path_enabled", "refresh_classic", "set_fast_path"]
__version__ = "0.2.0"  # 0.2.0: groups argument (grouped / depthwise convolution)
