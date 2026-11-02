"""Réseau d'un agent : backbone choisi par son nom + tête de classification."""
from typing import Any, Dict

import tensorflow as tf

from .layers import start_backbone
from .registry import get_backbone

L = tf.keras.layers


class AgentNetwork(tf.keras.Model):
    """
    Agent individuel : backbone spectral enregistré (ResNet, MobileNet, ...) suivi de
    GlobalAveragePooling, Dropout optionnel, Dense et Softmax. Les entrées peuvent être des
    images (batch, H, W, C) ou des signaux (batch, T, C) : la dimension des convolutions et du
    pooling suit la forme de l'entrée.

    La sortie est un vecteur de probabilités en float32 (même en précision mixte),
    l'interface attendue par les losses de MultiAgentModel.

    Args:
        arch: Nom d'un modèle enregistré (voir models.available_models()).
        num_classes: Nombre de classes de sortie.
        dropout_rate: Taux de dropout avant la couche de sortie (0 = désactivé).
        width: Multiplicateur du nombre de filtres de chaque couche (1.0 = modèle standard).
        name: Nom du modèle.
    """

    def __init__(
        self,
        arch: str = "spectral_resnet20",
        num_classes: int = 10,
        dropout_rate: float = 0.0,
        width: float = 1.0,
        name: str = "agent",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self._arch = arch
        self._backbone_fn = get_backbone(arch)  # échoue tôt si le nom est inconnu
        self._num_classes = num_classes
        self._dropout_rate = dropout_rate
        self._width = width

        self._backbone = None  # construit dans build(), qui connaît la taille des images
        self._pool = None      # GlobalAveragePooling1D (signaux) ou 2D (images), choisi dans build()
        self._dropout = L.Dropout(dropout_rate) if dropout_rate > 0 else None
        self._logits = L.Dense(num_classes)
        self._softmax = L.Softmax(dtype="float32")

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=tuple(input_shape[1:]))
        # (batch, T, C) = signal -> convolutions SpecCnn1D ; (batch, H, W, C) = image -> SpecCnn2D
        self._pool = L.GlobalAveragePooling1D() if len(input_shape) == 3 else L.GlobalAveragePooling2D()
        start_backbone()  # numérotation des convolutions pour la configuration spectrale
        features = self._backbone_fn(inputs, width=self._width)
        self._backbone = tf.keras.Model(inputs, features, name=f"{self.name}_{self._arch}")
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False, return_logits: bool = False) -> tf.Tensor:
        """Probabilités (batch, num_classes), ou logits float32 si return_logits=True."""
        x = self._pool(self._backbone(x, training=training))
        if self._dropout is not None:
            x = self._dropout(x, training=training)
        logits = tf.cast(self._logits(x), tf.float32)
        return logits if return_logits else self._softmax(logits)

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arch": self._arch,
            "num_classes": self._num_classes,
            "dropout_rate": self._dropout_rate,
            "width": self._width,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AgentNetwork":
        return cls(**config)

    @property
    def arch(self) -> str:
        return self._arch

    @property
    def width(self) -> float:
        return self._width

    @property
    def num_classes(self) -> int:
        return self._num_classes
