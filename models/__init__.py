"""
Architectures des agents.

    from models import AgentNetwork, available_models, configure_spectral
    configure_spectral("configs/spectral_config.json")   # poids entraînés, couche par couche
    agent = AgentNetwork(arch="spectral_resnet32", num_classes=10)

Chaque famille d'architectures a son module (resnet.py, mobilenet.py). Pour en
ajouter une, créer un module qui enregistre ses backbones avec @register_model,
puis l'importer ci-dessous.

Toutes les convolutions sont des SpecCnn2D pour des images (batch, H, W, C), des SpecCnn1D pour
des signaux (batch, T, C) (paquet spectralcnn, à placer à côté de models/ ou à installer) : les
mêmes architectures servent aux deux. configure_spectral() dit, couche par couche, quels poids elles entraînent ;
configs/classique.json (lambda_in = 1, lambda_out = 0, alpha = 1 figés) redonne les modèles
classiques.
"""
from .registry import available_models, get_backbone, register_model
from . import resnet, mobilenet  # noqa: F401  (enregistre les architectures)
from .agent import AgentNetwork
from .layers import configure_spectral, get_spectral_config, spectral_plan

__all__ = ["AgentNetwork", "available_models", "configure_spectral", "get_backbone",
           "get_spectral_config", "register_model", "spectral_plan"]
