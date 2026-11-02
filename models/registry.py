"""
Registre des architectures, indexées par un nom fixe.

Ajouter une architecture :
    1. écrire une fonction backbone(inputs) -> tenseur de caractéristiques (B, H, W, C) ;
    2. la décorer avec @register_model("mon_nom") ;
    3. importer son module dans models/__init__.py.
"""
from typing import Callable, Dict, List

import tensorflow as tf

BackboneFn = Callable[[tf.Tensor], tf.Tensor]

_REGISTRY: Dict[str, BackboneFn] = {}


def register_model(name: str) -> Callable[[BackboneFn], BackboneFn]:
    """Décorateur : enregistre une fonction backbone sous le nom donné."""
    def decorator(fn: BackboneFn) -> BackboneFn:
        if name in _REGISTRY:
            raise ValueError(f"Le modèle '{name}' est déjà enregistré.")
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_backbone(name: str) -> BackboneFn:
    """Renvoie la fonction backbone d'un modèle enregistré."""
    if name not in _REGISTRY:
        raise ValueError(f"Modèle inconnu : '{name}'. Modèles disponibles : {available_models()}")
    return _REGISTRY[name]


def available_models() -> List[str]:
    """Liste des noms de modèles enregistrés."""
    return sorted(_REGISTRY)
