"""
Vérifications faites avant un lancement, pour ne pas découvrir un problème après des heures de
calcul : format des dossiers, montage spectral incohérent avec le pruning, runs non comparables.
"""
import json
from typing import Dict, List, Sequence

from . import FORMAT_VERSION


def check_format(config: Dict, where: str) -> int:
    """Version du format d'un config.json (1 si absente). Erreur si le dossier vient d'une version plus récente."""
    version = int(config.get("format_version", 1))
    if version > FORMAT_VERSION:
        raise ValueError(f"{where} a été écrit par une version plus récente de specprune (format "
                         f"{version} > {FORMAT_VERSION}) : mettre le code à jour.")
    return version


def spectral_warnings(agent) -> List[str]:
    """
    Montages spectraux qui faussent l'interprétation du pruning de lambda_in.

    Le poids effectif est W = alpha * phi * (lambda_in - lambda_out) : si lambda_out est entraîné
    dans une couche où lambda_in l'est aussi, cette couche voit encore l'entrée quand tous ses
    lambda_in sont à zéro (W = -alpha * phi * lambda_out), et la précision à p = 1 ne tombe pas
    au hasard.
    """
    from .runs import spectral_layers
    both = [l.name for l in spectral_layers(agent) if l.trainable_lambda_in and l.trainable_lambda_out]
    if not both:
        return []
    return [f"lambda_out est entraîné dans {both}, où lambda_in l'est aussi : à p = 1, ces couches "
            "voient encore l'entrée par -alpha*phi*lambda_out. Pour mesurer le seul effet de "
            "lambda_in, mettre \"trainable_lambda_out\": false dans la configuration spectrale."]


# Ce qui doit être identique pour regrouper des runs dans un même pruning (médiane, quantiles)
_SAME = {
    "dataset": lambda c: c["dataset"]["name"],
    "input_shape": lambda c: c["dataset"]["input_shape"],
    "arch": lambda c: c["args"]["arch"],
    "width": lambda c: c["args"].get("width", 1.0),
    "spectral_config": lambda c: json.dumps(c.get("spectral_config"), sort_keys=True),
    "lambda_in": lambda c: [(m["layer"], m["shape"]) for m in c.get("lambda_in_files", [])],
}


def runs_differences(configs: Sequence[Dict], names: Sequence[str]) -> List[str]:
    """Différences entre runs qui empêchent de les agréger (vide si les runs sont comparables)."""
    out = []
    for key, get in _SAME.items():
        values = {}
        for name, config in zip(names, configs):
            values.setdefault(str(get(config)), []).append(name)
        if len(values) > 1:
            detail = " ; ".join(f"{v[:60]} : {', '.join(n)}" for v, n in values.items())
            out.append(f"{key} différent selon les runs ({detail})")
    return out
