"""
Pruning des poids spectraux (lambda_in, lambda_out, alpha) des couches SpecCnn1D / SpecCnn2D.

Pruning d'ordre p : dans chaque groupe de valeurs |w|, on calcule le percentile d'ordre p et on met
à zéro ce qui est en deçà. p est une fraction : 0 = rien, 1 = tout.

    from tools.prune import prune_spectral, freeze_spectral
    report = prune_spectral(agent, p=0.5, method="element")   # lambda_in des couches entraînées
    freeze_spectral(agent, train="phi")                         # post-training : seuls les phi bougent

Méthodes (lambda_in a la forme (F filtres, C canaux, L positions)) :
    gradient  filtre par filtre : percentile sur tous les |w| du filtre (tous canaux et positions) ;
              chaque poids est retiré séparément, un pixel peut garder une partie de ses canaux
              (le masque montre une dégradation progressive)
    element   filtre par filtre : score d'un pixel = somme sur les canaux des |w| ; percentile sur
              ces scores ; un pixel en deçà est retiré pour tous ses canaux (RGB nul), pour ce filtre
    mixte     comme element, mais sur tous les filtres : score d'un pixel = somme sur les filtres et
              les canaux des |w| ; un pixel en deçà est retiré pour tous les filtres et canaux
Chaque couche est traitée séparément. Les positions de padding de lambda_in (qui multiplient
toujours des zéros) sont exclues du calcul et laissées telles quelles, sauf include_padding=True.

Critère : "percentile" (défaut) met à zéro les scores < percentile d'ordre p (à p = 1, tout est
retiré) ; "rank" retire exactement round(p * n) scores, les plus petits (identique sauf ex aequo).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import keras
import numpy as np

from spectralcnn import SpecCnn1D, SpecCnn2D

from .runs import spectral_layers, valid_positions

PARAMS = ("lambda_in", "lambda_out", "alpha")
METHODS = ("gradient", "element", "mixte")
CRITERIA = ("percentile", "rank")
LayerSpec = Union[str, int, Sequence[Union[str, int]]]


# --- Choix des couches ----------------------------------------------------------------

def _parse_indices(spec: str, n: int) -> List[int]:
    """'0', '1-5', '*', '0,3,5-7' -> indices de convolution (même syntaxe que spectral_config)."""
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if part == "*":
            out += list(range(n))
        elif "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        raise ValueError(f"indices de convolution hors limites {bad} (le modèle en a {n} : 0 à {n - 1})")
    return sorted(set(out))


def select_layers(agent, layers: LayerSpec = "trained", param: str = "lambda_in") -> list:
    """
    Couches spectrales visées.

    layers : "trained"  couches où `param` est entraîné (défaut : les autres ont des valeurs
                        constantes, 1 ou 0, qu'un classement par magnitude ne départage pas)
             "*", "0", "1-5", "0,3,5-7"   indices de convolution (comme dans spectral_config.json)
             int, nom de couche, ou liste d'indices / de noms
    """
    spec = spectral_layers(agent)
    if isinstance(layers, str) and layers == "trained":
        chosen = [l for l in spec if getattr(l, f"trainable_{param}")]
        if not chosen:
            raise ValueError(f"Aucune couche n'entraîne {param} dans ce modèle : préciser les couches "
                             "(ex. layers='0').")
        return chosen
    items = [layers] if isinstance(layers, (str, int)) else list(layers)
    by_name = {l.name: l for l in spec}
    chosen = []
    for item in items:
        if isinstance(item, str) and item in by_name:
            chosen.append(by_name[item])
        else:
            chosen += [spec[i] for i in _parse_indices(item, len(spec))]
    return list(dict.fromkeys(chosen))  # sans doublons, dans l'ordre


def _weight(layer, param: str):
    if param not in PARAMS:
        raise ValueError(f"param doit être parmi {PARAMS}, reçu '{param}'")
    return getattr(layer, param)


# --- Pruning --------------------------------------------------------------------------

@dataclass
class PruneReport:
    """Résultat d'un pruning : masques gardés (True = conservé) et comptes."""
    p: float
    param: str
    method: str
    criterion: str
    masks: Dict[str, np.ndarray] = field(default_factory=dict)   # nom de couche -> masque (forme du poids)
    pruned: int = 0          # poids mis à zéro (positions de padding exclues)
    total: int = 0           # poids classés
    counts: Dict[str, tuple] = field(default_factory=dict)       # nom de couche -> (retirés, classés)

    @property
    def fraction(self) -> float:
        return self.pruned / self.total if self.total else 0.0

    def per_layer(self) -> Dict[str, float]:
        """Fraction retirée dans chaque couche (positions de padding exclues)."""
        return {name: pruned / total if total else 0.0 for name, (pruned, total) in self.counts.items()}


def _drop(scores: np.ndarray, p: float, criterion: str) -> np.ndarray:
    """Scores (1D) à retirer pour l'ordre p : en deçà du percentile d'ordre p, ou les round(p*n) plus petits."""
    if p >= 1.0:
        return np.ones(scores.size, bool)
    if criterion == "rank":
        drop = np.zeros(scores.size, bool)
        drop[np.argsort(scores, kind="stable")[:int(round(p * scores.size))]] = True
        return drop
    return scores < np.percentile(scores, 100.0 * p)


def _layer_mask(w: np.ndarray, valid: np.ndarray, p: float, method: str, criterion: str) -> np.ndarray:
    """Masque gardé (True = conservé) d'une couche, de la forme de w (F, C, L)."""
    a, keep = np.abs(w), np.ones(w.shape, bool)
    if method == "mixte":
        ok = valid[0, 0]
        pix = np.ones(ok.shape, bool)
        pix[ok] = ~_drop(a.sum(axis=(0, 1))[ok], p, criterion)
        return np.broadcast_to(pix, w.shape).copy()
    for f in range(w.shape[0]):
        if method == "gradient":
            keep[f][valid[f]] = ~_drop(a[f][valid[f]], p, criterion)
        else:  # element : pixels du filtre f, tous canaux ensemble
            ok = valid[f, 0]
            pix = np.ones(ok.shape, bool)
            pix[ok] = ~_drop(a[f].sum(axis=0)[ok], p, criterion)
            keep[f] = pix[None]
    return keep


def compute_masks(agent, p: float, layers: LayerSpec = "trained", param: str = "lambda_in",
                  method: str = "element", criterion: str = "percentile",
                  include_padding: bool = False,
                  values: Optional[Dict[str, np.ndarray]] = None) -> PruneReport:
    """
    Calcule les masques du pruning d'ordre p sans modifier le modèle.

    values : valeurs à classer {nom de couche: poids} ; défaut = poids actuels du modèle
             (permet de classer des valeurs d'origine après un premier pruning, ou une médiane).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p doit être une fraction dans [0, 1], reçu {p}")
    if method not in METHODS:
        raise ValueError(f"method doit être parmi {METHODS}, reçu '{method}'")
    if criterion not in CRITERIA:
        raise ValueError(f"criterion doit être parmi {CRITERIA}, reçu '{criterion}'")

    report = PruneReport(p=p, param=param, method=method, criterion=criterion)
    for layer in select_layers(agent, layers, param):
        w = values[layer.name] if values is not None else _weight(layer, param).numpy()
        if w.ndim != 3:  # lambda_out, alpha : pas de structure (filtre, canal, position)
            w = w.reshape(w.shape[0] if w.ndim else 1, 1, -1)
        valid = (valid_positions(layer) if param == "lambda_in" and not include_padding
                 else np.ones(w.shape, bool))
        mask = _layer_mask(w, valid, p, method, criterion)
        shape = tuple(_weight(layer, param).shape)
        report.masks[layer.name] = mask.reshape(shape)
        n_total, n_pruned = int(valid.sum()), int((~mask & valid).sum())
        report.counts[layer.name] = (n_pruned, n_total)
        report.total += n_total
        report.pruned += n_pruned
    return report


def apply_masks(agent, masks: Dict[str, np.ndarray], param: str = "lambda_in",
                values: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Met à zéro les poids hors masque (sur `values` si donné, sinon sur les poids actuels)."""
    by_name = {l.name: l for l in spectral_layers(agent)}
    for name, mask in masks.items():
        var = _weight(by_name[name], param)
        w = values[name] if values is not None else var.numpy()
        var.assign(np.where(mask, w, 0.0).astype(np.float32))


def prune_spectral(agent, p: float, layers: LayerSpec = "trained", param: str = "lambda_in",
                   method: str = "element", criterion: str = "percentile",
                   include_padding: bool = False) -> PruneReport:
    """Pruning d'ordre p appliqué au modèle (en place). Renvoie les masques et les comptes."""
    report = compute_masks(agent, p, layers, param, method, criterion, include_padding)
    apply_masks(agent, report.masks, param)
    return report


# --- Valeurs spectrales : lire, réassigner, figer ------------------------------------

def get_spectral_values(agent, param: str = "lambda_in", layers: LayerSpec = "*") -> Dict[str, np.ndarray]:
    """Copie des valeurs d'un poids spectral {nom de couche: tableau}."""
    return {l.name: _weight(l, param).numpy().copy() for l in select_layers(agent, layers, param)}


def assign_spectral_weights(agent, weights: Dict[str, np.ndarray], param: str = "lambda_in",
                            verbose: bool = True) -> int:
    """
    Assigne des valeurs calculées ailleurs {nom de couche: tableau} (ex. median_lambda_in de
    tools.runs, pour partir de la médiane de plusieurs runs). Les noms inconnus sont ignorés.
    Renvoie le nombre de couches modifiées.
    """
    n = 0
    for layer in spectral_layers(agent):
        if layer.name not in weights:
            continue
        var, arr = _weight(layer, param), np.asarray(weights[layer.name], np.float32)
        if arr.shape != tuple(var.shape):
            raise ValueError(f"{layer.name}.{param} : forme {arr.shape} au lieu de {tuple(var.shape)}")
        var.assign(arr)
        n += 1
        if verbose:
            print(f"  {layer.name}.{param} {arr.shape} assigné")
    if verbose and n == 0:
        print("  Aucune couche assignée : vérifier les noms de couches.")
    return n


def freeze_spectral(agent, train: str = "phi") -> Dict[str, int]:
    """
    Fige lambda_in, lambda_out et alpha de toutes les couches spectrales (post-training).

    train : "phi"           seuls les filtres phi sont entraînés (gamma/beta des BatchNorm et tête
                            figés ; les BatchNorm normalisent avec les statistiques du batch)
            "non_spectral"  phi + BatchNorm + tête Dense ; seuls les poids spectraux sont figés
    Les filtres phi sont rendus entraînables même s'ils étaient figés pendant l'entraînement.
    À appeler avant compile(). Renvoie le nombre de paramètres entraînables / figés.
    """
    if train not in ("phi", "non_spectral"):
        raise ValueError(f"train doit être 'phi' ou 'non_spectral', reçu '{train}'")
    spec = spectral_layers(agent)
    if train == "phi":
        # Dense, etc. : figés. BatchNorm : gamma et beta figés, mais la couche reste en mode
        # entraînement (normalisation par les statistiques du batch, moyennes mobiles mises à
        # jour). Une BatchNorm entièrement figée (layer.trainable = False) normalise avec ses
        # statistiques d'origine : quand les phi changent, les activations ne sont plus
        # renormalisées et le réentraînement diverge (perte nan, précision au hasard).
        for layer in agent._backbone.layers:
            if isinstance(layer, (SpecCnn1D, SpecCnn2D)) or not layer.weights:
                continue
            if isinstance(layer, keras.layers.BatchNormalization):
                for var in (layer.gamma, layer.beta):
                    if var is not None:
                        var.trainable = False
            else:
                layer.trainable = False
        for layer in (agent._logits,):
            layer.trainable = False
    for layer in spec:
        layer.phi_weights.trainable = True
        for name in PARAMS:
            getattr(layer, name).trainable = False
        if layer.use_bias and train == "phi":
            layer.bias.trainable = False
    trainable = int(sum(np.prod(v.shape) for v in agent.trainable_variables))
    frozen = int(sum(np.prod(v.shape) for v in agent.non_trainable_variables))
    return {"trainable": trainable, "frozen": frozen}


# --- Pré-training : pruning puis prédiction directe ----------------------------------

def masks_for(agent, p_values: Sequence[float], layers: LayerSpec = "trained",
              param: str = "lambda_in", method: str = "element", criterion: str = "percentile",
              include_padding: bool = False):
    """
    Valeurs d'origine des couches visées et masques de chaque p, classés sur ces valeurs.

    Returns:
        (original, reports) : {nom de couche: valeurs}, {p: PruneReport}
    """
    names = [l.name for l in select_layers(agent, layers, param)]
    original = get_spectral_values(agent, param, names)
    reports = {p: compute_masks(agent, p, names, param, method, criterion, include_padding,
                                values=original) for p in p_values}
    return original, reports


def evaluate_pruned(agent, report: PruneReport, original: Dict[str, np.ndarray], test_ds):
    """
    Pré-training : applique les masques de `report` aux valeurs d'origine puis évalue le modèle
    (compilé) sur test_ds. Renvoie (précision, perte). Le modèle reste élagué : rappeler avec
    un autre report, ou assign_spectral_weights(agent, original) pour revenir au modèle entraîné.
    """
    apply_masks(agent, report.masks, report.param, values=original)
    loss, acc = agent.evaluate(test_ds, verbose=0)
    return float(acc), float(loss)
