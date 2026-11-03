"""
Runs : ce que run_experiment.py enregistre, et comment le relire plus tard.

Un run est un dossier runs/<dataset>_<arch>_<date>/ (voir run_experiment.py). Pour le réutiliser
(pruning, analyse, prédiction), on ne relit pas un fichier .keras : le modèle est reconstruit
depuis config.json et la copie de spectral_config.json, puis ses poids sont rechargés depuis
weights.weights.h5. C'est plus robuste qu'un modèle sérialisé, qui dépendrait de la
configuration spectrale globale au moment du chargement.

    from specprune.runs import load_run, load_run_data
    agent, config = load_run("runs/cifar10_spectral_resnet20_20261030_101500")
    train, val, test, info = load_run_data(config)
"""
import csv
import json
import os
import platform
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from .models import AgentNetwork, configure_spectral
from .layers import SpecCnn1D, SpecCnn2D

from .checks import check_format
from .data import load_dataset

WEIGHTS_FILE = "weights.weights.h5"


# --- Couches spectrales d'un modèle ----------------------------------------------

def spectral_layers(agent) -> list:
    """Couches spectrales (SpecCnn1D / SpecCnn2D) dans l'ordre de construction (indice = conv_index)."""
    return [l for l in agent._backbone.layers if isinstance(l, (SpecCnn1D, SpecCnn2D))]


def lambda_in_layers(agent) -> list:
    """Couches spectrales dont lambda_in est entraîné (les autres valent 1 en permanence)."""
    return [l for l in spectral_layers(agent) if l.trainable_lambda_in]


def valid_positions(layer) -> np.ndarray:
    """
    Masque booléen des positions de lambda_in qui voient l'entrée (et non le padding).

    lambda_in couvre l'entrée avec son padding : (F, C, Lh*Lw) en 2D, (F, C, L) en 1D. Les
    positions de padding multiplient toujours des zéros : elles ne comptent pas.
    Renvoie un masque de la forme de lambda_in.
    """
    if isinstance(layer, SpecCnn1D):
        grid = np.zeros(layer._L, bool)
        grid[layer._P:layer._P + layer._L_raw] = True
    else:
        grid = np.zeros((layer._Lh, layer._Lw), bool)
        grid[layer._Ph:layer._Ph + layer._H, layer._Pw:layer._Pw + layer._W] = True
        grid = grid.ravel()
    return np.broadcast_to(grid, tuple(layer.lambda_in.shape)).copy()


def lambda_in_file(layer, conv_index: int) -> Dict:
    """Description d'un fichier lambda_in, pour le relire (notebook d'analyse)."""
    if isinstance(layer, SpecCnn1D):
        grid, padding, size = [layer._L], [layer._P], [layer._L_raw]
    else:
        grid, padding, size = [layer._Lh, layer._Lw], [layer._Ph, layer._Pw], [layer._H, layer._W]
    return {"layer": layer.name, "conv_index": conv_index, "dim": len(grid),
            "file": f"{layer.name}_lambda_in.npy", "shape": list(layer.lambda_in.shape),
            "padded_grid": grid, "padding": padding, "input_hw": size, "groups": layer.groups}


# --- Sauvegardes pendant un run ----------------------------------------------------

def save_lambda_in(agent, folder: str) -> None:
    """Enregistre les poids bruts lambda_in, (F, C/groups, Lh*Lw) ou (F, C/groups, L), sans traitement."""
    os.makedirs(folder, exist_ok=True)
    for layer in lambda_in_layers(agent):
        np.save(os.path.join(folder, f"{layer.name}_lambda_in.npy"), layer.lambda_in.numpy())


class LambdaInSnapshot(tf.keras.callbacks.Callback):
    """Enregistre lambda_in toutes les `every` epochs dans spectral/epoch_XXX/."""

    def __init__(self, agent, folder, every):
        super().__init__()
        self.agent, self.folder, self.every = agent, folder, every

    def on_epoch_end(self, epoch, logs=None):
        if self.every and (epoch + 1) % self.every == 0:
            save_lambda_in(self.agent, os.path.join(self.folder, f"epoch_{epoch + 1:03d}"))


def environment_info(gpus) -> Dict:
    """Versions et matériel, pour pouvoir reproduire le run."""
    build = tf.sysconfig.get_build_info()
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__,
        "cuda": build.get("cuda_version", ""), "cudnn": build.get("cudnn_version", ""),
        "gpus": [tf.config.experimental.get_device_details(g).get("device_name", g.name)
                 for g in gpus],
    }


def append_summary(path: str, row: Dict) -> None:
    """Ajoute une ligne au tableau récapitulatif (l'ancien est renommé si les colonnes changent)."""
    if os.path.exists(path):
        with open(path) as f:
            header = f.readline().strip().split(",")
        if header != list(row):
            os.replace(path, path.replace(".csv", f"_ancien_{datetime.now():%Y%m%d_%H%M%S}.csv"))
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            w.writeheader()
        w.writerow(row)


# --- Relire un run -----------------------------------------------------------------

def read_config(run_dir: str) -> Dict:
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{run_dir} : pas de config.json (ce n'est pas un dossier de run)")
    with open(path) as f:
        config = json.load(f)
    if not isinstance(config.get("dataset"), dict) or "lambda_in_files" not in config:
        raise ValueError(f"{run_dir} a été produit par une ancienne version de run_experiment.py : "
                         "relancer l'entraînement avec la version actuelle.")
    check_format(config, run_dir)
    return config


def run_spectral_config(run_dir: str, config: Dict) -> Optional[str]:
    """Configuration spectrale du run : la copie du dossier, sinon le fichier d'origine."""
    copy_path = os.path.join(run_dir, "spectral_config.json")
    return copy_path if os.path.exists(copy_path) else (config.get("spectral_config_file") or None)


def build_run_model(config: Dict, spectral_config: Optional[str]):
    """Reconstruit le modèle du run (architecture et réglages spectraux), sans ses poids."""
    args = config["args"]
    configure_spectral(spectral_config)
    agent = AgentNetwork(arch=args["arch"], num_classes=config["dataset"]["num_classes"],
                         dropout_rate=args.get("dropout", 0.0), width=args.get("width", 1.0),
                         name="agent")
    agent(tf.zeros([1] + list(config["dataset"]["input_shape"])))
    return agent


def load_run(run_dir: str, weights: bool = True) -> Tuple[object, Dict]:
    """
    Recharge le modèle entraîné d'un run.

    Returns:
        (agent, config) : AgentNetwork avec les poids de fin d'entraînement, et config.json.
    """
    config = read_config(run_dir)
    agent = build_run_model(config, run_spectral_config(run_dir, config))
    if weights:
        path = os.path.join(run_dir, WEIGHTS_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} introuvable : le run a été lancé avec --no-save_weights")
        agent.load_weights(path)
    return agent, config


def load_run_data(config: Dict, check: bool = True):
    """
    Recharge les données du run, avec les mêmes sous-ensembles et la même normalisation.

    data_path est relatif au dossier d'où le run a été lancé : s'il n'existe pas depuis le dossier
    courant, on utilise le chemin absolu enregistré (dataset.source).
    """
    args, saved = config["args"], config["dataset"]
    data_path = args.get("data_path") or None
    if data_path and not os.path.exists(data_path) and os.path.exists(saved.get("source", "")):
        data_path = saved["source"]
    train, val, test, info = load_dataset(
        args["dataset"], data_path, args.get("augment", "auto"), args.get("train_subset", 0),
        args.get("test_subset", 0), args.get("val_split", 0.0), args.get("image_size", 0),
        args.get("seed", 42), args.get("modality", "auto"))
    if check and not np.allclose(info.mean, saved["mean"], atol=1e-4):
        raise ValueError("Les données rechargées n'ont pas la normalisation du run : "
                         "le dataset a changé depuis l'entraînement.")
    return train, val, test, info


def test_dataset(test, info, run_args) -> tf.data.Dataset:
    """Jeu de test tf.data d'un run (batch 512 ; étiquettes one-hot si le run utilisait du label smoothing)."""
    ds = tf.data.Dataset.from_tensor_slices(test).batch(512).prefetch(tf.data.AUTOTUNE)
    if run_args.get("label_smoothing", 0.0):
        ds = ds.map(lambda x, y: (x, tf.one_hot(y, info.num_classes)))
    return ds


def median_lambda_in(run_dirs: List[str], snapshot: str = "final") -> Dict[str, np.ndarray]:
    """
    Médiane élément par élément des lambda_in de plusieurs runs (ex. plusieurs graines).

    Les runs doivent avoir la même architecture et la même configuration spectrale.
    Renvoie {nom de couche: médiane (F, C/groups, L)} ; à donner à assign_spectral_weights
    (specprune.prune) pour partir de ces valeurs.
    """
    stacks: Dict[str, list] = {}
    for run_dir in run_dirs:
        config = read_config(run_dir)
        folder = os.path.join(run_dir, "spectral", "" if snapshot == "final" else snapshot)
        for meta in config["lambda_in_files"]:
            stacks.setdefault(meta["layer"], []).append(np.load(os.path.join(folder, meta["file"])))
    return {name: np.median(np.stack(arrays), axis=0) for name, arrays in stacks.items()}
