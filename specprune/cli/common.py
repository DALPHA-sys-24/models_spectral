"""Options et utilitaires communs aux scripts (appareil, dossiers, configuration spectrale)."""
import argparse
import os
import sys

from .. import CONFIGS_DIR

DEFAULT_SPECTRAL_CONFIG = os.path.join(CONFIGS_DIR, "spectral_config.json")


class Formatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Aide avec la docstring du script telle quelle et les valeurs par défaut des options."""

    def _get_help_string(self, action):
        text = action.help or ""
        if "défaut" in text or action.default in (None, False, argparse.SUPPRESS) or not action.option_strings:
            return text
        return f"{text} (défaut : %(default)s)" if text else "défaut : %(default)s"


def parser(doc: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=doc, formatter_class=Formatter)


def add_device(g, default: str = "/GPU:0") -> None:
    g.add_argument("--device", default=default, help='"/GPU:0", "/GPU:1", "/CPU:0"...')


def add_verbose(g, default: int = 2) -> None:
    g.add_argument("--verbose", type=int, default=default, choices=[0, 1, 2],
                   help="affichage de Keras pendant l'entraînement")


def resolve_spectral_config(path: str) -> str:
    """Chemin de la configuration spectrale : tel quel, sinon dans configs/ (vérifié avant tout calcul)."""
    if not path or os.path.exists(path):
        return path
    candidate = os.path.join(CONFIGS_DIR, os.path.basename(path))
    if os.path.exists(candidate):
        print(f"--spectral_config {path} introuvable ici : {candidate} utilisé")
        return candidate
    sys.exit(f"--spectral_config {path} introuvable (ni depuis {os.getcwd()}, ni dans configs/). "
             f"Configurations disponibles : {sorted(os.listdir(CONFIGS_DIR))}")


def dataset_tag(name: str) -> str:
    """Nom de dataset utilisable dans un nom de dossier (tfds:x/y -> x_y, ucr:Nom -> ucr_Nom)."""
    return name.replace("tfds:", "").replace(":", "_").replace("/", "_")
