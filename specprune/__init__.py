"""
specprune : modèles à convolutions spectrales, runs, pruning d'ordre p de lambda_in et figures.

    specprune.layers     couches SpecCnn1D / SpecCnn2D
    specprune.models     architectures (ResNet, MobileNet), configuration spectrale par couche
    specprune.data       datasets images et signaux
    specprune.training   appareil, optimiseur, compilation
    specprune.runs       enregistrement et rechargement d'un run
    specprune.checks     vérifications avant un lancement (configuration, runs comparables)
    specprune.prune      pruning d'ordre p, gel des poids spectraux (post-training)
    specprune.plotting   figures de publication
    specprune.cli        scripts en ligne de commande (train, search, prune, plot, bench)
"""
import os

__version__ = "1.0.0"

# Version du format des dossiers écrits par les scripts (config.json des runs et des prunings).
# 1 : format d'origine (sans numéro) ; 2 : numéro de version, reprise d'un entraînement (backup/).
FORMAT_VERSION = 2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
