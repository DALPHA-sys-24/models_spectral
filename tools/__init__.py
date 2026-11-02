"""
Outils du projet, utilisés par les scripts run_experiment.py, search_hparams.py et pruning.py.

    tools.data      chargement des datasets (images et signaux), normalisation, tf.data
    tools.training  appareil, optimiseur et compilation (recette d'entraînement)
    tools.runs      sauvegarde d'un run et rechargement (load_run, load_run_data)
    tools.prune     pruning d'ordre p des poids spectraux, gel des lambda pour le post-training
"""
