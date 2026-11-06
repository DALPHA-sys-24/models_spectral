# Modèles spectraux et pruning de lambda_in

Modèles de référence (ResNet, MobileNet) dont toutes les convolutions sont spectrales :
`SpecCnn2D` pour des images, `SpecCnn1D` pour des signaux (choix automatique d'après les données).
Un fichier de configuration dit, couche par couche, quels poids sont entraînés (phi, lambda_in,
lambda_out, alpha). Après l'entraînement, les poids lambda_in servent au pruning d'ordre p.

## Installation

```bash
python3 -m venv ~/venvs/spectral && source ~/venvs/spectral/bin/activate
pip install -r requirements_gpu.txt    # GPU (requirements_cpu.txt sans GPU)
pip install -e .                       # le paquet specprune et les commandes specprune-*
```

`pip install -e .` est facultatif : les scripts de la racine marchent sans, lancés depuis
`models_spectral/`. Avec, on peut écrire `specprune-train ...` (= `python run_experiment.py ...`)
de n'importe quel dossier, et `import specprune` dans tout script ou notebook.

## Arborescence

```
models_spectral/
├── run_experiment.py        entraîne un ou plusieurs runs           (= specprune-train)
├── search_hparams.py        cherche lr et batch                     (= specprune-search)
├── pruning.py               pruning d'ordre p : pré et/ou post      (= specprune-prune)
├── plot_results.py          figures de publication                  (= specprune-plot)
├── bench_memory.py          mémoire des couches spectrales          (= specprune-bench)
├── plan_experiments.py      série découpée en tâches (Slurm, local) (= specprune-plan)
├── specprune/               le paquet (les scripts ci-dessus n'en sont que des raccourcis)
│   ├── layers/                  couches SpecCnn1D et SpecCnn2D
│   ├── models/                  architectures (registre, ResNet, MobileNet, AgentNetwork)
│   ├── data.py                  datasets images et signaux, normalisation, tf.data
│   ├── training.py              appareil, optimiseur, compilation (recette d'entraînement)
│   ├── runs.py                  sauvegarde d'un run et rechargement (load_run, load_run_data)
│   ├── checks.py                vérifications avant un lancement
│   ├── prune.py                 pruning d'ordre p, évaluation (pre), gel des lambda (post)
│   ├── plotting.py              figures style Nature Communications (sans titre, en anglais)
│   └── cli/                     code des commandes (train, search, prune, plot, bench, common)
├── configs/
│   ├── spectral_config.json     défaut : couche 0 = phi + lambda_in, les autres = Conv classiques
│   ├── classique.json           tous les lambda figés : modèle classique (seuls les phi)
│   └── all_spectral.json        lambda_in entraîné dans toutes les couches
├── experiments/
│   ├── serie.sh                 série complète sur une machine, étape par étape
│   ├── slurm/                   train.sh, prune.sh, finish.sh : tableaux Slurm écrits à la main
│   └── exemple.json             série pour plan_experiments.py (plusieurs datasets, cluster)
├── notebooks/
│   ├── analyse_lambda_in.ipynb  pruning d'ordre p en images : masques, exemples, précision (pre)
│   └── make_notebook.py         génère le notebook (le modifier ici, pas dans le .ipynb)
├── tests/                   pytest : couches, modèles, pruning, chaîne complète
├── pyproject.toml           paquet, dépendances, commandes, réglages de pytest
├── requirements_gpu.txt     GPU (WSL / Linux)
└── requirements_cpu.txt     CPU (ex. Windows pour les notebooks)
```

Les résultats vont dans `runs/`, `runs_search/` et `runs_pruning/` (ignorés par git).

## Déroulement

### 1. Entraîner

```bash
python run_experiment.py --dataset cifar10 --arch spectral_resnet20 --spectral_config configs/spectral_config.json
python run_experiment.py --dataset mnist1d --arch spectral_resnet20          # signaux téléchargés
python run_experiment.py --dataset npz --data_path signaux.npz --modality signal
```

Chaque run crée `runs/<dataset>_<arch>_<date>/` : `config.json` (tout ce qu'il faut pour
rejouer le run), `spectral_config.json` (copie), `weights.weights.h5` (poids finaux),
`spectral/<couche>_lambda_in.npy` (lambda_in bruts en fin d'entraînement, et `epoch_000/` avant),
`history.csv`, `metrics.json`. Rejouer : `python run_experiment.py --from_config runs/<run>/config.json`.

Plusieurs runs identiques (graines 42, 43, ...) avec `--n_runs` :

```bash
python run_experiment.py --dataset cifar10 --lr 0.1 --batch_size 128 --epochs 20 --n_runs 5 --run_name cifar10_5seeds
# -> runs/cifar10_5seeds/seed42/ ... seed46/
python pruning.py runs/cifar10_5seeds/seed* --mode pre
```

Un run interrompu (coupure, Ctrl-C, machine arrêtée) reprend à la dernière epoch terminée : un
point de reprise est écrit à chaque epoch dans `<run>/backup/` (effacé à la fin du run).

```bash
python run_experiment.py --resume runs/<run>              # un run
python run_experiment.py --resume runs/cifar10_5seeds     # un groupe --n_runs : runs non terminés
```

Recherche de lr et batch (20 epochs) : `python search_hparams.py --arch spectral_resnet20 --dataset cifar10`.

### 2. Pruning d'ordre p

```bash
# pré-training : pruning des lambda_in du modèle entraîné, prédiction directe sur le test
python pruning.py runs/<run>

# post-training : pruning, lambda figés, réentraînement des seuls phi, puis prédiction
python pruning.py runs/<run> --mode post --epochs 20

# les deux, sur plusieurs runs (graines) : médiane et bande 5 %-95 % par p
python pruning.py runs/cifar10_spectral_resnet20_* --mode both
```

Pour chaque p (par défaut 0, 0.1, ..., 1), les scores |lambda_in| en deçà du percentile d'ordre p
sont mis à zéro, dans chaque couche où lambda_in est entraîné ; chaque p repart du modèle entraîné.
Trois méthodes (`--method`) :

- `gradient` : filtre par filtre, percentile sur tous les |w| du filtre (tous canaux et positions).
  Chaque poids part séparément : un pixel peut garder une partie de ses canaux.
- `element` (défaut) : filtre par filtre, score d'un pixel = somme des |w| sur les canaux ; un pixel
  en deçà du percentile est retiré pour tous ses canaux (RGB nul), pour ce filtre.
- `mixte` : comme `element`, mais le score d'un pixel est sommé sur tous les filtres ; un pixel en
  deçà est retiré pour tous les filtres et tous les canaux.

Options principales :

| option | valeurs | rôle |
|---|---|---|
| `--mode` | `pre`, `post`, `both` | pré-training, post-training, ou les deux avec les mêmes masques |
| `--p` | fractions | ordres p testés |
| `--layers` | `trained`, `*`, `0`, `1-5`, `0,3` | couches élaguées (indices comme dans spectral_config) |
| `--method` | `gradient`, `element`, `mixte` | voir ci-dessus |
| `--criterion` | `percentile`, `rank` | scores < percentile d'ordre p, ou exactement round(p·n) retirés |
| `--train` | `phi`, `non_spectral` | post : seuls les phi, ou phi + BatchNorm + tête |
| `--epochs`, `--lr`, `--repeats` | | post : 20 epochs, lr du run, réentraînements par p |

Sorties dans `runs_pruning/<dataset>_<arch>_<mode>_<date>/` : `results.csv` (une ligne par run,
mode, p et répétition), `summary.csv` (médiane, quantiles 5 % et 95 % par mode et p),
les masques de chaque p et l'historique de chaque réentraînement.
`--resume <dossier>` reprend un pruning interrompu.

Plusieurs GPU : `--gpus 0 1` lance un processus par GPU sur le même dossier ; les couples
(run, p, répétition) sont répartis entre eux, chacun écrit `results_part<k>.csv` et son journal
`log_gpu<k>.txt`, et le résumé est fait sur l'ensemble à la fin.

```bash
python pruning.py runs/cifar10_5seeds/seed* --mode post --gpus 0 1
```

Vérifications avant le calcul : les runs donnés ensemble doivent être comparables (même dataset,
forme d'entrée, architecture, largeur et configuration spectrale), sinon le script s'arrête au
lieu de mélanger leurs résultats dans la médiane (`--allow_mixed` pour passer outre). Si
`lambda_out` est entraîné là où `lambda_in` l'est, un avertissement le signale : la précision à
p = 1 ne tombera pas au hasard (W = −alpha·phi·lambda_out reste).

### 3. Figures

Les scripts enregistrent les données ; les figures se font à part avec `plot_results.py`, sans
recalcul. Style Nature Communications : pas de titre, axes en anglais, 7 pt, une colonne (89 mm),
couleurs orange, bleu puis vert, export PDF (vectoriel) et PNG 600 dpi.

```bash
python plot_results.py pruning runs_pruning/<dossier>                       # pre et post selon p
python plot_results.py pruning runs_pruning/A runs_pruning/B runs_pruning/C --modes post \
       --labels Gradient Element Mixed --chance 0.1 --out figures/methods   # comparer des méthodes
python plot_results.py history runs/cifar10_5seeds/seed*                    # entraînement (plusieurs runs : médiane + bande)
python plot_results.py masks runs_pruning/<dossier> --p 0.2 0.5 0.8         # positions gardées
```

Tout se règle en option : `--xlabel`, `--ylabel`, `--labels`, `--colors`, `--width_mm` (183 =
deux colonnes), `--height_mm`, `--font`, `--font_size`, `--line_width`, `--no_markers`,
`--legend`, `--xlim`, `--ylim`, `--stat`, `--band`, `--individual`, `--formats`, `--dpi`, `--out`
(`python plot_results.py pruning -h`). Sans `--out`, la figure va dans `<dossier>/figures/`.

### 4. Série complète

`experiments/serie.sh` enchaîne tout (entraînement de N graines, pruning pré et post, figures) et
reprend là où il s'est arrêté si on le relance. Les réglages se changent par variables :

```bash
bash experiments/serie.sh                                         # cifar10, 5 runs, 20 epochs
DATASET=mnist N_RUNS=3 EPOCHS=30 LR=0.05 bash experiments/serie.sh
GPUS="0 1" bash experiments/serie.sh                              # pruning sur deux GPU
```

### 5. Analyser

`notebooks/analyse_lambda_in.ipynb` (à lancer depuis `notebooks/`) : version visuelle de
`pruning.py --mode pre`, avec les mêmes fonctions de `specprune` et les mêmes options (`P_VALUES`,
`PARAM`, `LAYERS`, `METHOD`, `CRITERION`) : valeurs brutes de lambda_in, importance de chaque
entrée (pixel ou instant), masques de chaque p sur la carte et sur des exemples, précision du
modèle élagué. Pour les mêmes réglages, il donne les mêmes précisions que `pruning.py`.

## Configuration spectrale

Les convolutions sont numérotées dans l'ordre de construction (0 = première couche). Une règle
s'applique par défaut (`default`), à toutes (`"*"`), à une couche (`"0"`) ou à un intervalle
(`"1-5"`) :

```json
{
  "default": {"trainable_phi": true, "trainable_lambda_in": false,
              "trainable_lambda_out": false, "trainable_alpha": false},
  "layers": {"0": {"trainable_lambda_in": true, "lambda_in_l1": 1e-2}}
}
```

Un poids non entraîné garde sa valeur neutre (lambda_in = 1, lambda_out = 0, alpha = 1) : la
couche est alors exactement une convolution classique.

## Réutiliser un run en Python

```python
from specprune.runs import load_run, load_run_data
from specprune.prune import prune_spectral

agent, config = load_run("runs/<run>")          # modèle reconstruit + poids du run
train, val, test, info = load_run_data(config)  # mêmes données, même normalisation
report = prune_spectral(agent, p=0.5, method="element")  # pruning en place, masques dans report.masks
probs = agent.predict(test[0])
```

Le modèle est reconstruit depuis `config.json` puis ses poids sont rechargés : c'est plus sûr
qu'un fichier `.keras`, qui dépendrait de la configuration spectrale active au chargement.

## Vitesse et mémoire

Une couche dont lambda_in, lambda_out et alpha sont figés à (1, 0, 1) est exactement une
convolution classique : elle est calculée avec `tf.nn.conv2d` (ou `conv1d`, `depthwise_conv2d`)
au lieu de la formulation spectrale (`specprune/layers/fast.py`). Avec `configs/spectral_config.json`,
c'est le cas de 20 couches sur 21 dans ResNet-20. Les résultats sont les mêmes (sorties et
gradients vérifiés par les tests), et sur CPU un pas d'entraînement est environ 5 fois plus rapide
pour ResNet-20 sur des images 32x32 (1 382 ms -> 281 ms au batch 128), 8 fois pour MobileNetV2.
La mémoire des couches spectrales passe d'environ 1 Gio à 120 Mio (ResNet-20, cifar10, batch 128).
Les runs déjà enregistrés en profitent aussi (pruning, notebook).

Une couche spectrale garde, pour la rétropropagation, les fenêtres de son entrée `X_win`
(B, O, K, C), soit K fois son entrée. `bench_memory.py` donne le détail par couche (colonne
`calcul` : spectral ou conv) et, sur GPU, le pic mesuré selon le batch :

```bash
python bench_memory.py --dataset cifar10 --arch spectral_resnet20 --batch_size 64 128 256
```

## Cluster (Slurm) et calcul en parallèle

Une série complète (plusieurs datasets, N graines, pruning pré et post) se décrit dans un fichier
JSON, puis `plan_experiments.py` la découpe en tâches indépendantes, en quatre étapes :

| étape | une tâche par | GPU |
|---|---|---|
| train | (dataset, graine) | oui |
| prepare | (dataset, pruning) : crée le dossier du pruning | non |
| prune | pré : (dataset, pruning, run) ; post : (dataset, pruning, run, p, répétition) | oui |
| finish | dataset : résumés et figures | non |

```bash
python plan_experiments.py experiments/exemple.json     # écrit jobs/serie1/
bash jobs/serie1/prefetch.sh                            # télécharge les datasets (nœud de connexion)
sbatch jobs/serie1/soumission.sh                        # soumet les 4 étapes, chaînées (afterok)
bash jobs/serie1/run_local.sh 2                         # ou : sur ta machine, 2 tâches à la fois
```

Exemple : 3 datasets x 5 graines = 15 tâches d'entraînement en même temps ; le post-training de
5 runs x 11 valeurs de p = 55 réentraînements en même temps par dataset. Chaque étape est un
tableau de tâches Slurm (`--array`) ; une tâche relancée reprend là où elle s'était arrêtée.
`"per_gpu": 2` (ou plus) dans `slurm` place plusieurs tâches sur un même GPU, utile pour les
petits modèles (une RTX 6000 Ada de Lyra a 48 Go). Les lignes `setup` du JSON (modules,
environnement Python) sont à adapter au cluster ; les journaux vont dans `jobs/<nom>/logs/`.

Sur Lyra (CECI) : Slurm, file `batch`, un GPU par nœud (`--gres=gpu:1`), 5 jours au plus par
job ; mettre `out_dir` et `pruning_dir` sous `$GLOBALSCRATCH` dans le JSON pour les gros volumes.
Le module TensorFlow de Lyra (2.15) utilise Keras 2, alors que ce projet demande TensorFlow 2.21
et Keras 3 : on installe donc un environnement Python une fois, sur le nœud de connexion :

```bash
module --force purge && ml releases/2023a && ml Python/3.11.3-GCCcore-12.3.0
python -m venv ~/venvs/spectral && source ~/venvs/spectral/bin/activate
pip install --upgrade pip && pip install -r requirements_gpu.txt && pip install -e .
```

Les scripts générés chargent ces mêmes modules et cet environnement (lignes `setup` du JSON).

Ou, sans plan, trois scripts Slurm à éditer directement (grille en tête de chaque script,
`--array` à mettre à jour), à lancer depuis `models_spectral/` l'un après l'autre :

```bash
sbatch experiments/slurm/train.sh     # dataset x graine
sbatch experiments/slurm/prune.sh     # MODE="post" : dataset x run x p ; MODE="pre" : dataset x run
sbatch experiments/slurm/finish.sh    # résumés et figures, par dataset (sans GPU)
```

Les mêmes découpages s'utilisent à la main :

```bash
python run_experiment.py --dataset cifar10 --n_runs 5 --run_name c10 --shard 0/5   # graine 42 seulement
python pruning.py runs/c10/seed* --mode post --name c10_post --prepare             # crée le dossier
python pruning.py --resume runs_pruning/c10_post --shard 3/55                      # une part du travail
python pruning.py --resume runs_pruning/c10_post --finish                          # résumé
```

## Tests

```bash
pip install pytest
pytest                      # couches, modèles, pruning, chaîne complète sur données synthétiques (CPU, ~5 min)
pytest tests/test_models.py -m slow     # en plus : tous les modèles, toutes les formes d'entrée
```

La chaîne complète (`tests/test_pipeline.py`) entraîne deux graines sur des images et sur des
signaux synthétiques, coupe un run puis le reprend, fait le pruning pré et post, vérifie que le
travail réparti en deux parties (comme `--gpus`) donne les mêmes résultats, refuse des runs non
comparables, puis produit les trois figures.

## Format des dossiers

`config.json` porte un numéro `format_version` (2 actuellement ; absent = 1). Les dossiers de
la version 1 restent lisibles ; un dossier écrit par une version plus récente du code est refusé
avec un message clair plutôt que mal interprété.

## Langue

Documentation, messages et options sont en français ; les figures sont en anglais. Le code des
couches (`specprune/layers/`) garde ses docstrings d'origine en anglais.
