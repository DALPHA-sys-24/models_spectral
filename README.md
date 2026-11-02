# Modèles spectraux et pruning de lambda_in

Modèles de référence (ResNet, MobileNet) dont toutes les convolutions sont spectrales :
`SpecCnn2D` pour des images, `SpecCnn1D` pour des signaux (choix automatique d'après les données).
Un fichier de configuration dit, couche par couche, quels poids sont entraînés (phi, lambda_in,
lambda_out, alpha). Après l'entraînement, les poids lambda_in servent au pruning d'ordre p.

## Arborescence

```
models_spectral/
├── run_experiment.py        entraîne un modèle et enregistre le run (config, poids, lambda_in)
├── search_hparams.py        cherche lr et batch (epochs fixés) en lançant run_experiment.py
├── pruning.py               pruning d'ordre p d'un ou plusieurs runs : pré et/ou post-training
├── plot_results.py          figures de publication à partir des résultats enregistrés
├── configs/
│   ├── spectral_config.json     défaut : couche 0 = phi + lambda_in, les autres = Conv classiques
│   ├── classique.json           tous les lambda figés : modèle classique (seuls les phi)
│   └── all_spectral.json        lambda_in entraîné dans toutes les couches
├── spectralcnn/             couches SpecCnn1D et SpecCnn2D
├── models/                  architectures (registre, ResNet, MobileNet, AgentNetwork)
├── tools/                   bibliothèque commune aux scripts
│   ├── data.py                  datasets images et signaux, normalisation, tf.data
│   ├── training.py              appareil, optimiseur, compilation (recette d'entraînement)
│   ├── runs.py                  sauvegarde d'un run et rechargement (load_run, load_run_data)
│   ├── prune.py                 pruning d'ordre p, évaluation (pre), gel des lambda (post)
│   └── plotting.py              figures style Nature Communications (sans titre, en anglais)
├── notebooks/
│   └── analyse_lambda_in.ipynb  pruning d'ordre p en images : masques, exemples, précision (pre)
├── tests/
│   ├── verifier_modeles.py      équivalences avec Conv1D / Conv2D, construction des modèles
│   └── test_pruning.py          vérifications de tools/prune.py
├── requirements.txt         GPU (WSL / Linux)
└── requirements-cpu.txt     CPU (ex. Windows pour les notebooks)
```

Les scripts se lancent depuis la racine `models_spectral/`. Les résultats vont dans `runs/`,
`runs_search/` et `runs_pruning/` (ignorés par git).

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

### 4. Analyser

`notebooks/analyse_lambda_in.ipynb` (à lancer depuis `notebooks/`) : version visuelle de
`pruning.py --mode pre`, avec les mêmes fonctions de `tools/` et les mêmes options (`P_VALUES`,
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
  "layers": {"0": {"trainable_lambda_in": true, "lambda_in_l1": 1e-6}}
}
```

Un poids non entraîné garde sa valeur neutre (lambda_in = 1, lambda_out = 0, alpha = 1) : la
couche est alors exactement une convolution classique.

## Réutiliser un run en Python

```python
from tools.runs import load_run, load_run_data
from tools.prune import prune_spectral

agent, config = load_run("runs/<run>")          # modèle reconstruit + poids du run
train, val, test, info = load_run_data(config)  # mêmes données, même normalisation
report = prune_spectral(agent, p=0.5, method="element")  # pruning en place, masques dans report.masks
probs = agent.predict(test[0])
```

Le modèle est reconstruit depuis `config.json` puis ses poids sont rechargés : c'est plus sûr
qu'un fichier `.keras`, qui dépendrait de la configuration spectrale active au chargement.

## Vérifications

```bash
python tests/verifier_modeles.py spectral_resnet20
python tests/test_pruning.py
```
