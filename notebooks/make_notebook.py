"""
Génère notebooks/analyse_lambda_in.ipynb (le notebook est produit par ce script : le modifier ici).

    python notebooks/make_notebook.py notebooks/analyse_lambda_in.ipynb
"""
import sys

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# Analyse de `lambda_in` et pruning d'ordre p

Ce notebook est la version visuelle de `python pruning.py <run> --mode pre`. Il utilise les mêmes fonctions du paquet `specprune` (`load_run`, `masks_for`, `evaluate_pruned`) avec les mêmes réglages : pour les mêmes paramètres, il donne **les mêmes masques et les mêmes précisions** que `pruning.py`.

1. Chargement du run (modèle entraîné et poids) et des mêmes données que l'entraînement (dataset, sous-ensembles, normalisation relus dans `config.json`).
2. Valeurs brutes de `lambda_in` et importance de chaque position d'entrée (pixel ou instant).
3. Masques du pruning d'ordre *p* (`METHOD`) : dans chaque filtre (`gradient`, `element`) ou sur tous les filtres (`mixte`), les scores |w| en deçà du percentile d'ordre *p* sont mis à zéro (positions de padding exclues).
4. Positions d'entrée encore vues par la couche, sur la carte et sur des exemples.
5. Précision sur le test du modèle élagué, sans réentraînement (pré-training). Le post-training (phi réentraînés) se lance avec `pruning.py --mode post`.

Seule la cellule **Paramètres** est à modifier. Le notebook se lance depuis le dossier `notebooks/` ; il marche pour des images (SpecCnn2D) comme pour des signaux (SpecCnn1D)."""))

cells.append(md("## Paramètres"))
cells.append(code('''# Dossier du run à analyser (contient config.json, weights.weights.h5, spectral/).
# None = le run le plus récent de RUNS_ROOT. Le dataset est relu dans le run.
RUN_DIR = None
RUNS_ROOT = "../runs"   # le notebook est dans notebooks/ : les runs sont dans ../runs

# Pruning : mêmes options que pruning.py (--p, --param, --layers, --method, --criterion, --include_padding)
P_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PARAM = "lambda_in"          # "lambda_in", "lambda_out" ou "alpha"
LAYERS = "trained"           # "trained" (couches où PARAM est entraîné), "*", "0", "1-5", "0,3"
# "gradient" : par filtre, percentile sur tous les |w| du filtre (poids par poids)
# "element"  : par filtre, score d'un pixel = somme des |w| sur les canaux ; pixel retiré pour tous ses canaux
# "mixte"    : comme element, mais score sommé sur tous les filtres ; pixel retiré pour tous les filtres
METHOD = "element"
CRITERION = "percentile"     # "percentile" (|w| < percentile d'ordre p mis à zéro) ou "rank" (round(p*n))
INCLUDE_PADDING = False      # True = classer aussi les positions de padding

# Couche affichée (cartes, exemples) : None = la première couche élaguée
SHOW_LAYER = None

# Exemples : indices dans le jeu de test, ou None = un exemple par classe (10 au plus)
EXAMPLE_INDICES = None

# Précision du modèle élagué sur le test (comme pruning.py --mode pre)
EVALUATE = True
DEVICE = "/GPU:0"            # "/CPU:0" sans GPU'''))

cells.append(md("## Imports"))
cells.append(code('''import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Paquet specprune : installé (pip install -e .), sinon pris dans la racine du projet, que le
# notebook soit lancé depuis notebooks/ ou depuis models_spectral/
PROJECT_DIR = next((d for d in (os.path.abspath(".."), os.path.abspath("."))
                    if os.path.exists(os.path.join(d, "specprune", "prune.py"))), None)
if PROJECT_DIR is not None and PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
for name in [m for m in sys.modules if m == "specprune" or m.startswith("specprune.")]:
    del sys.modules[name]  # recharge specprune si une ancienne version était déjà importée
try:
    # mêmes fonctions que pruning.py
    from specprune.prune import assign_spectral_weights, evaluate_pruned, masks_for, select_layers
    from specprune.runs import load_run, load_run_data, spectral_layers, test_dataset, valid_positions
    from specprune.training import compile_agent, recipe_from_run
    import specprune
except ImportError as e:
    raise ImportError(f"{e}\\nspecprune introuvable ou trop ancien : ouvrir le notebook depuis "
                      "models_spectral/notebooks/, ou installer le projet (pip install -e .).") from None
print("specprune", specprune.__version__, "chargé depuis", os.path.dirname(specprune.__file__))

np.set_printoptions(precision=4, suppress=True)'''))

cells.append(md("## 1. Chargement du run et des données"))
cells.append(code('''if RUN_DIR is None:
    candidates = [os.path.join(RUNS_ROOT, d) for d in os.listdir(RUNS_ROOT)
                  if os.path.exists(os.path.join(RUNS_ROOT, d, "config.json"))]
    if not candidates:
        raise FileNotFoundError(f"Aucun run dans {RUNS_ROOT}/ : renseigner RUN_DIR")
    RUN_DIR = max(candidates, key=os.path.getmtime)

with tf.device(DEVICE):
    agent, config = load_run(RUN_DIR)          # architecture, réglages spectraux et poids du run
args = config["args"]
train, val, test, info = load_run_data(config)  # mêmes données, même normalisation
x_test, y_test = test
test_ds = test_dataset(test, info, args)
compile_agent(agent, recipe_from_run(args), 1)  # compilation pour l'évaluation (perte du run)

SIGNAL = info.modality == "signal"   # signal (T, C) : SpecCnn1D ; image (H, W, C) : SpecCnn2D
C = info.input_shape[-1]
unit = "instants" if SIGNAL else "pixels"
mean, std = np.array(info.mean), np.array(info.std)
class_names = info.class_names or [str(i) for i in range(info.num_classes)]

print(f"Run      : {config['run_name']}  ({RUN_DIR})")
print(f"Modèle   : {args['arch']}  |  config spectrale : {os.path.basename(config['spectral_config_file'] or '')}")
print(f"Dataset  : {info.name} ({info.modality}) {info.input_shape}, {info.num_classes} classes, test = {len(x_test)}")
print(f"Précision test du run : {config.get('test_accuracy', float('nan')):.4f}")'''))

cells.append(md("## 2. Valeurs brutes et importance de chaque position\n\nImportance d'une position d'entrée = moyenne sur les filtres et les canaux de |lambda_in| (positions de padding retirées). C'est un affichage : le pruning, lui, suit `METHOD`."))
cells.append(code('''targets = select_layers(agent, LAYERS, PARAM)
layer = targets[0] if SHOW_LAYER is None else next(l for l in targets if l.name == SHOW_LAYER)
conv_index = spectral_layers(agent).index(layer)
print(f"Couches élaguées : {[l.name for l in targets]}  ->  affichage de '{layer.name}' (convolution {conv_index})")


def to_input_grid(a, layer):
    """Tableau (..., L) sur la grille avec padding -> grille d'entrée de la couche, (..., T) ou (..., H, W)."""
    if hasattr(layer, "_L_raw"):  # SpecCnn1D
        return a[..., layer._P:layer._P + layer._L_raw]
    grid = a.reshape(a.shape[:-1] + (layer._Lh, layer._Lw))
    return grid[..., layer._Ph:layer._Ph + layer._H, layer._Pw:layer._Pw + layer._W]


values = getattr(layer, PARAM).numpy()                    # poids bruts (F, C/groups, L)
importance = to_input_grid(np.abs(values), layer).mean(axis=(0, 1)) if PARAM == "lambda_in" else None
valid = valid_positions(layer) if PARAM == "lambda_in" and not INCLUDE_PADDING else np.ones(values.shape, bool)
print(f"{PARAM} : {values.shape}, valeurs classées : {valid.sum()} ; min {values[valid].min():.4f}, "
      f"médiane {np.median(values[valid]):.4f}, max {values[valid].max():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
axes[0].hist(values[valid].ravel(), bins=100, color="steelblue")
axes[0].set_yscale("log"); axes[0].set_title(f"{PARAM} brut ({layer.name})")
if importance is not None:
    if SIGNAL:
        axes[1].plot(importance, "k"); axes[1].set_xlabel("instant")
    else:
        im = axes[1].imshow(importance, cmap="viridis"); fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[1].set_title("importance de chaque position : moyenne de |lambda_in|")
else:
    axes[1].axis("off")
plt.tight_layout(); plt.show()'''))

cells.append(md("## 3. Masques du pruning d'ordre p\n\nMêmes masques que `pruning.py` (fichiers `masks/p*.npz`) : pour chaque *p*, les scores en deçà du percentile d'ordre *p* sont retirés, calculés sur les valeurs du modèle entraîné (`gradient` : chaque |w| d'un filtre ; `element` : somme des |w| sur les canaux pour chaque pixel d'un filtre ; `mixte` : somme sur les filtres et les canaux)."))
cells.append(code('''original, reports = masks_for(agent, P_VALUES, [l.name for l in targets], PARAM, METHOD,
                              CRITERION, INCLUDE_PADDING)

print(f"{'p':>5} | {'retirés':>9} | {'fraction':>8} | par couche")
print("-" * 70)
for p, rep in reports.items():
    per_layer = ", ".join(f"{n}: {f:.1%}" for n, f in rep.per_layer().items())
    print(f"{p:>5.2f} | {rep.pruned:>9} | {rep.fraction:>8.1%} | {per_layer}")

# Part des poids gardés à chaque position d'entrée de la couche affichée (1 = tous les filtres et canaux)
kept = {p: to_input_grid(rep.masks[layer.name].astype(float), layer).mean(axis=(0, 1))
        for p, rep in reports.items()} if PARAM == "lambda_in" else {}'''))

cells.append(md(f"## 4. Positions d'entrée encore vues par la couche\n\nPour chaque *p* : part des `lambda_in` gardés à chaque position (vert = tous gardés, orange = position retirée pour tous les filtres). Avec `METHOD = \"mixte\"`, une position est gardée ou retirée en entier ; avec `\"element\"` ou `\"gradient\"`, chaque filtre a ses propres positions, d'où une dégradation progressive."))
cells.append(code('''from matplotlib.colors import LinearSegmentedColormap

# part des poids gardés : orange = retiré, vert = gardé
KEPT_CMAP = LinearSegmentedColormap.from_list("gardé", ["#e8710a", "#f2c14e", "#1e8e3e"])

if kept:
    n = len(P_VALUES)
    fig, axes = plt.subplots(1, n, figsize=(1.9 * n, 2.2 if not SIGNAL else 1.6), squeeze=False)
    for a, p in zip(axes[0], P_VALUES):
        k = kept[p]
        a.imshow(np.atleast_2d(k), cmap=KEPT_CMAP, vmin=0, vmax=1, aspect="auto" if SIGNAL else "equal",
                 interpolation="nearest")
        a.set_title(f"p={p:g}", fontsize=8); a.axis("off")
    plt.suptitle(f"part des lambda_in gardés par position ({layer.name})", fontsize=9)
    plt.tight_layout(); plt.show()

# Exemples : uniquement pour la couche qui voit l'entrée (convolution 0)
if kept and conv_index == 0:
    if EXAMPLE_INDICES is None:
        EXAMPLE_INDICES = [int(np.where(y_test == k)[0][0]) for k in range(min(info.num_classes, 10))
                           if (y_test == k).any()]

    def to_display(x):
        if SIGNAL:
            return x * std + mean
        img = np.clip(x * std + mean, 0, 1)
        return img[..., 0] if img.shape[-1] == 1 else img

    n_cols = len(P_VALUES) + 1
    size = (2.6, 1.6) if SIGNAL else (1.9, 1.9)
    fig, axes = plt.subplots(len(EXAMPLE_INDICES), n_cols,
                             figsize=(size[0] * n_cols, size[1] * len(EXAMPLE_INDICES)), squeeze=False)
    for r, idx in enumerate(EXAMPLE_INDICES):
        x = to_display(x_test[idx])
        axes[r, 0].set_ylabel(class_names[y_test[idx]], fontsize=8)
        for c, p in enumerate([None] + P_VALUES):
            a = axes[r, c]
            if SIGNAL:
                lo, hi = x.min(), x.max()
                if p is not None:  # fond : part des poids gardés à chaque instant
                    a.imshow(kept[p][None], extent=[-0.5, len(x) - 0.5, lo, hi], aspect="auto",
                             cmap=KEPT_CMAP, vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
                for ch in range(x.shape[1]):  # signal en couleurs sombres, lisible sur les deux fonds
                    a.plot(x[:, ch], lw=0.9, color=["black", "navy", "purple", "saddlebrown"][ch % 4])
                a.set_ylim(lo, hi)
            else:
                w = np.ones(x.shape[:2]) if p is None else kept[p]
                shown = x * (w if x.ndim == 2 else w[..., None])  # affichage : image x part gardée
                a.imshow(shown, cmap="gray" if C == 1 else None, vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title("original" if p is None else f"p={p:g}", fontsize=8)
    plt.suptitle("entrée vue par la couche 0 : " + ("fond vert = instant gardé, "
                 "orange = instant retiré (teinte = part des poids gardés)" if SIGNAL else "image multipliée par la part des poids gardés"),
                 fontsize=9)
    plt.tight_layout(); plt.show()
elif kept:
    print("Couche intermédiaire : pas d'affichage sur les exemples (ses entrées sont des cartes internes).")'''))

cells.append(md("## 5. Précision du modèle élagué (pré-training)\n\nPour chaque *p* : masques appliqués aux valeurs d'origine, puis évaluation sur le test du run, comme `pruning.py --mode pre`. Le modèle est remis à ses valeurs entraînées à la fin."))
cells.append(code('''results = {}
if EVALUATE:
    with tf.device(DEVICE):
        for p, rep in reports.items():
            results[p] = evaluate_pruned(agent, rep, original, test_ds)  # (précision, perte)
        assign_spectral_weights(agent, original, PARAM, verbose=False)   # retour au modèle entraîné

    print(f"{'p':>5} | {'fraction retirée':>16} | {'précision test':>14} | {'perte':>8}")
    print("-" * 54)
    for p, (acc, loss) in results.items():
        print(f"{p:>5.2f} | {reports[p].fraction:>16.1%} | {acc:>14.4f} | {loss:>8.4f}")

    plt.figure(figsize=(6, 3.6))
    plt.plot(list(results), [a for a, _ in results.values()], "o-")
    plt.xlabel("ordre p (fraction des poids retirés)"); plt.ylabel("précision test"); plt.ylim(0, 1.02)
    plt.title(f"Pré-training : {PARAM}, {METHOD}, {CRITERION}"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()'''))

cells.append(md("## 6. Sauvegarde\n\nRéglages, fractions retirées et précisions dans `RUN_DIR/analyse_lambda_in/`."))
cells.append(code('''out_dir = os.path.join(RUN_DIR, "analyse_lambda_in", f"{PARAM}_{METHOD}_{CRITERION}")
os.makedirs(out_dir, exist_ok=True)
summary = {
    "run": config["run_name"], "param": PARAM, "layers": [l.name for l in targets], "method": METHOD,
    "criterion": CRITERION, "include_padding": INCLUDE_PADDING,
    "p": [{"p": p, "retires": rep.pruned, "total": rep.total, "fraction": rep.fraction,
           "par_couche": rep.per_layer(),
           **({"precision_test": results[p][0], "perte_test": results[p][1]} if p in results else {})}
          for p, rep in reports.items()],
}
with open(os.path.join(out_dir, "resume.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"Enregistré dans {out_dir}")'''))

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, sys.argv[1] if len(sys.argv) > 1 else "notebooks/analyse_lambda_in.ipynb")
