"""
Figures et table de l'article, à partir de runs/ (entraînement) et runs_pruning/ (pruning.py).

Le script retrouve seul, pour chaque dataset, le groupe de runs (graines) et ses dossiers de
pruning (pre pour chaque méthode, post), puis produit :

    1  importance   carte d'importance des pixels : médiane sur les runs de la moyenne de
                    |lambda_in| sur tous les filtres et canaux (une figure par dataset + une
                    figure réunissant tous les datasets)
    2  masks        dégradation du masque selon p, pour les trois méthodes (une par dataset)
    3  data         effet de la dégradation sur des exemples du test (une par dataset et exemple)
    4  accuracy     précision test selon p : les trois méthodes en pre-training et le
                    post-training (une par dataset + une réunissant tous les datasets)
    5  table        table récapitulative (CSV + LaTeX) : dataset, modèle, précision sans pruning,
                    pre et post à p = --table_p, pixels de l'image encore utilisés

Les masques sont recalculés depuis les lambda_in enregistrés par run_experiment.py
(<run>/spectral/), avec la même fonction et les mêmes réglages (critère, padding) que les
dossiers de pruning : ce sont exactement les masques de pruning.py. Aucun modèle n'est chargé ni
évalué ; les précisions viennent des summary.csv de pruning.py et des config.json des runs.

Figures : style Nature Communications, sans titre, textes en anglais, PDF vectoriel + PNG 600 dpi.

Exemples :
    # tout, avec les dossiers par défaut (runs/ et runs_pruning/) -> paper_figures/
    python paper_results.py

    # seulement deux datasets, la table à p = 0.9, des colonnes p choisies
    python paper_results.py --datasets mnist cifar10 --table_p 0.9 --p_show 0 0.5 0.8 0.95

    # seulement la figure de précision et la table, en deux colonnes Nature
    python paper_results.py --only accuracy table --width_mm 183

    # plusieurs groupes pour un même dataset : choisir par un morceau du nom
    python paper_results.py --tag 50ep_x5
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

from .common import Formatter
from ..plotting import COLORS, KEPT_CMAP, MM, Style, save

OUTPUTS = ["importance", "masks", "data", "accuracy", "table"]
METHODS = ["gradient", "element", "mixte"]
DATASET_LABELS = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "cifar10": "CIFAR-10",
                  "cifar100": "CIFAR-100", "mnist1d": "MNIST-1D"}
METHOD_LABELS = {"gradient": "Gradient", "element": "Element", "mixte": "Mixed"}
LAMBDA = r"$\lambda_{in}$"


# --- Ligne de commande ----------------------------------------------------------------

def key_values(items: List[str], what: str) -> Dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{what} : « {item} » doit être de la forme clé=valeur")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=Formatter)
    g = p.add_argument_group("données")
    g.add_argument("--runs_dir", default="runs", help="dossier des runs (run_experiment.py)")
    g.add_argument("--pruning_dir", default="runs_pruning", help="dossier des résultats de pruning.py")
    g.add_argument("--out", default="paper_figures", help="dossier de sortie")
    g.add_argument("--datasets", nargs="+", default=None,
                   help="datasets, dans l'ordre des figures (défaut : tous ceux trouvés)")
    g.add_argument("--groups", nargs="+", default=None,
                   help="groupes de runs à utiliser (noms dans --runs_dir ou chemins) au lieu de la recherche")
    g.add_argument("--tag", default="", help="ne garder que les groupes dont le nom contient ce texte")
    g.add_argument("--only", nargs="+", choices=OUTPUTS, default=OUTPUTS, help="sorties à produire")

    g = p.add_argument_group("pruning montré")
    g.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS, help="méthodes (pre-training)")
    g.add_argument("--post_methods", nargs="+", choices=METHODS, default=None,
                   help="méthodes du post-training montrées (défaut : toutes celles trouvées)")
    g.add_argument("--p_show", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 0.9],
                   help="colonnes des figures de masques et d'exemples")
    g.add_argument("--layer", type=int, default=None,
                   help="indice de convolution montré (défaut : première couche à lambda_in entraîné)")
    g.add_argument("--mask_run", default="0", help='run montré pour les masques et exemples : indice, ou "mean"')
    g.add_argument("--image_mask", choices=["all", "filter"], default="all",
                   help="exemples vus à travers la part gardée sur tous les filtres, ou le masque de --filter")
    g.add_argument("--filter", type=int, default=0, help="filtre utilisé avec --image_mask filter")
    g.add_argument("--examples", type=int, default=1, help="nombre d'exemples par dataset (classes 0, 1, ...)")
    g.add_argument("--example_indices", type=int, nargs="+", default=None,
                   help="indices dans le jeu de test (mêmes pour tous les datasets)")
    g.add_argument("--criterion", default=None,
                   help="critère des masques (défaut : celui des dossiers de pruning)")
    g.add_argument("--include_padding", action=argparse.BooleanOptionalAction, default=None,
                   help="défaut : comme les dossiers de pruning")

    g = p.add_argument_group("table")
    g.add_argument("--table_p", type=float, default=0.8, help="p de la table")
    g.add_argument("--table_methods", nargs="+", choices=METHODS, default=None,
                   help="lignes de la table (défaut : --methods)")
    g.add_argument("--table_scale", choices=["percent", "fraction"], default="percent")
    g.add_argument("--table_decimals", type=int, default=1)
    g.add_argument("--table_band", choices=["none", "quantile", "std"], default="none",
                   help="dans le .tex : rien, [q5, q95] ou ± écart-type à côté de la médiane")
    g.add_argument("--table_caption", default="", help="légende LaTeX (vide = aucune)")
    g.add_argument("--table_label", default="tab:results")

    g = p.add_argument_group("noms (anglais)")
    g.add_argument("--dataset_labels", nargs="+", default=None, metavar="CLE=NOM",
                   help='ex. mnist=MNIST "fashion_mnist=Fashion MNIST"')
    g.add_argument("--method_labels", nargs="+", default=None, metavar="CLE=NOM",
                   help="ex. mixte=Mixed element=Element")
    g.add_argument("--model_label", default="", help='nom du modèle dans la table (défaut : depuis arch, ex. "Spectral ResNet-20")')
    g.add_argument("--pre_label", default="{method}", help="légende des courbes pre ({method} = nom de la méthode)")
    g.add_argument("--post_label", default="{method}, retrained", help="légende des courbes post")
    g.add_argument("--xlabel", default="Pruning ratio p")
    g.add_argument("--ylabel", default="Accuracy")
    g.add_argument("--panel_labels", choices=["letters", "names", "both", "none"], default="letters",
                   help="figures réunissant les datasets : lettre a, b, c ; nom du dataset ; les deux ; rien")

    g = p.add_argument_group("style")
    g.add_argument("--formats", nargs="+", default=["pdf", "png"])
    g.add_argument("--dpi", type=int, default=600)
    g.add_argument("--width_mm", type=float, default=183.0, help="figures en grille (89 = une colonne, 183 = deux)")
    g.add_argument("--accuracy_size_mm", type=float, nargs=2, default=[89.0, 60.0],
                   help="largeur, hauteur de la figure de précision d'un dataset")
    g.add_argument("--importance_size_mm", type=float, default=55.0, help="largeur d'une carte d'importance")
    g.add_argument("--font", default="Arial")
    g.add_argument("--font_size", type=float, default=7.0)
    g.add_argument("--line_width", type=float, default=1.2)
    g.add_argument("--marker_size", type=float, default=3.5)
    g.add_argument("--no_markers", action="store_true")
    g.add_argument("--colors", nargs="+", default=COLORS, help="couleurs des méthodes, dans l'ordre de --methods")
    g.add_argument("--post_style", default="--", help="trait des courbes post (--, :, -.)")
    g.add_argument("--legend", default="best", help='position, ou "none"')
    g.add_argument("--ylim", type=float, nargs=2, default=[0.0, 1.02])
    g.add_argument("--stat", choices=["median", "mean"], default="median")
    g.add_argument("--band", choices=["quantile", "std", "none"], default="quantile",
                   help="bande : quantiles 5-95 %%, ± écart-type, ou rien")
    g.add_argument("--chance", choices=["auto", "none"], default="auto",
                   help="niveau du hasard en pointillés (auto = 1 / nombre de classes)")
    g.add_argument("--importance_colors", nargs="+", default=["#ffffff", "#9ecae1", COLORS[1], "#08306b"],
                   help="couleurs de la carte d'importance (de faible à fort)")
    return p.parse_args(argv)


# --- Recherche des runs et des dossiers de pruning -------------------------------------

def natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def read_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def runs_of(folder: str) -> List[str]:
    """Runs terminés d'un groupe (sous-dossiers seed*/) ou d'un run seul, triés par graine."""
    if os.path.exists(os.path.join(folder, "config.json")):
        found = [folder]
    else:
        found = [d for d in glob.glob(os.path.join(folder, "*")) if os.path.exists(os.path.join(d, "config.json"))]
    done = []
    for d in found:
        try:
            c = read_json(os.path.join(d, "config.json"))
        except (OSError, ValueError):
            continue
        if "finished_at" in c and c.get("lambda_in_files") and isinstance(c.get("dataset"), dict):
            done.append(d)
    return sorted(done, key=lambda d: natural(os.path.basename(d)))


def dataset_key(config: Dict) -> str:
    """Nom du dataset ; pour un fichier npz ou un dossier : nom du fichier."""
    args = config["args"]
    if args["dataset"] in ("npz", "folder") and args.get("data_path"):
        return os.path.splitext(os.path.basename(os.path.normpath(args["data_path"])))[0]
    return args["dataset"]


def find_groups(args) -> Dict[str, Dict]:
    """{dataset: {"dir", "runs"}} : un groupe par dataset."""
    if args.groups:
        folders = [g if os.path.isdir(g) else os.path.join(args.runs_dir, g) for g in args.groups]
    else:
        folders = sorted(d for d in glob.glob(os.path.join(args.runs_dir, "*")) if os.path.isdir(d))
    by_dataset: Dict[str, List] = {}
    for d in folders:
        if args.tag and args.tag not in os.path.basename(os.path.normpath(d)):
            continue
        runs = runs_of(d)
        if runs:
            by_dataset.setdefault(dataset_key(read_json(os.path.join(runs[0], "config.json"))), []).append((d, runs))
    if not by_dataset:
        raise SystemExit(f"Aucun run terminé dans {args.runs_dir}/" + (f" (--tag {args.tag})" if args.tag else ""))
    groups = {}
    for ds in args.datasets or sorted(by_dataset, key=lambda k: list(DATASET_LABELS).index(k)
                                     if k in DATASET_LABELS else len(DATASET_LABELS)):
        if ds not in by_dataset:
            raise SystemExit(f"Dataset « {ds} » introuvable dans {args.runs_dir}/ ; trouvés : {sorted(by_dataset)}")
        found = sorted(by_dataset[ds], key=lambda x: os.path.getmtime(x[0]))
        d, runs = found[-1]
        if len(found) > 1:
            print(f"  {ds} : plusieurs groupes {[os.path.basename(f[0]) for f in found]} -> "
                  f"{os.path.basename(d)} (le plus récent ; choisir avec --tag ou --groups)")
        groups[ds] = {"dir": os.path.normpath(d), "runs": runs}
    return groups


def find_pruning(pruning_dir: str, group_dir: str) -> Dict[str, Dict[str, str]]:
    """Dossiers de pruning.py d'un groupe : {"pre": {méthode: dossier}, "post": {...}}."""
    name = os.path.basename(group_dir)
    found: Dict[str, Dict[str, List[str]]] = {"pre": {}, "post": {}}
    for d in sorted(glob.glob(os.path.join(pruning_dir, "*"))):
        if not (os.path.exists(os.path.join(d, "config.json")) and os.path.exists(os.path.join(d, "summary.csv"))):
            continue
        c = read_json(os.path.join(d, "config.json"))
        parents = {os.path.basename(os.path.dirname(os.path.normpath(r))) for r in c.get("runs", [])}
        selves = {os.path.basename(os.path.normpath(r)) for r in c.get("runs", [])}
        if name not in parents and name not in selves:
            continue
        a = c.get("args", {})
        if a.get("param", "lambda_in") != "lambda_in":
            continue
        modes = ["pre", "post"] if a.get("mode") == "both" else [a.get("mode")]
        for mode in modes:
            if mode in found:
                found[mode].setdefault(a.get("method", "element"), []).append(d)
    out: Dict[str, Dict[str, str]] = {"pre": {}, "post": {}}
    for mode, by_method in found.items():
        for method, dirs in by_method.items():
            dirs = sorted(dirs, key=os.path.getmtime)
            if len(dirs) > 1:
                print(f"  {name} {mode} {method} : plusieurs dossiers -> {os.path.basename(dirs[-1])} (le plus récent)")
            out[mode][method] = dirs[-1]
    return out


def read_summary(pruning_dir: str, mode: str) -> Dict[str, np.ndarray]:
    with open(os.path.join(pruning_dir, "summary.csv")) as f:
        rows = sorted((r for r in csv.DictReader(f) if r["mode"] == mode), key=lambda r: float(r["p"]))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("p", "n", "mean", "median", "q05", "q95", "std")}


# --- lambda_in et masques, depuis les fichiers des runs --------------------------------

class Layer:
    """Géométrie de la couche montrée (lambda_in_files du run) et passage à la grille d'entrée."""

    def __init__(self, meta: Dict):
        self.meta = meta
        self.dim = meta["dim"]
        self.grid, self.pad, self.size = meta["padded_grid"], meta["padding"], meta["input_hw"]

    def to_input(self, a: np.ndarray) -> np.ndarray:
        """(..., L) sur la grille avec padding -> (..., T) ou (..., H, W)."""
        a = a.reshape(a.shape[:-1] + tuple(self.grid))
        crop = tuple(slice(p, p + s) for p, s in zip(self.pad, self.size))
        return a[(Ellipsis,) + crop]

    def valid(self, shape, include_padding: bool) -> np.ndarray:
        if include_padding:
            return np.ones(shape, bool)
        g = np.zeros(self.grid, bool)
        g[tuple(slice(p, p + s) for p, s in zip(self.pad, self.size))] = True
        return np.broadcast_to(g.ravel(), shape).copy()


def load_lambda(run_dir: str, conv_index: Optional[int]):
    """lambda_in final d'un run pour la couche voulue : (valeurs (F, C/groups, L), Layer)."""
    config = read_json(os.path.join(run_dir, "config.json"))
    files = sorted(config["lambda_in_files"], key=lambda m: m["conv_index"])
    meta = files[0] if conv_index is None else next((m for m in files if m["conv_index"] == conv_index), None)
    if meta is None:
        raise SystemExit(f"--layer {conv_index} : lambda_in non entraîné dans cette couche "
                         f"(couches entraînées : {[m['conv_index'] for m in files]})")
    return np.load(os.path.join(run_dir, "spectral", meta["file"])), Layer(meta)


# --- Figures -----------------------------------------------------------------------------

class Painter:
    def __init__(self, args):
        self.args = args
        self.style = Style(width_mm=args.width_mm, font=args.font, font_size=args.font_size,
                           line_width=args.line_width, marker_size=args.marker_size, colors=args.colors,
                           markers=not args.no_markers, legend=args.legend, formats=args.formats, dpi=args.dpi)
        from matplotlib.colors import LinearSegmentedColormap
        self.value_cmap = LinearSegmentedColormap.from_list("value", args.importance_colors)

    def figure(self, width_mm, height_mm, nrows=1, ncols=1, **kw):
        import matplotlib.pyplot as plt
        self.style.apply()
        plt.rcParams["mathtext.default"] = "regular"
        return plt.subplots(nrows, ncols, figsize=(width_mm * MM, height_mm * MM), **kw)

    def save(self, fig, name):
        files = save(fig, os.path.join(self.args.out, name), self.style)
        print("    ->", ", ".join(os.path.relpath(f) for f in files))

    @staticmethod
    def clean(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    def colorbar(self, fig, im, axes, label, fraction=0.025):
        cb = fig.colorbar(im, ax=axes, fraction=fraction, pad=0.02, aspect=25)
        cb.set_label(label)
        cb.outline.set_linewidth(self.style.axis_width)
        cb.ax.tick_params(width=self.style.axis_width)
        return cb

    def band(self, ax, x, curves, color, label, marker=None, ls="-", hollow=False):
        y = np.stack(curves)
        center = np.median(y, 0) if self.args.stat == "median" else y.mean(0)
        if len(y) > 1 and self.args.band != "none":
            lo, hi = ((np.quantile(y, 0.05, 0), np.quantile(y, 0.95, 0)) if self.args.band == "quantile"
                      else (y.mean(0) - y.std(0), y.mean(0) + y.std(0)))
            ax.fill_between(x, lo, hi, color=color, alpha=0.22, lw=0)
        ax.plot(x, center, color=color, label=label, marker=marker, ls=ls,
                markerfacecolor="white" if hollow else color,
                markeredgecolor=color if hollow else "white", markeredgewidth=0.8 if hollow else 0.4)

    def panel_label(self, ax, i, name, x=-0.16, y=1.04):
        mode = self.args.panel_labels
        text = {"letters": "abcdefgh"[i], "names": name, "both": f"{'abcdefgh'[i]}   {name}", "none": ""}[mode]
        if not text:
            return
        if mode == "names":
            ax.text(0.0, y, text, transform=ax.transAxes, va="bottom", fontsize=self.style.font_size)
        else:
            ax.text(x, y, "abcdefgh"[i], transform=ax.transAxes, va="bottom", fontweight="bold",
                    fontsize=self.style.font_size + 1)
            if mode == "both":
                ax.text(x + 0.08, y, name, transform=ax.transAxes, va="bottom", fontsize=self.style.font_size)


# --- Programme -----------------------------------------------------------------------------

def nearest(values, p):
    values = list(values)
    i = int(np.argmin([abs(v - p) for v in values]))
    return i, values[i]


def model_label(arch: str) -> str:
    m = re.fullmatch(r"spectral_(resnet|mobilenet)(.*)", arch)
    if not m:
        return arch
    family = {"resnet": "ResNet", "mobilenet": "MobileNet"}[m.group(1)]
    rest = m.group(2).lstrip("_")
    return f"Spectral {family}-{rest}" if rest.isdigit() else f"Spectral {family} {rest}".strip()


def main(argv=None):
    args = parse_args(argv)
    ds_labels = {**DATASET_LABELS, **key_values(args.dataset_labels, "--dataset_labels")}
    m_labels = {**METHOD_LABELS, **key_values(args.method_labels, "--method_labels")}
    mask_run = None if args.mask_run == "mean" else int(args.mask_run)
    table_methods = args.table_methods or args.methods
    os.makedirs(args.out, exist_ok=True)
    painter = Painter(args)
    style = painter.style

    from ..prune import _layer_mask   # fonction des masques de pruning.py (mêmes masques)

    print(f"Recherche dans {args.runs_dir}/ et {args.pruning_dir}/")
    groups = find_groups(args)
    data = {}
    for ds, g in groups.items():
        prune_dirs = find_pruning(args.pruning_dir, g["dir"])
        configs = [read_json(os.path.join(r, "config.json")) for r in g["runs"]]
        ref = next(iter(prune_dirs["pre"].values()), None) or next(iter(prune_dirs["post"].values()), None)
        ref_args = read_json(os.path.join(ref, "config.json"))["args"] if ref else {}
        criterion = args.criterion or ref_args.get("criterion", "percentile")
        padding = args.include_padding if args.include_padding is not None else ref_args.get("include_padding", False)
        lam, layers = zip(*(load_lambda(r, args.layer) for r in g["runs"]))
        layer = layers[0]
        print(f"  {ds:<14} {os.path.basename(g['dir'])} : {len(g['runs'])} runs, couche "
              f"{layer.meta['layer']} (conv {layer.meta['conv_index']}), lambda_in {lam[0].shape} ; "
              f"pre {sorted(prune_dirs['pre'])}, post {sorted(prune_dirs['post'])}")
        data[ds] = {"group": g, "configs": configs, "lam": np.stack(lam), "layer": layer, "pruning": prune_dirs,
                    "criterion": criterion, "include_padding": bool(padding),
                    "label": ds_labels.get(ds, ds), "num_classes": configs[0]["dataset"]["num_classes"]}

    def masks(ds, method, p):
        """Masques gardés (runs, F, C/groups, L) d'un dataset, méthode et p (mis en cache)."""
        d = data[ds]
        cache = d.setdefault("masks", {})
        if (method, p) not in cache:
            valid = d["layer"].valid(d["lam"].shape[1:], d["include_padding"])
            cache[(method, p)] = np.stack([_layer_mask(w, valid, p, method, d["criterion"]) for w in d["lam"]])
        return cache[(method, p)]

    def kept_by_position(ds, method, p, which="all"):
        """Part gardée par position (runs, C/groups, T|H, W) : moyenne sur les filtres, ou un filtre."""
        g = data[ds]["layer"].to_input(masks(ds, method, p).astype(np.float32))
        return g.mean(axis=1) if which == "all" else g[:, args.filter]

    def pick(a):
        return a.mean(0) if mask_run is None else a[min(mask_run, len(a) - 1)]

    p_cols = list(dict.fromkeys(args.p_show))
    n_ds = len(data)

    # 1. Carte d'importance des pixels
    if "importance" in args.only:
        print("1. Importance des pixels")
        maps = {ds: np.median(d["layer"].to_input(np.abs(d["lam"])).mean(axis=(1, 2)), 0) for ds, d in data.items()}
        curves = {ds: list(d["layer"].to_input(np.abs(d["lam"])).mean(axis=(1, 2))) for ds, d in data.items()}
        w = args.importance_size_mm
        for ds, d in data.items():
            if d["layer"].dim == 1:
                fig, ax = painter.figure(*args.accuracy_size_mm)
                t = np.arange(len(maps[ds]))
                painter.band(ax, t, curves[ds], style.color(1), None)
                ax.set_xlabel("Time step"); ax.set_ylabel(f"|{LAMBDA}|"); ax.set_xlim(t[0], t[-1])
            else:
                fig, ax = painter.figure(w, w * 0.82)
                im = ax.imshow(maps[ds], cmap=painter.value_cmap, interpolation="nearest")
                painter.clean(ax)
                painter.colorbar(fig, im, [ax], f"Median |{LAMBDA}|", fraction=0.046)
            painter.save(fig, f"importance_{ds}")
        images = [ds for ds in data if data[ds]["layer"].dim == 2]
        if len(images) > 1:
            fig, axes = painter.figure(args.width_mm, args.width_mm / len(images) * 0.82, 1, len(images),
                                       squeeze=False, gridspec_kw={"wspace": 0.35})
            for i, (ax, ds) in enumerate(zip(axes[0], images)):
                im = ax.imshow(maps[ds], cmap=painter.value_cmap, interpolation="nearest")
                painter.clean(ax)
                painter.colorbar(fig, im, [ax], f"Median |{LAMBDA}|", fraction=0.046)
                painter.panel_label(ax, i, data[ds]["label"], x=-0.05)
            painter.save(fig, "importance_all")

    # 2. Dégradation du masque selon p
    if "masks" in args.only:
        print("2. Masques selon p")
        for ds, d in data.items():
            n = len(args.methods)
            if d["layer"].dim == 1:
                fig, axes = painter.figure(args.width_mm, 16 * n + 8, n, 1, sharex=True, squeeze=False)
                axes = axes[:, 0]
                for a, m in zip(axes, args.methods):
                    img = np.stack([pick(kept_by_position(ds, m, p)).mean(0) for p in p_cols])
                    im = a.imshow(img, cmap=KEPT_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                                  origin="lower", extent=[-0.5, img.shape[1] - 0.5, -0.5, len(p_cols) - 0.5])
                    a.set_yticks(range(len(p_cols)), [f"{p:g}" for p in p_cols])
                    a.set_ylabel(f"{m_labels[m]}\np")
                axes[-1].set_xlabel("Time step")
            else:
                cols = len(p_cols)
                fig, axes = painter.figure(args.width_mm, args.width_mm / (cols + 0.6) * n, n, cols, squeeze=False,
                                           gridspec_kw={"wspace": 0.05, "hspace": 0.08})
                for i, m in enumerate(args.methods):
                    for j, p in enumerate(p_cols):
                        a = axes[i, j]
                        im = a.imshow(pick(kept_by_position(ds, m, p)).mean(0), cmap=KEPT_CMAP, vmin=0, vmax=1,
                                      interpolation="nearest")
                        painter.clean(a)
                        if i == n - 1:
                            a.set_xlabel(f"p = {p:g}")
                        if j == 0:
                            a.set_ylabel(m_labels[m])
            painter.colorbar(fig, im, list(np.ravel(axes)), "Fraction of weights kept")
            painter.save(fig, f"masks_{ds}")

    # 3. Effet sur les données
    if "data" in args.only:
        print("3. Effet sur les données")
        from ..runs import load_run_data
        import matplotlib.pyplot as plt
        for ds, d in data.items():
            if d["layer"].meta["conv_index"] != 0:
                print(f"  {ds} : la couche montrée n'est pas la première, pas d'exemples de données")
                continue
            _, _, (x_test, y_test), info = load_run_data(d["configs"][0])
            mean, std = np.array(info.mean), np.array(info.std)
            denorm = lambda x: x * std + mean
            if args.example_indices:
                indices = [i for i in args.example_indices if i < len(x_test)]
            else:
                classes = [k for k in range(info.num_classes) if (y_test == k).any()][:args.examples]
                indices = [int(np.where(y_test == k)[0][0]) for k in classes]
            which = "all" if args.image_mask == "all" else "filter"
            n = len(args.methods)
            for idx in indices:
                x = x_test[idx]
                if d["layer"].dim == 1:
                    cols = len(p_cols)
                    fig, axes = painter.figure(args.width_mm, 17 * n + 6, n, cols, sharex=True, sharey=True,
                                               squeeze=False, gridspec_kw={"wspace": 0.06, "hspace": 0.1})
                    xs = denorm(x)
                    lo, hi = xs.min() - 0.05 * np.ptp(xs), xs.max() + 0.05 * np.ptp(xs)
                    for i, m in enumerate(args.methods):
                        for j, p in enumerate(p_cols):
                            a = axes[i, j]
                            k = pick(kept_by_position(ds, m, p, which)).mean(0)
                            a.imshow(k[None], extent=[-0.5, len(k) - 0.5, lo, hi], aspect="auto", cmap=KEPT_CMAP,
                                     vmin=0, vmax=1, alpha=0.45, interpolation="nearest")
                            for ch in range(xs.shape[1]):
                                a.plot(xs[:, ch], lw=0.7, color="black" if ch == 0 else plt.cm.Greys(0.4 + 0.5 * ch / xs.shape[1]))
                            a.set_ylim(lo, hi); a.set_xlim(-0.5, len(k) - 0.5)
                            if i == n - 1:
                                a.set_xlabel(f"p = {p:g}")
                            if j == 0:
                                a.set_ylabel(m_labels[m])
                else:
                    ps = p_cols if 0.0 in p_cols else [0.0] + p_cols
                    cols = len(ps)
                    fig, axes = painter.figure(args.width_mm, args.width_mm / (cols + 0.2) * n, n, cols, squeeze=False,
                                               gridspec_kw={"wspace": 0.05, "hspace": 0.08})
                    ref = denorm(x)
                    lo, hi = (0.0, 1.0) if ref.min() >= 0 and ref.max() <= 1 else (ref.min(), ref.max())
                    for i, m in enumerate(args.methods):
                        for j, p in enumerate(ps):
                            a = axes[i, j]
                            if p == 0.0:
                                img = ref
                            else:
                                k = pick(kept_by_position(ds, m, p, which))          # (C/groups, H, W)
                                k = np.moveaxis(k, 0, -1) if k.shape[0] == x.shape[-1] else k.mean(0)[..., None]
                                img = denorm(x * k)   # pixel retiré = 0 normalisé = couleur moyenne du dataset
                            img = np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)
                            a.imshow(img[..., 0] if img.shape[-1] == 1 else img, cmap="gray" if img.shape[-1] == 1 else None,
                                     vmin=0, vmax=1, interpolation="nearest")
                            painter.clean(a)
                            if i == n - 1:
                                a.set_xlabel("Original" if p == 0.0 else f"p = {p:g}")
                            if j == 0:
                                a.set_ylabel(m_labels[m])
                painter.save(fig, f"data_{ds}_example{idx}")

    # 4. Précision selon p
    if "accuracy" in args.only:
        print("4. Précision selon p")

        def draw(ax, ds):
            d, n_curves = data[ds], 0
            for i, m in enumerate(args.methods):
                if m in d["pruning"]["pre"]:
                    s = read_summary(d["pruning"]["pre"][m], "pre")
                    draw_curve(ax, s, i, args.pre_label.format(method=m_labels[m]), "-", False)
                    n_curves += 1
            post = args.post_methods or [m for m in METHODS if m in d["pruning"]["post"]]
            for m in post:
                if m in d["pruning"]["post"]:
                    i = args.methods.index(m) if m in args.methods else METHODS.index(m)
                    s = read_summary(d["pruning"]["post"][m], "post")
                    draw_curve(ax, s, i, args.post_label.format(method=m_labels[m]), args.post_style, True)
                    n_curves += 1
            if args.chance == "auto":
                ax.axhline(1 / d["num_classes"], color="#8c8c8c", lw=0.6, ls=":", zorder=0)
            ax.set_xlabel(args.xlabel); ax.set_ylabel(args.ylabel)
            ax.set_ylim(*args.ylim)
            return n_curves

        def draw_curve(ax, s, i, label, ls, hollow):
            col = style.color(i)
            if args.band != "none" and s["n"].max() > 1:
                lo, hi = (s["q05"], s["q95"]) if args.band == "quantile" else (s["mean"] - s["std"], s["mean"] + s["std"])
                ax.fill_between(s["p"], lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(s["p"], s[args.stat], color=col, ls=ls, label=label, marker=style.marker(i),
                    markerfacecolor="white" if hollow else col, markeredgecolor=col if hollow else "white",
                    markeredgewidth=0.8 if hollow else 0.4)

        for ds in data:
            fig, ax = painter.figure(*args.accuracy_size_mm)
            if draw(ax, ds) == 0:
                print(f"  {ds} : aucun dossier de pruning trouvé")
                import matplotlib.pyplot as plt
                plt.close(fig)
                continue
            if args.legend != "none":
                ax.legend(loc=args.legend, handlelength=2.4)
            painter.save(fig, f"accuracy_{ds}")
        if n_ds > 1:
            fig, axes = painter.figure(args.width_mm, args.accuracy_size_mm[1], 1, n_ds, squeeze=False,
                                       sharey=True, gridspec_kw={"wspace": 0.12})
            for i, (ax, ds) in enumerate(zip(axes[0], data)):
                draw(ax, ds)
                if i:
                    ax.set_ylabel("")
                painter.panel_label(ax, i, data[ds]["label"], x=-0.22 if i == 0 else -0.08)
            if args.legend != "none":
                axes[0, 0].legend(loc=args.legend, handlelength=2.4)
            painter.save(fig, "accuracy_all")

    # 5. Table récapitulative
    if "table" in args.only:
        print(f"5. Table (p = {args.table_p:g})")
        rows = []
        for ds, d in data.items():
            acc0 = np.array([c.get("test_accuracy", np.nan) for c in d["configs"]], float)
            grid = d["layer"].size
            total = int(np.prod(grid))
            for m in table_methods:
                row = {"dataset": d["label"], "model": args.model_label or model_label(d["configs"][0]["args"]["arch"]),
                       "method": m_labels[m], "n_runs": len(acc0),
                       "unpruned_median": np.median(acc0), "unpruned_q05": np.quantile(acc0, .05),
                       "unpruned_q95": np.quantile(acc0, .95), "unpruned_std": acc0.std(), "p": args.table_p}
                for mode in ("pre", "post"):
                    folder = d["pruning"][mode].get(m)
                    vals = {"median": np.nan, "q05": np.nan, "q95": np.nan, "std": np.nan, "p": np.nan}
                    if folder:
                        s = read_summary(folder, mode)
                        if len(s["p"]):
                            k, p_used = nearest(s["p"], args.table_p)
                            if abs(p_used - args.table_p) > 1e-6:
                                print(f"  {ds} {mode} {m} : p = {args.table_p:g} absent de {os.path.basename(folder)}, "
                                      f"p = {p_used:g} utilisé")
                            vals = {"median": s["median"][k], "q05": s["q05"][k], "q95": s["q95"][k],
                                    "std": s["std"][k], "p": p_used}
                    row.update({f"{mode}_{k}": v for k, v in vals.items()})
                keep = d["layer"].to_input(masks(ds, m, args.table_p))   # (runs, F, C/groups, *grid)
                axes_pix = tuple(range(-len(grid), 0))
                used = keep.any(axis=(1, 2)).sum(axis=axes_pix)          # pixels gardés par au moins un filtre
                per_filter = keep.any(axis=2).sum(axis=axes_pix).mean(axis=1)
                row.update({"pixels_total": total, "pixels_used": float(np.median(used)),
                            "pixels_used_fraction": float(np.median(used)) / total,
                            "pixels_per_filter": float(np.median(per_filter)),
                            "pixels_per_filter_fraction": float(np.median(per_filter)) / total})
                rows.append(row)
        with open(os.path.join(args.out, "table.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

        scale = 100.0 if args.table_scale == "percent" else 1.0
        dec = args.table_decimals if args.table_scale == "percent" else args.table_decimals + 2

        def fmt(r, prefix):
            v = r[f"{prefix}_median"]
            if not np.isfinite(v):
                return "--"
            text = f"{scale * v:.{dec}f}"
            if args.table_band == "quantile":
                text += f" [{scale * r[f'{prefix}_q05']:.{dec}f}, {scale * r[f'{prefix}_q95']:.{dec}f}]"
            elif args.table_band == "std":
                text += f" $\\pm$ {scale * r[f'{prefix}_std']:.{dec}f}"
            return text

        unit = " (\\%)" if args.table_scale == "percent" else ""
        pl = f"{args.table_p:g}"
        head = ["Dataset", "Model", "Method", f"No pruning{unit}", f"Pre-training, $p={pl}${unit}",
                f"Post-training, $p={pl}${unit}", "Pixels used", "Pixels per filter"]
        lines = []
        if args.table_caption:
            lines += ["\\begin{table}[t]", "\\centering", f"\\caption{{{args.table_caption}}}",
                      f"\\label{{{args.table_label}}}"]
        lines += ["\\begin{tabular}{lll" + "c" * 5 + "}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
        last = None
        for r in rows:
            if last is not None and r["dataset"] != last:
                lines.append("\\midrule")
            first = r["dataset"] != last
            last = r["dataset"]
            pix = f"{r['pixels_used']:.0f}/{r['pixels_total']} ({100 * r['pixels_used_fraction']:.0f}\\%)"
            pf = f"{r['pixels_per_filter']:.0f} ({100 * r['pixels_per_filter_fraction']:.0f}\\%)"
            lines.append(" & ".join([r["dataset"] if first else "", r["model"] if first else "", r["method"],
                                     fmt(r, "unpruned") if first else "", fmt(r, "pre"), fmt(r, "post"), pix, pf]) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        if args.table_caption:
            lines.append("\\end{table}")
        with open(os.path.join(args.out, "table.tex"), "w") as f:
            f.write("% Nécessite \\usepackage{booktabs}\n" + "\n".join(lines) + "\n")
        print("    ->", os.path.relpath(os.path.join(args.out, "table.csv")), ",",
              os.path.relpath(os.path.join(args.out, "table.tex")))
        print(f"    {'dataset':<14} {'méthode':<9} {'sans':>6} {'pre':>6} {'post':>6} {'pixels':>11} {'par filtre':>10}")
        for r in rows:
            f6 = lambda v: f"{100 * v:6.1f}" if np.isfinite(v) else f"{'--':>6}"
            print(f"    {r['dataset']:<14} {r['method']:<9} {f6(r['unpruned_median'])} {f6(r['pre_median'])} "
                  f"{f6(r['post_median'])} {r['pixels_used']:>5.0f}/{r['pixels_total']:<5} {r['pixels_per_filter']:>10.0f}")

    settings = {"command": " ".join(sys.argv), "args": vars(args),
                "datasets": {ds: {"group": d["group"]["dir"], "runs": d["group"]["runs"],
                                  "pruning": d["pruning"], "layer": d["layer"].meta["layer"],
                                  "conv_index": d["layer"].meta["conv_index"], "criterion": d["criterion"],
                                  "include_padding": d["include_padding"]} for ds, d in data.items()}}
    with open(os.path.join(args.out, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
    print(f"Terminé : {args.out}/")


if __name__ == "__main__":
    main()
