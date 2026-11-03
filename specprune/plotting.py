"""
Figures de publication (style Nature Communications) à partir des résultats enregistrés.

Les scripts enregistrent toutes les données (results.csv, summary.csv, history.csv, masks/*.npz) ;
les figures se font ensuite, à part, avec ce module ou avec plot_results.py en ligne de commande.

Style : pas de titre, seulement les noms des axes (en anglais) ; police sans empattement de 7 pt ;
traits fins ; axes haut et droit retirés ; largeur d'une colonne Nature (89 mm) ou de deux
(183 mm) ; export vectoriel (PDF, SVG, polices incorporées) et PNG 600 dpi.
Couleurs : orange, bleu, puis vert (palette Okabe-Ito, lisible par les daltoniens).

    from specprune.plotting import Style, plot_pruning, save
    style = Style(width_mm=89)
    fig = plot_pruning(["runs_pruning/mnist1d_spectral_resnet20_both_20261102_093000"], style)
    save(fig, "figures/accuracy_vs_p", style)
"""
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib

# Hors notebook (scripts), les figures sont seulement enregistrées : backend sans affichage,
# qui ne dépend pas d'un serveur graphique (sous WSL, DISPLAY peut être défini sans écran).
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

MM = 1 / 25.4                                      # mm -> pouces
COLORS = ["#E07B00", "#0072B2", "#009E73"]         # orange, bleu, vert
MARKERS = ["o", "s", "^", "D", "v"]
MODE_LABELS = {"pre": "Pre-training", "post": "Post-training"}


@dataclass
class Style:
    """Réglages communs à toutes les figures (tous modifiables en ligne de commande)."""
    width_mm: float = 89.0           # 89 = une colonne, 183 = deux colonnes
    height_mm: float = 60.0
    font: str = "Arial"              # repli : Helvetica, Liberation Sans, DejaVu Sans
    font_size: float = 7.0           # texte ; graduations et légende 1 pt plus petites
    line_width: float = 1.2
    marker_size: float = 3.5
    axis_width: float = 0.6
    colors: List[str] = field(default_factory=lambda: list(COLORS))
    markers: bool = True
    grid: bool = False
    legend: str = "best"             # position matplotlib, ou "none"
    formats: List[str] = field(default_factory=lambda: ["pdf", "png"])
    dpi: int = 600

    def apply(self) -> None:
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": [self.font, "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": self.font_size, "axes.labelsize": self.font_size,
            "xtick.labelsize": self.font_size - 1, "ytick.labelsize": self.font_size - 1,
            "legend.fontsize": self.font_size - 1, "legend.frameon": False,
            "axes.linewidth": self.axis_width, "axes.spines.top": False, "axes.spines.right": False,
            "xtick.major.width": self.axis_width, "ytick.major.width": self.axis_width,
            "xtick.major.size": 3, "ytick.major.size": 3,
            "xtick.direction": "out", "ytick.direction": "out",
            "axes.grid": self.grid, "grid.linewidth": 0.4, "grid.color": "#d9d9d9",
            "lines.linewidth": self.line_width, "lines.markersize": self.marker_size,
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
            "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        })

    def figure(self, nrows: int = 1, ncols: int = 1, **kw):
        self.apply()
        return plt.subplots(nrows, ncols, figsize=(self.width_mm * MM, self.height_mm * MM), **kw)

    def color(self, i: int) -> str:
        return self.colors[i % len(self.colors)]

    def marker(self, i: int) -> Optional[str]:
        return MARKERS[i % len(MARKERS)] if self.markers else None


def save(fig, path: str, style: Style) -> List[str]:
    """Enregistre la figure dans chaque format (path sans extension). Renvoie les fichiers écrits."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = []
    for fmt in style.formats:
        f = f"{path}.{fmt}"
        fig.savefig(f, dpi=style.dpi if fmt in ("png", "jpg", "tif", "tiff") else None)
        out.append(f)
    plt.close(fig)
    return out


def _legend(ax, style: Style, n_series: int) -> None:
    if style.legend != "none" and n_series > 1:
        ax.legend(loc=style.legend, handlelength=1.8)


def _limits(ax, xlim, ylim) -> None:
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


# --- Précision selon p (pruning.py) -----------------------------------------------------

def pruning_curves(pruning_dirs: Sequence[str], modes: Sequence[str] = ("pre", "post")) -> List[Dict]:
    """
    Courbes d'un ou plusieurs dossiers de pruning.py (summary.csv) : une par (dossier, mode).
    Chaque courbe : {"dir", "mode", "method", "p", "mean", "median", "q05", "q95", "std", "n"}.
    """
    curves = []
    for d in pruning_dirs:
        rows = _read_csv(os.path.join(d, "summary.csv"))
        with open(os.path.join(d, "config.json")) as f:
            args = json.load(f)["args"]
        for mode in modes:
            r = sorted((x for x in rows if x["mode"] == mode), key=lambda x: float(x["p"]))
            if r:
                curves.append({"dir": d, "mode": mode, "method": args.get("method", ""),
                               "n": max(int(x["n"]) for x in r),
                               **{k: np.array([float(x[k]) for x in r])
                                  for k in ("p", "mean", "median", "q05", "q95", "std")}})
    if not curves:
        raise ValueError(f"aucune courbe des modes {list(modes)} dans {list(pruning_dirs)}")
    return curves


def default_label(curve: Dict, several_dirs: bool) -> str:
    label = MODE_LABELS.get(curve["mode"], curve["mode"])
    return f"{curve['method']}, {label.lower()}" if several_dirs and curve["method"] else label


def plot_pruning(pruning_dirs: Sequence[str], style: Style, modes: Sequence[str] = ("pre", "post"),
                 stat: str = "median", band: str = "quantile", labels: Optional[Sequence[str]] = None,
                 xlabel: str = "Pruning ratio p", ylabel: str = "Accuracy",
                 percent_x: bool = False, xlim=None, ylim=(0, 1.02), chance: Optional[float] = None):
    """
    Précision test selon p. stat : "median" ou "mean" ; band : "quantile" (5 %-95 %), "std" ou
    "none" (la bande n'est tracée que s'il y a plusieurs runs ou répétitions).
    chance : niveau du hasard (ex. 0.1 pour 10 classes), en pointillés.
    """
    curves = pruning_curves(pruning_dirs, modes)
    if labels and len(labels) != len(curves):
        raise ValueError(f"{len(labels)} noms pour {len(curves)} courbes "
                         f"({[default_label(c, True) for c in curves]})")
    fig, ax = style.figure()
    scale = 100.0 if percent_x else 1.0
    for i, c in enumerate(curves):
        x, y, col = c["p"] * scale, c[stat], style.color(i)
        if band != "none" and c["n"] > 1:
            lo, hi = (c["q05"], c["q95"]) if band == "quantile" else (c["mean"] - c["std"], c["mean"] + c["std"])
            ax.fill_between(x, lo, hi, color=col, alpha=0.2, lw=0)
        label = labels[i] if labels else default_label(c, len(pruning_dirs) > 1)
        ax.plot(x, y, color=col, marker=style.marker(i), label=label,
                markeredgecolor="white", markeredgewidth=0.4)
    if chance is not None:
        ax.axhline(chance, color="#8c8c8c", lw=0.6, ls="--", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _limits(ax, xlim, ylim)
    _legend(ax, style, len(curves))
    return fig


# --- Courbes d'entraînement (history.csv) -----------------------------------------------

METRIC_LABELS = {"accuracy": "Train", "val_accuracy": "Validation", "loss": "Train",
                 "val_loss": "Validation"}


def _metric_labels(history_file: str) -> Dict[str, str]:
    """Noms des métriques ; val_* = « Test » si le run n'avait pas de validation (val_split = 0)."""
    labels = dict(METRIC_LABELS)
    config = os.path.join(os.path.dirname(os.path.abspath(history_file)), "config.json")
    if os.path.exists(config):
        with open(config) as f:
            if json.load(f).get("validation_on") == "test":
                labels.update(val_accuracy="Test", val_loss="Test")
    return labels


def _aggregate(curves: List[np.ndarray], stat: str, band: str):
    """Courbes de plusieurs runs (même longueur) -> (centre, bas, haut) ; bande None si band="none"."""
    y = np.stack(curves)
    center = np.median(y, axis=0) if stat == "median" else y.mean(axis=0)
    if band == "quantile":
        return center, np.quantile(y, 0.05, axis=0), np.quantile(y, 0.95, axis=0)
    if band == "std":
        return center, y.mean(axis=0) - y.std(axis=0), y.mean(axis=0) + y.std(axis=0)
    return center, None, None


def plot_history(history_files: Sequence[str], style: Style,
                 metrics: Sequence[str] = ("accuracy", "val_accuracy"),
                 labels: Optional[Sequence[str]] = None, xlabel: str = "Epoch",
                 ylabel: str = "Accuracy", xlim=None, ylim=None, stat: str = "median",
                 band: str = "quantile", individual: bool = False):
    """
    Courbes d'entraînement. Un seul run : une courbe par métrique. Plusieurs runs (ex. plusieurs
    graines) : par défaut une courbe par métrique, médiane (ou moyenne) des runs avec une bande
    5 %-95 % (ou ± écart-type), sur les epochs communes à tous les runs ; individual=True trace
    plutôt une courbe par (run, métrique).
    """
    runs = []
    for path in history_files:
        rows = _read_csv(path)
        runs.append((os.path.basename(os.path.dirname(os.path.abspath(path))), rows))
    metrics = [m for m in metrics if any(rows and m in rows[0] for _, rows in runs)]
    names = _metric_labels(history_files[0])
    if not metrics:
        raise ValueError(f"aucune des métriques demandées dans {list(history_files)}")

    series = []  # (nom par défaut, x, centre, bas, haut)
    if len(runs) == 1 or individual:
        for run, rows in runs:
            for m in metrics:
                if rows and m in rows[0]:
                    name = names.get(m, m) if len(runs) == 1 else f"{run}, {names.get(m, m).lower()}"
                    series.append((name, np.arange(1, len(rows) + 1),
                                   np.array([float(r[m]) for r in rows]), None, None))
    else:
        for m in metrics:
            curves = [np.array([float(r[m]) for r in rows]) for _, rows in runs if rows and m in rows[0]]
            n = min(len(c) for c in curves)
            if any(len(c) != n for c in curves):
                print(f"  {m} : runs de longueurs différentes, {n} epochs communes tracées")
            center, lo, hi = _aggregate([c[:n] for c in curves], stat, band)
            series.append((names.get(m, m), np.arange(1, n + 1), center, lo, hi))
    if labels and len(labels) != len(series):
        raise ValueError(f"{len(labels)} noms pour {len(series)} courbes ({[s[0] for s in series]})")

    fig, ax = style.figure()
    for i, (name, x, y, lo, hi) in enumerate(series):
        col = style.color(i)
        if lo is not None:
            ax.fill_between(x, lo, hi, color=col, alpha=0.2, lw=0)
        ax.plot(x, y, color=col, label=labels[i] if labels else name,
                marker=style.marker(i) if len(x) <= 25 else None,
                markeredgecolor="white", markeredgewidth=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _limits(ax, xlim, ylim)
    _legend(ax, style, len(series))
    return fig


# --- Masques : part des lambda_in gardés à chaque position ------------------------------

def kept_fraction(mask: np.ndarray, meta: Dict) -> np.ndarray:
    """Masque (F, C, L) -> part des poids gardés par position d'entrée, padding retiré : (T,) ou (H, W)."""
    grid = mask.reshape(mask.shape[:2] + tuple(meta["padded_grid"])).mean(axis=(0, 1))
    crop = tuple(slice(pad, pad + size) for pad, size in zip(meta["padding"], meta["input_hw"]))
    return grid[crop]


def load_masks(pruning_dir: str, run: Optional[str] = None, layer: Optional[str] = None):
    """
    Masques enregistrés par pruning.py pour un run : {p: part gardée par position}, et la
    description de la couche (lambda_in_files du run).
    """
    with open(os.path.join(pruning_dir, "config.json")) as f:
        runs = json.load(f)["runs"]
    by_name = {os.path.basename(os.path.normpath(r)): r for r in runs}
    run = run or next(iter(by_name))
    if run not in by_name:
        raise ValueError(f"run '{run}' absent de {pruning_dir} : {list(by_name)}")
    with open(os.path.join(by_name[run], "config.json")) as f:
        files = {m["layer"]: m for m in json.load(f)["lambda_in_files"]}
    folder = os.path.join(pruning_dir, run, "masks")
    kept, meta = {}, None
    for name in sorted(os.listdir(folder)):
        data = np.load(os.path.join(folder, name))
        key = layer or data.files[0]
        if key not in files:
            raise ValueError(f"couche '{key}' inconnue : {list(files)}")
        meta = files[key]
        kept[float(name[1:-4])] = kept_fraction(data[key], meta)
    return kept, meta


KEPT_CMAP = LinearSegmentedColormap.from_list("kept", [COLORS[0], "#f2f2f2", COLORS[1]])


def plot_masks(pruning_dir: str, style: Style, run: Optional[str] = None, layer: Optional[str] = None,
               p_values: Optional[Sequence[float]] = None, xlabel: Optional[str] = None,
               ylabel: Optional[str] = None, cbar_label: str = "Fraction of weights kept"):
    """
    Part des lambda_in gardés à chaque position d'entrée (orange = retiré, bleu = gardé).
    Signaux : une carte p (axe y) x temps (axe x). Images : une carte par p, côte à côte.
    """
    kept, meta = load_masks(pruning_dir, run, layer)
    ps = sorted(kept) if p_values is None else [p for p in sorted(kept) if any(abs(p - q) < 1e-9 for q in p_values)]
    if not ps:
        raise ValueError(f"aucun des p demandés parmi {sorted(kept)}")
    if meta["dim"] == 1:
        fig, ax = style.figure()
        img = np.stack([kept[p] for p in ps])
        im = ax.imshow(img, cmap=KEPT_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                       origin="lower", extent=[-0.5, img.shape[1] - 0.5, -0.5, len(ps) - 0.5])
        ax.set_yticks(range(len(ps)), [f"{p:g}" for p in ps])
        ax.set_xlabel(xlabel or "Time step")
        ax.set_ylabel(ylabel or "Pruning ratio p")
        axes = [ax]
    else:
        fig, axes = style.figure(1, len(ps), squeeze=False)
        axes = list(axes[0])
        for a, p in zip(axes, ps):
            im = a.imshow(kept[p], cmap=KEPT_CMAP, vmin=0, vmax=1, interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            a.set_xlabel(f"p = {p:g}")
            for s in a.spines.values():
                s.set_visible(False)
        if xlabel:
            fig.supxlabel(xlabel, fontsize=style.font_size)
    cb = fig.colorbar(im, ax=axes, fraction=0.04 if meta["dim"] == 1 else 0.02, pad=0.02)
    cb.set_label(cbar_label)
    cb.outline.set_linewidth(style.axis_width)
    cb.ax.tick_params(width=style.axis_width)
    return fig
