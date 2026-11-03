"""
Figures de publication à partir des résultats enregistrés (aucun calcul, aucun modèle chargé).

Trois figures, chacune réglable en entier en ligne de commande (voir specprune/plotting.py) :
    pruning   précision test selon p, depuis un ou plusieurs dossiers de pruning.py
    history   courbes d'entraînement, depuis un ou plusieurs runs (history.csv)
    masks     part des lambda_in gardés à chaque position, depuis les masques de pruning.py

Exemples :
    # pré et post-training d'un pruning (orange, bleu), en PDF et PNG
    specprune-plot pruning runs_pruning/mnist1d_spectral_resnet20_both_20261102_093000

    # comparer trois méthodes en post-training (orange, bleu, vert), noms choisis
    specprune-plot pruning runs_pruning/*_post_* --modes post \
        --labels Gradient Element Mixed --chance 0.1 --out figures/methods

    # courbes d'entraînement de 5 graines : médiane et bande 5 %-95 % (--individual : une par run)
    specprune-plot history runs/cifar10_5seeds/seed* --width_mm 183

    # masques (signal : p en y, temps en x ; image : une carte par p)
    specprune-plot masks runs_pruning/<dossier> --p 0.2 0.5 0.8

Sans --out, la figure va dans <dossier>/figures/<nom>.{pdf,png} (history : dossier commun des runs).
"""
import argparse
import os
import sys

from .common import Formatter
from ..plotting import COLORS, Style, plot_history, plot_masks, plot_pruning, save


def style_args(p):
    g = p.add_argument_group("style")
    g.add_argument("--out", default="", help="fichier de sortie sans extension")
    g.add_argument("--formats", nargs="+", default=["pdf", "png"], help="pdf png svg tif...")
    g.add_argument("--dpi", type=int, default=600)
    g.add_argument("--width_mm", type=float, default=89.0, help="89 = une colonne, 183 = deux")
    g.add_argument("--height_mm", type=float, default=60.0)
    g.add_argument("--font", default="Arial")
    g.add_argument("--font_size", type=float, default=7.0)
    g.add_argument("--line_width", type=float, default=1.2)
    g.add_argument("--marker_size", type=float, default=3.5)
    g.add_argument("--no_markers", action="store_true")
    g.add_argument("--colors", nargs="+", default=COLORS, help="couleurs des courbes, dans l'ordre")
    g.add_argument("--grid", action="store_true")
    g.add_argument("--legend", default="best", help='position ("lower left"...) ou "none"')
    g.add_argument("--xlabel", default=None)
    g.add_argument("--ylabel", default=None)
    g.add_argument("--xlim", type=float, nargs=2, default=None)
    g.add_argument("--ylim", type=float, nargs=2, default=None)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=Formatter)
    sub = p.add_subparsers(dest="figure", required=True)

    s = sub.add_parser("pruning", formatter_class=Formatter, help="précision test selon p (pruning.py)")
    s.add_argument("dirs", nargs="+", help="dossiers de runs_pruning/")
    s.add_argument("--modes", nargs="+", choices=["pre", "post"], default=["pre", "post"])
    s.add_argument("--stat", choices=["median", "mean"], default="median")
    s.add_argument("--band", choices=["quantile", "std", "none"], default="quantile",
                   help="bande 5 %%-95 %%, ± écart-type, ou rien (si plusieurs runs)")
    s.add_argument("--labels", nargs="+", default=None, help="un nom par courbe, dans l'ordre")
    s.add_argument("--percent_x", action="store_true", help="p en pourcentage")
    s.add_argument("--chance", type=float, default=None, help="niveau du hasard (ex. 0.1)")
    style_args(s)

    s = sub.add_parser("history", formatter_class=Formatter, help="courbes d'entraînement (history.csv)")
    s.add_argument("runs", nargs="+", help="dossiers de run ou fichiers history.csv")
    s.add_argument("--metrics", nargs="+", default=["accuracy", "val_accuracy"])
    s.add_argument("--labels", nargs="+", default=None, help="un nom par courbe, dans l'ordre")
    s.add_argument("--stat", choices=["median", "mean"], default="median",
                   help="plusieurs runs : courbe centrale")
    s.add_argument("--band", choices=["quantile", "std", "none"], default="quantile",
                   help="plusieurs runs : bande 5 %%-95 %%, ± écart-type, ou rien")
    s.add_argument("--individual", action="store_true",
                   help="plusieurs runs : une courbe par run au lieu de la médiane et de la bande")
    style_args(s)

    s = sub.add_parser("masks", formatter_class=Formatter, help="part des lambda_in gardés par position (pruning.py)")
    s.add_argument("dir", help="dossier de runs_pruning/")
    s.add_argument("--run", default=None, help="run du dossier (défaut : le premier)")
    s.add_argument("--layer", default=None, help="couche (défaut : la première élaguée)")
    s.add_argument("--p", type=float, nargs="+", default=None, help="ordres p affichés")
    s.add_argument("--cbar_label", default="Fraction of weights kept")
    style_args(s)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    style = Style(width_mm=args.width_mm, height_mm=args.height_mm, font=args.font,
                  font_size=args.font_size, line_width=args.line_width,
                  marker_size=args.marker_size, colors=args.colors, markers=not args.no_markers,
                  grid=args.grid, legend=args.legend, formats=args.formats, dpi=args.dpi)
    labels = {k: v for k, v in (("xlabel", args.xlabel), ("ylabel", args.ylabel)) if v is not None}
    try:
        if args.figure == "pruning":
            fig = plot_pruning(args.dirs, style, args.modes, args.stat, args.band, args.labels,
                               percent_x=args.percent_x, xlim=args.xlim,
                               ylim=args.ylim or (0, 1.02), chance=args.chance, **labels)
            default_out = os.path.join(args.dirs[0], "figures", "accuracy_vs_p")
        elif args.figure == "history":
            files = [os.path.join(r, "history.csv") if os.path.isdir(r) else r for r in args.runs]
            fig = plot_history(files, style, args.metrics, args.labels, xlim=args.xlim,
                               ylim=args.ylim, stat=args.stat, band=args.band,
                               individual=args.individual, **labels)
            dirs = [os.path.dirname(os.path.abspath(f)) for f in files]
            # un run : son dossier ; plusieurs : leur dossier commun (ex. le groupe --n_runs)
            base = dirs[0] if len(set(dirs)) == 1 else os.path.commonpath(dirs)
            default_out = os.path.join(base, "figures", "history")
        else:
            fig = plot_masks(args.dir, style, args.run, args.layer, args.p,
                             cbar_label=args.cbar_label, **labels)
            default_out = os.path.join(args.dir, "figures", "masks")
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Erreur : {e}")
    for f in save(fig, args.out or default_out, style):
        print(f)


if __name__ == "__main__":
    main()
