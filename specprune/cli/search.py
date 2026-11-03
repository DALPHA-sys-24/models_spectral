"""
Recherche du learning rate et du batch size d'un modèle, à nombre d'epochs fixé (20 par défaut).

Chaque essai est un run complet de run_experiment.py, lancé dans un processus séparé
(la mémoire GPU est libérée entre deux essais). Le choix se fait sur un jeu de validation
tiré du train (--val_split, 0.1 par défaut) : le test n'intervient jamais dans le choix.

Exemples :
    # grille par défaut (lr 0.01 0.05 0.1 0.2 x batch 64 128 256) sur CIFAR-10
    python search_hparams.py --arch spectral_resnet20 --spectral_config configs/spectral_config.json

    # AdamW (la grille de lr par défaut s'adapte : 1e-4 3e-4 1e-3 3e-3)
    python search_hparams.py --arch spectral_resnet20 --optimizer adamw

    # grille choisie, autre dataset, et arguments supplémentaires pour run_experiment après "--"
    python search_hparams.py --arch spectral_mobilenetv2 --spectral_config configs/classique.json \
        --dataset fashion_mnist --lr 0.02 0.05 0.1 --batch_size 64 128 -- --weight_decay 1e-4

    # signaux (.npz avec x en (N, T, C)) : mêmes modèles en SpecCnn1D
    python search_hparams.py --arch spectral_resnet20 --dataset npz \
        -- --data_path signaux.npz --modality signal --crop_pad 16

    # signaux téléchargés automatiquement
    python search_hparams.py --arch spectral_resnet20 --dataset mnist1d

    # recherche aléatoire : 12 essais, lr log-uniforme dans [1e-3, 0.5]
    python search_hparams.py --arch spectral_resnet20 --search random --n_trials 12 \
        --lr_range 1e-3 0.5 --batch_size 64 128 256

    # reprendre une recherche interrompue (les essais déjà terminés sont sautés)
    python search_hparams.py --resume runs_search/cifar10_spectral_resnet20_20261023_101500

Sorties dans <out_dir>/<dataset>_<arch>_<date>/ :
    search.json      paramètres de la recherche (pour la reprendre)
    results.csv      un essai par ligne : lr, batch, précisions, temps, statut
    best.json        meilleur essai + commande d'entraînement complet à lancer
    trials/          dossiers de run de chaque essai (config, history, metrics, lambda_in...)
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

from .. import PROJECT_ROOT
from .common import (DEFAULT_SPECTRAL_CONFIG, Formatter, add_device, dataset_tag,
                     resolve_spectral_config)
RESULT_FIELDS = ["trial", "lr", "batch_size", "status", "best_val_accuracy", "best_epoch",
                 "final_val_accuracy", "final_train_accuracy", "test_accuracy",
                 "train_time_s", "run_dir"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=Formatter)
    p.add_argument("--resume", default="", help="dossier d'une recherche à reprendre")

    g = p.add_argument_group("modèle et données")
    g.add_argument("--arch", default="spectral_resnet20")
    g.add_argument("--spectral_config", default=DEFAULT_SPECTRAL_CONFIG)
    g.add_argument("--dataset", default="cifar10")
    g.add_argument("--val_split", type=float, default=0.1,
                   help="part du train gardée pour la validation (le choix se fait dessus)")

    g = p.add_argument_group("recherche")
    g.add_argument("--search", choices=["grid", "random"], default="grid")
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd",
                   help="sgd = Nesterov, momentum 0.9 (réglables après --)")
    g.add_argument("--lr", type=float, nargs="+", default=None,
                   help="valeurs de lr (grille) ; défaut : 0.01 0.05 0.1 0.2 (sgd), "
                        "1e-4 3e-4 1e-3 3e-3 (adamw)")
    g.add_argument("--batch_size", type=int, nargs="+", default=[64, 128, 256],
                   help="valeurs de batch (grille et aléatoire)")
    g.add_argument("--lr_range", type=float, nargs=2, default=None,
                   help="bornes du lr, tiré log-uniforme (aléatoire) ; défaut : 1e-3 0.5 (sgd), "
                        "1e-5 1e-2 (adamw)")
    g.add_argument("--n_trials", type=int, default=12, help="nombre d'essais (aléatoire)")
    g.add_argument("--metric", choices=["best_val_accuracy", "final_val_accuracy"],
                   default="best_val_accuracy", help="critère de choix")
    g.add_argument("--seed", type=int, default=42, help="graine des runs et du tirage aléatoire")

    g = p.add_argument_group("exécution")
    add_device(g)
    g.add_argument("--out_dir", default="runs_search")
    g.add_argument("--save_weights", action=argparse.BooleanOptionalAction, default=False,
                   help="enregistrer les poids de chaque essai")

    # Tout ce qui suit "--" est transmis tel quel à run_experiment.py
    argv = sys.argv[1:] if argv is None else list(argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = p.parse_args(argv)
    args.extra = extra
    if "--optimizer" in extra:
        p.error("donner --optimizer avant --, pas après")
    if args.lr is None:
        args.lr = [0.01, 0.05, 0.1, 0.2] if args.optimizer == "sgd" else [1e-4, 3e-4, 1e-3, 3e-3]
    if args.lr_range is None:
        args.lr_range = [1e-3, 0.5] if args.optimizer == "sgd" else [1e-5, 1e-2]
    return args


def make_trials(args):
    """Liste des couples (lr, batch) à essayer."""
    if args.search == "grid":
        return [(lr, bs) for bs in args.batch_size for lr in args.lr]
    rng = np.random.default_rng(args.seed)
    lo, hi = math.log10(args.lr_range[0]), math.log10(args.lr_range[1])
    return [(float(f"{10 ** rng.uniform(lo, hi):.3g}"), int(rng.choice(args.batch_size)))
            for _ in range(args.n_trials)]


def trial_name(lr, bs):
    return f"lr{lr:g}_bs{bs}"


def read_results(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {row["trial"]: row for row in csv.DictReader(f)}


def write_results(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)


def collect(run_dir):
    """Lit metrics.json et history.csv d'un essai terminé."""
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    with open(os.path.join(run_dir, "history.csv")) as f:
        last = list(csv.DictReader(f))[-1]
    return {
        "best_val_accuracy": f"{m.get('best_val_accuracy', float('nan')):.4f}",
        "best_epoch": m.get("best_epoch", ""),
        "final_val_accuracy": f"{float(last['val_accuracy']):.4f}",
        "final_train_accuracy": f"{m['final_train_accuracy']:.4f}",
        "test_accuracy": f"{m['test_accuracy']:.4f}",
        "train_time_s": m["train_time_s"],
    }


def run_trial(args, search_dir, lr, bs):
    name = trial_name(lr, bs)
    trials_dir = os.path.join(search_dir, "trials")
    run_dir = os.path.join(trials_dir, name)
    train = [sys.executable, "-m", "specprune.cli.train"]
    if os.path.isdir(os.path.join(run_dir, "backup")):  # essai interrompu : reprise à la dernière epoch
        cmd = train + ["--resume", run_dir, "--device", args.device]
    else:
        if os.path.isdir(run_dir):  # interrompu avant la première epoch : on repart de zéro
            os.replace(run_dir, f"{run_dir}_interrompu_{datetime.now():%H%M%S}")
        cmd = train + [
            "--arch", args.arch, "--spectral_config", args.spectral_config,
            "--dataset", args.dataset, "--val_split", str(args.val_split),
            "--epochs", str(args.epochs), "--optimizer", args.optimizer, "--lr", str(lr),
            "--batch_size", str(bs), "--seed", str(args.seed), "--device", args.device,
            "--out_dir", trials_dir, "--run_name", name,
            "--save_weights" if args.save_weights else "--no-save_weights", *args.extra]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [PROJECT_ROOT, os.environ.get("PYTHONPATH")]))}
    row = {"trial": name, "lr": lr, "batch_size": bs, "run_dir": run_dir}
    print(f"\n=== Essai {name} ===\n{' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0 or not os.path.exists(os.path.join(run_dir, "metrics.json")):
        # ex. mémoire GPU insuffisante pour ce batch : l'essai est noté en échec, on continue
        row.update(status=f"echec (code {proc.returncode})", train_time_s=round(time.time() - t0, 1))
        return row
    row.update(status="ok", **collect(run_dir))
    return row


def score(row, metric):
    try:
        return float(row[metric]) if row.get("status") == "ok" else -1.0
    except (TypeError, ValueError):
        return -1.0


def main(argv=None):
    args = parse_args(argv)
    if args.resume:
        search_dir = args.resume
        with open(os.path.join(search_dir, "search.json")) as f:
            saved = json.load(f)
        for key, value in saved["args"].items():
            setattr(args, key, value)
        trials = [tuple(t) for t in saved["trials"]]
    else:
        search_dir = os.path.join(args.out_dir, f"{dataset_tag(args.dataset)}_{args.arch}_{datetime.now():%Y%m%d_%H%M%S}")
        os.makedirs(os.path.join(search_dir, "trials"))
        config = resolve_spectral_config(args.spectral_config)
        args.spectral_config = os.path.abspath(config) if config else ""
        trials = make_trials(args)
        with open(os.path.join(search_dir, "search.json"), "w") as f:
            json.dump({"args": {k: v for k, v in vars(args).items() if k != "resume"},
                       "trials": trials, "command": " ".join([sys.executable] + sys.argv),
                       "started_at": datetime.now().isoformat(timespec="seconds")}, f, indent=2)

    results_path = os.path.join(search_dir, "results.csv")
    done = read_results(results_path)
    print(f"Recherche : {search_dir}\n{args.arch} | {args.dataset} | {os.path.basename(args.spectral_config)} "
          f"| {args.optimizer} | {args.epochs} epochs | {len(trials)} essais ({len(done)} déjà faits)")

    for i, (lr, bs) in enumerate(trials, 1):
        name = trial_name(lr, bs)
        if name in done and done[name]["status"] == "ok":
            continue
        print(f"\n[{i}/{len(trials)}]", end="")
        done[name] = run_trial(args, search_dir, lr, bs)
        rows = [done[trial_name(*t)] for t in trials if trial_name(*t) in done]
        write_results(results_path, rows)

    rows = [done[trial_name(*t)] for t in trials if trial_name(*t) in done]
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        sys.exit("Aucun essai n'a abouti : voir les messages ci-dessus.")

    # Tableau final trié par le critère choisi
    ok.sort(key=lambda r: score(r, args.metric), reverse=True)
    print(f"\nRésultats ({args.metric}, {args.epochs} epochs) :")
    print(f"{'lr':>8} {'batch':>6} {'val best':>9} {'val fin':>8} {'train fin':>9} {'temps(s)':>9}")
    for r in ok:
        print(f"{float(r['lr']):>8g} {int(r['batch_size']):>6} {r['best_val_accuracy']:>9} "
              f"{r['final_val_accuracy']:>8} {r['final_train_accuracy']:>9} {r['train_time_s']:>9}")
    for r in rows:
        if r["status"] != "ok":
            print(f"{float(r['lr']):>8g} {int(r['batch_size']):>6}  {r['status']}")

    best = ok[0]
    final_cmd = (f"python run_experiment.py --arch {args.arch} --spectral_config {args.spectral_config} "
                 f"--dataset {args.dataset} --optimizer {args.optimizer} --lr {float(best['lr']):g} --batch_size {int(best['batch_size'])} "
                 f"--device {args.device} {' '.join(args.extra)}").strip()
    with open(os.path.join(search_dir, "best.json"), "w") as f:
        json.dump({"metric": args.metric, "epochs": args.epochs, "best": best,
                   "final_training_command": final_cmd}, f, indent=2)
    print(f"\nMeilleur : lr={float(best['lr']):g}, batch={int(best['batch_size'])} "
          f"({args.metric} = {best[args.metric]})\nEntraînement complet :\n  {final_cmd}")


if __name__ == "__main__":
    main()
