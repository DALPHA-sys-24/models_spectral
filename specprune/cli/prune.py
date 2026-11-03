"""
Pruning d'ordre p des poids spectraux d'un ou plusieurs runs de run_experiment.py.

Pour chaque run et chaque ordre p (fraction de poids retirés, de 0 à 1) :
    pre   pruning des lambda_in du modèle entraîné, puis prédiction directe sur le test
    post  pruning des lambda_in, valeurs spectrales figées (lambda_in, lambda_out, alpha),
          réentraînement des seuls filtres phi, puis prédiction sur le test
    both  les deux, avec les mêmes masques (comparaison directe)
Chaque p repart du modèle entraîné du run (weights.weights.h5) ; les données sont celles du run
(même dataset, mêmes sous-ensembles, même normalisation).

Exemples :
    # pré-training : précision selon p, sans réentraînement
    python pruning.py runs/cifar10_spectral_resnet20_20261030_101500

    # post-training : 20 epochs de réentraînement des phi pour chaque p
    python pruning.py runs/cifar10_spectral_resnet20_20261030_101500 --mode post --epochs 20

    # les deux, sur plusieurs runs (graines) : médiane et bande 5 %-95 % par p
    python pruning.py runs/mnist1d_spectral_resnet20_* --mode both --p 0 0.25 0.5 0.75 0.9

    # pixels retirés pour tous les filtres (mixte), ou poids par poids dans chaque filtre (gradient)
    python pruning.py runs/<run> --method mixte

    # reprendre un pruning interrompu (les lignes déjà faites sont sautées)
    python pruning.py --resume runs_pruning/cifar10_spectral_resnet20_both_20261030_120000

    # répartir le travail (runs x p) sur deux GPU : un processus par GPU, résultats réunis à la fin
    python pruning.py runs/cifar10_5seeds/seed* --mode post --gpus 0 1

Réglages du pruning (voir specprune/prune.py) :
    --param        poids élagué : lambda_in (défaut), lambda_out ou alpha
    --layers       trained (couches où ce poids est entraîné, défaut), "*", "0", "1-5", "0,3"
    --method       gradient  par filtre, percentile sur tous les |w| du filtre, poids par poids
                   element   par filtre, score d'un pixel = somme des |w| sur les canaux (défaut)
                   mixte     comme element, mais score sommé sur tous les filtres
    --criterion    percentile (défaut : |w| < percentile d'ordre p mis à zéro) ou rank (round(p*n))
Chaque couche est traitée séparément ; les positions de padding de lambda_in sont exclues
(--include_padding pour les compter). Les runs donnés ensemble doivent être comparables (même
dataset, architecture et configuration spectrale) : sinon le script s'arrête (--allow_mixed pour
passer outre).

Post-training : --train phi (défaut, seuls les phi bougent ; gamma/beta des BatchNorm et tête
figés, les BatchNorm normalisant avec les statistiques du batch) ou
non_spectral (phi + BatchNorm + tête). Recette : celle du run (optimiseur, batch, weight decay,
schedule, augmentation), avec --epochs 20 et le même lr que le run par défaut ; tout se change en option.

Sorties dans <out_dir>/<dataset>_<arch>_<mode>_<date>/ (données seulement ; les figures se font
avec plot_results.py, qui les range dans <dossier>/figures/) :
    config.json       arguments, runs traités, commande, versions
    results.csv       une ligne par (run, mode, p, répétition) : poids retirés, précision, perte...
                      (avec --gpus : results_part<k>.csv, un par processus, lus ensemble)
    summary.csv       par (mode, p) : n, moyenne, médiane, quantiles 5 % et 95 %, écart-type
    <run>/masks/p0.50.npz          masques gardés (True = conservé) de chaque couche élaguée
    <run>/post/p0.50_r0/history.csv  réentraînement (+ weights.weights.h5 avec --save_weights)
"""
import argparse
import csv
import gc
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import tensorflow as tf

from .. import FORMAT_VERSION, PROJECT_ROOT
from ..checks import check_format, runs_differences, spectral_warnings
from ..data import make_tf_datasets
from ..prune import (CRITERIA, METHODS, PARAMS, apply_masks, evaluate_pruned, freeze_spectral,
                     get_spectral_values, masks_for, select_layers)
from ..runs import (WEIGHTS_FILE, environment_info, load_run, load_run_data, read_config,
                    test_dataset)
from ..training import compile_agent, onehot_datasets, recipe_from_run, setup_device
from .common import add_device, add_verbose, dataset_tag, parser

RESULT_FIELDS = ["run", "mode", "p", "repeat", "seed", "pruned", "total", "fraction_pruned",
                 "test_accuracy", "test_loss", "final_train_accuracy", "epochs", "lr",
                 "train_time_s"]
DEFAULT_P = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def parse_args(argv=None):
    p = parser(__doc__)
    p.add_argument("run_dirs", nargs="*", help="dossiers de run (run_experiment.py)")
    p.add_argument("--resume", default="", help="dossier d'un pruning à reprendre")

    g = p.add_argument_group("pruning")
    g.add_argument("--mode", choices=["pre", "post", "both"], default="pre")
    g.add_argument("--p", type=float, nargs="+", default=DEFAULT_P,
                   help="ordres p : fractions de poids retirés, de 0 à 1")
    g.add_argument("--param", choices=PARAMS, default="lambda_in")
    g.add_argument("--layers", default="trained",
                   help='"trained", "*", "0", "1-5", "0,3,5-7" (indices de convolution)')
    g.add_argument("--method", choices=METHODS, default="element",
                   help="gradient (poids par poids dans chaque filtre), element (pixels de chaque "
                        "filtre, tous canaux), mixte (pixels pour tous les filtres)")
    g.add_argument("--criterion", choices=CRITERIA, default="percentile")
    g.add_argument("--include_padding", action="store_true",
                   help="classer aussi les positions de padding de lambda_in")

    g = p.add_argument_group("post-training (réentraînement des phi)")
    g.add_argument("--train", choices=["phi", "non_spectral"], default="phi",
                   help="phi = seuls les filtres ; non_spectral = phi + BatchNorm + tête")
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--lr", type=float, default=None, help="défaut : lr du run")
    g.add_argument("--batch_size", type=int, default=None, help="défaut : celui du run")
    g.add_argument("--optimizer", choices=["sgd", "adamw"], default=None, help="défaut : celui du run")
    g.add_argument("--weight_decay", type=float, default=None, help="défaut : celui du run")
    g.add_argument("--schedule", choices=["cosine", "step", "constant"], default=None,
                   help="défaut : celui du run")
    g.add_argument("--warmup_epochs", type=int, default=0)
    g.add_argument("--repeats", type=int, default=1,
                   help="réentraînements par p (graines seed, seed+1...)")
    g.add_argument("--seed", type=int, default=None, help="défaut : graine du run")
    g.add_argument("--save_weights", action="store_true",
                   help="enregistrer les poids de chaque modèle réentraîné (volumineux)")

    g = p.add_argument_group("exécution")
    add_device(g)
    g.add_argument("--gpus", type=int, nargs="+", default=None,
                   help="répartir le travail sur ces GPU (ex. --gpus 0 1) : un processus par GPU")
    g.add_argument("--shard", default="", help=argparse.SUPPRESS)  # "k/n", posé par --gpus
    g.add_argument("--allow_mixed", action="store_true",
                   help="accepter des runs non comparables (dataset, architecture, configuration)")
    g.add_argument("--out_dir", default="runs_pruning")
    g.add_argument("--name", default="", help="nom du dossier (défaut : <dataset>_<arch>_<mode>_<date>)")
    add_verbose(g, default=0)
    args = p.parse_args(argv)
    if not args.resume and not args.run_dirs:
        p.error("donner au moins un dossier de run, ou --resume")
    if any(not 0.0 <= x <= 1.0 for x in args.p):
        p.error("--p : fractions entre 0 et 1")
    return args


# --- Résultats ------------------------------------------------------------------------

def row_key(row):
    return (row["run"], row["mode"], f"{float(row['p']):.4f}", int(row["repeat"]))


def read_results(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def results_path(out_dir, shard):
    """Fichier de résultats de ce processus : results.csv, ou results_part<k>.csv avec --gpus."""
    return os.path.join(out_dir, f"results_part{shard[0]}.csv" if shard else "results.csv")


def read_all_results(out_dir):
    """Toutes les lignes du dossier (results.csv et les parties des processus --gpus), sans doublon."""
    rows = {}
    for path in [os.path.join(out_dir, "results.csv")] + sorted(glob.glob(os.path.join(out_dir, "results_part*.csv"))):
        for r in read_results(path):
            rows[row_key(r)] = r
    return list(rows.values())


def write_results(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, out_dir):
    """summary.csv : statistiques de la précision test par (mode, p), tous runs et répétitions."""
    groups = {}
    for r in rows:
        groups.setdefault((r["mode"], float(r["p"])), []).append(float(r["test_accuracy"]))
    summary = []
    for (mode, p), accs in sorted(groups.items()):
        a = np.array(accs)
        summary.append({"mode": mode, "p": p, "n": len(a), "mean": a.mean(), "median": np.median(a),
                        "q05": np.quantile(a, 0.05), "q95": np.quantile(a, 0.95), "std": a.std()})
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in s.items()}
                    for s in summary)
    return summary


# --- Pruning d'un run -------------------------------------------------------------------

def memory_used() -> str:
    """Mémoire GPU utilisée (ou rien sans GPU), pour suivre une éventuelle accumulation."""
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return f"GPU {info['current'] / 2**30:.2f} Gio (pic {info['peak'] / 2**30:.2f} Gio)"
    except (ValueError, RuntimeError):
        return ""


def optimizer_reset(agent):
    """
    Crée les variables de l'optimiseur et renvoie une fonction qui les remet à leur état
    initial (moments, compteur d'itérations donc schedule), sans recompiler le modèle.
    """
    opt = agent.optimizer
    opt.build(agent.trainable_variables)
    initial = [v.numpy().copy() for v in opt.variables]

    def reset():
        for var, value in zip(opt.variables, initial):
            var.assign(value)
    return reset


def train_args(args, run_args):
    """Recette du post-training : celle du run, avec les options données en plus (même lr que le run par défaut)."""
    return recipe_from_run(run_args, epochs=args.epochs,
                           lr=args.lr if args.lr is not None else run_args["lr"],
                           batch_size=args.batch_size, optimizer=args.optimizer,
                           weight_decay=args.weight_decay, schedule=args.schedule,
                           warmup_epochs=args.warmup_epochs)


def evaluate(agent, test_ds):
    loss, acc = agent.evaluate(test_ds, verbose=0)
    return float(acc), float(loss)


def prune_run(args, run_index, run_dir, out_dir, done, rows, path):
    """Pré et/ou post-training pour toutes les valeurs de p d'un run ; ajoute les lignes à rows."""

    def mine(p_index, r=0):
        """Tâche de ce processus ? (avec --gpus, les (run, p, répétition) sont répartis)"""
        if not args.shard:
            return True
        k, n = args.shard
        return ((run_index * len(args.p) + p_index) * args.repeats + r) % n == k

    run_name = read_config(run_dir)["run_name"]
    todo = [(m, i, r) for m in (("pre",) if args.mode == "pre" else ("post",) if args.mode == "post"
                                else ("pre", "post"))
            for i, p in enumerate(args.p) for r in range(args.repeats if m == "post" else 1)
            if (run_name, m, f"{p:.4f}", r) not in done and mine(i, r)]
    if not todo:
        print(f"\n=== {run_name} : déjà fait")
        return
    agent, config = load_run(run_dir)
    run_args = config["args"]
    train, val, test, info = load_run_data(config)
    targets = select_layers(agent, args.layers, args.param)
    if run_index == 0 or not args.shard:
        for warning in spectral_warnings(agent):
            print(f"  Attention : {warning}")
    print(f"\n=== {run_name} : {info.name} {info.input_shape}, {run_args['arch']} ; "
          f"{args.param} de {[l.name for l in targets]}")

    run_out = os.path.join(out_dir, run_name)
    os.makedirs(os.path.join(run_out, "masks"), exist_ok=True)
    original, reports = masks_for(agent, args.p, [l.name for l in targets], args.param, args.method,
                                  args.criterion, args.include_padding)
    if all(np.ptp(v) == 0 for v in original.values()):
        print(f"  Attention : toutes les valeurs de {args.param} sont égales (poids non entraîné) : "
              "le classement par magnitude est arbitraire.")
    for p, rep in reports.items():
        if args.shard and args.shard[0] != 0:
            break  # avec --gpus, seul le premier processus écrit les masques (identiques partout)
        np.savez_compressed(os.path.join(run_out, "masks", f"p{p:.2f}.npz"), **rep.masks)

    targs = train_args(args, run_args)
    seed = args.seed if args.seed is not None else run_args.get("seed", 42)
    test_ds = test_dataset(test, info, run_args)

    def record(row):
        rows.append(row)
        done.add(row_key(row))
        write_results(path, rows)

    def base_row(mode, p, repeat, rep):
        return {"run": run_name, "mode": mode, "p": f"{p:.4f}", "repeat": repeat, "seed": "",
                "pruned": rep.pruned, "total": rep.total, "fraction_pruned": f"{rep.fraction:.4f}",
                "final_train_accuracy": "", "epochs": 0, "lr": "", "train_time_s": ""}

    # --- pré-training : prédiction directe du modèle élagué ---
    if args.mode in ("pre", "both"):
        compile_agent(agent, targs, 1)
        for i, p in enumerate(args.p):
            if (run_name, "pre", f"{p:.4f}", 0) in done or not mine(i):
                continue
            acc, loss = evaluate_pruned(agent, reports[p], original, test_ds)
            record({**base_row("pre", p, 0, reports[p]), "test_accuracy": f"{acc:.4f}",
                    "test_loss": f"{loss:.4f}"})
            print(f"  pre   p={p:.2f}  retirés {reports[p].fraction:6.1%}  précision {acc:.4f}")
            if p == 0.0 and abs(acc - config.get("test_accuracy", acc)) > 2e-3:
                print(f"  Attention : précision à p=0 ({acc:.4f}) différente de celle du run "
                      f"({config['test_accuracy']:.4f}) ; vérifier les données et les poids.")
    del agent
    tf.keras.backend.clear_session()

    # --- post-training : valeurs spectrales figées, réentraînement des phi ---
    if args.mode in ("post", "both"):
        train_ds, val_ds, test_ds = make_tf_datasets(train, val, test, info, targs.batch_size,
                                                     run_args.get("crop_pad", 4), seed)
        if targs.label_smoothing:
            train_ds, val_ds, test_ds = onehot_datasets([train_ds, val_ds, test_ds], info.num_classes)
        steps = len(train[0]) // targs.batch_size
        # Un seul modèle pour tous les p : on recharge ses poids entraînés à chaque p (créer un
        # modèle par p fait grossir la mémoire du GPU jusqu'à l'erreur « Out of memory »).
        agent, _ = load_run(run_dir)
        counts = freeze_spectral(agent, args.train)
        compile_agent(agent, targs, steps)
        reset_optimizer = optimizer_reset(agent)
        weights_path = os.path.join(run_dir, WEIGHTS_FILE)
        for i, p in enumerate(args.p):
            for r in range(args.repeats):
                if (run_name, "post", f"{p:.4f}", r) in done or not mine(i, r):
                    continue
                tf.keras.utils.set_random_seed(seed + r)
                with warnings.catch_warnings():           # modèle entraîné du run ; l'état de
                    warnings.simplefilter("ignore")       # son optimiseur ne sert pas (remis à zéro)
                    agent.load_weights(weights_path)
                apply_masks(agent, reports[p].masks, args.param)
                reset_optimizer()                         # momentum et schedule repartent de zéro
                folder = os.path.join(run_out, "post", f"p{p:.2f}_r{r}")
                os.makedirs(folder, exist_ok=True)
                t0 = time.time()
                history = agent.fit(train_ds, validation_data=val_ds, epochs=targs.epochs,
                                    callbacks=[tf.keras.callbacks.CSVLogger(
                                        os.path.join(folder, "history.csv")),
                                        tf.keras.callbacks.TerminateOnNaN()],
                                    verbose=args.verbose)
                elapsed = time.time() - t0
                if not np.isfinite(history.history["loss"][-1]):
                    print(f"  Attention : réentraînement divergé (perte nan) à p={p:.2f} : "
                          "réduire --lr (ex. --lr 0.01).")
                acc, loss = evaluate(agent, test_ds)
                # contrôle : les valeurs élaguées sont restées à zéro
                pruned_values = get_spectral_values(agent, args.param, list(reports[p].masks))
                assert all(np.all(v[~reports[p].masks[k]] == 0) for k, v in pruned_values.items())
                if args.save_weights:
                    agent.save_weights(os.path.join(folder, WEIGHTS_FILE))
                record({**base_row("post", p, r, reports[p]), "seed": seed + r,
                        "test_accuracy": f"{acc:.4f}", "test_loss": f"{loss:.4f}",
                        "final_train_accuracy": f"{history.history['accuracy'][-1]:.4f}",
                        "epochs": targs.epochs, "lr": targs.lr, "train_time_s": round(elapsed, 1)})
                print(f"  post  p={p:.2f}  r={r}  retirés {reports[p].fraction:6.1%}  précision {acc:.4f}"
                      f"  ({counts['trainable']} paramètres réentraînés, {elapsed:.0f} s)")
                del history, pruned_values
                gc.collect()
                mem = memory_used()
                if mem:
                    print(f"    mémoire {mem}", flush=True)
        del agent
        tf.keras.backend.clear_session()


def launch_shards(args, out_dir):
    """--gpus : un processus par GPU sur le même dossier, chacun avec sa part des tâches."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [PROJECT_ROOT, os.environ.get("PYTHONPATH")]))}
    procs = []
    for k, gpu in enumerate(args.gpus):
        cmd = [sys.executable, "-m", "specprune.cli.prune", "--resume", out_dir,
               "--shard", f"{k}/{len(args.gpus)}", "--device", f"/GPU:{gpu}", "--verbose", str(args.verbose)]
        log = open(os.path.join(out_dir, f"log_gpu{gpu}.txt"), "w")
        print(f"GPU {gpu} : {' '.join(cmd)}  (journal : {log.name})")
        procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    failed = []
    for gpu, proc, log in procs:
        if proc.wait() != 0:
            failed.append(gpu)
        log.close()
    if failed:
        print(f"Échec sur les GPU {failed} : voir log_gpu<k>.txt ; relancer avec --resume {out_dir}")


def main(argv=None):
    args = parse_args(argv)
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else ()
    saved = None
    if args.resume:
        out_dir = args.resume
        with open(os.path.join(out_dir, "config.json")) as f:
            saved = json.load(f)
        check_format(saved, out_dir)
        if "method" not in saved["args"]:
            sys.exit(f"{out_dir} vient de l'ancienne version (granularity/scope) : relancer le pruning.")
        cli = {k: getattr(args, k) for k in ("device", "gpus", "verbose")}
        for key, value in saved["args"].items():
            setattr(args, key, value)
        for key, value in cli.items():
            setattr(args, key, value)
    else:
        try:
            configs = [read_config(d) for d in args.run_dirs]
        except (ValueError, FileNotFoundError) as e:
            sys.exit(f"Erreur : {e}")
        diffs = runs_differences(configs, [os.path.basename(os.path.normpath(d)) for d in args.run_dirs])
        if diffs and not args.allow_mixed:
            sys.exit("Runs non comparables (leurs résultats seraient agrégés) :\n  " + "\n  ".join(diffs)
                     + "\nLancer un pruning par groupe de runs, ou --allow_mixed pour passer outre.")
        first = configs[0]
        name = args.name or f"{dataset_tag(first['dataset']['name'])}_{first['args']['arch']}_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}"
        out_dir = os.path.join(args.out_dir, name)
        os.makedirs(out_dir)
        args.run_dirs = [os.path.abspath(d) for d in args.run_dirs]
    args.shard = shard

    if not shard:  # le processus principal écrit la configuration ; les parties --gpus la relisent
        now = datetime.now().isoformat(timespec="seconds")
        config = {"format_version": FORMAT_VERSION,
                  "args": {k: v for k, v in vars(args).items() if k not in ("resume", "shard", "gpus")},
                  "runs": [os.path.abspath(d) for d in args.run_dirs],
                  "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
                  "environment": environment_info(tf.config.list_physical_devices("GPU")),
                  "started_at": now}
        if saved:
            config["started_at"] = saved.get("started_at", now)
            config["command"] = saved.get("command", config["command"])
            config["resumed_at"] = saved.get("resumed_at", []) + [now]
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        if args.gpus:
            launch_shards(args, out_dir)
            return finish(out_dir)

    setup_device(args.device)
    path = results_path(out_dir, shard)
    rows = read_results(path)                         # lignes de ce processus
    done = {row_key(r) for r in read_all_results(out_dir)}
    print(f"Pruning : {out_dir}{f'  (part {shard[0] + 1}/{shard[1]})' if shard else ''}\n"
          f"mode={args.mode}  param={args.param}  layers={args.layers}  "
          f"method={args.method}  criterion={args.criterion}  p={args.p}"
          + (f"  ({len(done)} lignes déjà faites)" if done else ""))

    try:
        with tf.device(args.device):
            for i, run_dir in enumerate(args.run_dirs):
                prune_run(args, i, run_dir, out_dir, done, rows, path)
    except (ValueError, FileNotFoundError) as e:
        if not done and not args.resume:
            shutil.rmtree(out_dir, ignore_errors=True)  # pas de dossier vide
        sys.exit(f"Erreur : {e}")
    if not shard:
        finish(out_dir)


def finish(out_dir):
    rows = read_all_results(out_dir)
    if not rows:
        return
    summary = summarize(rows, out_dir)
    print(f"\n{'mode':>5} {'p':>5} {'n':>3} {'médiane':>8} {'q05':>7} {'q95':>7}")
    for s in summary:
        print(f"{s['mode']:>5} {s['p']:>5.2f} {s['n']:>3} {s['median']:>8.4f} {s['q05']:>7.4f} {s['q95']:>7.4f}")
    print(f"\nRésultats : {out_dir}/results*.csv, summary.csv")
    print(f"Figures   : python plot_results.py pruning {out_dir}")


if __name__ == "__main__":
    main()
