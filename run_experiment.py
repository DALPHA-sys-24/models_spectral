"""
Run d'expérience : entraîne un modèle du registre sur un dataset (images ou signaux) et
sauvegarde le run.

Datasets (--dataset) : cifar10, cifar100, mnist, fashion_mnist, tfds:<nom>, npz, folder, et pour
des signaux mnist1d, ucr:<Nom> (voir tools/data.py). Exemple Fashion-MNIST :
    python run_experiment.py --dataset fashion_mnist --arch spectral_resnet20
Signaux : fichier .npz avec x en (N, T) ou (N, T, C) ; les mêmes modèles utilisent alors des
SpecCnn1D (--modality signal obligatoire pour (N, T, C), sinon lu comme des images (N, H, W)) :
    python run_experiment.py --dataset npz --data_path signaux.npz --modality signal --crop_pad 16
Signaux téléchargés automatiquement (MNIST-1D, archive UCR/UEA) :
    python run_experiment.py --dataset mnist1d --arch spectral_resnet20
    python run_experiment.py --dataset ucr:GunPoint --arch spectral_resnet20 --crop_pad 8

Recette par défaut (littérature ResNet CIFAR-10) : SGD momentum 0.9 Nesterov, lr 0.1,
batch 128, 200 epochs, cosine decay, weight decay 5e-4, augmentation pad 4 + crop +
flip horizontal (--augment auto : choisie selon le dataset, pas de flip pour MNIST),
normalisation par canal. Ordres de grandeur publiés sur CIFAR-10 : ResNet-20 ~91.3-92.5 %,
ResNet-32 ~92.5-93 %, ResNet-56 ~93-94 %.

Toutes les convolutions sont des SpecCnn2D (SpecCnn1D pour des signaux) ; le fichier
--spectral_config dit, couche par couche, quels poids elles entraînent (syntaxe dans
models/layers.py, exemples dans configs/), de la même façon en 1D et en 2D.
Premier test : la première couche entraîne phi + lambda_in, les autres sont figées en Conv2D :
    python run_experiment.py --arch spectral_resnet20 --spectral_config configs/spectral_config.json
Référence classique (lambda_in = 1, lambda_out = 0, alpha = 1 figés partout), même recette :
    python run_experiment.py --arch spectral_resnet20 --spectral_config configs/classique.json

Rejouer un run à l'identique (les arguments donnés en plus remplacent ceux du run) :
    python run_experiment.py --from_config runs/<run>/config.json
    python run_experiment.py --from_config runs/<run>/config.json --seed 1

Plusieurs runs identiques (graines 42 à 46), pour le pruning sur plusieurs graines :
    python run_experiment.py --dataset cifar10 --epochs 20 --n_runs 5 --run_name cifar10_5seeds
    -> runs/cifar10_5seeds/seed42/ ... seed46/ ; puis python pruning.py runs/cifar10_5seeds/seed*

Chaque run crée un dossier <out_dir>/<dataset>_<arch>_<date>_<heure>/ contenant :
    config.json          tous les arguments, le dataset (source, tailles, normalisation,
                         augmentation, classes), la configuration spectrale et le réglage de
                         chaque convolution, les paramètres, les versions, la commande
    command.sh           commande exacte du run
    spectral_config.json copie de la configuration spectrale utilisée
    history.csv          perte / précision par epoch
    metrics.json         précision finale et meilleure précision (test)
    spectral/            <couche>_lambda_in.npy : poids bruts lambda_in en fin d'entraînement,
                         (F, C/groups, Lh*Lw) en 2D, (F, C/groups, L) en 1D,
                         pour chaque couche où lambda_in est entraîné (aucun traitement)
    spectral/epoch_000/  les mêmes poids avant l'entraînement
    spectral/epoch_XXX/  instantanés toutes les N epochs (--spectral_every N)
    weights.weights.h5   poids finaux (sauf --no-save_weights)
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (AgentNetwork, available_models, configure_spectral,  # noqa: E402
                    get_spectral_config, spectral_plan)
from tools.data import available_datasets, load_dataset, make_tf_datasets  # noqa: E402
from tools.runs import (WEIGHTS_FILE, LambdaInSnapshot, append_summary,  # noqa: E402
                        environment_info, lambda_in_file, lambda_in_layers, save_lambda_in,
                        spectral_layers)
from tools.training import compile_agent, onehot_datasets, setup_device  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from_config", default="",
                   help="config.json d'un run précédent : reprend tous ses arguments")

    g = p.add_argument_group("données")
    g.add_argument("--dataset", default="cifar10",
                   help=f"un parmi {available_datasets()}")
    g.add_argument("--data_path", default="", help="fichier .npz ou dossier (npz, folder)")
    g.add_argument("--modality", choices=["auto", "image", "signal"], default="auto",
                   help="auto = signal si x est (N, T), image sinon ; signal pour un npz (N, T, C)")
    g.add_argument("--augment", choices=["auto", "none", "crop", "crop_flip"], default="auto",
                   help="auto = défaut du dataset (crop_flip ; crop pour mnist et les signaux)")
    g.add_argument("--crop_pad", type=int, default=4,
                   help="padding avant le crop aléatoire (pixels, ou pas de temps pour un signal)")
    g.add_argument("--image_size", type=int, default=0,
                   help="redimensionne les images en NxN (0 = taille d'origine)")
    g.add_argument("--val_split", type=float, default=0.0,
                   help="part du train gardée pour la validation (0 = valider sur le test)")
    g.add_argument("--train_subset", type=int, default=0, help="n premiers exemples (0 = tous)")
    g.add_argument("--test_subset", type=int, default=0)

    g = p.add_argument_group("modèle")
    g.add_argument("--arch", default="spectral_resnet20", choices=available_models())
    g.add_argument("--width", type=float, default=1.0, help="multiplicateur de filtres")
    g.add_argument("--dropout", type=float, default=0.0)

    g = p.add_argument_group("couches spectrales")
    g.add_argument("--spectral_config", default=os.path.join(HERE, "configs", "spectral_config.json"),
                   help="fichier JSON : quelles couches sont spectrales et quels poids elles "
                        "entraînent ; configs/classique.json = modèle classique ; \"\" = défauts des couches")
    g.add_argument("--spectral_every", type=int, default=0,
                   help="instantané de lambda_in toutes les N epochs (0 = fin seulement)")

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--batch_size", type=int, default=128)
    g.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    g.add_argument("--lr", type=float, default=0.1)
    g.add_argument("--momentum", type=float, default=0.9)
    g.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--weight_decay", type=float, default=5e-4,
                   help="appliqué aux noyaux, BN et Dense ; jamais aux poids spectraux")
    g.add_argument("--schedule", choices=["cosine", "step", "constant"], default="cosine")
    g.add_argument("--step_epochs", type=int, nargs="+", default=[100, 150],
                   help="epochs où lr est multiplié par 0.1 (--schedule step)")
    g.add_argument("--warmup_epochs", type=int, default=0)
    g.add_argument("--label_smoothing", type=float, default=0.0)

    g = p.add_argument_group("exécution")
    g.add_argument("--device", default="/GPU:0", help='ex. "/GPU:0", "/GPU:1", "/CPU:0"')
    g.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--n_runs", type=int, default=1,
                   help="nombre de runs identiques (graines seed, seed+1, ...), rangés dans "
                        "<out_dir>/<run_name ou dataset_arch_date_xN>/seed<k>/")
    g.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False,
                   help="opérations déterministes (résultats identiques d'un run à l'autre, plus lent)")
    g.add_argument("--out_dir", default="runs")
    g.add_argument("--run_name", default="", help="nom du dossier (défaut : <dataset>_<arch>_<date>)")
    g.add_argument("--save_weights", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2])
    return p


def parse_args(argv=None):
    """Arguments de la ligne de commande ; avec --from_config, ceux du run indiqué par défaut."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.from_config:
        with open(args.from_config) as f:
            saved = json.load(f)
        defaults = {k: v for k, v in saved["args"].items()
                    if k not in ("from_config", "run_name", "n_runs")}
        # La configuration spectrale est relue depuis la copie faite dans le dossier du run.
        copy_path = os.path.join(os.path.dirname(os.path.abspath(args.from_config)),
                                 "spectral_config.json")
        if os.path.exists(copy_path):
            defaults["spectral_config"] = copy_path
        parser.set_defaults(**defaults)
        args = parser.parse_args(argv)
    return args


def resolve_spectral_config(path: str) -> str:
    """Chemin de la configuration spectrale : tel quel, sinon dans configs/ (vérifié avant tout calcul)."""
    if not path or os.path.exists(path):
        return path
    candidate = os.path.join(HERE, "configs", os.path.basename(path))
    if os.path.exists(candidate):
        print(f"--spectral_config {path} introuvable ici : {candidate} utilisé")
        return candidate
    sys.exit(f"--spectral_config {path} introuvable (ni depuis {os.getcwd()}, ni dans configs/). "
             f"Configurations disponibles : {sorted(os.listdir(os.path.join(HERE, 'configs')))}")


def main(argv=None):
    args = parse_args(argv)
    if args.n_runs < 1:
        sys.exit("--n_runs doit valoir au moins 1")
    args.spectral_config = resolve_spectral_config(args.spectral_config)
    gpus = setup_device(args.device, args.mixed_precision)
    if args.n_runs == 1:
        run(args, gpus)
        return
    # Plusieurs runs : même recette, graines seed, seed+1, ... dans un dossier commun
    tag = args.dataset.replace("tfds:", "").replace(":", "_").replace("/", "_")
    group = args.run_name or f"{tag}_{args.arch}_{datetime.now():%Y%m%d_%H%M%S}_x{args.n_runs}"
    seeds = [args.seed + k for k in range(args.n_runs)]
    print(f"{args.n_runs} runs dans {os.path.join(args.out_dir, group)} (graines {seeds})")
    for k, seed in enumerate(seeds):
        print(f"\n=== run {k + 1}/{args.n_runs} : graine {seed}")
        one = argparse.Namespace(**{**vars(args), "seed": seed, "n_runs": 1,
                                    "out_dir": os.path.join(args.out_dir, group),
                                    "run_name": f"seed{seed}"})
        run(one, gpus)
        tf.keras.backend.clear_session()
    print(f"\nPruning des {args.n_runs} runs : python pruning.py {os.path.join(args.out_dir, group)}/seed*")


def run(args, gpus):
    """Entraîne et enregistre un run."""
    tf.keras.utils.set_random_seed(args.seed)
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    # Données (avant de créer le dossier : pas de run vide si le chargement échoue)
    train, val, test, info = load_dataset(
        args.dataset, args.data_path or None, args.augment, args.train_subset,
        args.test_subset, args.val_split, args.image_size, args.seed, args.modality)
    train_ds, val_ds, test_ds = make_tf_datasets(train, val, test, info, args.batch_size,
                                                 args.crop_pad, args.seed)
    steps_per_epoch = len(train[0]) // args.batch_size

    dataset_tag = info.name.replace("tfds:", "").replace(":", "_").replace("/", "_")
    run_name = args.run_name or f"{dataset_tag}_{args.arch}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)
    print(f"Run : {run_dir}  |  appareil : {args.device}  |  GPU visibles : {[d.name for d in gpus]}")
    print(f"Données : {info.name} ({info.modality}) {info.input_shape}, {info.num_classes} classes, "
          f"train={info.n_train} val={info.n_val} test={info.n_test}, augmentation={info.augment}")

    configure_spectral(args.spectral_config or None)
    if args.spectral_config:
        shutil.copy(args.spectral_config, os.path.join(run_dir, "spectral_config.json"))

    with tf.device(args.device):
        agent = AgentNetwork(arch=args.arch, num_classes=info.num_classes,
                             dropout_rate=args.dropout, width=args.width, name="agent")
        agent(tf.zeros([1] + info.input_shape))  # construit le backbone et la tête
        compile_agent(agent, args, steps_per_epoch)
        if args.label_smoothing:
            train_ds, val_ds, test_ds = onehot_datasets([train_ds, val_ds, test_ds], info.num_classes)

        n_spec = len(spectral_layers(agent))
        saved_layers = lambda_in_layers(agent)
        config = {
            "run_name": run_name,
            "run_dir": os.path.abspath(run_dir),
            "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
            "replayed_from": os.path.abspath(args.from_config) if args.from_config else None,
            "args": vars(args),
            "dataset": info.to_dict(),
            "validation_on": "val" if val_ds is not None else "test",
            "model": {
                "arch": args.arch,
                "params_total": int(agent.count_params()),
                "params_trainable": int(sum(np.prod(v.shape) for v in agent.trainable_variables)),
                "spectral_layers": [l.name for l in spectral_layers(agent)],
            },
            # Fichiers lambda_in enregistrés : couche, indice de convolution (voir spectral_plan),
            # forme (F, C/groups, Lh*Lw) en 2D ou (F, C/groups, L) en 1D, grille avec padding
            # ([Lh, Lw] ou [L]), padding ([Ph, Pw] ou [P]) et taille d'entrée ([H, W] ou [T])
            "lambda_in_files": [lambda_in_file(l, spectral_layers(agent).index(l))
                                for l in saved_layers],
            "spectral_config_file": args.spectral_config,
            "spectral_config": get_spectral_config(),
            "spectral_plan": spectral_plan(),
            "environment": environment_info(gpus),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }

        def write_config():
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        write_config()
        with open(os.path.join(run_dir, "command.sh"), "w") as f:
            f.write(config["command"] + "\n")
            f.write(f"# rejouer : python run_experiment.py --from_config {run_dir}/config.json\n")
        print(f"{args.arch} : {config['model']['params_total'] / 1e6:.2f} M paramètres "
              f"({config['model']['params_trainable'] / 1e6:.3f} M entraînables), "
              f"{n_spec} couche(s) spectrale(s)")

        spectral_dir = os.path.join(run_dir, "spectral")
        if saved_layers:
            save_lambda_in(agent, os.path.join(spectral_dir, "epoch_000"))
        callbacks = [tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "history.csv"))]
        if saved_layers and args.spectral_every:
            callbacks.append(LambdaInSnapshot(agent, spectral_dir, args.spectral_every))

        t0 = time.time()
        history = agent.fit(train_ds, validation_data=val_ds if val_ds is not None else test_ds,
                            epochs=args.epochs, callbacks=callbacks, verbose=args.verbose)
        train_time = time.time() - t0
        test_loss, test_acc = agent.evaluate(test_ds, verbose=0)

    # --- Sauvegardes de fin de run ---
    val_acc = history.history.get("val_accuracy", [test_acc])
    metrics = {
        "dataset": info.name,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        f"best_{config['validation_on']}_accuracy": float(max(val_acc)),
        "best_epoch": int(np.argmax(val_acc) + 1),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "train_time_s": round(train_time, 1),
        "epochs": args.epochs,
    }
    if saved_layers:
        save_lambda_in(agent, spectral_dir)
    if args.save_weights:
        agent.save_weights(os.path.join(run_dir, WEIGHTS_FILE))
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    config["finished_at"] = datetime.now().isoformat(timespec="seconds")
    config["test_accuracy"] = metrics["test_accuracy"]
    write_config()

    # Tableau récapitulatif de tous les runs du dossier out_dir
    append_summary(os.path.join(args.out_dir, "runs_summary.csv"), {
        "run": run_name, "dataset": info.name, "arch": args.arch,
        "spectral_config": os.path.basename(args.spectral_config) or "defauts_couches",
        "params_trainable": config["model"]["params_trainable"],
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch_size,
        "weight_decay": args.weight_decay, "seed": args.seed,
        "test_accuracy": f"{test_acc:.4f}",
        "best_val_accuracy": f"{max(val_acc):.4f}", "validation_on": config["validation_on"],
        "train_time_s": metrics["train_time_s"],
    })

    print(f"Précision test : {test_acc:.4f} (meilleure en validation [{config['validation_on']}] : "
          f"{max(val_acc):.4f} à l'epoch {metrics['best_epoch']})  ->  {run_dir}")


if __name__ == "__main__":
    main()
