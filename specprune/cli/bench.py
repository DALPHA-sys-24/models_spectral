"""
Mémoire des couches spectrales : ce que chaque couche alloue, et le pic mesuré d'un pas
d'entraînement selon la taille du batch (pour choisir --batch_size avant de lancer un run).

Par couche (calcul exact à partir des formes, en float32) :
    params      phi, lambda_in, lambda_out, alpha (et leurs moments dans l'optimiseur)
    W           poids effectif alpha*phi*(lambda_in - lambda_out), forme (F, Cg, O, K) :
                ne dépend pas du batch, recalculé à chaque pas
    X_win       fenêtres de l'entrée, forme (B, O, K, C) : K fois l'entrée de la couche,
                gardées pour la rétropropagation ; c'est le terme qui grandit avec le batch
    sortie      (B, O, F)
Le pic réel est plus haut (gradients, copies transposées de l'einsum, fusions de XLA) : sur GPU,
le script le mesure pour un pas d'entraînement complet, avec et sans XLA (--jit).

Exemples :
    # cifar10, ResNet-20 spectral : tableau par couche et pic mesuré pour 64, 128 et 256
    python bench_memory.py --dataset cifar10 --arch spectral_resnet20 --batch_size 64 128 256

    # sans télécharger de données : forme d'entrée donnée directement
    python bench_memory.py --input_shape 32 32 3 --num_classes 10 --arch spectral_resnet20

    # signal 1D, configuration spectrale choisie, sans mesure (CPU)
    python bench_memory.py --input_shape 40 1 --arch spectral_resnet20 --device /CPU:0
"""
import csv
import os
import sys
import time

import numpy as np
import tensorflow as tf

from ..models import AgentNetwork, available_models, configure_spectral
from ..runs import spectral_layers
from ..training import setup_device
from .common import DEFAULT_SPECTRAL_CONFIG, add_device, parser, resolve_spectral_config

BYTES = 4  # float32


def parse_args(argv=None):
    p = parser(__doc__)
    g = p.add_argument_group("modèle")
    g.add_argument("--arch", default="spectral_resnet20", choices=available_models())
    g.add_argument("--width", type=float, default=1.0)
    g.add_argument("--spectral_config", default=DEFAULT_SPECTRAL_CONFIG,
                   help='configuration spectrale ("" : aucune, tous les poids entraînés)')
    g = p.add_argument_group("données (une des deux)")
    g.add_argument("--dataset", default="", help="nom du dataset (forme lue sans entraîner)")
    g.add_argument("--input_shape", type=int, nargs="+", default=None,
                   help="forme d'une entrée sans le batch : 32 32 3 (image) ou 40 1 (signal)")
    g.add_argument("--num_classes", type=int, default=10)
    g = p.add_argument_group("mesure")
    g.add_argument("--batch_size", type=int, nargs="+", default=[32, 64, 128, 256])
    g.add_argument("--jit", choices=["auto", "on", "off", "both"], default="both",
                   help="XLA pour la mesure (auto = choix de Keras, celui des entraînements)")
    g.add_argument("--steps", type=int, default=2, help="pas mesurés (le premier compile)")
    add_device(g)
    g.add_argument("--out", default="", help="fichier CSV du tableau par couche (optionnel)")
    args = p.parse_args(argv)
    if not args.dataset and not args.input_shape:
        p.error("donner --dataset ou --input_shape")
    return args


def input_shape_of(args):
    if args.input_shape:
        return list(args.input_shape), args.num_classes
    from ..data import load_dataset
    info = load_dataset(args.dataset)[-1]
    return list(info.input_shape), info.num_classes


def build(args, input_shape, num_classes):
    configure_spectral(args.spectral_config or None)
    agent = AgentNetwork(arch=args.arch, num_classes=num_classes, width=args.width, name="agent")
    agent(tf.zeros([1] + input_shape))
    return agent


def layer_table(agent, batch):
    """Une ligne par couche spectrale : formes et mémoire (octets) des tenseurs principaux."""
    rows = []
    for i, l in enumerate(spectral_layers(agent)):
        F, Cg = int(l.phi_weights.shape[0]), int(l.phi_weights.shape[1])
        K = int(np.prod(l.phi_weights.shape[2:]))
        C, O = l._C, l._O
        params = sum(int(np.prod(w.shape)) for w in (l.phi_weights, l.lambda_in, l.lambda_out, l.alpha))
        trainable = sum(int(np.prod(w.shape)) for w in l.trainable_weights)
        rows.append({"index": i, "layer": l.name, "F": F, "C": C, "groups": l.groups, "O": O, "K": K,
                     "params": params * BYTES, "trainable": trainable * BYTES,
                     "W": F * Cg * O * K * BYTES, "X_win": batch * O * K * C * BYTES,
                     "output": batch * O * F * BYTES})
    return rows


def human(n):
    for unit, size in (("Gio", 2**30), ("Mio", 2**20), ("Kio", 2**10)):
        if n >= size:
            return f"{n / size:.2f} {unit}"
    return f"{n} o"


def print_table(rows, batch):
    print(f"\nPar couche, batch {batch} (float32) :")
    print(f"{'#':>3} {'couche':<22} {'F':>4} {'C':>4} {'G':>3} {'O':>6} {'K':>3} "
          f"{'params':>10} {'W':>10} {'X_win':>10} {'sortie':>10}")
    for r in rows:
        print(f"{r['index']:>3} {r['layer'][:22]:<22} {r['F']:>4} {r['C']:>4} {r['groups']:>3} {r['O']:>6} "
              f"{r['K']:>3} {human(r['params']):>10} {human(r['W']):>10} {human(r['X_win']):>10} "
              f"{human(r['output']):>10}")
    tot = {k: sum(r[k] for r in rows) for k in ("params", "trainable", "W", "X_win", "output")}
    print(f"    {'total':<22} {'':>24} {human(tot['params']):>10} {human(tot['W']):>10} "
          f"{human(tot['X_win']):>10} {human(tot['output']):>10}")
    # gardé pour la rétropropagation : X_win et sortie de chaque couche, W de chaque couche ;
    # poids entraînés : valeur + gradient + 2 moments (AdamW) au plus
    estimate = tot["X_win"] + tot["output"] + tot["W"] + 4 * tot["trainable"]
    print(f"Estimation basse du pic d'un pas d'entraînement (couches spectrales seules) : {human(estimate)}")
    biggest = max(rows, key=lambda r: r["X_win"])
    print(f"Plus gros X_win : {biggest['layer']} ({human(biggest['X_win'])}) ; "
          f"plus gros W : {max(rows, key=lambda r: r['W'])['layer']} ({human(max(r['W'] for r in rows))})")
    return tot


def measure(agent, input_shape, num_classes, batch, jit, steps):
    """Pic de mémoire GPU (octets) et durée d'un pas d'entraînement ; None si pas de GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return None
    agent.compile(optimizer=tf.keras.optimizers.SGD(0.01, momentum=0.9),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), jit_compile=jit)
    x = tf.random.normal([batch] + input_shape)
    y = tf.random.uniform([batch], 0, num_classes, dtype=tf.int32)
    agent.train_on_batch(x, y)                          # compilation (et XLA)
    tf.config.experimental.reset_memory_stats("GPU:0")
    t0 = time.time()
    for _ in range(steps):
        agent.train_on_batch(x, y)
    elapsed = (time.time() - t0) / steps
    return tf.config.experimental.get_memory_info("GPU:0")["peak"], elapsed


def main(argv=None):
    args = parse_args(argv)
    args.spectral_config = resolve_spectral_config(args.spectral_config)
    setup_device(args.device)
    input_shape, num_classes = input_shape_of(args)
    print(f"{args.arch} (width {args.width}), entrée {input_shape}, {num_classes} classes, "
          f"configuration {args.spectral_config or 'aucune'}")
    with tf.device(args.device):
        agent = build(args, input_shape, num_classes)
        rows = layer_table(agent, args.batch_size[0])
        print_table(rows, args.batch_size[0])
        if args.out:
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"Tableau : {args.out}")

        if not tf.config.list_physical_devices("GPU") or "CPU" in args.device.upper():
            print("\nPas de GPU utilisé : pas de mesure du pic (seulement le tableau ci-dessus).")
            return
        jits = {"both": [False, True], "on": [True], "off": [False], "auto": ["auto"]}[args.jit]
        print("\nPic mesuré d'un pas d'entraînement (GPU:0, poids compris) :")
        print(f"{'batch':>6} " + " ".join(f"{('XLA ' + str(j)):>22}" for j in jits))
        for batch in args.batch_size:
            cells = []
            for jit in jits:
                try:
                    peak, dt = measure(agent, input_shape, num_classes, batch, jit, args.steps)
                    cells.append(f"{human(peak)} ({dt * 1000:.0f} ms)")
                except tf.errors.ResourceExhaustedError:
                    cells.append("mémoire insuffisante")
                tf.keras.backend.clear_session()
                agent = build(args, input_shape, num_classes)
            print(f"{batch:>6} " + " ".join(f"{c:>22}" for c in cells), flush=True)


if __name__ == "__main__":
    sys.exit(main())
