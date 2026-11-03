"""
Entraînement : appareil, optimiseur et compilation, communs à run_experiment.py et pruning.py.
"""
import sys
from types import SimpleNamespace

import tensorflow as tf

# Poids spectraux : ils ont leur propre régularisation (L1), donc pas de weight decay.
SPECTRAL_WEIGHTS = ("lambda_in", "lambda_out", "alpha")


def setup_device(device: str, mixed_precision: bool = False):
    """Vérifie que le GPU demandé existe, active la croissance mémoire ; renvoie les GPU visibles."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if device.upper().startswith("/GPU") and not gpus:
        sys.exit(f"Aucun GPU visible par TensorFlow : impossible d'utiliser {device}. "
                 "Vérifiez l'installation, ou passez --device /CPU:0.")
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    return gpus


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Montée linéaire pendant warmup_steps, puis le schedule de base (décalé)."""

    def __init__(self, base, warmup_steps, peak_lr):
        self.base, self.warmup_steps, self.peak_lr = base, warmup_steps, peak_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.peak_lr * (step + 1.0) / max(1, self.warmup_steps)
        return tf.where(step < self.warmup_steps, warm, self.base(step - self.warmup_steps))

    def get_config(self):
        return {"warmup_steps": self.warmup_steps, "peak_lr": self.peak_lr}


def make_optimizer(args, steps_per_epoch):
    total = args.epochs * steps_per_epoch
    warmup = args.warmup_epochs * steps_per_epoch
    if args.schedule == "cosine":
        lr = tf.keras.optimizers.schedules.CosineDecay(args.lr, max(1, total - warmup))
    elif args.schedule == "step":
        bounds = [e * steps_per_epoch - warmup for e in args.step_epochs]
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            bounds, [args.lr * 0.1 ** i for i in range(len(bounds) + 1)])
    else:
        lr = lambda step: args.lr  # noqa: E731
    if warmup:
        lr = WarmupSchedule(lr, warmup, args.lr)
    elif args.schedule == "constant":
        lr = args.lr

    if args.optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(lr, momentum=args.momentum, nesterov=args.nesterov,
                                      weight_decay=args.weight_decay or None)
    else:
        opt = tf.keras.optimizers.AdamW(lr, weight_decay=args.weight_decay)
    if args.weight_decay:
        # Les poids spectraux ont leur propre régularisation (L1) : pas de weight decay.
        opt.exclude_from_weight_decay(var_names=list(SPECTRAL_WEIGHTS))
    return opt



def compile_agent(agent, args, steps_per_epoch):
    """Compile avec l'optimiseur de la recette (args) et la perte adaptée au label smoothing."""
    agent.compile(
        optimizer=make_optimizer(args, steps_per_epoch),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        if args.label_smoothing else tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def onehot_datasets(datasets, num_classes):
    """Étiquettes en one-hot (nécessaire avec le label smoothing) ; None reste None."""
    to_onehot = lambda x, y: (x, tf.one_hot(y, num_classes))  # noqa: E731
    return [ds.map(to_onehot) if ds is not None else None for ds in datasets]


def recipe_from_run(run_args, epochs=None, lr=None, batch_size=None, optimizer=None,
                    weight_decay=None, schedule=None, warmup_epochs=0):
    """
    Recette d'entraînement (pour make_optimizer / compile_agent) reprise d'un run (config["args"]).
    Les arguments donnés remplacent ceux du run ; epochs et lr valent ceux du run par défaut.
    """
    return SimpleNamespace(
        epochs=epochs if epochs is not None else run_args.get("epochs", 1),
        warmup_epochs=warmup_epochs,
        lr=lr if lr is not None else run_args.get("lr", 0.1),
        schedule=schedule or run_args.get("schedule", "cosine"),
        step_epochs=run_args.get("step_epochs", [100, 150]),
        optimizer=optimizer or run_args.get("optimizer", "sgd"),
        momentum=run_args.get("momentum", 0.9), nesterov=run_args.get("nesterov", True),
        weight_decay=weight_decay if weight_decay is not None else run_args.get("weight_decay", 0.0),
        label_smoothing=run_args.get("label_smoothing", 0.0),
        batch_size=batch_size or run_args.get("batch_size", 128),
    )
