"""
Chargement des datasets (images ou signaux) pour run_experiment.py, pruning.py et les notebooks.

Datasets disponibles (--dataset) :
    cifar10, cifar100, mnist, fashion_mnist   via tf.keras.datasets (téléchargés au 1er appel)
    tfds:<nom>        tout dataset image de tensorflow_datasets (ex. tfds:svhn_cropped) ;
                      nécessite `pip install tensorflow-datasets`
    npz               fichier .npz (--data_path) contenant x_train, y_train, x_test, y_test :
                      images (N, H, W) ou (N, H, W, C), ou signaux (N, T) ou (N, T, C)
    folder            dossier (--data_path) avec train/<classe>/*.png et test/<classe>/*.png

Signaux (téléchargés au 1er appel dans ~/.keras/datasets/, puis relus depuis ce cache) :
    mnist1d           MNIST-1D (Greydanus 2020) : 4000 + 1000 signaux de longueur 40, 10 classes
    ucr:<Nom>         un dataset de l'archive UCR/UEA (timeseriesclassification.com), ex.
                      ucr:GunPoint, ucr:ECG200, ucr:FordA (univarié), ucr:BasicMotions (multivarié)
    Sans accès internet : télécharger le fichier à la main et le donner avec --data_path
    (mnist1d_data.pkl pour mnist1d ; dossier ou .zip contenant <Nom>_TRAIN.ts et <Nom>_TEST.ts,
    ou .txt / .tsv, pour ucr:<Nom>).

Modalité (--modality) : "image" ou "signal". "auto" : signal pour mnist1d et ucr:<Nom> ; pour un
.npz, signal si x est en (N, T), image sinon ; un .npz de signaux multicanaux (N, T, C) doit donc
être chargé avec --modality signal (sans quoi il serait lu comme des images (N, H, W)).

Les données sont rendues en float32, (N, H, W, C) pour des images et (N, T, C) pour des signaux,
normalisées par canal avec la moyenne et l'écart-type du jeu d'entraînement ; ces valeurs sont
renvoyées pour être enregistrées. Les images en entiers 0-255 sont d'abord ramenées dans [0, 1] ;
les signaux ne sont pas mis à l'échelle avant la normalisation. Les étiquettes qui ne sont pas
0..K-1 (ex. 1/2 ou -1/1) sont renumérotées, leurs valeurs d'origine deviennent les noms de classes.

Augmentation : crop = décalage aléatoire (padding de crop_pad par réflexion puis découpe à la
taille d'origine, en espace pour une image, dans le temps pour un signal), flip = symétrie
horizontale (images seulement). auto : crop seul pour MNIST et les signaux, crop_flip sinon.
"""
import glob
import os
import pickle
import zipfile
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

Arrays = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass
class DatasetInfo:
    """Tout ce qu'il faut savoir (et enregistrer) sur les données d'un run."""
    name: str
    source: str
    num_classes: int
    input_shape: List[int]
    n_train: int
    n_val: int
    n_test: int
    mean: List[float]
    std: List[float]
    augment: str
    class_names: List[str] = field(default_factory=list)
    modality: str = "image"   # "image" (N, H, W, C) ou "signal" (N, T, C)

    def to_dict(self):
        return asdict(self)


# --- Sources ----------------------------------------------------------------------

def _keras(loader: Callable) -> Callable[[], Arrays]:
    def load():
        (x_train, y_train), (x_test, y_test) = loader()
        return x_train, y_train, x_test, y_test
    return load


# (fonction de chargement, augmentation par défaut, noms des classes)
_KERAS_DATASETS: Dict[str, Tuple[Callable, str, List[str]]] = {
    "cifar10": (_keras(tf.keras.datasets.cifar10.load_data), "crop_flip",
                ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]),
    "cifar100": (_keras(lambda: tf.keras.datasets.cifar100.load_data("fine")), "crop_flip", []),
    "mnist": (_keras(tf.keras.datasets.mnist.load_data), "crop", [str(i) for i in range(10)]),
    "fashion_mnist": (_keras(tf.keras.datasets.fashion_mnist.load_data), "crop_flip",
                      ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]),
}


def _load_tfds(name: str) -> Tuple[Arrays, List[str]]:
    try:
        import tensorflow_datasets as tfds
    except ImportError as e:
        raise SystemExit("tfds:<nom> nécessite `pip install tensorflow-datasets`.") from e
    (tr, te), info = tfds.load(name, split=["train", "test"], batch_size=-1,
                               as_supervised=True, with_info=True)
    tr, te = tfds.as_numpy(tr), tfds.as_numpy(te)
    names = list(info.features["label"].names) if hasattr(info.features["label"], "names") else []
    return (tr[0], tr[1], te[0], te[1]), names


def _load_npz(path: str) -> Arrays:
    data = np.load(path)
    missing = {"x_train", "y_train", "x_test", "y_test"} - set(data.files)
    if missing:
        raise SystemExit(f"{path} : clés manquantes {sorted(missing)}")
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


def _load_folder(path: str) -> Tuple[Arrays, List[str]]:
    # image_dataset_from_directory impose une taille : on lit la taille de la 1re image.
    first_class = sorted(os.listdir(os.path.join(path, "train")))[0]
    first_file = sorted(os.listdir(os.path.join(path, "train", first_class)))[0]
    img = tf.io.decode_image(tf.io.read_file(os.path.join(path, "train", first_class, first_file)))
    size = tuple(img.shape[:2])
    mode = {1: "grayscale", 3: "rgb", 4: "rgba"}[img.shape[2]]

    out, names = [], []
    for split in ("train", "test"):
        ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(path, split), labels="inferred", label_mode="int", color_mode=mode,
            image_size=size, batch_size=None, shuffle=False)
        names = ds.class_names
        xs, ys = zip(*[(x.numpy(), y.numpy()) for x, y in ds])
        out += [np.stack(xs), np.array(ys)]
    return tuple(out), names


# --- Signaux téléchargeables ----------------------------------------------------------

_MNIST1D_URL = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
_UCR_URLS = ["https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip",
             "https://timeseriesclassification.com/Downloads/{name}.zip"]


def _download(fname: str, urls: List[str], subdir: str) -> str:
    """Télécharge le premier lien qui répond dans ~/.keras/datasets/<subdir>/ (cache)."""
    errors = []
    for url in urls:
        try:
            return tf.keras.utils.get_file(fname, url, cache_subdir=os.path.join("datasets", subdir))
        except Exception as e:  # lien mort, pas d'accès internet...
            errors.append(f"  {url} : {e}")
    raise SystemExit(f"Téléchargement de {fname} impossible :\n" + "\n".join(errors) +
                     "\nTélécharger le fichier à la main et le donner avec --data_path.")


def _load_mnist1d(data_path: Optional[str]) -> Tuple[Arrays, str]:
    path = data_path or _download("mnist1d_data.pkl", [_MNIST1D_URL], "mnist1d")
    with open(path, "rb") as f:
        data = pickle.load(f)  # fichier publié par l'auteur de MNIST-1D (dictionnaire numpy)
    return (data["x"], data["y"], data["x_test"], data["y_test"]), os.path.abspath(path)


def _read_ts(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fichier .ts (sktime / aeon) : une ligne par série, canaux séparés par ':', étiquette à la fin."""
    xs, ys, in_data = [], [], False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                in_data = line.lower().startswith("@data")
                continue
            *channels, label = line.split(":")
            xs.append([[float(v) for v in c.replace("?", "nan").split(",")] for c in channels])
            ys.append(label.strip())
    lengths = {len(c) for x in xs for c in x}
    if len(lengths) > 1:
        raise SystemExit(f"{path} : séries de longueurs différentes ({min(lengths)} à "
                         f"{max(lengths)}), non prises en charge")
    return np.asarray(xs, np.float32).transpose(0, 2, 1), np.asarray(ys)  # (N, T, C)


def _read_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fichier .txt / .tsv de l'archive UCR : une série univariée par ligne, étiquette en 1re colonne."""
    data = np.genfromtxt(path, delimiter="\t" if path.endswith(".tsv") else None)
    return data[:, 1:].astype(np.float32), data[:, 0]


def _load_ucr(name: str, data_path: Optional[str]) -> Tuple[Arrays, str]:
    path = data_path or _download(f"{name}.zip", [u.format(name=name) for u in _UCR_URLS], "ucr")
    folder = path
    if zipfile.is_zipfile(path):
        folder = os.path.join(os.path.dirname(path), name)
        if not os.path.isdir(folder):
            with zipfile.ZipFile(path) as z:
                z.extractall(folder)
    found = {}
    for ext, reader in [(".ts", _read_ts), (".tsv", _read_txt), (".txt", _read_txt)]:
        files = {split: glob.glob(os.path.join(folder, "**", f"*{name}_{split}{ext}"), recursive=True)
                 for split in ("TRAIN", "TEST")}
        if files["TRAIN"] and files["TEST"]:
            found = {split: reader(paths[0]) for split, paths in files.items()}
            break
    if not found:
        raise SystemExit(f"{folder} : fichiers {name}_TRAIN / {name}_TEST (.ts, .tsv ou .txt) introuvables")
    (x_train, y_train), (x_test, y_test) = found["TRAIN"], found["TEST"]
    if np.isnan(x_train).any() or np.isnan(x_test).any():
        raise SystemExit(f"ucr:{name} contient des valeurs manquantes, non prises en charge")
    try:  # étiquettes numériques écrites en texte : ordre numérique (2 avant 10)
        y_train, y_test = y_train.astype(float), y_test.astype(float)
        if np.all(np.mod(np.concatenate([y_train, y_test]), 1) == 0):
            y_train, y_test = y_train.astype(int), y_test.astype(int)
    except ValueError:
        pass
    return (x_train, y_train, x_test, y_test), os.path.abspath(folder)


# --- Chargement principal -----------------------------------------------------------

def _modality(name: str, x: np.ndarray, modality: str) -> str:
    """Images ou signaux, d'après --modality, le dataset et la forme des données."""
    signal_source = name == "mnist1d" or name.startswith("ucr:")
    if modality == "auto":
        return "signal" if signal_source or np.ndim(x) == 2 else "image"
    if signal_source and modality == "image":
        raise SystemExit(f"--modality image : '{name}' contient des signaux")
    if modality == "signal":
        if not (signal_source or name == "npz"):
            raise SystemExit(f"--modality signal : '{name}' contient des images ; les signaux "
                             "viennent de npz, mnist1d ou ucr:<Nom>.")
        if np.ndim(x) not in (2, 3):
            raise SystemExit(f"--modality signal : x doit être (N, T) ou (N, T, C), reçu {np.shape(x)}")
    elif np.ndim(x) not in (3, 4):
        raise SystemExit(f"--modality image : x doit être (N, H, W) ou (N, H, W, C), reçu {np.shape(x)}")
    return modality


def available_datasets() -> List[str]:
    return sorted(_KERAS_DATASETS) + ["tfds:<nom>", "npz", "folder", "mnist1d", "ucr:<Nom>"]


def load_dataset(name: str, data_path: Optional[str] = None, augment: str = "auto",
                 train_subset: int = 0, test_subset: int = 0, val_split: float = 0.0,
                 image_size: int = 0, seed: int = 42, modality: str = "auto"):
    """
    Charge, découpe et normalise un dataset (images ou signaux, voir --modality).

    Returns:
        (x_train, y_train), (x_val, y_val) ou None, (x_test, y_test), DatasetInfo
    """
    class_names: List[str] = []
    default_augment = "crop_flip"
    if name in _KERAS_DATASETS:
        loader, default_augment, class_names = _KERAS_DATASETS[name]
        arrays, source = loader(), "tf.keras.datasets"
    elif name.startswith("tfds:"):
        arrays, class_names = _load_tfds(name[5:])
        source = "tensorflow_datasets"
    elif name == "mnist1d":
        arrays, source = _load_mnist1d(data_path or None)
        class_names = [str(i) for i in range(10)]
    elif name.startswith("ucr:"):
        arrays, source = _load_ucr(name[4:], data_path or None)
    elif name in ("npz", "folder"):
        if not data_path:
            raise SystemExit(f"--dataset {name} nécessite --data_path")
        if name == "npz":
            arrays = _load_npz(data_path)
        else:
            arrays, class_names = _load_folder(data_path)
        source = os.path.abspath(data_path)
    else:
        raise SystemExit(f"Dataset inconnu : '{name}'. Disponibles : {available_datasets()}")

    x_train, y_train, x_test, y_test = arrays
    modality = _modality(name, x_train, modality)
    y_train, y_test = np.asarray(y_train).reshape(-1), np.asarray(y_test).reshape(-1)
    x_train, x_test = _with_channels(x_train, modality), _with_channels(x_test, modality)

    # Étiquettes renumérotées 0..K-1 si besoin (ex. 1/2 ou -1/1 dans des fichiers de signaux)
    labels = np.unique(np.concatenate([y_train, y_test]))
    if not np.array_equal(labels, np.arange(len(labels))):
        y_train, y_test = np.searchsorted(labels, y_train), np.searchsorted(labels, y_test)
        class_names = class_names or [str(v) for v in labels]
    num_classes = int(max(y_train.max(), y_test.max()) + 1)

    if train_subset:
        x_train, y_train = x_train[:train_subset], y_train[:train_subset]
    if test_subset:
        x_test, y_test = x_test[:test_subset], y_test[:test_subset]

    # Validation tirée du train (mélange reproductible par seed) ; sinon on valide sur le test.
    val = None
    if val_split > 0:
        idx = np.random.default_rng(seed).permutation(len(x_train))
        n_val = int(round(len(x_train) * val_split))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        val = (x_train[val_idx], y_train[val_idx])
        x_train, y_train = x_train[train_idx], y_train[train_idx]

    if image_size and modality == "signal":
        raise SystemExit("--image_size ne s'applique qu'aux images")
    if image_size:
        resize = lambda x: tf.image.resize(x, (image_size, image_size)).numpy()  # noqa: E731
        x_train, x_test = resize(x_train), resize(x_test)
        val = (resize(val[0]), val[1]) if val else None

    # Normalisation par canal avec les statistiques du train uniquement
    scale = 1.0
    if modality == "image" and (x_train.dtype == np.uint8 or x_train.max() > 1.5):
        scale = 255.0
    x_train = x_train.astype(np.float32) / scale
    axes = tuple(range(x_train.ndim - 1))  # tout sauf le canal
    mean = x_train.mean(axis=axes)
    std = x_train.std(axis=axes) + 1e-7
    norm = lambda x: ((x.astype(np.float32) / scale) - mean) / std  # noqa: E731
    x_train = (x_train - mean) / std
    x_test = norm(x_test)
    val = (norm(val[0]), val[1]) if val else None

    if augment == "auto":
        augment = "crop" if modality == "signal" else default_augment
    if modality == "signal" and "flip" in augment:
        raise SystemExit("--augment crop_flip : pas de flip pour des signaux (utiliser crop ou none)")

    info = DatasetInfo(
        name=name, source=source, num_classes=num_classes,
        input_shape=list(x_train.shape[1:]),
        n_train=len(x_train), n_val=len(val[0]) if val else 0, n_test=len(x_test),
        mean=[float(m) for m in mean], std=[float(s) for s in std],
        augment=augment,
        class_names=class_names,
        modality=modality,
    )
    return (x_train, y_train), val, (x_test, y_test), info


def _with_channels(x: np.ndarray, modality: str) -> np.ndarray:
    """Ajoute l'axe des canaux s'il manque : (N, H, W) -> (N, H, W, 1), (N, T) -> (N, T, 1)."""
    x = np.asarray(x)
    missing = x.ndim == (3 if modality == "image" else 2)
    return x[..., None] if missing else x


def make_tf_datasets(train, val, test, info: DatasetInfo, batch_size: int,
                     crop_pad: int = 4, seed: int = 42):
    """tf.data : mélange + augmentation (crop / flip selon info.augment) + batch."""
    shape = info.input_shape                 # [H, W, C] ou [T, C]
    pads = [[crop_pad, crop_pad]] * (len(shape) - 1) + [[0, 0]]

    def augment(x, y):
        if "crop" in info.augment:
            # image : décalage en espace ; signal : décalage dans le temps
            x = tf.pad(x, pads, mode="REFLECT")
            x = tf.image.random_crop(x, shape)
        if "flip" in info.augment:
            x = tf.image.random_flip_left_right(x)
        return x, y

    train_ds = tf.data.Dataset.from_tensor_slices(train).shuffle(
        len(train[0]), seed=seed, reshuffle_each_iteration=True)
    if info.augment != "none":
        train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    batch = lambda d: tf.data.Dataset.from_tensor_slices(d).batch(512).prefetch(tf.data.AUTOTUNE)  # noqa: E731
    return train_ds, batch(val) if val else None, batch(test)
