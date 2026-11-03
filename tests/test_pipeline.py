"""
Chaîne complète sur des données synthétiques (sur CPU, quelques secondes par étape) :
entraînement (plusieurs graines) -> reprise d'un run interrompu -> pruning pré et post
(y compris réparti en parties, comme avec --gpus) -> figures -> mesure mémoire.

    pytest tests/test_pipeline.py
"""
import csv
import json
import os

import numpy as np
import pytest
import tensorflow as tf

from specprune.cli import bench, plot, prune, train

ARCH = "spectral_resnet20"
# lambda_in entraîné dans les deux premières couches seulement, lambda_out figé
SPECTRAL = {"default": {"trainable_lambda_in": False, "trainable_lambda_out": False,
                        "trainable_alpha": False},
            "layers": {"0-1": {"trainable_lambda_in": True}}}


def make_npz(path, shape, n_train=64, n_test=32, classes=3, seed=0):
    """Classes séparables : chaque classe a sa propre moyenne."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, (classes,) + shape).astype("float32")

    def draw(n):
        y = rng.integers(0, classes, n)
        return (centers[y] + 0.3 * rng.normal(0, 1, (n,) + shape)).astype("float32"), y

    (x_tr, y_tr), (x_te, y_te) = draw(n_train), draw(n_test)
    np.savez(path, x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
    return str(path)


def train_argv(data, config, out, *extra):
    modality = "image" if np.load(data)["x_train"].ndim == 4 else "signal"
    return ["--dataset", "npz", "--data_path", data, "--modality", modality, "--arch", ARCH, "--spectral_config", config,
            "--epochs", "2", "--batch_size", "16", "--lr", "0.05", "--crop_pad", "0",
            "--augment", "none", "--device", "/CPU:0", "--verbose", "0", "--out_dir", out, *extra]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module", params=[(16, 16, 3), (32, 2)], ids=["image", "signal"])
def workspace(request, tmp_path_factory):
    """Données, configuration spectrale et 2 runs (graines 1 et 2), partagés par les tests du module."""
    root = tmp_path_factory.mktemp("ws")
    data = make_npz(root / "data.npz", request.param)
    config = root / "spectral.json"
    config.write_text(json.dumps(SPECTRAL))
    train.main(train_argv(data, str(config), str(root / "runs"), "--n_runs", "2", "--seed", "1",
                          "--run_name", "grp"))
    tf.keras.backend.clear_session()
    return {"root": root, "data": data, "config": str(config), "group": root / "runs" / "grp"}


def test_train_group(workspace):
    group = workspace["group"]
    assert json.loads((group / "group.json").read_text())["seeds"] == [1, 2]
    for seed in (1, 2):
        run = group / f"seed{seed}"
        config = json.loads((run / "config.json").read_text())
        assert config["format_version"] == 2 and "finished_at" in config
        assert (run / "weights.weights.h5").exists()
        assert not (run / "backup").exists()            # point de reprise effacé en fin de run
        assert len(read_csv(run / "history.csv")) == 2
    assert len(read_csv(group / "runs_summary.csv")) == 2
    # tout est déjà fait : la reprise du groupe ne relance rien
    train.main(["--resume", str(group), "--device", "/CPU:0", "--verbose", "0"])


class Interrupt(tf.keras.callbacks.CSVLogger):
    """CSVLogger qui coupe l'entraînement après la première epoch (point de reprise déjà écrit)."""

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if epoch == 0:
            raise KeyboardInterrupt


def test_resume_interrupted_run(workspace, monkeypatch):
    out = str(workspace["root"] / "runs_resume")
    argv = train_argv(workspace["data"], workspace["config"], out, "--epochs", "3", "--run_name", "cut")
    monkeypatch.setattr(tf.keras.callbacks, "CSVLogger", Interrupt)
    with pytest.raises(KeyboardInterrupt):
        train.main(argv)
    monkeypatch.undo()
    run = os.path.join(out, "cut")
    assert os.path.isdir(os.path.join(run, "backup"))
    assert "finished_at" not in json.load(open(os.path.join(run, "config.json")))

    tf.keras.backend.clear_session()
    train.main(["--resume", run, "--device", "/CPU:0", "--verbose", "0"])
    config = json.load(open(os.path.join(run, "config.json")))
    assert "finished_at" in config and len(config["resumed_at"]) == 1
    assert [int(r["epoch"]) for r in train._read_history(os.path.join(run, "history.csv"))] == [0, 1, 2]
    assert not os.path.exists(os.path.join(run, "backup"))


def prune_argv(workspace, name, *extra):
    runs = [str(workspace["group"] / "seed1"), str(workspace["group"] / "seed2")]
    return [*runs, "--p", "0", "0.5", "1", "--epochs", "1", "--device", "/CPU:0",
            "--out_dir", str(workspace["root"] / "pruning"), "--name", name, *extra]


def test_prune_pre_post(workspace):
    prune.main(prune_argv(workspace, "both", "--mode", "both"))
    out = workspace["root"] / "pruning" / "both"
    rows = read_csv(out / "results.csv")
    assert len(rows) == 2 * 3 * 2                       # 2 runs x 3 p x (pre, post)
    pre = {(r["run"], float(r["p"])): r for r in rows if r["mode"] == "pre"}
    for run in ("seed1", "seed2"):
        # p = 0 : précision du run ; p = 1 : tous les lambda_in entraînés retirés
        run_acc = json.load(open(workspace["group"] / run / "config.json"))["test_accuracy"]
        assert abs(float(pre[(run, 0.0)]["test_accuracy"]) - run_acc) < 2e-3
        assert float(pre[(run, 1.0)]["fraction_pruned"]) == 1.0
        assert (out / run / "masks" / "p0.50.npz").exists()
        assert (out / run / "post" / "p0.50_r0" / "history.csv").exists()
    summary = read_csv(out / "summary.csv")
    assert {(s["mode"], float(s["p"])) for s in summary} == {(m, p) for m in ("pre", "post") for p in (0, 0.5, 1)}
    assert all(int(s["n"]) == 2 for s in summary)


def test_prune_shards_match(workspace):
    """Deux parties (ce que fait --gpus, un processus par GPU) = le même travail qu'un seul processus."""
    prune.main(prune_argv(workspace, "sharded", "--mode", "pre"))
    out = workspace["root"] / "pruning" / "sharded"
    whole = {(r["run"], r["p"]): r["test_accuracy"] for r in read_csv(out / "results.csv")}
    os.remove(out / "results.csv")
    for k in range(2):
        prune.main(["--resume", str(out), "--shard", f"{k}/2", "--device", "/CPU:0"])
    parts = [read_csv(out / f"results_part{k}.csv") for k in range(2)]
    assert parts[0] and parts[1]
    assert {(r["run"], r["p"]) for r in parts[0]}.isdisjoint({(r["run"], r["p"]) for r in parts[1]})
    merged = {(r["run"], r["p"]): r["test_accuracy"] for r in parts[0] + parts[1]}
    assert merged == whole
    prune.finish(str(out))
    assert len(read_csv(out / "summary.csv")) == 3


def test_prune_refuses_mixed_runs(workspace):
    other = workspace["root"] / "other"
    data = make_npz(workspace["root"] / "other.npz", (8, 8, 1), classes=4)
    train.main(train_argv(data, workspace["config"], str(other), "--epochs", "1", "--run_name", "r"))
    argv = [str(workspace["group"] / "seed1"), str(other / "r"), "--device", "/CPU:0",
            "--out_dir", str(workspace["root"] / "pruning"), "--p", "0"]
    with pytest.raises(SystemExit, match="non comparables"):
        prune.main(argv)
    prune.main(argv + ["--allow_mixed", "--name", "mixed"])


def test_plots(workspace):
    out = workspace["root"] / "pruning" / "both"
    plot.main(["pruning", str(out), "--formats", "png"])
    assert (out / "figures" / "accuracy_vs_p.png").exists()
    plot.main(["history", str(workspace["group"] / "seed1"), str(workspace["group"] / "seed2"),
               "--formats", "png"])
    assert (workspace["group"] / "figures" / "history.png").exists()
    plot.main(["masks", str(out), "--p", "0.5", "--formats", "png"])
    assert (out / "figures" / "masks.png").exists()


def test_bench(workspace, capsys):
    shape = [int(s) for s in json.load(open(workspace["group"] / "seed1" / "config.json"))["dataset"]["input_shape"]]
    bench.main(["--input_shape", *map(str, shape), "--num_classes", "3", "--batch_size", "16",
                "--device", "/CPU:0", "--spectral_config", workspace["config"]])
    assert "X_win" in capsys.readouterr().out
