"""
Vérifie les couches et les modèles : python tests/verifier_modeles.py [noms...] (tous par défaut).

Chaque modèle est construit avec configs/classique.json (lambda_in = 1, lambda_out = 0,
alpha = 1 figés : mêmes paramètres entraînables que le modèle classique) et avec
configs/all_spectral.json (lambda_in entraîné partout), sur des images (32x32x3, 28x28x1) et
sur des signaux (128x3, 100x1, convolutions SpecCnn1D).
"""
import sys, time, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # racine du projet
sys.path.insert(0, ROOT)
import numpy as np, tensorflow as tf
from models import AgentNetwork, available_models, configure_spectral
from spectralcnn import SpecCnn1D, SpecCnn2D

FROZEN = dict(use_bias=False, trainable_lambda_in=False, trainable_lambda_out=False,
              trainable_alpha=False)

# Équivalences : lambdas figés (1, 0, 1) => couches Keras standard
x2 = tf.random.normal((2, 9, 9, 6))
for groups, filters, stride in [(1, 4, 1), (2, 4, 1), (3, 6, 1), (6, 6, 1), (6, 6, 2), (6, 12, 1)]:
    sc = SpecCnn2D(filters, 3, strides=stride, padding="same", groups=groups, **FROZEN)
    y = sc(x2)
    c = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", groups=groups, use_bias=False)
    c(x2); c.set_weights([np.transpose(sc.phi_weights.numpy(), (2, 3, 1, 0))])
    print(f"SpecCnn2D groups={groups} filters={filters} stride={stride} vs Conv2D : "
          f"max|diff| = {float(tf.reduce_max(tf.abs(y - c(x2)))):.1e}")
sc = SpecCnn2D(6, 3, padding="same", groups=6, **FROZEN); y = sc(x2)
dw = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False); dw(x2)
dw.set_weights([np.transpose(sc.phi_weights.numpy(), (2, 3, 0, 1))])
print(f"SpecCnn2D groups=C vs DepthwiseConv2D : max|diff| = {float(tf.reduce_max(tf.abs(y - dw(x2)))):.1e}")
# "same" des couches spectrales = padding symétrique (K-1)/2 de chaque côté (convention PyTorch).
# Avec stride 2 sur une longueur paire, Keras "same" met tout le padding à droite : on compare
# donc à Conv1D "valid" précédée du même padding symétrique.
x1 = tf.random.normal((2, 16, 6))
for groups, filters, stride in [(1, 4, 1), (3, 6, 1), (6, 6, 1), (6, 6, 2), (1, 4, 2)]:
    sc = SpecCnn1D(filters, 3, strides=stride, padding="same", groups=groups, **FROZEN); y = sc(x1)
    c = tf.keras.layers.Conv1D(filters, 3, strides=stride, padding="valid", groups=groups, use_bias=False)
    xp = tf.pad(x1, [[0, 0], [1, 1], [0, 0]])
    c(xp); c.set_weights([np.transpose(sc.phi_weights.numpy(), (2, 1, 0))])
    print(f"SpecCnn1D groups={groups} filters={filters} stride={stride} vs Conv1D : "
          f"max|diff| = {float(tf.reduce_max(tf.abs(y - c(xp)))):.1e}")
cfg = SpecCnn2D(6, 3, groups=3).get_config()
assert SpecCnn2D.from_config(cfg)(x2).shape == (2, 7, 7, 6)
print("get_config/from_config avec groups OK")

names = sys.argv[1:] or available_models()
for mode in ["classique", "all_spectral"]:
    cfg = os.path.join(ROOT, "configs", f"{mode}.json")
    configure_spectral(cfg)
    for name in names:
        for shape in [(32, 32, 3), (28, 28, 1), (128, 3), (100, 1)]:
            t = time.time()
            agent = AgentNetwork(arch=name, num_classes=10)
            out = agent(tf.random.normal((2,) + shape))
            ok = out.shape == (2, 10) and bool(np.allclose(out.numpy().sum(1), 1, atol=1e-4))
            trainable = sum(int(np.prod(v.shape)) for v in agent.trainable_variables)
            print(f"{mode:12s} {name:25s} {str(shape):12s} out={tuple(out.shape)} ok={ok} "
                  f"params={agent.count_params()/1e6:8.2f}M entraînables={trainable/1e6:7.2f}M "
                  f"feat={agent._backbone.output.shape[1:]} {time.time()-t:.1f}s", flush=True)
            del agent; tf.keras.backend.clear_session()
