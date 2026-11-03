"""speccnn2d.py

Keras Layer implementation of SpecCnn2D.
2D spectral convolution with multi-channel support and lambda modulation.

Depends on spectral_conv_op.py for all tensor operations.

Flattened formulation
---------------------
The spatial grids (Oh,Ow) and (Lh,Lw) are flattened immediately after padding:

    O = Oh·Ow    (flattened output positions)
    L = Lh·Lw    (flattened padded input positions)

The effective kernel per (filter f, channel c) is then:

    kernel[f,c, o, l] = Phi[f,c, o, l]
                      * (lambda_in[f,c, l] - lambda_out[f,c, o])

This is the same matrix structure as SpecCnn1D — just with O=Oh·Ow, L=Lh·Lw.

Forward pass:

    X_flat     = reshape(X, (B, H*W, C))
    X_pad_flat = pad_matrix @ X_flat              (B, L, C)
    Z          = einsum('fcol,blc->bofc', kernel, X_pad_flat)
    Y_flat     = einsum('bofc,fco->bof',  Z, alpha)
    output     = reshape(Y_flat, (B, Oh, Ow, F))  [+ bias]  [activation]

Weight shapes  (C is replaced by C/groups when groups > 1)
-------------
    phi_weights  : (F, C, Kh, Kw)   shared kernel  (weight sharing)
    lambda_in    : (F, C, Lh*Lw)    input modulation per flattened position
    lambda_out   : (F, C, Oh*Ow)    output modulation per flattened position
    alpha        : (F, C, Oh*Ow)    inter-channel aggregation
    bias         : (F,)
"""

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, activations, regularizers

from .spectral_conv_op import (
    compute_conv_dims_2d,
    spectral_conv_2d,
)


class SpecCnn2D(layers.Layer):
    """Spectral 2D convolution layer.

    Flattened formulation — spatial grids are flattened to 1D vectors
    immediately after padding. The convolution is then a plain matrix product.

    Effective kernel (per filter f, channel c):

        kernel[f,c, o, l] = Phi[f,c,o,l]
                          * (lambda_in[f,c,l] - lambda_out[f,c,o])

        o = oh·Ow + ow   (flattened output position)
        l = lh·Lw + lw   (flattened padded input position)

    Forward pass:

        X_pad_flat = pad(X)                              (B, Lh*Lw, C)
        Z          = einsum('fcol,blc->bofc', kernel, X_pad_flat)
        Y_flat     = einsum('bofc,fco->bof', Z, alpha)
        output     = reshape(Y_flat, (B, Oh, Ow, F)) [+ bias] [activation]

    Trainable weights
    -----------------
    phi_weights  : (F, C, Kh, Kw)    if trainable_phi=False  : frozen ones
    lambda_in    : (F, C, Lh*Lw)     if trainable_lambda_in=False : frozen ones
    lambda_out   : (F, C, Oh*Ow)     if trainable_lambda_out=False : frozen zeros
    alpha        : (F, C, Oh*Ow)     if trainable_alpha=False : frozen ones
    bias         : (F,)              if use_bias=False : absent

    Non-trainable constants (built once in build())
    -----------------------------------------------
    _sparse_indices : (F*C*Oh*Ow*Kh*Kw, 4)   receptive-field connectivity
    _pad_matrix     : (Lh*Lw, H*W)            Kronecker padding matrix

    Parameters
    ----------
    filters : int
    kernel_size : int or (int, int), default 3
    strides : int or (int, int), default 1
    padding : 'valid' | 'same', default 'valid'
    use_bias : bool, default True
    activation : str or callable, default None
    groups : int, default 1
        As in Keras Conv2D: channels and filters are split into `groups`
        groups, each filter only sees the C/groups channels of its group.
        Weights then use C/groups instead of C (shapes below).
        groups=C with filters=C => depthwise spectral convolution (MobileNet).
    trainable_phi : bool, default True
    trainable_lambda_in : bool, default True
    trainable_lambda_out : bool, default True
    trainable_alpha : bool, default True

    phi_regularizer        : Regularizer, default None
    lambda_in_regularizer  : Regularizer, default L1(1e-2) when trainable_lambda_in=True,
                             else None. Applied only when the weight is trained.
    lambda_out_regularizer : Regularizer, default L1(1e-2) when trainable_lambda_out=True,
                             else None. Applied only when the weight is trained.
    alpha_regularizer      : Regularizer, default None
    bias_regularizer       : Regularizer, default None

    Special cases
    -------------
    trainable_phi=True, lambda_in=1, lambda_out=0, alpha=1
        => standard Conv2D (unweighted channel sum)
    trainable_phi=False
        => fixed sliding-sum window; lambdas learn spatial focus
    trainable_alpha=True
        => each channel contributes independently per output position
    groups=G (with lambda_in=1, lambda_out=0, alpha=1)
        => Conv2D(groups=G) ; groups=C, filters=C => DepthwiseConv2D
    """

    def __init__(
        self,
        filters                : int,
        kernel_size                   = 3,
        strides                       = 1,
        padding                : str  = "valid",
        use_bias               : bool = True,
        activation                    = None,
        groups                 : int  = 1,
        trainable_phi          : bool = True,
        trainable_lambda_in    : bool = True,
        trainable_lambda_out   : bool = True,
        trainable_alpha        : bool = True,
        phi_initializer               = "glorot_uniform",
        lambda_in_initializer         = None,
        lambda_out_initializer        = None,
        alpha_initializer             = None,
        bias_initializer              = "zeros",
        phi_regularizer               = None,
        lambda_in_regularizer         = None,
        lambda_out_regularizer        = None,
        alpha_regularizer             = None,
        bias_regularizer              = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # -- Hyperparameter validation ----------------------------------------
        if filters < 1:
            raise ValueError(f"filters must be >= 1, got {filters}")

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)

        if len(kernel_size) != 2 or any(k < 1 for k in kernel_size):
            raise ValueError(
                f"kernel_size must be a positive int or (int, int), got {kernel_size}"
            )
        if len(strides) != 2 or any(s < 1 for s in strides):
            raise ValueError(
                f"strides must be a positive int or (int, int), got {strides}"
            )
        if padding not in ("valid", "same"):
            raise ValueError(f"padding must be 'valid' or 'same', got '{padding}'")
        if padding == "same" and any(k % 2 == 0 for k in kernel_size):
            raise ValueError(
                f"padding='same' requires odd kernel sizes, got {kernel_size}"
            )

        if groups < 1 or filters % groups != 0:
            raise ValueError(
                f"filters ({filters}) must be divisible by groups ({groups})"
            )

        self.filters              = filters
        self.groups               = groups
        self.kernel_size          = kernel_size
        self.strides              = strides
        self.padding_mode         = padding
        self.use_bias             = use_bias
        self.trainable_phi        = trainable_phi
        self.trainable_lambda_in  = trainable_lambda_in
        self.trainable_lambda_out = trainable_lambda_out
        self.trainable_alpha      = trainable_alpha

        self._activation_identifier = activation
        self.activation = activations.get(activation)

        # Default initializers (mirror speccnn1d.py)
        if lambda_in_initializer is None:
            lambda_in_initializer = (
                initializers.TruncatedNormal(mean=1.0, stddev=0.05)
                if trainable_lambda_in else initializers.Ones()
            )
        if lambda_out_initializer is None:
            lambda_out_initializer = (
                initializers.TruncatedNormal(mean=0.0, stddev=0.05)
                if trainable_lambda_out else initializers.Zeros()
            )
        if alpha_initializer is None:
            alpha_initializer = (
                initializers.TruncatedNormal(mean=1.0, stddev=0.05)
                if trainable_alpha else initializers.Ones()
            )

        self.phi_initializer        = initializers.get(phi_initializer)
        self.lambda_in_initializer  = initializers.get(lambda_in_initializer)
        self.lambda_out_initializer = initializers.get(lambda_out_initializer)
        self.alpha_initializer      = initializers.get(alpha_initializer)
        self.bias_initializer       = initializers.get(bias_initializer)

        # Default regularizers (L1 when trained)
        if lambda_in_regularizer is None and trainable_lambda_in:
            lambda_in_regularizer = regularizers.L1(1e-2)
        if lambda_out_regularizer is None and trainable_lambda_out:
            lambda_out_regularizer = regularizers.L1(1e-2)

        self.phi_regularizer        = regularizers.get(phi_regularizer)
        self.lambda_in_regularizer  = regularizers.get(lambda_in_regularizer)
        self.lambda_out_regularizer = regularizers.get(lambda_out_regularizer)
        self.alpha_regularizer      = regularizers.get(alpha_regularizer)
        self.bias_regularizer       = regularizers.get(bias_regularizer)

    # --------------------------------------------------------------------------
    # build
    # --------------------------------------------------------------------------

    def build(self, input_shape):
        """Creates weights and architectural constants from input_shape (B,H,W,C).

        Weight shapes after build:
            phi_weights  : (F, C, Kh, Kw)
            lambda_in    : (F, C, Lh*Lw)   ← flattened input positions
            lambda_out   : (F, C, Oh*Ow)   ← flattened output positions
            alpha        : (F, C, Oh*Ow)
            bias         : (F,)
        """
        H = int(input_shape[1])
        W = int(input_shape[2])
        C_in = int(input_shape[3])
        if C_in % self.groups != 0:
            raise ValueError(
                f"input channels ({C_in}) must be divisible by groups ({self.groups})"
            )
        C  = C_in // self.groups   # channels seen by each filter
        F  = self.filters
        Kh, Kw = self.kernel_size
        Sh, Sw = self.strides

        # -- Padding ----------------------------------------------------------
        Ph = (Kh - 1) // 2 if self.padding_mode == "same" else 0
        Pw = (Kw - 1) // 2 if self.padding_mode == "same" else 0

        Oh, Ow, Lh, Lw = compute_conv_dims_2d(H, W, Kh, Kw, Sh, Sw, Ph, Pw)
        O = Oh * Ow   # flattened output size
        L = Lh * Lw   # flattened padded input size

        # Store for call() and compute_output_shape()
        self._C  = C
        self._H  = H;   self._W  = W
        self._Oh = Oh;  self._Ow = Ow
        self._Lh = Lh;  self._Lw = Lw
        self._O  = O;   self._L  = L
        self._Ph = Ph;  self._Pw = Pw
        self._Sh = Sh;  self._Sw = Sw

        # -- phi_weights : (F, C, Kh, Kw) ------------------------------------
        if self.trainable_phi:
            self.phi_weights = self.add_weight(
                name="phi_weights", shape=(F, C, Kh, Kw),
                initializer=self.phi_initializer,
                regularizer=self.phi_regularizer, trainable=True,
            )
        else:
            self.phi_weights = self.add_weight(
                name="phi_weights", shape=(F, C, Kh, Kw),
                initializer="ones", trainable=False,
            )

        # -- lambda_in : (F, C, L)  where L = Lh*Lw --------------------------
        if self.trainable_lambda_in:
            self.lambda_in = self.add_weight(
                name="lambda_in", shape=(F, C, L),
                initializer=self.lambda_in_initializer,
                regularizer=self.lambda_in_regularizer, trainable=True,
            )
        else:
            self.lambda_in = self.add_weight(
                name="lambda_in", shape=(F, C, L),
                initializer=self.lambda_in_initializer,
                trainable=False,
            )

        # -- lambda_out : (F, C, O)  where O = Oh*Ow --------------------------
        if self.trainable_lambda_out:
            self.lambda_out = self.add_weight(
                name="lambda_out", shape=(F, C, O),
                initializer=self.lambda_out_initializer,
                regularizer=self.lambda_out_regularizer, trainable=True,
            )
        else:
            self.lambda_out = self.add_weight(
                name="lambda_out", shape=(F, C, O),
                initializer=self.lambda_out_initializer,
                trainable=False,
            )

        # -- alpha : (F, C, O) ------------------------------------------------
        if self.trainable_alpha:
            self.alpha = self.add_weight(
                name="alpha", shape=(F, C, O),
                initializer=self.alpha_initializer,
                regularizer=self.alpha_regularizer, trainable=True,
            )
        else:
            self.alpha = self.add_weight(
                name="alpha", shape=(F, C, O),
                initializer=self.alpha_initializer,
                trainable=False,
            )

        # -- bias : (F,) ------------------------------------------------------
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=(F,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer, trainable=True,
            )
        else:
            self.bias = None

        # -- Architectural constants ------------------------------------------
        #
        # Ph, Pw, Sh, Sw stored for call() — no sparse indices or pad matrix.
        #

        super().build(input_shape)

    # --------------------------------------------------------------------------
    # call
    # --------------------------------------------------------------------------

    def call(self, X, training=None):
        """Forward pass.

        Parameters
        ----------
        X : tf.Tensor, (B, H, W, C)

        Returns
        -------
        Y : tf.Tensor, (B, Oh, Ow, F)
        """
        Y = spectral_conv_2d(
            X           = X,
            phi_weights = self.phi_weights,
            lambda_in   = self.lambda_in,
            lambda_out  = self.lambda_out,
            alpha       = self.alpha,
            stride_h    = self._Sh,
            stride_w    = self._Sw,
            pad_h       = self._Ph,
            pad_w       = self._Pw,
            out_h       = self._Oh,
            out_w       = self._Ow,
            pad_lh      = self._Lh,
            pad_lw      = self._Lw,
            groups      = self.groups,
        )   # (B, Oh, Ow, F)

        if self.use_bias:
            Y = Y + self.bias

        if self.activation is not None:
            Y = self.activation(Y)

        return Y

    # --------------------------------------------------------------------------
    # compute_output_shape
    # --------------------------------------------------------------------------

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self._Oh, self._Ow, self.filters)

    # --------------------------------------------------------------------------
    # Serialization
    # --------------------------------------------------------------------------

    def get_config(self):
        base = super().get_config()
        base.update(dict(
            filters                = self.filters,
            kernel_size            = self.kernel_size,
            strides                = self.strides,
            padding                = self.padding_mode,
            use_bias               = self.use_bias,
            activation             = self._activation_identifier,
            groups                 = self.groups,
            trainable_phi          = self.trainable_phi,
            trainable_lambda_in    = self.trainable_lambda_in,
            trainable_lambda_out   = self.trainable_lambda_out,
            trainable_alpha        = self.trainable_alpha,
            phi_initializer        = initializers.serialize(self.phi_initializer),
            lambda_in_initializer  = initializers.serialize(self.lambda_in_initializer),
            lambda_out_initializer = initializers.serialize(self.lambda_out_initializer),
            alpha_initializer      = initializers.serialize(self.alpha_initializer),
            bias_initializer       = initializers.serialize(self.bias_initializer),
            phi_regularizer        = regularizers.serialize(self.phi_regularizer),
            lambda_in_regularizer  = regularizers.serialize(self.lambda_in_regularizer),
            lambda_out_regularizer = regularizers.serialize(self.lambda_out_regularizer),
            alpha_regularizer      = regularizers.serialize(self.alpha_regularizer),
            bias_regularizer       = regularizers.serialize(self.bias_regularizer),
        ))
        return base

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ------------------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    BATCH, H, W, C, F, K = 2, 8, 8, 3, 4, 3
    X = tf.random.normal((BATCH, H, W, C))

    # valid + relu
    layer = SpecCnn2D(filters=F, kernel_size=K, padding="valid", activation="relu")
    Y = layer(X)
    assert Y.shape == (BATCH, H - K + 1, W - K + 1, F)
    print(f"valid OK — output {tuple(Y.shape)}")

    # same padding
    layer_same = SpecCnn2D(filters=F, kernel_size=K, padding="same")
    Y_same = layer_same(X)
    assert Y_same.shape == (BATCH, H, W, F)
    print(f"same  OK — output {tuple(Y_same.shape)}")

    # stride=2
    layer_s2 = SpecCnn2D(filters=F, kernel_size=K, strides=2, padding="valid")
    Y_s2 = layer_s2(X)
    Oh_s2 = math.floor((H - K) / 2) + 1
    assert Y_s2.shape == (BATCH, Oh_s2, Oh_s2, F)
    print(f"stride=2 OK — output {tuple(Y_s2.shape)}")

    # all frozen
    layer_fr = SpecCnn2D(filters=F, kernel_size=K,
                         trainable_phi=False, trainable_lambda_in=False,
                         trainable_lambda_out=False, trainable_alpha=False,
                         use_bias=False)
    layer_fr(X)
    assert len(layer_fr.trainable_weights) == 0
    print("frozen OK — 0 trainable params")

    # serialization
    cfg = layer.get_config()
    assert SpecCnn2D.from_config(cfg)(X).shape == Y.shape
    print("get_config/from_config OK")