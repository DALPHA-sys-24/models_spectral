"""speccnn1d.py

Keras Layer implementation of SpecCnn1D.
1D spectral convolution with multi-channel support and lambda modulation.

Depends on spectral_conv_op.py for all tensor operations.
"""

import tensorflow as tf
from tensorflow.keras import layers, initializers, activations, regularizers

from .spectral_conv_op import (
    compute_conv_dims_1d,
    spectral_conv_1d,
)


class SpecCnn1D(layers.Layer):
    """
    Spectral 1D convolution layer with multi-channel support.

    Effective kernel per (filter f, channel c):

        W_eff[f,c,o,k] = alpha[f,c,o] * phi[f,c,k]
                         * (lambda_in[f,c,o*S+k] - lambda_out[f,c,o])

    Forward pass via a single einsum on sliding windows:

        Y[b,o,f] = sum_c sum_k  W_eff[f,c,o,k] * X_pad[b, o*S+k, c]

    Trainable weights
    -----------------
    phi_weights  : (F, C, K)  -- convolution weights
                                 if trainable_phi=False  : frozen at ones
    lambda_in    : (F, C, L)  -- input modulation  (L = L_raw + 2*P)
                                 if trainable_lambda_in=False  : frozen at ones
    lambda_out   : (F, C, O)  -- output modulation
                                 if trainable_lambda_out=False : frozen at zeros
    alpha        : (F, C, O)  -- inter-channel aggregation
                                 if trainable_alpha=False : frozen at ones
    bias         : (F,)       -- bias (if use_bias=True)

    Parameters
    ----------
    filters : int
    kernel_size : int
    strides : int, default 1
    padding : str, "valid" | "same"
        "same" requires an odd kernel_size.
    use_bias : bool, default True
    activation : str or callable, default None
    groups : int, default 1
        As in Keras Conv1D: channels and filters are split into `groups`
        groups, each filter only sees the C/groups channels of its group.
        Weights then use C/groups instead of C. groups=C, filters=C gives
        the depthwise spectral convolution.

    trainable_phi        : bool, default True
    trainable_lambda_in  : bool, default True
    trainable_lambda_out : bool, default True
    trainable_alpha      : bool, default True

    phi_initializer        : default "glorot_uniform"
    lambda_in_initializer  : default glorot_uniform if trainable, else ones
    lambda_out_initializer : default glorot_uniform if trainable, else zeros
    alpha_initializer      : default "ones"
    bias_initializer       : default "zeros"

    phi_regularizer        : default None
    lambda_in_regularizer  : default None
    lambda_out_regularizer : default None
    alpha_regularizer      : default None
    bias_regularizer       : default None

    Special cases
    -------------
    trainable_phi=True, lambda_in frozen at 1, lambda_out frozen at 0,
    trainable_alpha=False => standard Conv1D
    + groups=G            => Conv1D(groups=G)
    """

    def __init__(
        self,
        filters                : int,
        kernel_size            : int,
        strides                : int  = 1,
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
        alpha_initializer             = "ones",
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
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        if strides < 1:
            raise ValueError(f"strides must be >= 1, got {strides}")
        if padding not in ("valid", "same"):
            raise ValueError(
                f"padding must be 'valid' or 'same', got '{padding}'"
            )
        if padding == "same" and kernel_size % 2 == 0:
            raise ValueError(
                f"padding='same' requires an odd kernel_size, got {kernel_size}"
            )

        if groups < 1 or filters % groups != 0:
            raise ValueError(
                f"filters ({filters}) must be divisible by groups ({groups})"
            )

        self.filters               = filters
        self.groups                = groups
        self.kernel_size           = kernel_size
        self.strides               = strides
        self.padding_mode          = padding
        self.use_bias              = use_bias
        self.trainable_phi         = trainable_phi
        self.trainable_lambda_in   = trainable_lambda_in
        self.trainable_lambda_out  = trainable_lambda_out
        self.trainable_alpha       = trainable_alpha

        # Activation stored as identifier for get_config serialization
        self._activation_identifier = activation
        self.activation = activations.get(activation)

        # Initializer defaults are trainable-aware:
        #   lambda_in  : glorot_uniform if trained, ones  if frozen
        #   lambda_out : glorot_uniform if trained, zeros if frozen
        if lambda_in_initializer is None:
            lambda_in_initializer = (
                "glorot_uniform" if trainable_lambda_in else "ones"
            )
        if lambda_out_initializer is None:
            lambda_out_initializer = (
                "glorot_uniform" if trainable_lambda_out else "zeros"
            )

        self.phi_initializer        = initializers.get(phi_initializer)
        self.lambda_in_initializer  = initializers.get(lambda_in_initializer)
        self.lambda_out_initializer = initializers.get(lambda_out_initializer)
        self.alpha_initializer      = initializers.get(alpha_initializer)
        self.bias_initializer       = initializers.get(bias_initializer)

        self.phi_regularizer        = regularizers.get(phi_regularizer)
        self.lambda_in_regularizer  = regularizers.get(lambda_in_regularizer)
        self.lambda_out_regularizer = regularizers.get(lambda_out_regularizer)
        self.alpha_regularizer      = regularizers.get(alpha_regularizer)
        self.bias_regularizer       = regularizers.get(bias_regularizer)

    # --------------------------------------------------------------------------
    # build
    # --------------------------------------------------------------------------

    def build(self, input_shape):
        """Creates trainable weights from input_shape (batch, L_raw, C)."""
        L_raw = int(input_shape[1])
        C_in  = int(input_shape[2])
        if C_in % self.groups != 0:
            raise ValueError(
                f"input channels ({C_in}) must be divisible by groups ({self.groups})"
            )
        C     = C_in // self.groups   # channels seen by each filter
        F     = self.filters
        K     = self.kernel_size
        S     = self.strides

        # -- Padding and dimension computation --------------------------------
        P       = (K - 1) // 2 if self.padding_mode == "same" else 0
        O, L, _ = compute_conv_dims_1d(L_raw, K, S, P)

        # Stored for call() and compute_output_shape()
        self._C     = C
        self._L_raw = L_raw
        self._L     = L    # padded length
        self._O     = O    # output length
        self._P     = P
        self._S     = S

        # -- phi_weights : (F, C, K) ------------------------------------------
        if self.trainable_phi:
            self.phi_weights = self.add_weight(
                name        = "phi_weights",
                shape       = (F, C, K),
                initializer = self.phi_initializer,
                regularizer = self.phi_regularizer,
                trainable   = True,
            )
        else:
            self.phi_weights = self.add_weight(
                name        = "phi_weights",
                shape       = (F, C, K),
                initializer = "ones",
                trainable   = False,
            )

        # -- lambda_in : (F, C, L) --------------------------------------------
        if self.trainable_lambda_in:
            self.lambda_in = self.add_weight(
                name        = "lambda_in",
                shape       = (F, C, L),
                initializer = self.lambda_in_initializer,
                regularizer = self.lambda_in_regularizer,
                trainable   = True,
            )
        else:
            self.lambda_in = self.add_weight(
                name        = "lambda_in",
                shape       = (F, C, L),
                initializer = self.lambda_in_initializer,
                trainable   = False,
            )

        # -- lambda_out : (F, C, O) -------------------------------------------
        if self.trainable_lambda_out:
            self.lambda_out = self.add_weight(
                name        = "lambda_out",
                shape       = (F, C, O),
                initializer = self.lambda_out_initializer,
                regularizer = self.lambda_out_regularizer,
                trainable   = True,
            )
        else:
            self.lambda_out = self.add_weight(
                name        = "lambda_out",
                shape       = (F, C, O),
                initializer = self.lambda_out_initializer,
                trainable   = False,
            )

        # -- alpha : (F, C, O) ------------------------------------------------
        if self.trainable_alpha:
            self.alpha = self.add_weight(
                name        = "alpha",
                shape       = (F, C, O),
                initializer = self.alpha_initializer,
                regularizer = self.alpha_regularizer,
                trainable   = True,
            )
        else:
            self.alpha = self.add_weight(
                name        = "alpha",
                shape       = (F, C, O),
                initializer = self.alpha_initializer,
                trainable   = False,
            )

        # -- bias : (F,) ------------------------------------------------------
        if self.use_bias:
            self.bias = self.add_weight(
                name        = "bias",
                shape       = (F,),
                initializer = self.bias_initializer,
                regularizer = self.bias_regularizer,
                trainable   = True,
            )
        else:
            self.bias = None

        super().build(input_shape)

    # --------------------------------------------------------------------------
    # call
    # --------------------------------------------------------------------------

    def call(self, X, training=None):
        """Forward pass.

        Parameters
        ----------
        X        : tf.Tensor, (batch, L_raw, C)
        training : bool or None

        Returns
        -------
        Y : tf.Tensor, (batch, O, F)
        """
        Y = spectral_conv_1d(
            X           = X,
            phi_weights = self.phi_weights,
            lambda_in   = self.lambda_in,
            lambda_out  = self.lambda_out,
            alpha       = self.alpha,
            stride      = self._S,
            padding     = self._P,
            groups      = self.groups,
        )   # (batch, O, F)

        if self.use_bias:
            Y = Y + self.bias

        if self.activation is not None:
            Y = self.activation(Y)

        return Y

    # --------------------------------------------------------------------------
    # compute_output_shape
    # --------------------------------------------------------------------------

    def compute_output_shape(self, input_shape):
        """Returns (batch, O, filters) given input shape (batch, L_raw, C)."""
        return (input_shape[0], self._O, self.filters)

    # --------------------------------------------------------------------------
    # Serialization
    # --------------------------------------------------------------------------

    def get_config(self):
        """Returns the layer config for serialization and cloning."""
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
        """Instantiates a SpecCnn1D layer from its config dict."""
        return cls(**config)


# ------------------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    BATCH, L_RAW, C, F, K = 2, 32, 3, 8, 3
    X = tf.random.normal((BATCH, L_RAW, C))

    # valid + relu
    layer = SpecCnn1D(filters=F, kernel_size=K, padding="valid", activation="relu")
    Y = layer(X)
    assert Y.shape == (BATCH, L_RAW - K + 1, F)
    print(f"valid OK — output {tuple(Y.shape)}")

    # same padding
    layer_same = SpecCnn1D(filters=F, kernel_size=K, padding="same")
    Y_same = layer_same(X)
    assert Y_same.shape == (BATCH, L_RAW, F)
    print(f"same  OK — output {tuple(Y_same.shape)}")

    # all frozen
    layer_fr = SpecCnn1D(filters=F, kernel_size=K,
                         trainable_phi=False, trainable_lambda_in=False,
                         trainable_lambda_out=False, trainable_alpha=False,
                         use_bias=False)
    layer_fr(X)
    assert len(layer_fr.trainable_weights) == 0
    print("frozen OK — 0 trainable params")

    # serialization
    cfg = layer.get_config()
    assert SpecCnn1D.from_config(cfg)(X).shape == Y.shape
    print("get_config/from_config OK")