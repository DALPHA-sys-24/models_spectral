"""spectral_conv_op.py

Window-based spectral convolution operations -- pure TensorFlow, no Keras dependency.

Key design
----------
Instead of materializing a dense (F, C, O, L) kernel tensor at every forward
pass, sliding windows are extracted once and the full computation is reduced
to a single einsum on compact tensors.

Mathematical identity
---------------------
The spectral kernel K[f,c,o,l] is non-zero only at the K positions inside the
receptive field of output o:  l ∈ [o·S, o·S + K).  Denoting those positions
as a window index k ∈ [0, K):

    Y[b,o,f] = Σ_c Σ_k  α[f,c,o] · φ[f,c,k]
                         · (λ_in[f,c,o·S+k] − λ_out[f,c,o])
                         · X_pad[b, o·S+k, c]

             = einsum('fcok, bokc → bof',  W,  X_win)

where W[f,c,o,k] = α[f,c,o] · φ[f,c,k] · (λ_in_win[f,c,o,k] − λ_out[f,c,o])

This avoids the (F,C,O,L) dense matrix entirely.  Only (F,C,O,K) and (B,O,K,C)
tensors are needed, with K << L always.

Grouped convolution (groups = G)
--------------------------------
As in Keras Conv1D/Conv2D(groups=G), the C input channels and the F filters are
split into G contiguous groups (Cg = C/G, Fg = F/G); filter f only sees the
channels of its own group g = f // Fg.  Weights then have Cg channels instead
of C:  phi (F, Cg, K), lambda_in (F, Cg, L), lambda_out/alpha (F, Cg, O).

    Y[b,o,g·Fg+d] = Σ_c Σ_k  W[g·Fg+d, c, o, k] · X_win[b, o, k, g·Cg+c]

groups=1 is the standard (fully connected over channels) spectral convolution;
groups=C with F=C is the depthwise spectral convolution (MobileNet).

Public API -- 1D
----------------
    compute_conv_dims_1d : O = floor((L+2P-K)/S)+1
    spectral_conv_1d     : full forward pass (B,L,C) → (B,O,F)

Public API -- 2D
----------------
    compute_conv_dims_2d : (Oh, Ow, Lh, Lw) from (H,W,Kh,Kw,Sh,Sw,Ph,Pw)
    spectral_conv_2d     : full forward pass (B,H,W,C) → (B,Oh,Ow,F)
"""

import math
import tensorflow as tf
from typing import Tuple


# ------------------------------------------------------------------------------
# Dimension helpers  (unchanged)
# ------------------------------------------------------------------------------

def compute_conv_dims_1d(
    input_length : int,
    kernel_size  : int,
    stride       : int,
    padding      : int,
) -> Tuple[int, int, int]:
    """O = floor((L + 2P - K) / S) + 1.

    Returns
    -------
    (output_length, padded_input_length, padding)
    """
    padded = input_length + 2 * padding
    output = math.floor((padded - kernel_size) / stride) + 1
    return output, padded, padding


def compute_conv_dims_2d(
    input_height : int,
    input_width  : int,
    kernel_h     : int,
    kernel_w     : int,
    stride_h     : int,
    stride_w     : int,
    pad_h        : int,
    pad_w        : int,
) -> Tuple[int, int, int, int]:
    """Applies the standard formula independently on each spatial axis.

    Returns
    -------
    (Oh, Ow, Lh, Lw)
    """
    Lh = input_height + 2 * pad_h
    Lw = input_width  + 2 * pad_w
    Oh = math.floor((Lh - kernel_h) / stride_h) + 1
    Ow = math.floor((Lw - kernel_w) / stride_w) + 1
    return Oh, Ow, Lh, Lw


# ------------------------------------------------------------------------------
# Grouped contraction (shared by 1D and 2D)
# ------------------------------------------------------------------------------

def _grouped_einsum(
    eff_weights : tf.Tensor,
    X_win       : tf.Tensor,
    F           : int,
    Cg          : int,
    groups      : int,
) -> tf.Tensor:
    """Y[b,o,f] = Σ_c Σ_k W[f,c,o,k] · X_win[b,o,k,(f//Fg)·Cg + c].

    eff_weights : (F, Cg, O, K)
    X_win       : (B, O, K, C)    with C = groups · Cg
    returns     : (B, O, F)
    """
    if groups == 1:
        return tf.einsum('fcok,bokc->bof', eff_weights, X_win)

    G, Fg = groups, F // groups
    B, O, K = tf.shape(X_win)[0], tf.shape(X_win)[1], tf.shape(X_win)[2]
    X_grp = tf.reshape(X_win, (B, O, K, G, Cg))                         # (B, O, K, G, Cg)
    W_grp = tf.reshape(eff_weights, (G, Fg, Cg, O, K))                  # (G, Fg, Cg, O, K)
    Y_grp = tf.einsum('gdcok,bokgc->bogd', W_grp, X_grp)                # (B, O, G, Fg)
    return tf.reshape(Y_grp, (B, O, F))


# ------------------------------------------------------------------------------
# 1D spectral convolution
# ------------------------------------------------------------------------------

def spectral_conv_1d(
    X           : tf.Tensor,
    phi_weights : tf.Tensor,
    lambda_in   : tf.Tensor,
    lambda_out  : tf.Tensor,
    alpha       : tf.Tensor,
    stride      : int,
    padding     : int,
    groups      : int = 1,
) -> tf.Tensor:
    """Full 1D spectral forward pass in a single einsum.

    Parameters
    ----------
    X           : (B, L_raw, C)   -- raw (unpadded) input
    phi_weights : (F, Cg, K)      -- convolution weights  (Cg = C / groups)
    lambda_in   : (F, Cg, L)      -- input modulation  (L = L_raw + 2*P)
    lambda_out  : (F, Cg, O)      -- output modulation
    alpha       : (F, Cg, O)      -- inter-channel aggregation weights
    stride      : int             -- S
    padding     : int             -- P  (0 = valid,  (K-1)//2 = same)
    groups      : int             -- G  (1 = standard, C = depthwise)

    Returns
    -------
    Y : (B, O, F)

    Notes
    -----
    Complexity  :  O(B · O · F · C · K)  vs.  O(B · O · F · C · L)  previously.
    Memory saved:  (F, C, O, L) dense tensor completely avoided.
    """
    F  = int(phi_weights.shape[0])
    Cg = int(phi_weights.shape[1])
    K  = int(phi_weights.shape[2])
    G  = groups

    # -- 1. Pad input ---------------------------------------------------------
    if padding > 0:
        X_pad = tf.pad(X, [[0, 0], [padding, padding], [0, 0]])
    else:
        X_pad = X

    # -- 2. Extract sliding windows -------------------------------------------
    #
    #   X_win[b, o, k, c]      = X_pad[b, o·S + k, c]   via tf.signal.frame
    #   L_in_win[f, c, o, k]   = lambda_in[f, c, o·S+k] via tf.signal.frame
    #
    X_win   = tf.signal.frame(X_pad,    K, stride, axis=1)   # (B, O, K, C)
    L_in_win = tf.signal.frame(lambda_in, K, stride, axis=2)  # (F, Cg, O, K)

    # -- 3. Effective weight --------------------------------------------------
    #
    #   W[f,c,o,k] = α[f,c,o] · φ[f,c,k] · (λ_in_win[f,c,o,k] − λ_out[f,c,o])
    #
    modulation  = L_in_win - lambda_out[:, :, :, tf.newaxis]             # (F,C,O,K)
    eff_weights = (
        alpha[:, :, :, tf.newaxis]
        * phi_weights[:, :, tf.newaxis, :]
        * modulation
    )   # (F, Cg, O, K)

    # -- 4. Single einsum (per group) -----------------------------------------
    return _grouped_einsum(eff_weights, X_win, F, Cg, G)     # (B, O, F)


# ------------------------------------------------------------------------------
# 2D spectral convolution
# ------------------------------------------------------------------------------

def spectral_conv_2d(
    X           : tf.Tensor,
    phi_weights : tf.Tensor,
    lambda_in   : tf.Tensor,
    lambda_out  : tf.Tensor,
    alpha       : tf.Tensor,
    stride_h    : int,
    stride_w    : int,
    pad_h       : int,
    pad_w       : int,
    out_h       : int,
    out_w       : int,
    pad_lh      : int,
    pad_lw      : int,
    groups      : int = 1,
) -> tf.Tensor:
    """Full 2D spectral forward pass in a single einsum (flattened formulation).

    Parameters
    ----------
    X           : (B, H, W, C)
    phi_weights : (F, Cg, Kh, Kw)   (Cg = C / groups)
    lambda_in   : (F, Cg, Lh*Lw)
    lambda_out  : (F, Cg, Oh*Ow)
    alpha       : (F, Cg, Oh*Ow)
    stride_h, stride_w : int
    pad_h, pad_w       : int
    out_h, out_w       : int    -- Oh, Ow
    pad_lh, pad_lw     : int    -- Lh, Lw (padded input spatial dims)
    groups             : int    -- G  (1 = standard, C = depthwise)

    Returns
    -------
    Y : (B, Oh, Ow, F)
    """
    F  = int(phi_weights.shape[0])
    Cg = int(phi_weights.shape[1])
    C  = Cg * groups
    Kh = int(phi_weights.shape[2])
    Kw = int(phi_weights.shape[3])
    Oh, Ow   = out_h, out_w
    Lh, Lw   = pad_lh, pad_lw
    O = Oh * Ow
    K = Kh * Kw
    B = tf.shape(X)[0]

    # -- 1. Pad input ---------------------------------------------------------
    if pad_h > 0 or pad_w > 0:
        X_pad = tf.pad(X, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
    else:
        X_pad = X

    # -- 2. Extract input patches  (B, Oh, Ow, Kh*Kw*C) → (B, O, K, C) ------
    patches = tf.image.extract_patches(
        X_pad,
        sizes   = [1, Kh, Kw, 1],
        strides = [1, stride_h, stride_w, 1],
        rates   = [1, 1, 1, 1],
        padding = 'VALID',
    )   # (B, Oh, Ow, K*C)
    X_win = tf.reshape(patches, (B, O, K, C))                           # (B, O, K, C)

    # -- 3. Extract lambda_in patches  (F, Cg, Lh, Lw) → (F, Cg, O, K) ------
    #
    #   Treat the F*Cg maps as a pseudo-batch for extract_patches.
    #
    lin_4d   = tf.reshape(lambda_in, (F * Cg, Lh, Lw, 1))              # (F*Cg, Lh, Lw, 1)
    lin_ptch = tf.image.extract_patches(
        lin_4d,
        sizes   = [1, Kh, Kw, 1],
        strides = [1, stride_h, stride_w, 1],
        rates   = [1, 1, 1, 1],
        padding = 'VALID',
    )   # (F*Cg, Oh, Ow, K)
    L_in_win = tf.reshape(lin_ptch, (F, Cg, O, K))                      # (F, Cg, O, K)

    # -- 4. Flatten phi weights -----------------------------------------------
    phi_flat = tf.reshape(phi_weights, (F, Cg, K))                      # (F, Cg, K)

    # -- 5. Effective weight --------------------------------------------------
    modulation  = L_in_win - lambda_out[:, :, :, tf.newaxis]            # (F, Cg, O, K)
    eff_weights = (
        alpha[:, :, :, tf.newaxis]
        * phi_flat[:, :, tf.newaxis, :]
        * modulation
    )   # (F, Cg, O, K)

    # -- 6. Single einsum (per group) + reshape to spatial output -------------
    Y_flat = _grouped_einsum(eff_weights, X_win, F, Cg, groups)         # (B, O, F)
    return tf.reshape(Y_flat, (B, Oh, Ow, F))                           # (B, Oh, Ow, F)


# ------------------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    # ── 1D ────────────────────────────────────────────────────────────────────
    B, L, C, F, K, S = 2, 16, 3, 4, 3, 1
    P = (K - 1) // 2  # same padding

    O, Lp, _ = compute_conv_dims_1d(L, K, S, P)
    phi_w  = tf.random.normal((F, C, K))
    lin    = tf.ones((F, C, Lp))
    lout   = tf.zeros((F, C, O))
    alpha  = tf.ones((F, C, O))
    X1     = tf.random.normal((B, L, C))

    Y1 = spectral_conv_1d(X1, phi_w, lin, lout, alpha, stride=S, padding=P)
    assert Y1.shape == (B, O, F), f"1D shape mismatch: {Y1.shape}"
    print(f"1D OK  — input {tuple(X1.shape)}  output {tuple(Y1.shape)}")

    # ── 2D ────────────────────────────────────────────────────────────────────
    H, W = 8, 8
    Kh, Kw, Sh, Sw = 3, 3, 1, 1
    Ph, Pw = (Kh - 1) // 2, (Kw - 1) // 2

    Oh, Ow, Lh, Lw = compute_conv_dims_2d(H, W, Kh, Kw, Sh, Sw, Ph, Pw)
    O2 = Oh * Ow

    phi_2d  = tf.random.normal((F, C, Kh, Kw))
    lin_2d  = tf.ones((F, C, Lh * Lw))
    lout_2d = tf.zeros((F, C, O2))
    alp_2d  = tf.ones((F, C, O2))
    X2      = tf.random.normal((B, H, W, C))

    Y2 = spectral_conv_2d(
        X2, phi_2d, lin_2d, lout_2d, alp_2d,
        Sh, Sw, Ph, Pw, Oh, Ow, Lh, Lw,
    )
    assert Y2.shape == (B, Oh, Ow, F), f"2D shape mismatch: {Y2.shape}"
    print(f"2D OK  — input {tuple(X2.shape)}  output {tuple(Y2.shape)}")