"""attention functions"""

import math
from typing import Optional

from compyute.nn.functional.activation_funcs import SoftmaxFn
from compyute.nn.functional.functions import Function, FunctionCache, PseudoCache
from compyute.nn.functional.regularization_funcs import DropoutFn
from compyute.tensor_ops.creation_ops import full
from compyute.tensor_ops.selection_ops import tril, triu
from compyute.tensors import ShapeError, ShapeLike, Tensor


def get_causal_mask(shape: ShapeLike) -> Tensor:
    """Returns a causal mask used for the self-attention mechanism.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the mask.

    Returns
    -------
    Tensor
        Causal mask.
    """
    return triu(full(shape, float("-inf")), diag_index=1)


def get_sliding_window_mask(shape: ShapeLike, window_size: int) -> Tensor:
    """Returns a sliding window mask used for the self-attention mechanism.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the mask.
    window_size : int
        Size of the sliding window.

    Returns
    -------
    Tensor
        Sliding window mask.
    """
    upper_mask = triu(full(shape, float("-inf")), diag_index=1)
    lower_mask = tril(full(shape, float("-inf")), diag_index=-window_size)
    return upper_mask + lower_mask


class SDPAttentionFn(Function):
    """Computes the scaled dot product attention scores."""

    @staticmethod
    def forward(
        cache: FunctionCache,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        dropout: float,
        return_attn_w: bool,
    ) -> tuple[Tensor, Optional[Tensor]]:
        if q.ndim < 2:
            raise ShapeError(f"Expected query to be at least 2D, got {q.ndim}D.")
        if k.ndim < 2:
            raise ShapeError(f"Expected key to be at least 2D, got {k.ndim}D.")
        if v.ndim < 2:
            raise ShapeError(f"Expected value to be at least 2D, got {v.ndim}D.")
        *_, seq_len, head_size = q.shape

        attn_w = q @ k.T / math.sqrt(head_size)
        if mask is not None:
            attn_w += mask[:seq_len, :seq_len]
        attn_w = SoftmaxFn.forward(cache, attn_w, dim=-1)
        attn_w = DropoutFn.forward(cache, attn_w, dropout, dropout > 0)
        y = attn_w @ v

        cache.push(q, k, v, attn_w)
        return y, (None if not return_attn_w else attn_w)

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q, k, v, attn_w = cache.pop()
        head_size = q.shape[-1]

        # attention gradients
        dattn_w = dy @ v.T
        dattn_w = DropoutFn.backward(cache, dattn_w)
        dattn_w = SoftmaxFn.backward(cache, dattn_w) / math.sqrt(head_size)

        # query, key, value gradients
        dq = dattn_w @ k
        dk = dattn_w.T @ q
        dv = attn_w.T @ dy

        return dq, dk, dv


def sdp_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    dropout: float = 0.0,
    return_attn_w: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:
    r"""Computes the scaled dot product attention scores.

    Parameters
    ----------
    q : Tensor
        Query tensor.
    k : Tensor
        Key tensor.
    v : Tensor
        Value tensor.
    mask : Tensor, optional
        Attention-mask. Defaults to ``None``.
        Must be a zeros-tensor with values of ```-inf`` indicating elements to be masked out.
    dropout : float, optional
        Dropout probability of attention weights. Defaults to ``0``.
    return_attn_w : bool, optional
        Whether to also return the computed attention weights. Defaults to ``False``.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], tuple[Tensor, Tensor, Optional[Tensor]]], optional
        Gradient function.

    See Also
    ----------
    :class:`compyute.nn.MultiHeadAttention`
    """
    return SDPAttentionFn.forward(PseudoCache(), q, k, v, mask, dropout, return_attn_w)
