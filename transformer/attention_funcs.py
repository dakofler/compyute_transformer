"""attention functions"""

import math
from typing import Optional

from compyute.nn.functional.activation_funcs import SoftmaxFn
from compyute.nn.functional.functions import Function, FunctionCache, PseudoCache
from compyute.nn.functional.regularization_funcs import DropoutFn
from compyute.tensor_ops.creation_ops import full
from compyute.tensor_ops.selection_ops import tril, triu
from compyute.tensors import ShapeError, Tensor


def get_causal_mask(max_context_len: int) -> Tensor:
    """Returns a causal mask used for the self-attention mechanism.

    Parameters
    ----------
    max_context_len : int
        Maximum sequence length.

    Returns
    -------
    Tensor
        Causal mask.
    """
    shape = (max_context_len, max_context_len)
    return triu(full(shape, float("-inf")), diag_index=1)


def get_sliding_window_mask(max_context_len: int, window_size: int) -> Tensor:
    """Returns a sliding window mask used for the self-attention mechanism.

    Parameters
    ----------
    max_context_len : int
        Maximum sequence length.
    window_size : int
        Size of the sliding window.

    Returns
    -------
    Tensor
        Sliding window mask.
    """
    shape = (max_context_len, max_context_len)
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
        return_attn_weights: bool,
    ) -> tuple[Tensor, Optional[Tensor]]:
        if q.ndim < 2:
            raise ShapeError(f"Expected query to be at least 2D, got {q.ndim}D.")
        if k.ndim < 2:
            raise ShapeError(f"Expected key to be at least 2D, got {k.ndim}D.")
        if v.ndim < 2:
            raise ShapeError(f"Expected value to be at least 2D, got {v.ndim}D.")
        *_, context_len, head_size = q.shape

        attn_weights = q @ k.T / math.sqrt(head_size)
        if mask is not None:
            attn_weights += mask[:context_len, :context_len]
        attn_weights = SoftmaxFn.forward(cache, attn_weights, dim=-1)
        attn_weights = DropoutFn.forward(cache, attn_weights, dropout, dropout > 0)
        y = attn_weights @ v

        cache.push(q, k, v, attn_weights)
        return y, (None if not return_attn_weights else attn_weights)

    @staticmethod
    def backward(cache: FunctionCache, dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q, k, v, attn_weights = cache.pop()
        head_size = q.shape[-1]

        # attention gradients
        dattn_weights = dy @ v.T
        dattn_weights = DropoutFn.backward(cache, dattn_weights)
        dattn_weights = SoftmaxFn.backward(cache, dattn_weights) / math.sqrt(head_size)

        # query, key, value gradients
        dq = dattn_weights @ k
        dk = dattn_weights.T @ q
        dv = attn_weights.T @ dy

        return dq, dk, dv


def sdp_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    dropout: float = 0.0,
    return_attn_weights: bool = False,
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
    return_attn_weights : bool, optional
        Whether to also return the computed attention weights. Defaults to ``False``.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Optional[Tensor]
        Attention weights if ``return_attn_weights`` is set to ``True``.

    See Also
    ----------
    :class:`compyute.nn.MultiHeadAttention`
    """
    return SDPAttentionFn.forward(
        PseudoCache(), q, k, v, mask, dropout, return_attn_weights
    )
