"""scaled dot product attention functions"""

import math
from typing import Optional

from compyute.nn.functional.activation_funcs import SoftmaxFunction
from compyute.nn.functional.functions import Function, FunctionContext
from compyute.nn.functional.regularization_funcs import DropoutFunction
from compyute.tensor_ops.creation_ops import full
from compyute.tensor_ops.selection_ops import tril, triu
from compyute.tensors import Tensor


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


class SDPAttentionFunction(Function):

    @staticmethod
    def forward(
        ctx: FunctionContext,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        dropout: float,
        return_attn_weights: bool,
    ) -> tuple[Tensor, Optional[Tensor]]:
        *_, context_len, head_size = q.shape

        attn_weights = q @ k.T / math.sqrt(head_size)
        if mask is not None:
            attn_weights += mask[:context_len, :context_len]
        attn_weights = SoftmaxFunction.forward(ctx, attn_weights, dim=-1)
        attn_weights = DropoutFunction.forward(ctx, attn_weights, dropout, dropout > 0)
        y = attn_weights @ v

        ctx.add(q, k, v, attn_weights)
        return y, (None if not return_attn_weights else attn_weights)

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q, k, v, attn_weights = ctx.get()
        head_size = q.shape[-1]

        # attention gradients
        dattn_weights = dy @ v.T
        dattn_weights = DropoutFunction.backward(ctx, dattn_weights)
        dattn_weights = SoftmaxFunction.backward(ctx, dattn_weights)
        dattn_weights /= math.sqrt(head_size)

        # query, key, value gradients
        dq = dattn_weights @ k
        dk = dattn_weights.T @ q
        dv = attn_weights.T @ dy

        return dq, dk, dv
