"""attention functions"""

import math
from typing import Optional

from compyute.nn.functional.activation_funcs import SoftmaxFunction
from compyute.nn.functional.functions import Function, FunctionContext, PseudoContext
from compyute.nn.functional.linear_funcs import LinearFunction
from compyute.nn.functional.regularization_funcs import DropoutFunction
from compyute.tensor_ops.shape_ops import concat, split
from compyute.tensors import ShapeError, Tensor


class MultiHeadSelfAttentionFunction(Function):
    """Applies multi head self-attention to a tensor."""

    @staticmethod
    def forward(
        ctx: FunctionContext,
        x: Tensor,
        w_i: Tensor,
        b_i: Optional[Tensor],
        w_o: Tensor,
        b_o: Optional[Tensor],
        n_heads: int,
        mask: Optional[Tensor],
        dropout: float,
        return_attn_weights: bool,
    ) -> tuple[Tensor, Optional[Tensor]]:
        if x.ndim < 3:
            raise ShapeError(f"Expected input to be 3D, got {x.ndim}D.")

        # input projection
        qkv = LinearFunction.forward(ctx, x, w_i, b_i)
        q, k, v = split(qkv, splits=3, dim=-1)

        # split into heads (B, T, H, Ch) and transpose to (B, H, T, Ch)
        head_shape = (*x.shape[:2], n_heads, -1)
        q = q.view(head_shape).transpose(1, 2).to_contiguous()
        k = k.view(head_shape).transpose(1, 2).to_contiguous()
        v = v.view(head_shape).transpose(1, 2).to_contiguous()

        # multi head attention
        attn, attn_weights = SDPAttentionFunction.forward(
            ctx, q, k, v, mask, dropout, return_attn_weights
        )

        # transpose back to (B, T, H, Ch) and merge heads to (B, T, C)
        attn = attn.transpose(1, 2).view(x.shape)

        # output projection
        y = LinearFunction.forward(ctx, attn, w_o, b_o)

        ctx.add(x.shape, head_shape)
        return y, attn_weights

    @staticmethod
    def backward(
        ctx: FunctionContext, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        x_shape, head_shape = ctx.get()

        # output gradients
        dy, dw_o, db_o = LinearFunction.backward(ctx, dy)

        # split head grads to (B, T, H, Ch), transpose to (B, H, T, Ch)
        dy = dy.view(head_shape).transpose(1, 2).to_contiguous()

        # multi head attention gradients
        dq, dk, dv = SDPAttentionFunction.backward(ctx, dy)

        # transpose back to (B, T, H, Ch) and  merge head grads to (B, T, C)
        dq = dq.transpose(1, 2).view(x_shape)
        dk = dk.transpose(1, 2).view(x_shape)
        dv = dv.transpose(1, 2).view(x_shape)

        # input projection gradients
        dx = concat([dq, dk, dv])
        dx, dw_i, db_i = LinearFunction.backward(ctx, dx)

        return dx, dw_i, db_i, dw_o, db_o


def multi_head_self_attention(
    x: Tensor,
    w_i: Tensor,
    b_i: Optional[Tensor],
    w_o: Tensor,
    b_o: Optional[Tensor],
    n_heads: int,
    mask: Optional[Tensor] = None,
    dropout: float = 0.0,
    return_attn_weights: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:
    r"""Applies multi head attention to a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor of tokens.
    w_i : Tensor
        Weight tensor for the input projection.
    b_i : Tensor, optional
        Bias tensor for the input projection.
    w_h : Tensor
        Weight tensor for the output projection.
    b_h : Tensor, optional
        Bias tensor for the output projection.
    n_heads: int
        Number of attention heads.
    mask : Tensor, optional
        Attention-mask. Defaults to ``None``.
        Must be a zeros-tensor with values of ```-inf`` indicating elements to be masked out.
    dropout : float, optional
        Dropout probability of attention weights. Defaults to ``0``.
    return_attn_weights : bool, optional
        Whether to also return the computed attention weights. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Optional[Tensor]
        Attention weights if ``return_attn_weights`` is set to ``True``.

    See Also
    ----------
    :class:`compyute.nn.MultiHeadSelfAttention`
    """
    return MultiHeadSelfAttentionFunction.forward(
        PseudoContext(),
        x,
        w_i,
        b_i,
        w_o,
        b_o,
        n_heads,
        mask,
        dropout,
        return_attn_weights,
    )


class SDPAttentionFunction(Function):
    """Computes the scaled dot product attention scores."""

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
    return SDPAttentionFunction.forward(
        PseudoContext(), q, k, v, mask, dropout, return_attn_weights
    )
