"""transformer neural network module"""

import math
from typing import Optional

from compyute.nn.functional.activation_funcs import SoftmaxFn
from compyute.nn.functional.functions import Function, FunctionCache, PseudoCache
from compyute.nn.functional.regularization_funcs import DropoutFn
from compyute.nn.modules.linear import Linear
from compyute.nn.modules.module import Module, ModuleList, validate_input_axes
from compyute.nn.parameter import Buffer
from compyute.tensor_ops.creation_ops import concat, full, split
from compyute.tensor_ops.reduction_ops import tensorsum
from compyute.tensor_ops.selection_ops import triu
from compyute.tensors import ShapeLike, Tensor
from compyute.typing import DType


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


class MultiHeadAttention(Module):
    r"""Multi Head Self-Attention as described by
    `Vaswani et al., 2017 <https://arxiv.org/pdf/1706.03762>`_.

    .. math::
        \begin{array}{ll} \\
            Q = xW_Q^T \\
            K = xW_K^T \\
            V = xW_V^T \\
            \text{MultiHeadAttention}(x) = \text{concat}(\text{Attention}(Q_1, K_1, V_1), ..., \text{Attention}(Q_n, K_n, V_n))W_o^T \\
            \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{N}}) \cdot V \\
        \end{array}

    where :math:`N` is the number of attention heads.

    Shapes:
        - Input :math:`(B, S, C_{in})`
        - Output :math:`(B, S, C_{in})`
    where
        - :math:`B` ... batch dimension
        - :math:`S` ... sequence
        - :math:`C_{in}` ... input channels

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    n_heads : int
        Number of attention heads.
    mask : Tensor, optional
        Attention-mask. Defaults to ``None``.
        Must be a zeros-tensor with values of ```-inf`` indicating elements to be masked out.
    dropout_p : float, optional
        Dropout probability. Defaults to ``0``.
    out_scale : float, optional
        Scale for the output projection. Defaults to ``1.0``.
    dtype: DtypeLike, optional
        Datatype of weights and biases. Defaults to ``None``.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.


    .. note::
        All weights are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in}}}`. Biases are initialized as zeros.

    .. note::
        Input projections do not use bias.
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0,
        out_scale: float = 1.0,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        if in_channels % n_heads != 0:
            raise ValueError("Number of input channels must be divisible by n_heads.")
        super().__init__(label)
        self.n_heads = n_heads
        head_size = in_channels // n_heads
        self.heads = ModuleList(
            AttentionHead(in_channels, head_size, mask, dropout_p, dtype)
            for _ in range(n_heads)
        )
        self.out_proj = Linear(in_channels, in_channels, dtype=dtype, label="OutProj")
        self.out_proj.w.data *= out_scale

    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [3])
        y = concat([h(x) for h in self.heads])
        return self.out_proj(y)

    def backward(self, dy: Tensor) -> Tensor:
        dy = self.out_proj.backward(dy)
        dys = split(dy, self.n_heads)
        return tensorsum(h.backward(dy) for h, dy in zip(self.heads, dys))


class AttentionHead(Module):

    def __init__(
        self,
        in_channels: int,
        head_size: int,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.mask = None if not mask else Buffer(mask)
        self.dropout_p = dropout_p
        self.attn_w: Optional[Tensor] = None

        self.query_proj = Linear(in_channels, head_size, False, dtype, "QueryProj")
        self.key_proj = Linear(in_channels, head_size, False, dtype, "KeyProj")
        self.value_proj = Linear(in_channels, head_size, False, dtype, "ValueProj")

    def forward(self, x: Tensor) -> Tensor:
        dropout_p = self.dropout_p if self._is_training else 0

        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        y, self.attn_w = SDPAttentionFn.forward(
            self.fcache,
            q,
            k,
            v,
            self.mask,
            dropout_p,
            self._retain_values,
        )
        return y

    def backward(self, dy: Tensor) -> Tensor:
        dq, dk, dv = SDPAttentionFn.backward(self.fcache, dy)
        dx1 = self.query_proj.backward(dq)
        dx2 = self.key_proj.backward(dk)
        dx3 = self.value_proj.backward(dv)
        return dx1 + dx2 + dx3


class SDPAttentionFn(Function):
    """Computes the scaled dot product attention scores."""

    @staticmethod
    def forward(
        cache: FunctionCache,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        dropout_p: float,
        return_attn_w: bool,
    ) -> tuple[Tensor, Optional[Tensor]]:
        _, seq_len, head_size = q.shape

        attn_w = q @ k.T / math.sqrt(head_size)
        if mask is not None:
            attn_w += mask[:seq_len, :seq_len]
        attn_w = SoftmaxFn.forward(cache, attn_w)
        attn_w = DropoutFn.forward(cache, attn_w, dropout_p, dropout_p > 0)
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
    dropout_p: float = 0,
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
    dropout_p : float, optional
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
    return SDPAttentionFn.forward(
        PseudoCache(), q, k, v, mask, dropout_p, return_attn_w
    )
