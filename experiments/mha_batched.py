"""attention neural network module"""

from typing import Optional

from attention_funcs import SDPAttentionFn
from compyute.nn.modules.linear import Linear
from compyute.nn.modules.module import Module
from compyute.nn.parameter import Buffer
from compyute.tensor_ops.creation_ops import concat, split
from compyute.tensors import Tensor
from compyute.typing import DType


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
    dropout : float, optional
        Dropout probability. Defaults to ``0``.
    out_scale : float, optional
        Scale for the output projection. Defaults to ``1.0``.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
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
        dropout: float = 0.0,
        out_scale: float = 1.0,
        bias: bool = True,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        if in_channels % n_heads != 0:
            raise ValueError("Number of input channels must be divisible by n_heads.")
        super().__init__(label)

        self.n_heads = n_heads
        self.mask = None if not mask else Buffer(mask)
        self.dropout = dropout
        self.attn_w: Optional[Tensor] = None

        self.q_proj = Linear(in_channels, in_channels, bias, dtype, "QueryProj")
        self.k_proj = Linear(in_channels, in_channels, bias, dtype, "KeyProj")
        self.v_proj = Linear(in_channels, in_channels, bias, dtype, "ValueProj")

        self.out_proj = Linear(in_channels, in_channels, bias, dtype, "OutProj")
        self.out_proj.w.data *= out_scale

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        dropout = self.dropout if self._is_training else 0

        # input projection
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # split heads to (B, T, H, Ch), transpose to (B, H, T, Ch)
        head_shape = (B, T, self.n_heads, C // self.n_heads)
        q = q.view(head_shape).transpose((0, 2, 1, 3))
        k = k.view(head_shape).transpose((0, 2, 1, 3))
        v = v.view(head_shape).transpose((0, 2, 1, 3))

        # multi head attention
        attn, self.attn_w = SDPAttentionFn.forward(
            self.fcache, q, k, v, self.mask, dropout, self._retain_values
        )

        # transpose back to (B, T, H, Ch) and merge heads to (B, T, C)
        attn = attn.transpose((0, 2, 1, 3)).view(x.shape)

        # output projection
        y = self.out_proj(attn)

        self.fcache.push(x.shape, head_shape)
        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        x_shape, head_shape = self.fcache.pop()

        # output gradients
        dy = self.out_proj.backward(dy)

        # split head grads to (B, T, H, Ch), transpose to (B, H, T, Ch)
        dy = dy.view(head_shape).transpose((0, 2, 1, 3))

        # multi head attention gradients
        dq, dk, dv = SDPAttentionFn.backward(self.fcache, dy)

        # transpose back to (B, T, H, Ch)and  merge head grads to (B, T, C)
        dq = dq.transpose((0, 2, 1, 3)).view(x_shape)
        dk = dk.transpose((0, 2, 1, 3)).view(x_shape)
        dv = dv.transpose((0, 2, 1, 3)).view(x_shape)

        # input projection gradients
        dx = self.q_proj.backward(dq)
        dx += self.k_proj.backward(dk)
        dx += self.v_proj.backward(dv)

        return dx
