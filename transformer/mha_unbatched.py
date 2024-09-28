"""transformer neural network module"""

from typing import Optional

from compyute.nn.modules.linear import Linear
from compyute.nn.modules.module import Module, ModuleList
from compyute.nn.parameter import Buffer
from compyute.tensor_ops.reduction_ops import tensorsum
from compyute.tensor_ops.shape_ops import concat, split
from compyute.tensors import Tensor
from compyute.typing import DType

from .attention_funcs import SDPAttentionFn


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
        head_size = in_channels // n_heads
        self.heads = ModuleList(
            AttentionHead(in_channels, head_size, mask, dropout, bias, dtype)
            for _ in range(n_heads)
        )
        self.out_proj = Linear(in_channels, in_channels, bias, dtype, "OutProj")
        self.out_proj.w.data *= out_scale

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        # multi head attention
        attn = concat([h(x) for h in self.heads])

        # output projection
        y = self.out_proj(attn)

        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        # output projection gradients
        dattn = self.out_proj.backward(dy)

        # multi head attention gradients
        dattn_heads = split(dattn, self.n_heads)
        dx = tensorsum(
            h.backward(dattn_head) for h, dattn_head in zip(self.heads, dattn_heads)
        )

        return dx


class AttentionHead(Module):

    def __init__(
        self,
        in_channels: int,
        head_size: int,
        mask: Optional[Tensor] = None,
        dropout: float = 0.0,
        bias: bool = True,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.mask = None if not mask else Buffer(mask)
        self.dropout = dropout
        self.attn_w: Optional[Tensor] = None

        self.q_proj = Linear(in_channels, head_size, bias, dtype, "QueryProj")
        self.k_proj = Linear(in_channels, head_size, bias, dtype, "KeyProj")
        self.v_proj = Linear(in_channels, head_size, bias, dtype, "ValueProj")

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        dropout = self.dropout if self._is_training else 0

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        y, self.attn_w = SDPAttentionFn.forward(
            self.fcache,
            q,
            k,
            v,
            self.mask,
            dropout,
            self._retain_values,
        )
        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        # attention gradients
        dq, dk, dv = SDPAttentionFn.backward(self.fcache, dy)

        # input projection gradients
        dx = self.q_proj.backward(dq)
        dx += self.k_proj.backward(dk)
        dx += self.v_proj.backward(dv)

        return dx
