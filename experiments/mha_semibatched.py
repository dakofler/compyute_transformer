"""transformer neural network module"""

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
        self.attn_w: list[Optional[Tensor]] = []

        self.q_proj = Linear(in_channels, in_channels, bias, dtype, "QueryProj")
        self.k_proj = Linear(in_channels, in_channels, bias, dtype, "KeyProj")
        self.v_proj = Linear(in_channels, in_channels, bias, dtype, "ValueProj")

        self.out_proj = Linear(in_channels, in_channels, bias, dtype, "OutProj")
        self.out_proj.w.data *= out_scale

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        dropout = self.dropout if self._is_training else 0
        attn_heads = []

        # input projection for self-attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # split projections for each head
        q_heads = split(q, self.n_heads)
        k_heads = split(k, self.n_heads)
        v_heads = split(v, self.n_heads)

        # multi head attention: compute attention weights for each head, concat results
        for q_head, k_head, v_head in zip(q_heads, k_heads, v_heads):
            attn_head, attn_w_head = SDPAttentionFn.forward(
                self.fcache,
                q_head,
                k_head,
                v_head,
                self.mask,
                dropout,
                self._retain_values,
            )
            attn_heads.append(attn_head)
            self.attn_w.append(attn_w_head)
        attn = concat(attn_heads)

        # output projection
        y = self.out_proj(attn)

        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dq_heads, dk_heads, dv_heads = [], [], []

        # output gradients
        dattn = self.out_proj.backward(dy)

        # split output gradients for each head
        dattn_heads = split(dattn, self.n_heads)

        # multi head attention gradients
        for dattn_head in reversed(dattn_heads):  # reversed, because of cache order
            dq_head, dk_head, dv_head = SDPAttentionFn.backward(self.fcache, dattn_head)
            dq_heads.append(dq_head)
            dk_heads.append(dk_head)
            dv_heads.append(dv_head)

        # merge heads
        dq = concat(dq_heads[::-1])
        dk = concat(dk_heads[::-1])
        dv = concat(dv_heads[::-1])

        # input projection gradients
        dx = self.q_proj.backward(dq)
        dx += self.k_proj.backward(dk)
        dx += self.v_proj.backward(dv)

        return dx
