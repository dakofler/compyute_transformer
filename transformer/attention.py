"""attention functions"""

import math
from typing import Optional

from compyute.nn.modules.module import Module
from compyute.nn.parameter import Buffer, Parameter
from compyute.random import uniform
from compyute.tensors import Tensor

from .attention_funcs import MultiHeadSelfAttentionFunction


class MultiHeadSelfAttention(Module):
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
        label: Optional[str] = None,
    ) -> None:
        if in_channels % n_heads != 0:
            raise ValueError("Number of input channels must be divisible by n_heads.")
        super().__init__(label)

        self.n_heads = n_heads
        self.mask = None if not mask else Buffer(mask)
        self.dropout = dropout
        self.attn_weights: Optional[Tensor] = None

        # init parameters
        k = 1.0 / math.sqrt(in_channels)
        self.w_i = Parameter(uniform((3 * in_channels, in_channels), -k, k))
        self.b_i = None if not bias else Parameter(uniform((3 * in_channels,), -k, k))
        self.w_o = Parameter(uniform((in_channels, in_channels), -k, k))
        self.w_o *= out_scale
        self.b_o = None if not bias else Parameter(uniform((in_channels,), -k, k))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        dropout = self.dropout if self._is_training else 0
        y, self.attn_weights = MultiHeadSelfAttentionFunction.forward(
            self.function_ctx,
            x,
            self.w_i,
            self.b_i,
            self.w_o,
            self.b_o,
            self.n_heads,
            self.mask,
            dropout,
            False,
        )
        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw_i, db_i, dw_o, db_o = MultiHeadSelfAttentionFunction.backward(
            self.function_ctx, dy
        )
        self.update_parameter_grad(self.w_o, dw_o)
        self.update_parameter_grad(self.b_o, db_o)
        self.update_parameter_grad(self.w_i, dw_i)
        self.update_parameter_grad(self.b_i, db_i)
        return dx
