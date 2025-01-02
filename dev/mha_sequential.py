"""multi head self attention module where heads are computed sequentially"""

from typing import Optional

from compyute.nn.modules.linear import Linear
from compyute.nn.modules.module import Module, ModuleList
from compyute.nn.parameter import Buffer
from compyute.tensor_ops.reduction_ops import tensorsum
from compyute.tensor_ops.shape_ops import concat, split
from compyute.tensors import Tensor
from sdp import SDPAttentionFunction


class SequentialMHA(Module):

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
        super().__init__(label)
        self.n_heads = n_heads
        head_size = in_channels // n_heads
        self.heads = ModuleList(
            SelfAttentionHead(in_channels, head_size, mask, dropout, bias)
            for _ in range(n_heads)
        )
        self.out_proj = Linear(in_channels, in_channels, bias, "OutProj")
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


class SelfAttentionHead(Module):

    def __init__(
        self,
        in_channels: int,
        head_size: int,
        mask: Optional[Tensor] = None,
        dropout: float = 0.0,
        bias: bool = True,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.mask = None if not mask else Buffer(mask)
        self.dropout = dropout
        self.attn_w: Optional[Tensor] = None

        self.q_proj = Linear(in_channels, head_size, bias, "QueryProj")
        self.k_proj = Linear(in_channels, head_size, bias, "KeyProj")
        self.v_proj = Linear(in_channels, head_size, bias, "ValueProj")

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        dropout = self.dropout if self._is_training else 0

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        y, self.attn_w = SDPAttentionFunction.forward(
            self.function_ctx, q, k, v, self.mask, dropout, self._retain_values
        )
        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        # attention gradients
        dq, dk, dv = SDPAttentionFunction.backward(self.function_ctx, dy)

        # input projection gradients
        dx = self.q_proj.backward(dq)
        dx += self.k_proj.backward(dk)
        dx += self.v_proj.backward(dv)

        return dx
