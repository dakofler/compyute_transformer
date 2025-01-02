"""multi head self attention module where heads are partially computed in parallel"""

from typing import Optional

from compyute.nn.modules.linear import Linear
from compyute.nn.modules.module import Module
from compyute.nn.parameter import Buffer
from compyute.tensor_ops.shape_ops import concat, split
from compyute.tensors import Tensor
from sdp import SDPAttentionFunction


class SemiparallelMHA(Module):

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
        self.mask = None if not mask else Buffer(mask)
        self.dropout = dropout
        self.attn_w: list[Optional[Tensor]] = []

        self.q_proj = Linear(in_channels, in_channels, bias, "QueryProj")
        self.k_proj = Linear(in_channels, in_channels, bias, "KeyProj")
        self.v_proj = Linear(in_channels, in_channels, bias, "ValueProj")

        self.out_proj = Linear(in_channels, in_channels, bias, "OutProj")
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
            attn_head, attn_w_head = SDPAttentionFunction.forward(
                self.function_ctx,
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
            dq_head, dk_head, dv_head = SDPAttentionFunction.backward(
                self.function_ctx, dattn_head
            )
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
