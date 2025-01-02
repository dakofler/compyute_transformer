"""multi head self attention module where heads are computed in parallel"""

from typing import Optional

from compyute.nn.modules.linear import Linear
from compyute.nn.modules.module import Module
from compyute.nn.parameter import Buffer
from compyute.tensor_ops.shape_ops import concat, split
from compyute.tensors import Tensor
from sdp import SDPAttentionFunction


class ParallelMHA(Module):

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

        self.in_proj = Linear(in_channels, 3 * in_channels, bias, "InProj")
        self.out_proj = Linear(in_channels, in_channels, bias, "OutProj")
        self.out_proj.w.data *= out_scale

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        dropout = self.dropout if self._is_training else 0

        # input projection
        q, k, v = split(self.in_proj(x), splits=3, dim=-1)

        # split into heads (B, T, H, Ch) and transpose to (B, H, T, Ch)
        head_shape = (*x.shape[:2], self.n_heads, x.shape[-1] // self.n_heads)
        q = q.view(head_shape).transpose(1, 2).to_contiguous()
        k = k.view(head_shape).transpose(1, 2).to_contiguous()
        v = v.view(head_shape).transpose(1, 2).to_contiguous()

        # multi head attention
        attn, self.attn_weights = SDPAttentionFunction.forward(
            self.function_ctx, q, k, v, self.mask, dropout, self._retain_values
        )

        # transpose back to (B, T, H, Ch) and merge heads to (B, T, C)
        attn = attn.transpose(1, 2).view(x.shape)

        # output projection
        y = self.out_proj(attn)

        self.function_ctx.add(x.shape, head_shape)
        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        x_shape, head_shape = self.function_ctx.get()

        # output gradients
        dy = self.out_proj.backward(dy)

        # split head grads to (B, T, H, Ch), transpose to (B, H, T, Ch)
        dy = dy.view(head_shape).transpose(1, 2).to_contiguous()

        # multi head attention gradients
        dq, dk, dv = SDPAttentionFunction.backward(self.function_ctx, dy)

        # transpose back to (B, T, H, Ch) and  merge head grads to (B, T, C)
        dq = dq.transpose(1, 2).view(x_shape)
        dk = dk.transpose(1, 2).view(x_shape)
        dv = dv.transpose(1, 2).view(x_shape)

        # input projection gradients
        dx = self.in_proj.backward(concat([dq, dk, dv]))

        return dx
