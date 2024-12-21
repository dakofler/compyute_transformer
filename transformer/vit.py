"""transformer neural network module"""

import math
from typing import Optional

from compyute.nn.modules.activations import GELU
from compyute.nn.modules.convolutions import Conv2D
from compyute.nn.modules.embeddings import Embedding
from compyute.nn.modules.linear import Linear
from compyute.nn.modules.module import Module, ModuleList
from compyute.nn.modules.normalizations import LayerNorm
from compyute.nn.modules.regularizations import Dropout
from compyute.nn.parameter import Buffer, Parameter
from compyute.nn.utils.initializers import init_normal
from compyute.random import normal
from compyute.tensor_ops.creation_ops import arange, empty
from compyute.tensor_ops.shape_ops import broadcast_to, concat, insert_dim
from compyute.tensors import Tensor
from compyute.typing import int32

from .mha_batched import MultiHeadSelfAttention

# pre resid layernorm
# additional ln before lm head
# learned pos embeddings
# no dropout bc large corpus, following karpathy do in attn and post resid
# shared weights of token emb and lm head
# scale out proj weights by 1/sqrt(2*layers)


class VisionTransformer(Module):
    r"""Docoder-only vision transformer model following
    `Dosovitskiy et al., 2020 <https://arxiv.org/pdf/2010.11929>`_.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        n_classes: int,
        mlp_channels: int,
        n_heads: int,
        n_blocks: int,
        dropout: float = 0.0,
        bias: bool = True,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)

        if image_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size.")
        n_patches = (image_size // patch_size) ** 2

        # Embeddings
        self.patch_emb = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.pos_emb = Embedding(n_patches + 1, embed_dim, "PosEmbedding")
        std = 1 / math.sqrt(embed_dim)
        init_normal(self.pos_emb.w, std=std)
        self.class_emb = Parameter(normal((1, 1, embed_dim)))

        # embedding dropout
        self.emb_dropout = Dropout(dropout)

        # Transformer blocks
        out_scale = 1 / math.sqrt(2 * n_blocks)
        self.blocks = ModuleList(
            TransformerBlock(embed_dim, mlp_channels, n_heads, out_scale, dropout, bias)
            for _ in range(n_blocks)
        )

        # Model head
        self.ln = LayerNorm((embed_dim,))
        self.lm_head = Linear(embed_dim, n_classes, bias)

        self.pos = Buffer(insert_dim(arange(n_patches + 1, dtype=int32), 0))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        patch_emb = self.patch_emb(x)
        class_emb = broadcast_to(self.class_emb, (x.shape[0], 1, patch_emb.shape[-1]))
        x = concat([class_emb, patch_emb], dim=1) + self.pos_emb(self.pos)
        x = self.emb_dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln(x))

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dy = self.ln.backward(self.lm_head.backward(dy))
        for module in reversed(self.blocks):
            dy = module.backward(dy)
        dy = self.emb_dropout.backward(dy)
        self.pos_emb.backward(dy.sum(0))
        self.class_emb.grad = dy[:, 0].sum(0, keepdims=True)
        self.patch_emb.backward(dy[:, 1:])
        return empty((0,))


class TransformerBlock(Module):

    def __init__(
        self,
        in_channels: int,
        mlp_channels: int,
        n_heads: int,
        out_scale: float,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()

        self.ln1 = LayerNorm((in_channels,))
        self.attn = MultiHeadSelfAttention(
            in_channels, n_heads, None, dropout, out_scale, bias
        )
        self.dropout = Dropout(dropout)
        self.ln2 = LayerNorm((in_channels,))
        self.mlp = MLP(in_channels, mlp_channels, out_scale, bias)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dy = dy + self.ln2.backward(self.mlp.backward(self.dropout.backward(dy)))
        dy = dy + self.ln1.backward(self.attn.backward(self.dropout.backward(dy)))
        return dy


class MLP(Module):

    def __init__(
        self, in_channels: int, h_channels: int, dropout: float, bias: bool
    ) -> None:
        super().__init__()
        self.up_proj = Linear(in_channels, h_channels, bias)
        self.act = GELU()
        self.dropout = Dropout(dropout)
        self.down_proj = Linear(h_channels, in_channels, bias)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        x = self.up_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dy = self.down_proj.backward(dy)
        dy = self.dropout.backward(dy)
        dy = self.act.backward(dy)
        dy = self.up_proj.backward(dy)
        return dy


class PatchEmbedding(Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        label: Optional[str] = None,
    ):
        super().__init__(label)
        self.conv = Conv2D(in_channels, embed_dim, patch_size, stride=patch_size)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:

        # Create embeddings (B, C, H, W) -> (B, E, P, P)
        x = self.conv(x)

        # Flatten patches and transpose (B, E, P, P) -> (B, E, P**2) -> (B, P**2, E)
        y = x.view((*x.shape[:-2], -1)).transpose(1, 2).to_contiguous()

        self.fcache.push(x.shape)
        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        (conv_shape,) = self.fcache.pop()
        dy = dy.transpose(1, 2).to_contiguous().view(conv_shape)
        return self.conv.backward(dy)
