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
from compyute.nn.parameter import Buffer
from compyute.nn.utils.initializers import init_normal
from compyute.tensor_ops.creation_ops import arange, empty
from compyute.tensor_ops.shape_ops import insert_dim
from compyute.tensors import Tensor
from compyute.typing import int32

from .mha_batched import MultiHeadAttention

# pre resid layernorm
# additional ln before lm head
# learned pos embeddings
# no dropout bc large corpus, following karpathy do in attn and post resid
# shared weights of token emb and lm head
# scale out proj weights by 1/sqrt(2*layers)


class VisionTransformer(Module):
    r"""Docoder-only vision transformer model following
    `Dosovitskiy et al., 2021 <https://arxiv.org/pdf/2010.11929>`_.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        embedding_dim: int,
        n_classes: int,
        ffwd_channels: int,
        n_heads: int,
        n_blocks: int,
        dropout: float = 0.0,
        bias: bool = True,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)

        # Embeddings
        self.patch_emb = VisionEmbedding(
            in_channels,
            image_size,
            patch_size,
            embedding_dim,
            label="PatchEmbedding",
        )
        n_patches = self.patch_emb.n_patches

        self.pos_emb = Embedding(n_patches, embedding_dim, "PosEmbedding")
        std = 1 / math.sqrt(embedding_dim)
        init_normal(self.pos_emb.w, std=std)

        # embedding dropout
        self.emb_dropout = Dropout(dropout)

        # Transformer blocks
        out_scale = 1 / math.sqrt(2 * n_blocks)
        self.blocks = ModuleList(
            TransformerBlock(
                embedding_dim, ffwd_channels, n_heads, out_scale, None, dropout, bias
            )
            for _ in range(n_blocks)
        )

        # Model head
        self.ln = LayerNorm((embedding_dim,))
        self.lm_head = Linear(n_patches * embedding_dim, n_classes, bias)

        self.pos = Buffer(insert_dim(arange(n_patches, dtype=int32), 0))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_emb(x) + self.pos_emb(self.pos)
        x = self.emb_dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln(x)[:, 0])

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dy = self.lm_head.backward(dy)
        # dy_ext = zeros
        # dy = self.ln.backward(self.flatten.backward(self.lm_head.backward(dy)))
        # TODO: Add this special classification token
        for module in reversed(self.blocks):
            dy = module.backward(dy)
        dy = self.emb_dropout.backward(dy)
        self.patch_emb.backward(dy)
        self.pos_emb.backward(dy.sum(0))
        return empty((0,))


class TransformerBlock(Module):

    def __init__(
        self,
        in_channels: int,
        ffwd_channels: int,
        n_heads: int,
        out_scale: float,
        mask: Optional[Tensor],
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()

        self.ln1 = LayerNorm((in_channels,))
        self.attn = MultiHeadAttention(
            in_channels, n_heads, mask, dropout, out_scale, bias
        )
        self.dropout = Dropout(dropout)
        self.ln2 = LayerNorm((in_channels,))
        self.ffwd = FeedForward(in_channels, ffwd_channels, out_scale, bias)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffwd(self.ln2(x)))
        return x

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dy = dy + self.ln2.backward(self.ffwd.backward(self.dropout.backward(dy)))
        dy = dy + self.ln1.backward(self.attn.backward(self.dropout.backward(dy)))
        return dy


class FeedForward(Module):

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


class VisionEmbedding(Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        embedding_dim: int,
        label: Optional[str] = None,
    ):
        super().__init__(label)
        self.n_patches = (image_size // patch_size) ** 2
        self.conv = Conv2D(in_channels, embedding_dim, patch_size, stride=patch_size)

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
