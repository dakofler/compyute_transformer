"""transformer neural network module"""

import math
from typing import Optional, Literal

from compyute.nn.modules.activations import GELU
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

from ..mha_batched import MultiHeadSelfAttention as BMHA
from ..mha_semibatched import MultiHeadSelfAttention as SMHA
from ..mha_unbatched import MultiHeadSelfAttention as UMHA

# pre resid layernorm
# additional ln before lm head
# learned pos embeddings
# no dropout bc large corpus, following karpathy do in attn and post resid
# shared weights of token emb and lm head
# scale out proj weights by 1/sqrt(2*layers)


class GPTTransformer(Module):
    r"""Docoder-only transformer model following
    `Radford et al., 2019 <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>`_ and
    `Brown et al., 2020 <https://arxiv.org/pdf/2005.14165>`_.

    Parameters
    ----------
    n_embeds : int
        Number of embedding vectors.
    embed_dim : int
        Number of embedding dimensions.
    mlp_channels : int
        Number of channels of the hidden layer in the MLP.
    n_heads : int
        Number of attention heads.
    n_blocks : int
        Number of transformer blocks.
    max_context_len : int
        Maximum possible length of the input sequence.
    mask : Tensor, optional
        Attention-mask. Defaults to ``None``.
        Must be a zeros-tensor with values of ```-inf`` indicating elements to be masked out.
    dropout : float, optional
        Dropout probability. Defaults to ``0.0``.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.


    .. note::
        Embeddings are initialized from :math:`\mathcal{N}(0, \sqrt{\frac{1}{C_{in}}})`.
        Linear layer weights are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in}}}`. Biases are initialized as zeros.

    .. note::
        Dropout is applied to the output of each residual block and to attention weights.

    .. note::
        The weights of the token embedding and the language model head are shared.

    .. note::
        Normalization is applied before weight layers within
        residual blocks (pre-weight-normalization).
    """

    def __init__(
        self,
        n_embeds: int,
        embed_dim: int,
        mlp_channels: int,
        n_heads: int,
        n_blocks: int,
        max_context_len: int,
        mask: Optional[Tensor] = None,
        dropout: float = 0.0,
        bias: bool = True,
        label: Optional[str] = None,
        implementation: Literal["batched", "semibatched", "unbatched"] = "batched",
    ) -> None:
        super().__init__(label)

        match implementation:
            case "unbatched":
                attn_impl = UMHA
            case "semibatched":
                attn_impl = SMHA
            case _:
                attn_impl = BMHA

        # Embeddings
        self.token_emb = Embedding(n_embeds, embed_dim, "TokenEmbedding")
        self.pos_emb = Embedding(max_context_len, embed_dim, "PosEmbedding")
        std = 1 / math.sqrt(embed_dim)
        init_normal(self.token_emb.w, self.pos_emb.w, std=std)

        # Transformer blocks
        out_scale = 1 / math.sqrt(2 * n_blocks)
        self.blocks = ModuleList(
            TransformerBlock(
                attn_impl,
                embed_dim,
                mlp_channels,
                n_heads,
                out_scale,
                mask,
                dropout,
                bias,
            )
            for _ in range(n_blocks)
        )

        # Language model head
        self.ln = LayerNorm((embed_dim,))
        self.lm_head = Linear(embed_dim, n_embeds, bias)
        self.lm_head.w = self.token_emb.w  # weight sharing

        self.pos = Buffer(insert_dim(arange(max_context_len, dtype=int32), 0))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        pos = self.pos[:, : x.shape[-1]]
        x = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln(x))

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dy = self.ln.backward(self.lm_head.backward(dy))
        for module in reversed(self.blocks):
            dy = module.backward(dy)
        self.token_emb.backward(dy)
        self.pos_emb.backward(dy.sum(0))
        return empty((0,))


class TransformerBlock(Module):

    def __init__(
        self,
        attn_impl: Module,
        in_channels: int,
        mlp_channels: int,
        n_heads: int,
        out_scale: float,
        mask: Optional[Tensor],
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()

        self.ln1 = LayerNorm((in_channels,))
        self.attn = attn_impl(in_channels, n_heads, mask, dropout, out_scale, bias)
        self.dropout1 = Dropout(dropout)

        self.ln2 = LayerNorm((in_channels,))
        self.mlp = MLP(in_channels, mlp_channels, out_scale, bias)
        self.dropout2 = Dropout(dropout)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout1(self.attn(self.ln1(x)))
        x = x + self.dropout2(self.mlp(self.ln2(x)))
        return x

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dy = dy + self.ln2.backward(self.mlp.backward(self.dropout2.backward(dy)))
        dy = dy + self.ln1.backward(self.attn.backward(self.dropout1.backward(dy)))
        return dy


class MLP(Module):

    def __init__(
        self, in_channels: int, h_channels: int, out_scale: float, bias: bool
    ) -> None:
        super().__init__()
        self.up_proj = Linear(in_channels, h_channels, bias)
        self.act = GELU()
        self.down_proj = Linear(h_channels, in_channels, bias)
        self.down_proj.w.data *= out_scale

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        x = self.up_proj(x)
        x = self.act(x)
        x = self.down_proj(x)
        return x

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dy = self.down_proj.backward(dy)
        dy = self.act.backward(dy)
        dy = self.up_proj.backward(dy)
        return dy
