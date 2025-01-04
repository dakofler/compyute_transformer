"""transformer neural network module"""

import math
from typing import Optional

from compyute.nn.modules.activations import GELU
from compyute.nn.modules.embeddings import Embedding
from compyute.nn.modules.linear import Linear
from compyute.nn.modules.module import Module, ModuleList
from compyute.nn.modules.normalizations import LayerNorm
from compyute.nn.modules.regularizations import Dropout
from compyute.nn.parameter import Buffer
from compyute.nn.utils.initializers import init_normal, init_uniform
from compyute.tensor_ops.creation_ops import arange, empty
from compyute.tensors import Tensor
from compyute.typing import int32

from .attention import MultiHeadSelfAttention

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
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)

        # Embeddings
        self.token_emb = Embedding(n_embeds, embed_dim, "TokenEmbedding")
        self.pos_emb = Embedding(max_context_len, embed_dim, "PosEmbedding")
        std = 1 / math.sqrt(embed_dim)
        init_normal(self.token_emb.w, self.pos_emb.w, std=std)

        # Transformer blocks
        out_scale = 1 / math.sqrt(2 * n_blocks)
        self.blocks = ModuleList(
            TransformerBlock(embed_dim, mlp_channels, n_heads, out_scale, mask, dropout)
            for _ in range(n_blocks)
        )

        # Language model head
        self.ln = LayerNorm((embed_dim,))
        self.head = Linear(embed_dim, n_embeds, bias=False)
        self.head.w = self.token_emb.w  # weight sharing

        self.pos = Buffer(arange(max_context_len, dtype=int32).view((1, -1)))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        x = self.token_emb(x) + self.pos_emb(self.pos[:, : x.shape[-1]])
        for block in self.blocks:
            x = block(x)
        x = self.head(self.ln(x))
        return x

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dy = self.ln.backward(self.head.backward(dy))
        for module in reversed(self.blocks):
            dy = module.backward(dy)
        self.token_emb.backward(dy)
        self.pos_emb.backward(dy.sum(0))
        return empty((0,))


class TransformerBlock(Module):

    def __init__(
        self,
        embed_dim: int,
        mlp_channels: int,
        n_heads: int,
        out_scale: float,
        mask: Optional[Tensor],
        dropout: float,
    ) -> None:
        super().__init__()

        self.ln1 = LayerNorm((embed_dim,))
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, mask, dropout, True)
        self.dropout1 = Dropout(dropout)

        std = out_scale / math.sqrt(embed_dim)
        init_uniform(self.attn.w_o, low=-std, high=std)

        self.ln2 = LayerNorm((embed_dim,))
        self.mlp = MLP(embed_dim, mlp_channels)
        self.dropout2 = Dropout(dropout)

        std = out_scale / math.sqrt(mlp_channels)
        init_uniform(self.mlp.down_proj.w, low=-std, high=std)

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

    def __init__(self, embed_dim: int, mlp_channels: int) -> None:
        super().__init__()
        self.up_proj = Linear(embed_dim, mlp_channels)
        self.act = GELU()
        self.down_proj = Linear(mlp_channels, embed_dim)

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
