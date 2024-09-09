"""transformer neural network module"""

import math
from typing import Optional

from attention_s import MultiHeadAttention
from compyute.nn.modules.activations import ReLU
from compyute.nn.modules.embedding import Embedding
from compyute.nn.modules.linear import Linear
from compyute.nn.modules.module import Module, ModuleList
from compyute.nn.modules.normalization import LayerNorm
from compyute.nn.modules.regularization import Dropout
from compyute.nn.parameter import Buffer
from compyute.nn.utils.initializers import init_normal
from compyute.tensor_ops.creating import arange, zeros
from compyute.tensor_ops.reshaping import insert_dim
from compyute.tensor_ops.transforming import cos, exp, sin
from compyute.tensors import Tensor
from compyute.typing import DType

# post resid layernorm
# sinusoidal pos encodings
# dropout after embeddings and resid blocks
# shared weights of token emb and lm head


class OrigTransformer(Module):
    r"""Docoder-only transformer model following
    `Vaswani et al., 2017 <https://arxiv.org/pdf/1706.03762>`_.

    Parameters
    ----------
    n_embeddings : int
        Number of embedding vectors.
    embedding_dim : int
        Number of embedding dimensions.
    ffwd_channels : int
        Number of channels of the hidden layer in the feed forward block.
    n_heads : int
        Number of attention heads.
    n_blocks : int
        Number of transformer blocks.
    max_seq_len : int
        Maximum possible length of the input sequence.
    mask : Tensor, optional
        Attention-mask. Defaults to ``None``.
        Must be a zeros-tensor with values of ```-inf`` indicating elements to be masked out.
    dropout : float, optional
        Dropout probability. Defaults to ``0.1``.
    dtype: DtypeLike, optional
        Datatype of weights and biases. Defaults to ``None``.
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
        n_embeddings: int,
        embedding_dim: int,
        ffwd_channels: int,
        n_heads: int,
        n_blocks: int,
        max_seq_len: int,
        pos_enc_base: float = 10000.0,
        mask: Optional[Tensor] = None,
        dropout: float = 0.1,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)

        # Embeddings
        self.token_emb = Embedding(n_embeddings, embedding_dim, dtype, "TokenEmbedding")
        init_normal(self.token_emb.w, std=1 / math.sqrt(embedding_dim))
        self.pos_emb = PositionalEncoding(
            max_seq_len, embedding_dim, pos_enc_base, dtype, "PosEncoding"
        )

        # Embedding dropout
        self.emb_dropout = Dropout(dropout)

        # Transformer blocks
        block_kwargs = {
            "in_channels": embedding_dim,
            "ffwd_channels": ffwd_channels,
            "n_heads": n_heads,
            "mask": mask,
            "dropout": dropout,
            "dtype": dtype,
        }
        self.blocks = ModuleList(
            TransformerBlock(**block_kwargs) for _ in range(n_blocks)
        )

        # Language model head
        self.lm_head = Linear(embedding_dim, n_embeddings, dtype=dtype)
        self.lm_head.w = self.token_emb.w  # weight sharing

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        x = self.token_emb(x) + self.pos_emb(x)
        x = self.emb_dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> None:
        dy = self.lm_head.backward(dy)
        for module in reversed(self.blocks):
            dy = module.backward(dy)
        dy = self.emb_dropout.backward(dy)
        self.token_emb.backward(dy)


class TransformerBlock(Module):
    """Decoder-only transformer block consisting of a multi head attention block
    and a feed forward block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    ffwd_channels : int
        Number of channels of the hidden layer in the feed forward block.
    n_heads : int
        Number of attention heads.
    mask : Tensor, optional
        Attention-mask.
        Must be a zeros-tensor with values of ```-inf`` indicating elements to be masked out.
    dropout : float
        Dropout probability.
    dtype: DtypeLike, optional
        Datatype of weights and biases.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.
    """

    def __init__(
        self,
        in_channels: int,
        ffwd_channels: int,
        n_heads: int,
        mask: Optional[Tensor],
        dropout: float,
        dtype: Optional[DType],
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)

        self.attn = MultiHeadAttention(in_channels, n_heads, mask, 0.0, 1.0, dtype)
        self.dropout_1 = Dropout(dropout)
        self.ln_1 = LayerNorm((in_channels,), dtype=dtype)

        self.ffwd = FeedForward(in_channels, ffwd_channels, dtype)
        self.dropout_2 = Dropout(dropout)
        self.ln_2 = LayerNorm((in_channels,), dtype=dtype)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        x = self.ln_1(x + self.dropout_1(self.attn(x)))
        x = self.ln_2(x + self.dropout_2(self.ffwd(x)))
        return x

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dy = self.ln_2.backward(dy)
        dy = dy + self.ffwd.backward(self.dropout_2.backward(dy))
        dy = self.ln_1.backward(dy)
        dy = dy + self.attn.backward(self.dropout_1.backward(dy))
        return dy


class FeedForward(Module):
    """FeedForward block for transformers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    h_channels : int
        Number of channels of the hidden layer.
    dtype: DtypeLike, optional
        Datatype of weights and biases.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        dtype: Optional[DType],
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.up_proj = Linear(in_channels, h_channels, dtype=dtype)
        self.act = ReLU()
        self.down_proj = Linear(h_channels, in_channels, dtype=dtype)

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


class PositionalEncoding(Module):
    r"""Sinusoidal Positional Encoding layer as described by
    `Vaswani et al., 2017 <https://arxiv.org/pdf/1706.03762>`_.

    .. math::
        \begin{array}{ll} \\
            PE_{(pos, 2i)} = \text{sin}(pos \cdot e^{-2i \frac{log(b)}{E})
            PE_{(pos, 2i+1)} = \text{cos}(pos \cdot e^{-2i \frac{log(b)}{E})
        \end{array}

    where :math:`E` is the embedding dimension and :math:`b` is the base.

    Shapes:
        - Input :math:`(B_1, ... , B_n, S)`
        - Output :math:`(B_1, ... , B_n, S, E)`
    where
        - :math:`B_1, ... , B_n` ... batch axes
        - :math:`S` ... sequence
        - :math:`E` ... embedding dimension

    Parameters
    ----------
    max_seq_len : int
        Maximum possible length of the input sequence.
    embedding_dim : int
        Embedding vector dimensions.
    base : float, optional
        Base for computing the positional encoding. Defaults to ``1e4``.
    dtype : DType, optional
        Datatype of weights and biases. Defaults to ``None``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(
        self,
        max_seq_len: int,
        embedding_dim: int,
        base: float = 1e4,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        # compute positional encodings
        encodings = zeros((max_seq_len, embedding_dim), dtype=dtype)
        positions = insert_dim(arange(max_seq_len, dtype=dtype), -1)
        emb_range = arange(embedding_dim, step=2, dtype=dtype)
        div_term = exp(emb_range * (-(math.log(base) / embedding_dim)))
        encodings[:, 0::2] = sin(positions * div_term)
        encodings[:, 1::2] = cos(positions * div_term)

        self.encodings = Buffer(encodings)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return self.encodings[: x.shape[1]]

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return dy
