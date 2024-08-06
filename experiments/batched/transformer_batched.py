"""transformer neural network module"""

from typing import Optional

from compyute.base_tensor import Tensor, _ShapeLike
from compyute.dtypes import Dtype, _DtypeLike
from compyute.nn import (
    Buffer,
    Container,
    Dropout,
    Embedding,
    Layernorm,
    Linear,
    ReLU,
    ResidualBlock,
    Sequential,
)
from compyute.nn.functional import softmax
from compyute.tensor_functions.combining import concatenate, split
from compyute.tensor_functions.creating import zeros_like
from compyute.tensor_functions.reshaping import reshape, transpose


class Transformer(Container):
    """Transformer model"""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        ffd_dim: int,
        n_heads: int,
        n_layers: int,
        sequence_length: int,
        mask: Optional[Tensor] = None,
        dropout: Optional[float] = None,
        attention_bias: bool = True,
        feedforward_bias: bool = True,
        layernorm_eps: float = 1e-5,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """Transformer model.

        Parameters
        ----------
        vocab_size : int
            Vocabulary size.
        emb_dim : int
            Number of embedding dimensions of the input.
        ffd_dim : int
            Number of dimensions of the hidden layer in the feed forward block.
        n_heads : int
            Number of attention heads.
        sequence_length : int
            Length of the input sequence.
        mask : Tensor, optional
            Mask for the attention, by default None.
        dropout : float, optional
            Dropout probability, by default None.
        attention_bias : bool, optional
            Whether to use bias values in the attention block, by default True.
        feedforward_bias : bool, optional
            Whether to use bias values in the feedforward block, by default True.
        layernorm_eps : float, optional
            Constant for numerical stability, by default 1e-5.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default Dtype.FLOAT32.
        label: str, optional
            Module label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
        super().__init__(label=label, training=training)
        dtype = Dtype(dtype)

        self.token_embedding = Embedding(
            vocab_size=vocab_size,
            embedding_dim=emb_dim,
            dtype=dtype,
            label="TokenEmbedding",
            training=training,
        )

        self.pos_embedding = Embedding(
            vocab_size=sequence_length,
            embedding_dim=emb_dim,
            dtype=dtype,
            label="PosEmbedding",
            training=training,
        )

        transformer_kwargs = {
            "emb_dim": emb_dim,
            "ffd_dim": ffd_dim,
            "n_heads": n_heads,
            "sequence_length": sequence_length,
            "mask": mask,
            "dropout": dropout,
            "attention_bias": attention_bias,
            "feedforward_bias": feedforward_bias,
            "layernorm_eps": layernorm_eps,
            "dtype": dtype,
            "label": "TransformerBlock",
            "training": training,
        }
        if n_layers == 1:
            self.transformer_blocks = TransformerBlock(**transformer_kwargs)
        else:
            self.transformer_blocks = Sequential(
                *[TransformerBlock(**transformer_kwargs) for _ in range(n_layers)],
                label="TransformerBlocks",
                training=training
            )
        self.out_projection = Linear(emb_dim, vocab_size, label="OutProjection", training=training)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embedding(x) + self.pos_embedding(x)
        x = self.transformer_blocks(x)
        y = self.out_projection(x)

        def _backward(dy: Tensor) -> Tensor:
            dy = self.out_projection.backward(dy)
            dy = self.transformer_blocks.backward(dy)
            self.token_embedding.backward(dy)
            self.pos_embedding.backward(dy)
            return zeros_like(x)

        self._backward = _backward

        return y


class TransformerBlock(Sequential):
    """Transformer Block"""

    def __init__(
        self,
        emb_dim: int,
        ffd_dim: int,
        n_heads: int,
        sequence_length: int,
        mask: Optional[Tensor] = None,
        dropout: Optional[float] = None,
        attention_bias: bool = True,
        feedforward_bias: bool = True,
        layernorm_eps: float = 1e-5,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """Transformer Block.

        Parameters
        ----------
        emb_dim : int
            Number of embedding dimensions of the input.
        ffd_dim : int
            Number of dimensions of the hidden layer in the feed forward block.
        n_heads : int
            Number of attention heads.
        sequence_length : int
            Length of the input sequence.
        mask : Tensor, optional
            Mask for the attention, by default None.
        dropout : float, optional
            Dropout probability, by default None.
        attention_bias : bool, optional
            Whether to use bias values in the attention block, by default True.
        feedforward_bias : bool, optional
            Whether to use bias values in the feedforward block, by default True.
        layernorm_eps : float, optional
            Constant for numerical stability, by default 1e-5.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default Dtype.FLOAT32.
        label: str, optional
            Module label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
        dtype = Dtype(dtype)

        layernorm_kwargs = {
            "normalized_shape": (sequence_length, emb_dim),
            "eps": layernorm_eps,
            "dtype": dtype,
            "training": training,
        }

        # attention block
        attention_block_kwargs = {
            "emb_dim": emb_dim,
            "n_heads": n_heads,
            "mask": mask,
            "dropout": dropout,
            "bias": attention_bias,
            "dtype": dtype,
            "training": training,
        }
        attention_block = ResidualBlock(
            Layernorm(**layernorm_kwargs),
            MultiHeadAttention(**attention_block_kwargs),
            training=training,
        )

        # feedforward block
        feedforward_block_kwargs = {
            "emb_dim": emb_dim,
            "ffd_dim": ffd_dim,
            "dropout": dropout,
            "bias": feedforward_bias,
            "dtype": dtype,
            "training": training,
        }
        feedforward_block = ResidualBlock(
            Layernorm(**layernorm_kwargs),
            FeedForward(**feedforward_block_kwargs),
            training=training,
        )

        super().__init__(attention_block, feedforward_block, label=label, training=training)


class FeedForward(Sequential):
    """FeedForward Block"""

    def __init__(
        self,
        emb_dim: int,
        ffd_dim: int,
        dropout: Optional[float] = None,
        bias: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """FeedForward Block.

        Parameters
        ----------
        emb_dim : int
            Number of embedding dimensions of the input.
        ffd_dim : int
            Number of dimensions of the hidden layer in the feed forward block.
        dropout : float, optional
            Dropout probability, by default None.
        bias : bool, optional
            Whether to use bias values, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default Dtype.FLOAT32.
        label: str, optional
            Module label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
        dtype = Dtype(dtype)

        layers = [
            Linear(emb_dim, ffd_dim, bias=bias, dtype=dtype, training=training),
            ReLU(training=training),
            Linear(ffd_dim, emb_dim, bias=bias, dtype=dtype, training=training),
        ]

        if dropout is not None:
            layers.append(Dropout(p=dropout, training=training))

        super().__init__(*layers, label=label, training=training)


class MultiHeadAttention(Container):
    """Multi Head Attention"""

    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        mask: Optional[Tensor] = None,
        dropout: Optional[float] = None,
        bias: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """Multi Head Attention.
        Input: (B, T, C)
            B ... batch, T ... time, C ... embedding dimension
        Output: (B, T, C)
            B ... batch, T ... time, C ... embedding dimension

        Parameters
        ----------
        emb_dim : int
            Number of embedding dimensions of the input.
        n_heads : int
            Number of attention heads.
        mask : Tensor, optional
            Attention mask, by default None.
        dropout : float, optional
            Dropout probability of attention output weights, by default None.
        bias : bool, optional
            Whether to use bias values, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default Dtype.FLOAT32.
        label: str, optional
            Module label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
        super().__init__(label=label, training=training)
        self.n_heads = n_heads
        self.head_size = emb_dim // n_heads
        self.mask = Buffer(mask, "attention_mask") if mask is not None else None
        self.dtype = Dtype(dtype)

        self.in_proj = Linear(emb_dim, 3 * emb_dim, bias, self.dtype, "InProjection", training)
        self.dropout = Dropout(p=dropout, training=training) if dropout else None
        self.out_proj = Linear(emb_dim, emb_dim, bias, dtype, "OutProjection", training)

    def forward(self, x: Tensor) -> Tensor:
        # perform input projections and split into query, key, value
        # (B, T, C) -> (B, T, 3*C) -> 3 * (B, T, C)
        q, k, v = split(self.in_proj(x), 3)

        # reshape q, k, v into multiple heads and transpose
        # (B, T, C) -> (B, T, nHeads, H) -> (B, nHeads, T, H)
        head_shape = (*x.shape[:-1], self.n_heads, self.head_size)
        q = transpose(reshape(q, head_shape), (1, 2))
        k = transpose(reshape(k, head_shape), (1, 2))
        v = transpose(reshape(v, head_shape), (1, 2))

        # attention
        attn_weights = q @ k.T * self.head_size**-0.5
        if self.mask is not None:
            attn_weights += self.mask
        attn_weights, sm_grad_func = softmax(attn_weights, self._training)

        # dropout attention weights
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # (B, nHeads, T, T) @ (B, nHeads, T, H) -> (B, nHeads, T, H)
        y = attn_weights @ v

        # transpose, then concatinate outputs of each head
        # (B, nHeads, T, H) -> (B, T, nHeads, H) -> (B, T, C)
        y = reshape(transpose(y, (1, 2)), x.shape)

        # output projection
        # (B, T, C) -> (B, T, C)
        y = self.out_proj(y)

        if self._training and sm_grad_func is not None:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)

                dy = self.out_proj.backward(dy)
                dy = transpose(reshape(dy, head_shape), (1, 2))

                dattn_weights = dy @ v.T

                if self.dropout is not None:
                    dattn_weights = self.dropout.backward(dattn_weights)

                dattn_weights = sm_grad_func(dattn_weights) * self.head_size**-0.5

                # transpose, then concatinate gradients of each head
                dq = reshape(transpose(dattn_weights @ k, (1, 2)), x.shape)
                dk = reshape(transpose(dattn_weights.T @ q, (1, 2)), x.shape)
                dv = reshape(transpose(attn_weights.T @ dy, (1, 2)), x.shape)
                dinputproj = concatenate([dq, dk, dv])
                return self.in_proj.backward(dinputproj)

            self._backward = _backward

        return y
