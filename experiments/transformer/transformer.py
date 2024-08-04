"""transformer neural network module"""

from typing import Optional

from compyute.base_tensor import Tensor
from compyute.dtypes import Dtype, _DtypeLike
from compyute.nn import (
    Buffer,
    Container,
    Dropout,
    Embedding,
    Layernorm,
    Linear,
    ParallelConcat,
    ReLU,
    ResidualBlock,
    Sequential,
)
from compyute.nn.functional import softmax
from compyute.tensor_functions.creating import zeros_like


class Transformer(Container):
    """Transformer model"""

    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        ffd_dim: int,
        n_heads: int,
        n_layers: int,
        sequence_length: int,
        mask: Optional[Tensor] = None,
        dropout: Optional[float] = None,
        attention_bias: bool = False,
        feedforward_bias: bool = True,
        layernorm_eps: float = 1e-5,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """Transformer model.

        Parameters
        ----------
        n_embeddings : int
            Number of embedding vectors.
        embedding_dim : int
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

        self.token_embedding = Embedding(
            n_embeddings=n_embeddings,
            embedding_dim=embedding_dim,
            dtype=dtype,
            label="TokenEmbedding",
            training=training,
        )

        self.pos_embedding = Embedding(
            n_embeddings=sequence_length,
            embedding_dim=embedding_dim,
            dtype=dtype,
            label="PosEmbedding",
            training=training,
        )

        transformer_kwargs = {
            "embedding_dim": embedding_dim,
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
            self.blocks = TransformerBlock(**transformer_kwargs)
        else:
            blocks = [TransformerBlock(**transformer_kwargs) for _ in range(n_layers)]
            self.blocks = Sequential(*blocks, label="Blocks", training=training)

        self.out_projection = Linear(
            embedding_dim, n_embeddings, label="OutProjection", training=training
        )

        super().__init__(
            self.token_embedding,
            self.pos_embedding,
            self.blocks,
            self.out_projection,
            label=label,
            training=training,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embedding(x) + self.pos_embedding(x)
        x = self.blocks(x)
        y = self.out_projection(x)

        def _backward(dy: Tensor) -> Tensor:
            dy = self.out_projection.backward(dy)
            dy = self.blocks.backward(dy)
            self.token_embedding.backward(dy)
            self.pos_embedding.backward(dy)
            return zeros_like(x)  # dummy output

        self._backward = _backward

        return y


class TransformerBlock(Sequential):
    """Transformer block consisting of a multi head attention block and a feed forward block"""

    def __init__(
        self,
        embedding_dim: int,
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
        """Transformer block consisting of a multi head attention block and a feed forward block.

        Parameters
        ----------
        embedding_dim : int
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
            Whether to use bias values in the input and output projection
            of the multi head attention block, by default True.
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
        layernorm_kwargs = {
            "normalized_shape": (sequence_length, embedding_dim),
            "eps": layernorm_eps,
            "dtype": dtype,
            "training": training,
        }

        # attention block
        attention_block_kwargs = {
            "embedding_dim": embedding_dim,
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
            "embedding_dim": embedding_dim,
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
    """FeedForward block for transformers"""

    def __init__(
        self,
        embedding_dim: int,
        ffd_dim: int,
        dropout: Optional[float] = None,
        bias: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """FeedForward block for transformers.

        Parameters
        ----------
        embedding_dim : int
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
            Linear(embedding_dim, ffd_dim, bias, dtype, training=training),
            ReLU(training=training),
            Linear(ffd_dim, embedding_dim, bias, dtype, training=training),
        ]
        layers += [Dropout(dropout, training=training)] if dropout is not None else []

        super().__init__(*layers, label=label, training=training)


class MultiHeadAttention(Sequential):
    """Multi Head Attention"""

    def __init__(
        self,
        embedding_dim: int,
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
        embedding_dim : int
            Number of embedding dimensions of the input.
        n_heads : int
            Number of attention heads.
        mask : Tensor, optional
            Mask for the attention, by default None.
        dropout : float, optional
            Dropout probability of attention output weights, by default None.
        bias : bool, optional
            Whether to use bias values to input and output projection, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default Dtype.FLOAT32.
        label: str, optional
            Module label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
        if embedding_dim % n_heads != 0:
            raise ValueError("Embedding dim must be divisible by number of heads.")

        attention_head_kwargs = {
            "embedding_dim": embedding_dim,
            "head_size": embedding_dim // n_heads,
            "mask": mask,
            "dropout": dropout,
            "bias": bias,
            "dtype": dtype,
            "training": training,
        }
        attention_heads = [AttentionHead(**attention_head_kwargs) for _ in range(n_heads)]
        layers = [
            ParallelConcat(*attention_heads, label="AttentionHeads", training=training),
            Linear(embedding_dim, embedding_dim, bias, dtype, "OutProjection", training),
        ]
        layers += [Dropout(dropout, training=training)] if dropout is not None else []

        super().__init__(*layers, label=label, training=training)


class AttentionHead(Container):
    """Attention Head"""

    def __init__(
        self,
        embedding_dim: int,
        head_size: int,
        mask: Optional[Tensor] = None,
        dropout: Optional[float] = None,
        bias: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """Attention Head.

        Input: (B, T, C)
            B ... batch, T ... time, C ... embedding dimension
        Output: (B, T, H)
            B ... batch, T ... time, H ... head size

        Parameters
        ----------
        embedding_dim : int
            Number of embedding dimensions of the input.
        head_size : int
            Head size of the attention head.
        mask : Tensor, optional
            Mask for the attention, by default None.
        dropout : float, optional
            Dropout probability of attention output weights, by default None.
        bias : bool, optional
            Whether to use bias values to input projection, by default True.
        dtype: DtypeLike, optional
            Datatype of weights and biases, by default Dtype.FLOAT32.
        label: str, optional
            Module label.
        training: bool, optional
            Whether the module should be in training mode, by default False.
        """
        super().__init__(label=label, training=training)
        self.dtype = Dtype(dtype)

        self.query = Linear(embedding_dim, head_size, bias, dtype, "Query", training)
        self.key = Linear(embedding_dim, head_size, bias, dtype, "Key", training)
        self.value = Linear(embedding_dim, head_size, bias, dtype, "Value", training)
        self.dropout = Dropout(dropout, training=training) if dropout else None

        self.head_size = head_size
        self.mask = Buffer(mask) if mask is not None else None

    def forward(self, x: Tensor) -> Tensor:
        # input projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # attention
        qk = q @ k.T * self.head_size**-0.5
        if self.mask is not None:
            qk += self.mask
        attn_weights, sm_grad_func = softmax(qk, self._training)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        y = attn_weights @ v

        if self._training and sm_grad_func is not None:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)

                # attention gradients
                dattn_weights = dy @ v.T
                if self.dropout is not None:
                    dattn_weights = self.dropout.backward(dattn_weights)
                dqk = sm_grad_func(dattn_weights) * self.head_size**-0.5

                # input projection gradients
                dx1 = self.query.backward(dqk @ k)  # dq = dqk @ k
                dx2 = self.key.backward(dqk.T @ q)  # dk = dqk.T @ q
                dx3 = self.value.backward(attn_weights.T @ dy)  # dq = attn_weights.T @ dy

                return dx1 + dx2 + dx3

            self._backward = _backward

        return y
