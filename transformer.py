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
    Sequential,
    SkipConnection,
)
from compyute.nn.functional import softmax
from compyute.tensor_functions.creating import zeros_like


class Transformer(Container):
    """Transformer Neural Network"""

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
        attention_bias: bool = False,
        feedforward_bias: bool = True,
        layernorm_eps: float = 1e-5,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """Transformer Block.

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
        self.blocks = Sequential(
            *[TransformerBlock(**transformer_kwargs) for _ in range(n_layers)], training=training
        )
        self.out_projection = Linear(emb_dim, vocab_size, label="OutProjection", training=training)

    def forward(self, x: Tensor) -> Tensor:
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(x)
        x = token_emb + pos_emb
        x = self.blocks(x)
        y = self.out_projection(x)

        def _backward(dy: Tensor) -> Tensor:
            dy = self.out_projection.backward(dy)
            dy = self.blocks.backward(dy)
            self.token_embedding.backward(dy)
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
        attention_bias: bool = False,
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
        attention_block = SkipConnection(
            Sequential(
                Layernorm(**layernorm_kwargs),
                MultiHeadAttention(**attention_block_kwargs),
                training=training,
            ),
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
        feedforward_block = SkipConnection(
            Sequential(
                Layernorm(**layernorm_kwargs),
                FeedForward(**feedforward_block_kwargs),
                training=training,
            ),
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


class MultiHeadAttention(Sequential):
    """Multi Head Attention Block"""

    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        mask: Optional[Tensor] = None,
        dropout: Optional[float] = None,
        bias: bool = False,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        """Multi Head Attention Block.

        Parameters
        ----------
        emb_dim : int
            Number of embedding dimensions of the input.
        n_heads : int
            Number of attention heads.
        mask : Tensor, optional
            Mask for the attention, by default None.
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
        attention_head_kwargs = {
            "emb_dim": emb_dim,
            "head_size": emb_dim // n_heads,
            "mask": mask,
            "dropout": dropout,
            "bias": bias,
            "dtype": dtype,
            "training": training,
        }
        layers = [
            ParallelConcat(
                *[AttentionHead(**attention_head_kwargs) for _ in range(n_heads)],
                label="Heads",
                training=training
            ),
            Linear(emb_dim, emb_dim, bias, dtype, label="OutProjection", training=training),
        ]

        if dropout is not None:
            layers.append(Dropout(p=dropout, training=training))

        super().__init__(*layers, label=label, training=training)


class AttentionHead(Container):
    """Attention Head"""

    def __init__(
        self,
        emb_dim: int,
        head_size: int,
        mask: Optional[Tensor] = None,
        dropout: Optional[float] = None,
        bias: bool = False,
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
        emb_dim : int
            Number of embedding dimensions of the input.
        head_size : int
            Head size of the attention head.
        mask : Tensor, optional
            Mask for the attention, by default None.
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
        super().__init__(label=label, training=training)
        self.dtype = Dtype(dtype)

        self.q = Linear(emb_dim, head_size, bias, self.dtype, label="Query", training=training)
        self.k = Linear(emb_dim, head_size, bias, self.dtype, label="Key", training=training)
        self.v = Linear(emb_dim, head_size, bias, self.dtype, label="Value", training=training)
        self.dropout = Dropout(p=dropout, training=training) if dropout else None

        self.head_size = head_size
        self.mask = Buffer(mask) if mask is not None else None

    def forward(self, x: Tensor) -> Tensor:
        # input projections
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # attention
        qk = q @ k.T * self.head_size**-0.5
        if self.mask is not None:
            qk += self.mask
        sm, sm_grad_func = softmax(qk, self._training)
        if self.dropout is not None:
            sm = self.dropout(sm)
        y = sm @ v

        if self._training and sm_grad_func is not None:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.as_type(self.dtype)

                dsm = dy @ v.T

                if self.dropout is not None:
                    dsm = self.dropout.backward(dsm)

                dqk = sm_grad_func(dsm) * self.head_size**-0.5
                dq = self.q.backward(dqk @ k)
                dk = self.k.backward(dqk.T @ q)
                dv = self.v.backward(sm.T @ dy)

                return dq + dk + dv

            self._backward = _backward

        return y
