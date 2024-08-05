"""transformer neural network module"""

from typing import Callable, Optional

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
from compyute.nn.functional import dropout, softmax
from compyute.tensor_functions.creating import zeros_like


class Transformer(Container):
    r"""Docoder-only transformer model.

    Parameters
    ----------
    n_embeddings : int
        Number of embedding vectors.
    embedding_dim : int
        Number of embedding dimensions.
    ffd_channels : int
        Number of channels of the hidden layer in the feed forward block.
    n_heads : int
        Number of attention heads.
    sequence_length : int
        Length of the input sequence.
    mask : Tensor, optional
        Mask for the attention. Defaults to ``None``.
    dropout_p : float, optional
        Dropout probability. Defaults to ``0``.
    attention_bias : bool, optional
        Whether to use bias values in the attention block. Defaults to ``True``.
    feedforward_bias : bool, optional
        Whether to use bias values in the feedforward block. Defaults to ``True``.
    layernorm_eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.
    dtype: DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.
    training: bool, optional
        Whether the module should be in training mode. Defaults to ``False``.


    .. note::
        All weights are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in} * k * k}}`. Biases are initialized as zeros.
    """

    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        ffd_channels: int,
        n_heads: int,
        n_layers: int,
        sequence_length: int,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0,
        attention_bias: bool = False,
        feedforward_bias: bool = True,
        layernorm_eps: float = 1e-5,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:

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
            "in_channels": embedding_dim,
            "ffd_channels": ffd_channels,
            "n_heads": n_heads,
            "sequence_length": sequence_length,
            "mask": mask,
            "dropout_p": dropout_p,
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
    """Decoder-only transformer block consisting of a multi head attention block
    and a feed forward block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    ffd_channels : int
        Number of channels of the hidden layer in the feed forward block.
    n_heads : int
        Number of attention heads.
    sequence_length : int
        Length of the input sequence.
    mask : Tensor, optional
        Mask for the attention. Defaults to ``None``.
    dropout_p : float, optional
        Dropout probability. Defaults to ``0``.
    attention_bias : bool, optional
        Whether to use bias values in the input and output projection
        of the multi head attention block. Defaults to ``True``.
    feedforward_bias : bool, optional
        Whether to use bias values in the feedforward block. Defaults to ``True``.
    layernorm_eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.
    dtype: DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.
    training: bool, optional
        Whether the module should be in training mode. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        ffd_channels: int,
        n_heads: int,
        sequence_length: int,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0,
        attention_bias: bool = True,
        feedforward_bias: bool = True,
        layernorm_eps: float = 1e-5,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        layernorm_kwargs = {
            "normalized_shape": (sequence_length, in_channels),
            "eps": layernorm_eps,
            "dtype": dtype,
            "training": training,
        }

        attention_block_kwargs = {
            "in_channels": in_channels,
            "n_heads": n_heads,
            "mask": mask,
            "dropout_p": dropout_p,
            "bias": attention_bias,
            "dtype": dtype,
            "training": training,
        }

        feedforward_block_kwargs = {
            "in_channels": in_channels,
            "ffd_channels": ffd_channels,
            "dropout_p": dropout_p,
            "bias": feedforward_bias,
            "dtype": dtype,
            "training": training,
        }

        attention_block = ResidualBlock(
            Layernorm(**layernorm_kwargs),
            MultiHeadAttention(**attention_block_kwargs),
            training=training,
        )

        feedforward_block = ResidualBlock(
            Layernorm(**layernorm_kwargs),
            FeedForward(**feedforward_block_kwargs),
            training=training,
        )

        super().__init__(attention_block, feedforward_block, label=label, training=training)


class FeedForward(Sequential):
    """FeedForward block for transformers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    ffd_channels : int
        Number of channels of the hidden layer.
    dropout_p : float, optional
        Dropout probability. Defaults to ``0``.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    dtype: DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.
    training: bool, optional
        Whether the module should be in training mode. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        ffd_channels: int,
        dropout_p: float = 0,
        bias: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        dtype = Dtype(dtype)

        layers = [
            Linear(in_channels, ffd_channels, bias, dtype, training=training),
            ReLU(training=training),
            Linear(ffd_channels, in_channels, bias, dtype, training=training),
        ]
        layers += [Dropout(dropout_p, training=training)] if dropout_p is not None else []

        super().__init__(*layers, label=label, training=training)


class MultiHeadAttention(Sequential):
    r"""Multi Head Attention.

    .. math::
        MultiHeadAttention(x) = concatenate(Head_1(x), ..., Head_n(x))W_o^T

    Shapes:
        - Input :math:`(B, S, C_{in})`
        - Output :math:`(B, S, C_{in})`
    where
        - :math:`B` ... batch axis
        - :math:`S` ... sequence
        - :math:`C_{in}` ... input channels

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    n_heads : int
        Number of attention heads.
    mask : Tensor, optional
        Mask for the attention. Defaults to ``None``.
    dropout_p : float, optional
        Dropout probability. Defaults to ``0``.
    bias : bool, optional
        Whether to use bias values to input and output projection. Defaults to ``True``.
    dtype: DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.
    training: bool, optional
        Whether the module should be in training mode. Defaults to ``False``.


    .. note::
        All weights are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in} * k * k}}`. Biases are initialized as zeros.
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0,
        bias: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        if in_channels % n_heads != 0:
            raise ValueError("Number of input channels must be divisible by number of heads.")

        attention_head_kwargs = {
            "in_channels": in_channels,
            "h_channels": in_channels // n_heads,
            "mask": mask,
            "dropout_p": dropout_p,
            "bias": bias,
            "dtype": dtype,
            "training": training,
        }
        attention_heads = [AttentionHead(**attention_head_kwargs) for _ in range(n_heads)]
        layers = [
            ParallelConcat(*attention_heads, label="AttentionHeads", training=training),
            Linear(in_channels, in_channels, bias, dtype, "OutProjection", training),
        ]
        layers += [Dropout(dropout_p, training=training)] if dropout_p is not None else []

        super().__init__(*layers, label=label, training=training)


class AttentionHead(Container):
    r"""Attention Head.

    .. math::
        \begin{array}{ll} \\
            Q = xW_q^T \\
            K = xW_q^T \\
            V = xW_q^T \\
            Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{|C_h|}}) \cdot V \\
        \end{array}
        
    Shapes:
        - Input :math:`(B, S, C_{in})`
        - Output :math:`(B, S, C_h)`
    where
        - :math:`B` ... batch axis
        - :math:`S` ... sequence
        - :math:`C_{in}` ... input channels
        - :math:`C_h` ... hidden channels

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    h_channels : int
        Number of hidden channels (head size) of the attention head.
    mask : Tensor, optional
        Mask for the attention. Defaults to ``None``.
    dropout : float, optional
        Dropout probability of attention output weights. Defaults to ``0``.
    bias : bool, optional
        Whether to use bias values to input projection. Defaults to ``True``.
    dtype: DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.
    training: bool, optional
        Whether the module should be in training mode. Defaults to ``False``.


    .. note::
        All weights are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in} * k * k}}`. Biases are initialized as zeros.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0,
        bias: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
        training: bool = False,
    ) -> None:
        super().__init__(label=label, training=training)
        self.dtype = Dtype(dtype)

        self.query = Linear(in_channels, h_channels, bias, dtype, "Query", training)
        self.key = Linear(in_channels, h_channels, bias, dtype, "Key", training)
        self.value = Linear(in_channels, h_channels, bias, dtype, "Value", training)

        self.dropout_p = dropout_p
        self.mask = Buffer(mask) if mask is not None else None

    def forward(self, x: Tensor) -> Tensor:
        self._check_dims(x, [3])
        x = x.to_type(self.dtype)

        # input projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # attention
        y, attn_backward = scaled_dot_product_attention(
            q, k, v, self.mask, self.dropout_p, self._training
        )

        if self._training and attn_backward is not None:

            def _backward(dy: Tensor) -> Tensor:
                dy = dy.to_type(self.dtype)

                # attention gradients
                dq, dk, dv = attn_backward(dy)

                # input projection gradients
                dx1 = self.query.backward(dq)
                dx2 = self.key.backward(dk)
                dx3 = self.value.backward(dv)

                return dx1 + dx2 + dx3

            self._backward = _backward

        return y


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0,
    return_grad_fn: bool = False,
) -> tuple[Tensor, Optional[Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]]]:
    """Computes the scaled dot product attention scores.

    Parameters
    ----------
    q : Tensor
        Query tensor.
    k : Tensor
        Key tensor.
    v : Tensor
        Value tensor.
    mask : Tensor, optional
        Mask for the attention. Defaults to ``None``.
    dropout_p : float, optional
        Dropout probability of attention weights. Defaults to ``0``.
    return_grad_fn : bool, optional
        Whether to also return the according gradient function. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Callable[[Tensor], tuple[Tensor, Tensor, Optional[Tensor]]], optional
        Gradient function.

    See Also
    ----------
    :class:`compyute.nn.AttentionHead`
    """
    scale_factor = q.shape[-1] ** -0.5

    qk = q @ k.T * scale_factor
    if mask is not None:
        qk += mask
    attn_weights, sm_grad_func = softmax(qk, return_grad_fn)
    if dropout_p > 0:
        attn_weights, do_grad_func = dropout(attn_weights, dropout_p, return_grad_fn)
    y = attn_weights @ v

    if return_grad_fn:

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            # attention gradients
            dattn_weights = dy @ v.T
            if dropout_p > 0:
                dattn_weights = do_grad_func(dattn_weights)
            dqk = sm_grad_func(dattn_weights) * scale_factor

            # query, key, value gradients
            dq = dqk @ k
            dk = dqk.T @ q
            dv = attn_weights.T @ dy

            return dq, dk, dv

        return y, grad_fn

    return y, None
