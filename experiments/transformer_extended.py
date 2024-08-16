"""transformer neural network module"""

import math
from typing import Callable, Optional

from compyute.base_tensor import Tensor, _ShapeLike
from compyute.dtypes import Dtype, _DtypeLike
from compyute.nn.functional import dropout, softmax
from compyute.nn.modules.activations import _ActivationLike, get_activation
from compyute.nn.modules.containers import ResidualConnection, Sequential
from compyute.nn.modules.embedding import Embedding
from compyute.nn.modules.linear import Linear
from compyute.nn.modules.module import Module, validate_input_axes
from compyute.nn.modules.normalization import LayerNorm
from compyute.nn.modules.regularization import Dropout
from compyute.nn.parameter import Buffer, Parameter
from compyute.random.random import normal, uniform
from compyute.tensor_ops.creating import arange, concatenate, full, split, zeros_like
from compyute.tensor_ops.selecting import triu


def get_causal_mask(shape: _ShapeLike) -> Tensor:
    """Returns a causal mask used for the self-attention mechanism.

    Parameters
    ----------
    shape : _ShapeLike
        Shape of the mask.

    Returns
    -------
    Tensor
        Causal mask.
    """
    return triu(full(shape, value=float("-inf"), dtype=Dtype.FLOAT32), d=1)


class Transformer(Module):
    r"""Docoder-only transformer model.

    Parameters
    ----------
    n_embeddings : int
        Number of embedding vectors.
    embedding_dim : int
        Number of embedding dimensions.
    feedforward_channels : int
        Number of channels of the hidden layer in the feed forward block.
    n_heads : int
        Number of attention heads.
    n_blocks : int
        Number of transformer blocks.
    max_sequence_length : int
        Maximum possible length of the input sequence.
    mask : Tensor, optional
        Attention-mask. Defaults to ``None``.
        Must be a zeros-tensor with values of ```-inf`` indicating elements to be masked out.
    dropout_p : float, optional
        Dropout probability. Defaults to ``0.2``.
    activation : _ActivationLike
        Activation function to use in the feedforward blocks. Defaults to ``relu``.
        See :ref:`activations` for more details.
    attention_bias : bool, optional
        Whether to use bias values in the attention heads. Defaults to ``True``.
    feedforward_bias : bool, optional
        Whether to use bias values in the feedforward block. Defaults to ``True``.
    layernorm_eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.
    dtype: DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.


    .. note::
        Embedding weights are initialized from :math:`\mathcal{N}(0, \sqrt{\frac{1}{C_{in})`, where
        :math:`C_{in}` is the embedding dimension.
        Output projection weights of residual blocks are initialized from :math:`\mathcal{U}(-k, k)`
        where :math:`k = \sqrt{\frac{1}{C_{in} * N}}` and :math:`N` is the number of residual connecitons.
        Biases are initialized as zeros.

    .. note::
        Dropout is applied to the output of each residual block and to attention weights.

    .. note::
        The weights of the token embedding and the language model head are shared.

    .. note::
        Normalization is applied before weight layers within residual blocks.
    """

    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        feedforward_channels: int,
        n_heads: int,
        n_blocks: int,
        sequence_length: int,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0.2,
        activation: _ActivationLike = "relu",
        attention_bias: bool = True,
        feedforward_bias: bool = True,
        layernorm_eps: float = 1e-5,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        w_std = math.sqrt(embedding_dim)
        block_kwargs = {
            "in_channels": embedding_dim,
            "feedforward_channels": feedforward_channels,
            "n_heads": n_heads,
            "mask": mask,
            "dropout_p": dropout_p,
            "out_proj_std": w_std / math.sqrt(2 * n_blocks),
            "activation": activation,
            "attention_bias": attention_bias,
            "feedforward_bias": feedforward_bias,
            "layernorm_eps": layernorm_eps,
            "dtype": dtype,
        }

        # Embeddings
        self.token_emb = Embedding(n_embeddings, embedding_dim, dtype, "TokenEmbedding")
        self.token_emb.w = Parameter(
            normal((n_embeddings, embedding_dim), std=w_std, dtype=dtype)
        )

        self.pos_emb = Embedding(sequence_length, embedding_dim, dtype, "PosEmbedding")
        self.pos_emb.w = Parameter(
            normal((sequence_length, embedding_dim), std=w_std, dtype=dtype)
        )

        # Transformer blocks
        self.blocks = [TransformerBlock(**block_kwargs) for _ in range(n_blocks)]

        # Language model head
        self.ln = LayerNorm((embedding_dim,), layernorm_eps, dtype)
        self.lm_head = Linear(embedding_dim, n_embeddings, False, dtype, "LmHead")
        self.lm_head.w = self.token_emb.w  # weight sharing

        self.modules = [
            self.token_emb,
            self.pos_emb,
            *self.blocks,
            self.ln,
            self.lm_head,
        ]

    def forward(self, x: Tensor) -> Tensor:
        pos = arange(x.shape[-1], dtype=Dtype.INT64, device=x.device)
        x = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        y = self.lm_head(self.ln(x))

        def _backward(dy: Tensor) -> Tensor:
            dy = self.ln.backward(self.lm_head.backward(dy))
            for module in reversed(self.blocks):
                dy = module.backward(dy)
            self.token_emb.backward(dy)
            self.pos_emb.backward(dy)
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
    feedforward_channels : int
        Number of channels of the hidden layer in the feed forward block.
    n_heads : int
        Number of attention heads.
    mask : Tensor, optional
        Attention-mask. Defaults to ``None``.
        Must be a zeros-tensor with values of ```-inf`` indicating elements to be masked out.
    dropout_p : float, optional
        Dropout probability. Defaults to ``0.2``.
    out_proj_std : float, optional
        Standard deviation of output projection weights to compensate the growing
        variance of the residual block. Defaults to ``None``.
        If ``None``, the default initialization of ``Linear`` is used.
    activation : _ActivationLike
        Activation function to use in the feedforward blocks. Defaults to ``relu``.
        See :ref:`activations` for more details.
    attention_bias : bool, optional
        Whether to use bias values in the input and output projections
        of the multi head attention block. Defaults to ``True``.
    feedforward_bias : bool, optional
        Whether to use bias values in the feedforward block. Defaults to ``True``.
    layernorm_eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.
    dtype: DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.
    """

    def __init__(
        self,
        in_channels: int,
        feedforward_channels: int,
        n_heads: int,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0.2,
        out_proj_std: float = 0.02,
        activation: _ActivationLike = "relu",
        attention_bias: bool = True,
        feedforward_bias: bool = True,
        layernorm_eps: float = 1e-5,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
    ) -> None:
        ln_kwargs = {
            "normalized_shape": (in_channels,),
            "eps": layernorm_eps,
            "dtype": dtype,
        }

        attention_block = ResidualConnection(
            LayerNorm(**ln_kwargs),
            MultiHeadAttention(
                in_channels,
                n_heads,
                mask=mask,
                dropout_p=dropout_p,
                out_proj_std=out_proj_std,
                bias=attention_bias,
                dtype=dtype,
            ),
            Dropout(dropout_p),
        )

        feedforward_block = ResidualConnection(
            LayerNorm(**ln_kwargs),
            FeedForward(
                in_channels,
                feedforward_channels,
                out_proj_std,
                activation,
                feedforward_bias,
                dtype,
            ),
            Dropout(dropout_p),
        )

        super().__init__(attention_block, feedforward_block, label=label)


class FeedForward(Sequential):
    """FeedForward block for transformers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    h_channels : int
        Number of channels of the hidden layer.
    out_proj_std : float, optional
        Standard deviation of output projection weights to compensate the growing
        variance of the residual block. Defaults to ``None``.
        If ``None``, the default initialization of ``Linear`` is used.
    activation : _ActivationLike
        Activation function to use. Defaults to ``relu``.
        See :ref:`activations` for more details.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    dtype: DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        out_proj_std: Optional[float] = None,
        activation: _ActivationLike = "relu",
        bias: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
    ) -> None:
        linear = Linear(in_channels, h_channels, bias, dtype)
        act = get_activation(activation)()
        out_proj = Linear(h_channels, in_channels, bias, dtype, "OutProj")
        if out_proj_std is not None:
            out_proj.w = Parameter(
                uniform((in_channels, h_channels), -out_proj_std, out_proj_std, dtype)
            )

        super().__init__(linear, act, out_proj, label=label)


class MultiHeadAttention(Module):
    r"""Multi Head Attention.

    .. math::
        \begin{array}{ll} \\
            Q = xW_q^T \\
            K = xW_q^T \\
            V = xW_q^T \\
            MultiHeadAttention(x) = concatenate(Attention_1(xW_q^T, xW_k^T, xW_v^T), ..., Attention_n(xW_q^T, xW_k^T, xW_v^T))W_o^T \\
        \end{array}
    
    where

    .. math::
            Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{N}}) \cdot V

    Shapes:
        - Input :math:`(B, S, C_{in})`
        - Output :math:`(B, S, C_{in})`
    where
        - :math:`B` ... batch axis
        - :math:`S` ... sequence
        - :math:`C_{in}` ... input channels
        - :math:`N` ... number of attention heads 

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    n_heads : int
        Number of attention heads.
    mask : Tensor, optional
        Attention-mask. Defaults to ``None``.
        Must be a zeros-tensor with values of ```-inf`` indicating elements to be masked out.
    dropout_p : float, optional
        Dropout probability. Defaults to ``0``.
    out_proj_std : float, optional
        Standard deviation of output projection weights to compensate the growing
        variance of the residual block. Defaults to ``None``.
        If ``None``, the default initialization of ``Linear`` is used.
    bias : bool, optional
        Whether to use bias values to input and output projection. Defaults to ``True``.
    dtype: DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.


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
        out_proj_std: float = 0.02,
        bias: bool = True,
        dtype: _DtypeLike = Dtype.FLOAT32,
        label: Optional[str] = None,
    ) -> None:
        if in_channels % n_heads != 0:
            raise ValueError(
                "Number of input channels must be divisible by number of heads."
            )
        super().__init__(label)
        self.n_heads = n_heads
        self.mask = Buffer(mask) if mask is not None else None
        self.dropout_p = dropout_p
        self.attn_w: Optional[list[Tensor]] = None

        self.query_proj = Linear(in_channels, in_channels, bias, dtype, "QueryProj")
        self.key_proj = Linear(in_channels, in_channels, bias, dtype, "KeyProj")
        self.value_proj = Linear(in_channels, in_channels, bias, dtype, "ValueProj")
        self.out_proj = Linear(in_channels, in_channels, bias, dtype, "OutProj")
        if out_proj_std is not None:
            self.out_proj.w = Parameter(
                uniform((in_channels, in_channels), -out_proj_std, out_proj_std, dtype)
            )

        self.modules = [self.query_proj, self.key_proj, self.value_proj, self.out_proj]

    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [3])

        head_grad_functions, ys, attn_w = [], [], []
        sdp_dropout_p = self.dropout_p if self._training else 0

        # input projection for self-attention (B, S, C_in) -> (B, S, C_in)
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # split projections into heads (B, S, C_in) -> (B, S, C_in // n_heads)
        q_heads = split(q, self.n_heads)
        k_heads = split(k, self.n_heads)
        v_heads = split(v, self.n_heads)

        # multi head attention: compute attention weights for each head, concat results
        for q_head, k_head, v_head in zip(q_heads, k_heads, v_heads):
            y_head, attn_w_head, attn_grad_fn = scaled_dot_product_attention(
                q_head,
                k_head,
                v_head,
                self.mask,
                sdp_dropout_p,
                self.is_retaining_values,
                self._training,
            )
            ys.append(y_head)
            attn_w.append(attn_w_head)
            head_grad_functions.append(attn_grad_fn)

        y = concatenate(ys)
        if self.is_retaining_values:
            self.attn_w = attn_w

        # output projection (B, S, C_in) -> (B, S, C_in)
        y = self.out_proj(y)

        if self._training:

            def _backward(dy: Tensor) -> Tensor:
                dy = self.out_proj.backward(dy)
                dy_splits = split(dy, self.n_heads)
                dq_heads, dk_heads, dv_heads = [], [], []
                for grad_fn, dy_head in zip(head_grad_functions, dy_splits):
                    dq_head, dk_head, dv_head = grad_fn(dy_head)
                    dq_heads.append(dq_head)
                    dk_heads.append(dk_head)
                    dv_heads.append(dv_head)

                dq = concatenate(dq_heads)
                dk = concatenate(dk_heads)
                dv = concatenate(dv_heads)

                dx1 = self.query_proj.backward(dq)
                dx2 = self.key_proj.backward(dk)
                dx3 = self.value_proj.backward(dv)

                return dx1 + dx2 + dx3

            self._backward = _backward

        return y


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0,
    return_attn_w: bool = False,
    return_grad_fn: bool = False,
) -> tuple[
    Tensor,
    Optional[Tensor],
    Optional[Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]],
]:
    r"""Computes the scaled dot product attention scores.

    Parameters
    ----------
    q : Tensor
        Query tensor.
    k : Tensor
        Key tensor.
    v : Tensor
        Value tensor.
    mask : Tensor, optional
        Attention-mask. Defaults to ``None``.
        Must be a zeros-tensor with values of ```-inf`` indicating elements to be masked out.
    dropout_p : float, optional
        Dropout probability of attention weights. Defaults to ``0``.
    return_attn_w : bool, optional
        Whether to also return the computed attention weights. Defaults to ``False``.
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
    :class:`compyute.nn.MultiHeadAttention`
    """
    _, seq_len, head_size = q.shape
    scale_factor = math.sqrt(head_size)

    attn_w = q @ k.T / scale_factor
    if mask is not None:
        attn_w += mask[:seq_len, :seq_len]  # truncate mask for smaller context lengths
    attn_w, sm_grad_fn = softmax(attn_w, return_grad_fn)
    if dropout_p > 0:
        attn_w, drouput_grad_fn = dropout(attn_w, dropout_p, return_grad_fn)
    y = attn_w @ v

    if return_grad_fn:

        def grad_fn(dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            # attention gradients
            dattn_w = dy @ v.T
            if dropout_p > 0:
                dattn_w = drouput_grad_fn(dattn_w)
            dattn_w = sm_grad_fn(dattn_w) / scale_factor

            # query, key, value gradients
            dq = dattn_w @ k
            dk = dattn_w.T @ q
            dv = attn_w.T @ dy

            return dq, dk, dv

        return y, attn_w if return_attn_w else None, grad_fn

    return y, attn_w if return_attn_w else None, None
