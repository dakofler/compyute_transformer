"""transformer neural network module"""

import math
from typing import Callable, Optional

from compyute.nn.functional import dropout, softmax
from compyute.nn.modules.activations import ReLU
from compyute.nn.modules.containers import ResidualConnection, Sequential
from compyute.nn.modules.embedding import Embedding
from compyute.nn.modules.linear import Linear
from compyute.nn.modules.module import Module, ModuleList, validate_input_axes
from compyute.nn.modules.normalization import LayerNorm
from compyute.nn.modules.regularization import Dropout
from compyute.nn.parameter import Buffer
from compyute.nn.utils.initializers import Normal
from compyute.tensor_ops.creating import arange, concat, full, split
from compyute.tensor_ops.selecting import triu
from compyute.tensor_ops.transforming import sum as cpsum
from compyute.tensors import ShapeLike, Tensor
from compyute.typing import DType


def get_causal_mask(shape: ShapeLike) -> Tensor:
    """Returns a causal mask used for the self-attention mechanism.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the mask.

    Returns
    -------
    Tensor
        Causal mask.
    """
    return triu(full(shape, float("-inf")), d=1)


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
        feedforward_channels: int,
        n_heads: int,
        n_blocks: int,
        sequence_length: int,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0.2,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)

        # Embeddings
        self.token_emb = Embedding(n_embeddings, embedding_dim, dtype, "TokenEmbedding")
        self.pos_emb = Embedding(sequence_length, embedding_dim, dtype, "PosEmbedding")

        # initialize embedding weights
        w_std = 1 / math.sqrt(embedding_dim)
        init = Normal(std=w_std)
        init(self.token_emb.w)
        init(self.pos_emb.w)

        # Transformer blocks
        block_kwargs = {
            "in_channels": embedding_dim,
            "feedforward_channels": feedforward_channels,
            "n_heads": n_heads,
            "mask": mask,
            "dropout_p": dropout_p,
            "dtype": dtype,
        }
        self.blocks = ModuleList(
            TransformerBlock(**block_kwargs) for _ in range(n_blocks)
        )

        # Language model head
        self.ln = LayerNorm((embedding_dim,), dtype=dtype)
        self.lm_head = Linear(embedding_dim, n_embeddings, dtype=dtype)
        self.lm_head.w = self.token_emb.w  # weight sharing

    def forward(self, x: Tensor) -> Tensor:
        pos = arange(x.shape[-1], device=x.device).to_int()
        x = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        y = self.lm_head(self.ln(x))

        def _backward(dy: Tensor) -> None:
            dy = self.ln.backward(self.lm_head.backward(dy))
            for module in reversed(self.blocks):
                dy = module.backward(dy)
            self.token_emb.backward(dy)
            self.pos_emb.backward(cpsum(dy, axis=0))  # undo broadcasting by summing

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
    dtype: DtypeLike, optional
        Datatype of weights and biases. Defaults to ``None``.
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
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:

        attention_block = ResidualConnection(
            LayerNorm((in_channels,), dtype=dtype),
            MultiHeadAttention(in_channels, n_heads, mask, dropout_p, dtype),
            Dropout(dropout_p),
        )

        feedforward_block = ResidualConnection(
            LayerNorm((in_channels,), dtype=dtype),
            FeedForward(in_channels, feedforward_channels, dtype),
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
    dtype: DtypeLike, optional
        Datatype of weights and biases. Defaults to ``None``.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        linear = Linear(in_channels, h_channels, dtype=dtype)
        activation = ReLU()
        out_proj = Linear(h_channels, in_channels, dtype=dtype, label="OutProj")

        super().__init__(linear, activation, out_proj, label=label)


class MultiHeadAttention(Module):
    r"""Multi Head Self-Attention as described by
    `Vaswani et al., 2017 <https://arxiv.org/pdf/1706.03762>`_.

    .. math::
        \begin{array}{ll} \\
            Q = xW_Q^T \\
            K = xW_K^T \\
            V = xW_V^T \\
            \text{MultiHeadAttention}(x) = \text{concat}(\text{Attention}(Q_1, K_1, V_1), ..., \text{Attention}(Q_n, K_n, V_n))W_o^T \\
            \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{N}}) \cdot V \\
        \end{array}

    where :math:`N` is the number of attention heads.

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
        Attention-mask. Defaults to ``None``.
        Must be a zeros-tensor with values of ```-inf`` indicating elements to be masked out.
    dropout_p : float, optional
        Dropout probability. Defaults to ``0``.
    dtype: DtypeLike, optional
        Datatype of weights and biases. Defaults to ``None``.
    label: str, optional
        Module label. Defaults to ``None``. If `None`, the class name is used.


    .. note::
        All weights are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in}}}`. Biases are initialized as zeros.

    .. note::
        Input projections do not use bias.
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0,
        dtype: Optional[DType] = None,
        label: Optional[str] = None,
    ) -> None:
        if in_channels % n_heads != 0:
            raise ValueError("Number of input channels must be divisible by n_heads.")
        super().__init__(label)

        self.n_heads = n_heads
        self.mask = Buffer(mask) if mask is not None else None
        self.dropout_p = dropout_p
        self.attn_w: list[Tensor | None] = []

        self.query_proj = Linear(in_channels, in_channels, False, dtype, "QueryProj")
        self.key_proj = Linear(in_channels, in_channels, False, dtype, "KeyProj")
        self.value_proj = Linear(in_channels, in_channels, False, dtype, "ValueProj")
        self.out_proj = Linear(in_channels, in_channels, dtype=dtype, label="OutProj")

    def forward(self, x: Tensor) -> Tensor:
        validate_input_axes(self, x, [3])

        head_grad_functions, ys = [], []
        sdp_dropout_p = self.dropout_p if self._is_training else 0

        # input projection for self-attention (B, S, C_in) -> (B, S, C_in)
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # split projections into heads (B, S, C_in) -> (B, S, C_in // n_heads)
        q_heads, k_heads, v_heads = (split(x, self.n_heads) for x in (q, k, v))

        # multi head attention: compute attention weights for each head, concat results
        for q_head, k_head, v_head in zip(q_heads, k_heads, v_heads):
            y_head, attn_w_head, attn_grad_fn = scaled_dot_product_attention(
                q_head,
                k_head,
                v_head,
                self.mask,
                sdp_dropout_p,
                self._is_retaining_values,
                self._is_training,
            )
            ys.append(y_head)
            self.attn_w.append(attn_w_head)
            head_grad_functions.append(attn_grad_fn)

        y = concat(ys)

        # output projection (B, S, C_in) -> (B, S, C_in)
        y = self.out_proj(y)

        if self._is_training:

            def _backward(dy: Tensor) -> Tensor:
                dy = self.out_proj.backward(dy)
                dy_splits = split(dy, self.n_heads)
                dq_heads, dk_heads, dv_heads = [], [], []

                for grad_fn, dy_head in zip(head_grad_functions, dy_splits):
                    dq_head, dk_head, dv_head = grad_fn(dy_head)
                    dq_heads.append(dq_head)
                    dk_heads.append(dk_head)
                    dv_heads.append(dv_head)

                dq, dk, dv = (concat(x) for x in (dq_heads, dk_heads, dv_heads))

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

        return y, (attn_w if return_attn_w else None), grad_fn

    return y, (attn_w if return_attn_w else None), None
