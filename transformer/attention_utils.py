"""attention functions"""

from compyute.tensor_ops.creation_ops import full
from compyute.tensor_ops.selection_ops import tril, triu
from compyute.tensors import Tensor


def get_causal_mask(max_context_len: int) -> Tensor:
    """Returns a causal mask used for the self-attention mechanism.

    Parameters
    ----------
    max_context_len : int
        Maximum sequence length.

    Returns
    -------
    Tensor
        Causal mask.
    """
    shape = (max_context_len, max_context_len)
    return triu(full(shape, float("-inf")), diag_index=1)


def get_sliding_window_mask(max_context_len: int, window_size: int) -> Tensor:
    """Returns a sliding window mask used for the self-attention mechanism.

    Parameters
    ----------
    max_context_len : int
        Maximum sequence length.
    window_size : int
        Size of the sliding window.

    Returns
    -------
    Tensor
        Sliding window mask.
    """
    shape = (max_context_len, max_context_len)
    upper_mask = triu(full(shape, float("-inf")), diag_index=1)
    lower_mask = tril(full(shape, float("-inf")), diag_index=-window_size)
    return upper_mask + lower_mask
