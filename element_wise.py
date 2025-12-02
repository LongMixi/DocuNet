import torch
import torch.nn as nn


class ElementWiseMatrixAttention(nn.Module):
    """
    Element-wise interaction between two sequences.
    Produces tensor: (batch, len1, len2, dim)
    Equivalent to einsum('iaj,ibj->ijab') from AllenNLP implementation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        """
        tensor_1: (batch, len1, dim)
        tensor_2: (batch, len2, dim)
        return:   (batch, len1, len2, dim)
        """

        # Expand so shapes become broadcastable
        # tensor_1 → (batch, len1, 1, dim)
        # tensor_2 → (batch, 1, len2, dim)
        t1 = tensor_1.unsqueeze(2)
        t2 = tensor_2.unsqueeze(1)

        # Element-wise multiplication
        result = t1 * t2  # (batch, len1, len2, dim)

        return result
