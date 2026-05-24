"""Sequence-level auxiliary loss for NanoPoor.

Adapted from BiBo's NTILLoss (arXiv:2505.13077).
Only the sequence-level component: L1 between predicted token values
and target values (soft argmax vs hard labels).

For natural language tokens, treats token ids as ordinal values.
This gives a "how wrong is the whole sequence" gradient signal
on top of standard CCE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SequenceLoss']


class SequenceLoss(nn.Module):
    """
    Sequence-level loss: L1 between predicted token values and target values.

    Predicted value = expected value under predicted distribution (soft argmax).
    This captures "whole sequence correctness" — not just per-token.

    Args:
        vocab_size: vocabulary size (ordinal range = [0, vocab_size))
        coeff: scaling coefficient (0 = disabled)
    """
    def __init__(self, vocab_size: int = 50257, coeff: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.coeff = coeff

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, T, V] raw logits
            targets: [B, T] target token ids
        Returns:
            seq_loss: scalar (already scaled by coeff)
        """
        if self.coeff == 0:
            return torch.tensor(0.0, device=logits.device)

        # Soft predicted value: E[token_value] under predicted distribution
        probs = F.softmax(logits, dim=-1)  # [B, T, V]
        ordinal_values = torch.arange(self.vocab_size, device=logits.device, dtype=logits.dtype)
        predicted_values = (probs * ordinal_values).sum(dim=-1)  # [B, T]

        # Target values
        target_values = targets.float()  # [B, T]

        # L1 distance normalized by vocab_size
        l1 = (predicted_values - target_values).abs() / self.vocab_size  # [B, T]

        return l1.mean() * self.coeff
