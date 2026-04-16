"""Compression loss for calibration."""

from __future__ import annotations

import torch


def compression_loss(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute the average relative difference between two sets of tensors.

    Used during calibration to measure quality degradation from compression.
    """
    ls = []
    for ai, bi in zip(a, b):
        diff = (ai - bi) / (torch.max(ai, bi) + 1e-6)
        l = diff.abs().clip(0, 10).mean()
        ls.append(l)
    return sum(ls) / len(ls)
