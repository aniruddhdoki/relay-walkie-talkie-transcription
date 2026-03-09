"""Loss functions for denoiser training."""

import torch
import torch.nn as nn


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss between predicted and target spectrogram."""
    return nn.functional.mse_loss(pred, target)


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss between predicted and target spectrogram."""
    return nn.functional.l1_loss(pred, target)
