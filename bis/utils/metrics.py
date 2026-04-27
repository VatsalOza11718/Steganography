"""Quality and accuracy metrics for steganography evaluation."""

from __future__ import annotations

import torch
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


def compute_psnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """Compute PSNR between two image tensors.

    Args:
        original: (N, C, H, W) or (C, H, W) tensor in [-1, 1].
        reconstructed: Same shape as original.

    Returns:
        PSNR value in dB.
    """
    if original.dim() == 3:
        original = original.unsqueeze(0)
    if reconstructed.dim() == 3:
        reconstructed = reconstructed.unsqueeze(0)

    # TorchMetrics expects ordinary image ranges, while BIS image tensors are
    # normalized to [-1, 1] for model training and inference.
    original = (original + 1.0) / 2.0
    reconstructed = (reconstructed + 1.0) / 2.0

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(original.device)
    return psnr(reconstructed, original).item()


def compute_ssim(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """Compute SSIM between two image tensors.

    Args:
        original: (N, C, H, W) or (C, H, W) tensor in [-1, 1].
        reconstructed: Same shape as original.

    Returns:
        SSIM value in [0, 1].
    """
    if original.dim() == 3:
        original = original.unsqueeze(0)
    if reconstructed.dim() == 3:
        reconstructed = reconstructed.unsqueeze(0)

    # SSIM is also defined on a standard intensity range rather than the
    # normalized training range used internally by the project.
    original = (original + 1.0) / 2.0
    reconstructed = (reconstructed + 1.0) / 2.0

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(original.device)
    return ssim(reconstructed, original).item()


def compute_bit_accuracy(
    original_bits: torch.Tensor,
    decoded_bits: torch.Tensor,
) -> float:
    """Compute bit-wise accuracy between original and decoded bit tensors.

    Args:
        original_bits: Binary tensor (values 0 or 1).
        decoded_bits: Tensor with continuous values, thresholded at 0.5.

    Returns:
        Accuracy in [0, 1].
    """
    # Threshold soft predictions so the metric compares actual binary symbols.
    decoded_binary = (decoded_bits > 0.5).float()
    original_binary = (original_bits > 0.5).float()
    correct = (decoded_binary == original_binary).float()
    return correct.mean().item()
