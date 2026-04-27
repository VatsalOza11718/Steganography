"""Image loading, saving, and tensor conversion utilities.

These helpers standardize the image representation used by BIS:
- PIL images for file I/O and human-readable operations
- Torch tensors in ``[-1, 1]`` for model-facing pipelines
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def load_image(
    path: str | Path,
    size: Optional[tuple[int, int]] = None,
) -> Image.Image:
    """Load an image from disk as RGB PIL Image, optionally resized."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        # Resizing at load time ensures every downstream consumer sees the
        # same dimensions instead of each call site resizing independently.
        img = img.resize(size, Image.LANCZOS)
    return img


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a tensor (C, H, W) in [-1, 1] as a PNG image.

    Always saves as PNG to preserve lossless quality for stego images.
    """
    path = Path(path)
    if path.suffix.lower() not in (".png", ".bmp", ".tiff", ".tif"):
        path = path.with_suffix(".png")

    img = tensor_to_pil(tensor)
    img.save(str(path), format="PNG")


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a normalized tensor in [-1, 1] of shape (C, H, W)."""
    transform = transforms.Compose([
        transforms.ToTensor(),          # [0, 1]
        transforms.Normalize(           # [-1, 1]
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])
    return transform(img)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor (C, H, W) in [-1, 1] back to a PIL Image."""
    # Denormalize: [-1, 1] -> [0, 1]
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0
    # Torch stores channels first, while PIL expects channels last.
    array = tensor.permute(1, 2, 0).numpy()
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)


def prepare_cover(
    path: str | Path,
    image_size: int = 256,
) -> torch.Tensor:
    """Load and prepare a cover image as a batch tensor (1, 3, H, W) in [-1, 1]."""
    img = load_image(path, size=(image_size, image_size))
    tensor = image_to_tensor(img)
    return tensor.unsqueeze(0)


def prepare_secret_image(
    path: str | Path,
    image_size: int = 256,
) -> torch.Tensor:
    """Load and prepare a secret image as a batch tensor (1, 3, H, W) in [-1, 1].

    The secret image is resized to match the cover image dimensions.
    """
    return prepare_cover(path, image_size)


def compute_difference_image(
    cover: torch.Tensor,
    stego: torch.Tensor,
    amplification: float = 10.0,
) -> Image.Image:
    """Compute an amplified difference image between cover and stego tensors.

    Both tensors should be (C, H, W) in [-1, 1].
    Returns a PIL Image showing the amplified absolute difference.
    """
    # The raw difference is usually visually imperceptible, so the optional
    # amplification makes changed pixels easier to inspect during debugging.
    diff = (stego - cover).abs() * amplification
    diff = diff.clamp(0, 1)
    # Convert to [0, 255]
    array = diff.detach().cpu().permute(1, 2, 0).numpy()
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)
