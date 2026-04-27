"""AI Image Generation Module — Unified Interface for Multiple Models.

Provides a plug-and-play architecture for text-to-image generation
using independently verified models from Hugging Face:

  1. Stable Diffusion 1.5     — U-Net baseline (~3.5 GB VRAM)
  2. SDXL-Turbo               — Fast single-step (~6.5 GB VRAM, optimized)
  3. PixArt-Σ (Sigma)         — DiT-based, efficient (~2 GB w/ quant)
  4. Stable Diffusion 3 Medium — MMDiT architecture (~4 GB fp16)
  5. FLUX.1 Schnell            — Distilled for speed (~6 GB w/ offload)

All models share a unified ``ImageGenerationRequest`` /
``ImageGenerationResult`` interface and are managed through a
central ``ImageGenerationManager``.

The goal is that callers can swap backends by model id without having to
rewrite request validation, response handling, or output serialization.
"""

from bis.generation.image_gen.interface import (
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageModelInfo,
    ImageModelCapability,
)
from bis.generation.image_gen.manager import ImageGenerationManager
from bis.generation.image_gen.registry import ModelRegistry
from bis.generation.image_gen.upscale import upscale_image, UpscaleTarget, image_to_bytes

__all__ = [
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "ImageModelInfo",
    "ImageModelCapability",
    "ImageGenerationManager",
    "ModelRegistry",
    "upscale_image",
    "UpscaleTarget",
    "image_to_bytes",
]
