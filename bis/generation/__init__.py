"""Generation helpers package.

The web app now uses deterministic local template generation for cover media.
This package remains for image-generation and fine-tuning support modules.
"""

from bis.generation.image_gen import (
    ImageGenerationManager,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageModelCapability,
    ImageModelInfo,
    ModelRegistry,
    UpscaleTarget,
    image_to_bytes,
    upscale_image,
)

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
