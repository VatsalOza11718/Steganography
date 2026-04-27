"""Adapter exports for each supported image generation backend."""

from bis.generation.image_gen.adapters.sd15 import StableDiffusion15Adapter
from bis.generation.image_gen.adapters.sdxl_turbo import SDXLTurboAdapter
from bis.generation.image_gen.adapters.pixart_sigma import PixArtSigmaAdapter
from bis.generation.image_gen.adapters.sd3_medium import StableDiffusion3Adapter
from bis.generation.image_gen.adapters.flux_schnell import FluxSchnellAdapter

__all__ = [
    "StableDiffusion15Adapter",
    "SDXLTurboAdapter",
    "PixArtSigmaAdapter",
    "StableDiffusion3Adapter",
    "FluxSchnellAdapter",
]
