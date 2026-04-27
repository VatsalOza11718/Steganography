"""Unified interface types for all image generation models.

Every model adapter conforms to these types so callers never
need to know which backend is running.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch
from PIL import Image


# ── Capability Flags ─────────────────────────────────────────

class ImageModelCapability(Enum):
    """Feature flags for what a model supports."""
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    INPAINTING = "inpainting"
    CONTROLNET = "controlnet"
    UPSCALING = "upscaling"
    NEGATIVE_PROMPT = "negative_prompt"
    GUIDANCE_SCALE = "guidance_scale"
    SCHEDULER_SWAP = "scheduler_swap"


# ── Model Metadata ───────────────────────────────────────────

@dataclass
class ImageModelInfo:
    """Static metadata describing a model backend.

    Populated once during registration and never changes.
    """
    model_id: str                                    # e.g. "sd15", "sdxl_turbo"
    display_name: str                                # human-friendly
    huggingface_repo: str                            # e.g. "stable-diffusion-v1-5/…"
    architecture: str                                # "unet", "dit", "mmdit"
    parameter_count: str                             # e.g. "860M"
    vram_estimate_gb: float                          # peak GPU memory in fp16
    default_resolution: tuple[int, int] = (512, 512)
    max_resolution: tuple[int, int] = (1024, 1024)
    default_steps: int = 20
    supports_fp16: bool = True
    license: str = ""
    description: str = ""
    capabilities: list[ImageModelCapability] = field(default_factory=list)
    version: str = "1.0.0"


# ── Request / Result ─────────────────────────────────────────

@dataclass
class ImageGenerationRequest:
    """Uniform input for every image model.

    Fields that a specific backend does not support are silently
    ignored — never raises.
    """
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 0       # 0 → use model default
    guidance_scale: float = 0.0        # 0 → use model default
    seed: int = -1                     # -1 → random
    num_images: int = 1
    # Image-to-image
    init_image: Optional[Image.Image] = None
    strength: float = 0.75
    # Scheduler override
    scheduler: Optional[str] = None    # e.g. "ddim", "euler", "euler_a"
    # Extra keyword args forwarded to the pipeline
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Clamp user input to safe ranges before it reaches a backend."""
        # This catches obviously invalid requests early and keeps adapters
        # focused on model-specific work rather than basic input hygiene.
        self.width = max(64, min(self.width, 2048))
        self.height = max(64, min(self.height, 2048))
        self.num_images = max(1, min(self.num_images, 8))
        self.strength = max(0.0, min(self.strength, 1.0))


@dataclass
class ImageGenerationResult:
    """Uniform output from every image model."""
    images: list[Image.Image]
    model_id: str
    generation_time_sec: float = 0.0
    seed_used: int = -1
    # Quality / diagnostic metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Report whether generation returned images without an error string."""
        return self.error is None and len(self.images) > 0


# ── Base Adapter ─────────────────────────────────────────────

class BaseImageModelAdapter:
    """Abstract base class for model adapters.

    Each concrete adapter wraps exactly one Hugging Face model,
    implementing load/unload/generate on top of this interface.
    """

    # Subclasses MUST set this
    MODEL_INFO: ImageModelInfo

    def __init__(self) -> None:
        """Initialize common adapter state shared by all model backends."""
        self._loaded = False
        self._device: str = "cpu"
        self._pipe: Any = None

    # ── Lifecycle ────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """Expose whether the underlying pipeline has completed loading."""
        return self._loaded

    @property
    def info(self) -> ImageModelInfo:
        """Return the static metadata declared on the adapter class."""
        return self.MODEL_INFO

    def load(self, device: str = "auto", **kwargs) -> None:
        """Download weights (if needed) and move to *device*."""
        raise NotImplementedError

    def unload(self) -> None:
        """Move model to CPU and release VRAM."""
        if self._pipe is not None:
            if hasattr(self._pipe, "to"):
                try:
                    self._pipe.to("cpu")
                except Exception:
                    pass
            del self._pipe
            self._pipe = None
        self._loaded = False
        self._gc()

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Produce image(s) from *request*."""
        raise NotImplementedError

    # ── Helpers ──────────────────────────────────────────

    def _resolve_device(self, device: str) -> str:
        """Convert ``auto`` into a concrete Torch device string."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _resolve_seed(self, seed: int) -> int:
        """Turn ``-1`` into a random seed while preserving explicit seeds."""
        if seed < 0:
            return torch.randint(0, 2**31, (1,)).item()
        return seed

    def _make_generator(self, seed: int, device: str) -> torch.Generator:
        """Build a deterministic Torch generator for reproducible results."""
        gen = torch.Generator(device=device if device != "cuda" else "cpu")
        gen.manual_seed(seed)
        return gen

    @staticmethod
    def _gc() -> None:
        """Release Python objects and ask CUDA to drop cached tensors."""
        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
