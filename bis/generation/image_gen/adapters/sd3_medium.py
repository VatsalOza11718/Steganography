"""Stable Diffusion 3 Medium — MMDiT architecture adapter.

═══════════════════════════════════════════════════════════
Model:         stabilityai/stable-diffusion-3-medium-diffusers
Architecture:  Multi-Modal Diffusion Transformer (MMDiT)
               2B params, triple text encoder (CLIP L/14,
               OpenCLIP G/14, T5-XXL)
VRAM:          ~4 GB (fp16, T5 dropped), ~11 GB full
Resolution:    1024×1024 native
Steps:         28 (default)
Capabilities:  text-to-image, img2img, negative prompt,
               guidance scale, scheduler swap
═══════════════════════════════════════════════════════════

Performance Metrics:
 • FID (internal):  Competitive with DALL-E 3
 • Human Eval:      Preferred over SDXL in blind tests
 • Text rendering:  Significantly improved over SD 1.5/SDXL
 • Inference:       ~5s on RTX 3090 (28 steps)

Compatibility:
 • Requires: diffusers ≥ 0.29, torch ≥ 2.1, transformers ≥ 4.38
 • Min VRAM: 4 GB (fp16, T5 dropped), 11 GB (full)
 • Can drop T5 encoder (10 GB) with minor quality loss
 • Gated model — requires HF token with access granted

Limitations:
 • Gated access (requires accepting license on HF)
 • Full model with T5 requires ~11 GB VRAM
 • Slower inference than SDXL-Turbo
 • Non-commercial license (Stability AI Community License)
 • May produce artifacts at non-square aspect ratios
"""

from __future__ import annotations

import logging
import time

import torch
from PIL import Image

from bis.generation.image_gen.interface import (
    BaseImageModelAdapter,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageModelCapability,
    ImageModelInfo,
)

logger = logging.getLogger(__name__)


class StableDiffusion3Adapter(BaseImageModelAdapter):
    """Adapter for Stable Diffusion 3 Medium (MMDiT)."""

    MODEL_INFO = ImageModelInfo(
        model_id="sd3_medium",
        display_name="Stable Diffusion 3 Medium",
        huggingface_repo="stabilityai/stable-diffusion-3-medium-diffusers",
        architecture="mmdit",
        parameter_count="2B",
        vram_estimate_gb=4.0,
        default_resolution=(1024, 1024),
        max_resolution=(1024, 1024),
        default_steps=28,
        supports_fp16=True,
        license="Stability AI Community License",
        description=(
            "SD3 Medium uses a novel Multi-Modal Diffusion Transformer "
            "(MMDiT) with flow matching. Triple text encoder (two CLIPs "
            "+ T5-XXL) delivers superior prompt understanding and text "
            "rendering. T5 can be dropped to reduce VRAM to ~4 GB."
        ),
        capabilities=[
            ImageModelCapability.TEXT_TO_IMAGE,
            ImageModelCapability.IMAGE_TO_IMAGE,
            ImageModelCapability.NEGATIVE_PROMPT,
            ImageModelCapability.GUIDANCE_SCALE,
            ImageModelCapability.SCHEDULER_SWAP,
        ],
        version="3.0.0",
    )

    def load(self, device: str = "auto", **kwargs) -> None:
        """Load SD3 Medium, optionally dropping the T5 encoder to save VRAM."""
        from diffusers import StableDiffusion3Pipeline

        device = self._resolve_device(device)
        dtype = torch.float16 if device == "cuda" else torch.float32
        drop_t5 = kwargs.get("drop_t5", True)  # Default: drop T5 for VRAM savings

        logger.info(
            "Loading %s on %s (drop_t5=%s)…",
            self.MODEL_INFO.display_name, device, drop_t5,
        )

        load_kwargs: dict = {"torch_dtype": dtype}
        if drop_t5:
            # Pass None for T5 text encoder to skip loading it
            load_kwargs["text_encoder_3"] = None
            load_kwargs["tokenizer_3"] = None

        self._pipe = StableDiffusion3Pipeline.from_pretrained(
            self.MODEL_INFO.huggingface_repo,
            **load_kwargs,
        )

        if device == "cuda":
            self._pipe = self._pipe.to(device)
            self._pipe.enable_attention_slicing()
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        else:
            self._pipe.enable_model_cpu_offload()

        self._device = device
        self._loaded = True
        self._t5_dropped = drop_t5
        logger.info("✓ %s loaded (T5 %s)",
                     self.MODEL_INFO.display_name,
                     "dropped" if drop_t5 else "included")

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Generate with SD3 and report whether the optional T5 encoder was used."""
        if not self._loaded:
            return ImageGenerationResult(images=[], model_id=self.MODEL_INFO.model_id,
                                          error="Model not loaded")

        seed = self._resolve_seed(request.seed)
        generator = self._make_generator(seed, self._device)

        steps = request.num_inference_steps or self.MODEL_INFO.default_steps
        guidance = request.guidance_scale or 7.0

        start = time.time()
        try:
            result = self._pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                width=request.width,
                height=request.height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                num_images_per_prompt=request.num_images,
                generator=generator,
                **request.extra_kwargs,
            )
            elapsed = time.time() - start

            return ImageGenerationResult(
                images=result.images,
                model_id=self.MODEL_INFO.model_id,
                generation_time_sec=elapsed,
                seed_used=seed,
                metadata={
                    "steps": steps,
                    "guidance_scale": guidance,
                    "resolution": f"{request.width}x{request.height}",
                    "t5_encoder": "dropped" if self._t5_dropped else "active",
                    "scheduler": type(self._pipe.scheduler).__name__,
                },
            )
        except Exception as e:
            return ImageGenerationResult(
                images=[], model_id=self.MODEL_INFO.model_id,
                error=f"Generation failed: {e}",
            )
