"""Stable Diffusion 1.5 — U-Net baseline adapter.

═══════════════════════════════════════════════════════════
Model:         stable-diffusion-v1-5/stable-diffusion-v1-5
Architecture:  U-Net (860M params) + CLIP text encoder
VRAM:          ~3.5 GB (fp16)
Resolution:    512×512 native, up to 768×768
Steps:         20–50 (default 25)
Capabilities:  text-to-image, img2img, negative prompt,
               guidance scale, scheduler swap
═══════════════════════════════════════════════════════════

Performance Metrics (from HF model card):
 • FID-30K (COCO):  ~8.5
 • CLIP Score:       ~0.31
 • Inference:        ~2s on RTX 3090 (25 steps)

Compatibility:
 • Requires: diffusers ≥ 0.25, torch ≥ 2.0, transformers
 • Optional: xformers for memory-efficient attention
 • Min VRAM: 3.5 GB (fp16), 2 GB with CPU offloading

Limitations:
 • Lower fidelity than SDXL/SD3 at high resolutions
 • Occasional anatomical errors in human figures
 • Limited understanding of spatial relationships
 • 512×512 native; resolutions > 768 may tile/degrade
"""

from __future__ import annotations

import logging
import time
from typing import Any

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


class StableDiffusion15Adapter(BaseImageModelAdapter):
    """Adapter for Stable Diffusion v1.5."""

    MODEL_INFO = ImageModelInfo(
        model_id="sd15",
        display_name="Stable Diffusion 1.5",
        huggingface_repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
        architecture="unet",
        parameter_count="860M",
        vram_estimate_gb=3.5,
        default_resolution=(512, 512),
        max_resolution=(768, 768),
        default_steps=25,
        supports_fp16=True,
        license="CreativeML Open RAIL-M",
        description=(
            "The original Stable Diffusion v1.5 model. Reliable baseline "
            "with extensive community fine-tunes and LoRA support. Best "
            "suited for 512×512 generation with moderate quality."
        ),
        capabilities=[
            ImageModelCapability.TEXT_TO_IMAGE,
            ImageModelCapability.IMAGE_TO_IMAGE,
            ImageModelCapability.NEGATIVE_PROMPT,
            ImageModelCapability.GUIDANCE_SCALE,
            ImageModelCapability.SCHEDULER_SWAP,
        ],
        version="1.5.0",
    )

    _SCHEDULER_MAP = {
        "ddim": "DDIMScheduler",
        "ddpm": "DDPMScheduler",
        "euler": "EulerDiscreteScheduler",
        "euler_a": "EulerAncestralDiscreteScheduler",
        "dpm++": "DPMSolverMultistepScheduler",
        "lms": "LMSDiscreteScheduler",
        "pndm": "PNDMScheduler",
    }

    def load(self, device: str = "auto", **kwargs) -> None:
        """Load the SD 1.5 text-to-image and img2img pipelines into memory."""
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

        device = self._resolve_device(device)
        dtype = torch.float16 if (device == "cuda" and self.MODEL_INFO.supports_fp16) else torch.float32

        logger.info("Loading %s on %s (dtype=%s)…", self.MODEL_INFO.display_name, device, dtype)

        self._pipe = StableDiffusionPipeline.from_pretrained(
            self.MODEL_INFO.huggingface_repo,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
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

        # Store img2img pipe reference
        self._img2img_pipe = StableDiffusionImg2ImgPipeline(**self._pipe.components)

        self._device = device
        self._loaded = True
        logger.info("✓ %s loaded", self.MODEL_INFO.display_name)

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Generate images with SD 1.5 and attach runtime metadata."""
        if not self._loaded:
            return ImageGenerationResult(images=[], model_id=self.MODEL_INFO.model_id,
                                          error="Model not loaded")

        seed = self._resolve_seed(request.seed)
        generator = self._make_generator(seed, self._device)

        steps = request.num_inference_steps or self.MODEL_INFO.default_steps
        guidance = request.guidance_scale or 7.5

        # Apply scheduler override
        if request.scheduler and request.scheduler in self._SCHEDULER_MAP:
            self._apply_scheduler(request.scheduler)

        start = time.time()
        try:
            if request.init_image is not None:
                # img2img
                result = self._img2img_pipe(
                    prompt=request.prompt,
                    image=request.init_image.resize((request.width, request.height)),
                    strength=request.strength,
                    negative_prompt=request.negative_prompt or None,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    num_images_per_prompt=request.num_images,
                    generator=generator,
                    **request.extra_kwargs,
                )
            else:
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
                    "scheduler": type(self._pipe.scheduler).__name__,
                    "resolution": f"{request.width}x{request.height}",
                },
            )
        except Exception as e:
            return ImageGenerationResult(
                images=[], model_id=self.MODEL_INFO.model_id,
                error=f"Generation failed: {e}",
            )

    def _apply_scheduler(self, name: str) -> None:
        """Swap the diffusion sampler without rebuilding the whole pipeline."""
        import diffusers
        cls_name = self._SCHEDULER_MAP.get(name)
        if cls_name and hasattr(diffusers, cls_name):
            sched_cls = getattr(diffusers, cls_name)
            self._pipe.scheduler = sched_cls.from_config(self._pipe.scheduler.config)
