"""SDXL-Turbo — Fast single-step image generation adapter.

═══════════════════════════════════════════════════════════
Model:         stabilityai/sdxl-turbo
Architecture:  U-Net (2.6B params) + dual CLIP text encoders
               Adversarial Diffusion Distillation (ADD)
VRAM:          ~6.5 GB (fp16), ~4 GB with CPU offloading
Resolution:    512×512 native (designed for speed)
Steps:         1–4 (default 1, ADD-distilled)
Capabilities:  text-to-image, img2img, guidance scale,
               scheduler swap
═══════════════════════════════════════════════════════════

Performance Metrics:
 • FID-30K (COCO):  ~14 (1 step), ~9 (4 steps)
 • CLIP Score:       ~0.33
 • Inference:        ~0.5s on RTX 3090 (1 step!)
 • Throughput:       15–20 images/sec (batch, 1-step)

Compatibility:
 • Requires: diffusers ≥ 0.25, torch ≥ 2.0, transformers
 • Min VRAM: 4 GB (CPU offload), 6.5 GB (full GPU)
 • Scheduler: Must use EulerAncestralDiscreteScheduler for 1-step

Limitations:
 • Lower quality than full SDXL (distillation trade-off)
 • guidance_scale must be 0.0 for 1-step, >0 for multi-step
 • 512×512 only — larger resolutions not supported
 • No negative prompt support in 1-step mode
 • Research license (non-commercial, Stability AI SDXL Turbo License)
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


class SDXLTurboAdapter(BaseImageModelAdapter):
    """Adapter for SDXL-Turbo (Adversarial Diffusion Distillation)."""

    MODEL_INFO = ImageModelInfo(
        model_id="sdxl_turbo",
        display_name="SDXL-Turbo",
        huggingface_repo="stabilityai/sdxl-turbo",
        architecture="unet_add",
        parameter_count="2.6B",
        vram_estimate_gb=6.5,
        default_resolution=(512, 512),
        max_resolution=(512, 512),
        default_steps=1,
        supports_fp16=True,
        license="Stability AI SDXL Turbo License (research only)",
        description=(
            "SDXL-Turbo uses Adversarial Diffusion Distillation to "
            "generate high-quality images in a single forward pass. "
            "Extremely fast inference (~0.5s) at cost of some quality. "
            "Best for prototyping and real-time preview."
        ),
        capabilities=[
            ImageModelCapability.TEXT_TO_IMAGE,
            ImageModelCapability.IMAGE_TO_IMAGE,
            ImageModelCapability.GUIDANCE_SCALE,
        ],
        version="1.0.0",
    )

    def load(self, device: str = "auto", **kwargs) -> None:
        """Load the distilled SDXL-Turbo pipelines for fast inference."""
        from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

        device = self._resolve_device(device)
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("Loading %s on %s…", self.MODEL_INFO.display_name, device)

        self._pipe = AutoPipelineForText2Image.from_pretrained(
            self.MODEL_INFO.huggingface_repo,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )

        if device == "cuda":
            self._pipe = self._pipe.to(device)
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        else:
            self._pipe.enable_model_cpu_offload()

        # img2img variant
        self._img2img_pipe = AutoPipelineForImage2Image.from_pipe(self._pipe)

        self._device = device
        self._loaded = True
        logger.info("✓ %s loaded", self.MODEL_INFO.display_name)

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Generate with SDXL-Turbo while honoring its step-specific CFG rules."""
        if not self._loaded:
            return ImageGenerationResult(images=[], model_id=self.MODEL_INFO.model_id,
                                          error="Model not loaded")

        seed = self._resolve_seed(request.seed)
        generator = self._make_generator(seed, self._device)

        steps = request.num_inference_steps or self.MODEL_INFO.default_steps
        # SDXL-Turbo: guidance_scale must be 0.0 for 1-step
        guidance = 0.0 if steps <= 1 else (request.guidance_scale or 7.5)

        start = time.time()
        try:
            if request.init_image is not None:
                result = self._img2img_pipe(
                    prompt=request.prompt,
                    image=request.init_image.resize((512, 512)),
                    strength=request.strength,
                    num_inference_steps=max(steps, 2),
                    guidance_scale=guidance,
                    num_images_per_prompt=request.num_images,
                    generator=generator,
                )
            else:
                result = self._pipe(
                    prompt=request.prompt,
                    width=512,
                    height=512,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    num_images_per_prompt=request.num_images,
                    generator=generator,
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
                    "mode": "1-step ADD" if steps <= 1 else f"{steps}-step",
                    "resolution": "512x512",
                },
            )
        except Exception as e:
            return ImageGenerationResult(
                images=[], model_id=self.MODEL_INFO.model_id,
                error=f"Generation failed: {e}",
            )
