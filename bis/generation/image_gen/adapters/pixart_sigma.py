"""PixArt-Σ (Sigma) — Efficient DiT-based image generation adapter.

═══════════════════════════════════════════════════════════
Model:         PixArt-alpha/PixArt-Sigma-XL-2-512-MS
Architecture:  Diffusion Transformer (DiT) — 600M params
               T5-XXL text encoder (shared w/ offloading)
VRAM:          ~2 GB (8-bit T5 + fp16 DiT)
               ~6 GB (full fp16)
Resolution:    256–1024px, multi-scale trained
Steps:         20 (default), good at 14+
Capabilities:  text-to-image, negative prompt,
               guidance scale, scheduler swap
═══════════════════════════════════════════════════════════

Performance Metrics:
 • FID-30K (COCO):  ~6.3 (competitive with SDXL)
 • CLIP Score:       ~0.32
 • Inference:        ~3s on RTX 3090 (20 steps)
 • 4.6× smaller than SDXL, faster training

Compatibility:
 • Requires: diffusers ≥ 0.27, torch ≥ 2.0, transformers,
   sentencepiece
 • Min VRAM: 2 GB (8-bit T5 offloaded), 6 GB (full fp16)
 • T5 text encoder can be offloaded to CPU independently

Limitations:
 • Text rendering in images is poor (T5 doesn't help here)
 • Complex multi-object scenes can be inconsistent
 • Fewer community fine-tunes than SD 1.5 / SDXL
 • T5-XXL text encoder is large (~10 GB on disk)
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


class PixArtSigmaAdapter(BaseImageModelAdapter):
    """Adapter for PixArt-Σ Diffusion Transformer."""

    MODEL_INFO = ImageModelInfo(
        model_id="pixart_sigma",
        display_name="PixArt-Σ",
        huggingface_repo="PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        architecture="dit",
        parameter_count="600M",
        vram_estimate_gb=2.0,
        default_resolution=(512, 512),
        max_resolution=(1024, 1024),
        default_steps=20,
        supports_fp16=True,
        license="Apache 2.0",
        description=(
            "PixArt-Σ is a Diffusion Transformer (DiT) model that "
            "achieves quality comparable to SDXL with only 600M "
            "parameters. Uses T5-XXL for text understanding. Very "
            "memory-efficient with quantisation and CPU offloading."
        ),
        capabilities=[
            ImageModelCapability.TEXT_TO_IMAGE,
            ImageModelCapability.NEGATIVE_PROMPT,
            ImageModelCapability.GUIDANCE_SCALE,
            ImageModelCapability.SCHEDULER_SWAP,
        ],
        version="1.0.0",
    )

    def load(self, device: str = "auto", **kwargs) -> None:
        """Load PixArt-Sigma with optional 8-bit text-encoder support."""
        from diffusers import PixArtSigmaPipeline

        device = self._resolve_device(device)
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("Loading %s on %s…", self.MODEL_INFO.display_name, device)

        # Load with memory optimizations
        load_kwargs: dict = {
            "torch_dtype": dtype,
        }

        # Try loading T5 in 8-bit if bitsandbytes is available
        use_8bit = kwargs.get("use_8bit_text_encoder", False)
        if use_8bit:
            try:
                from transformers import T5EncoderModel
                text_encoder = T5EncoderModel.from_pretrained(
                    self.MODEL_INFO.huggingface_repo,
                    subfolder="text_encoder",
                    load_in_8bit=True,
                    device_map="auto",
                )
                load_kwargs["text_encoder"] = text_encoder
            except ImportError:
                logger.warning("bitsandbytes not available, using fp16 T5")

        self._pipe = PixArtSigmaPipeline.from_pretrained(
            self.MODEL_INFO.huggingface_repo,
            **load_kwargs,
        )

        if device == "cuda":
            if not use_8bit:
                self._pipe = self._pipe.to(device)
            self._pipe.enable_attention_slicing()
        else:
            self._pipe.enable_model_cpu_offload()

        self._device = device
        self._loaded = True
        logger.info("✓ %s loaded", self.MODEL_INFO.display_name)

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Run a PixArt inference pass and return timing plus scheduler info."""
        if not self._loaded:
            return ImageGenerationResult(images=[], model_id=self.MODEL_INFO.model_id,
                                          error="Model not loaded")

        seed = self._resolve_seed(request.seed)
        generator = self._make_generator(seed, self._device)

        steps = request.num_inference_steps or self.MODEL_INFO.default_steps
        guidance = request.guidance_scale or 4.5  # PixArt works well at ~4.5

        # Apply scheduler override
        if request.scheduler:
            self._apply_scheduler(request.scheduler)

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
                    "scheduler": type(self._pipe.scheduler).__name__,
                },
            )
        except Exception as e:
            return ImageGenerationResult(
                images=[], model_id=self.MODEL_INFO.model_id,
                error=f"Generation failed: {e}",
            )

    def _apply_scheduler(self, name: str) -> None:
        """Replace the active scheduler when the caller requests one."""
        import diffusers
        scheduler_map = {
            "ddim": "DDIMScheduler",
            "euler": "EulerDiscreteScheduler",
            "euler_a": "EulerAncestralDiscreteScheduler",
            "dpm++": "DPMSolverMultistepScheduler",
        }
        cls_name = scheduler_map.get(name)
        if cls_name and hasattr(diffusers, cls_name):
            sched_cls = getattr(diffusers, cls_name)
            self._pipe.scheduler = sched_cls.from_config(self._pipe.scheduler.config)
