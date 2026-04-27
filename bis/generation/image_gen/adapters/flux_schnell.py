"""FLUX.1 Schnell — Fast distilled image generation adapter.

═══════════════════════════════════════════════════════════
Model:         black-forest-labs/FLUX.1-schnell
Architecture:  Rectified Flow Transformer (~12B params),
               distilled for 1–4 step inference
VRAM:          ~6 GB (fp16, CPU offloaded text encoder)
               ~23 GB (full fp16, no offloading)
Resolution:    256–1024px (best at 1024×1024)
Steps:         4 (default, distilled)
Capabilities:  text-to-image, guidance scale
═══════════════════════════════════════════════════════════

Performance Metrics:
 • Quality:      Competitive with DALL-E 3 and Midjourney
 • Inference:    ~2s on RTX 4090 (4 steps)
 • Architecture: Rectified Flow matching (state-of-the-art)

Compatibility:
 • Requires: diffusers ≥ 0.30, torch ≥ 2.1, transformers ≥ 4.40,
   sentencepiece
 • Min VRAM: ~6 GB with aggressive CPU offloading
 • T5-XXL text encoder can be pre-computed and offloaded
 • Apache 2.0 license (commercial use OK)

Limitations:
 • Very large model (~23 GB full, needs offloading for ≤8 GB GPU)
 • CPU offloading adds ~5–10s overhead per image
 • No negative prompt support (rectified flow doesn't use CFG)
 • Requires significant disk space for weights (~24 GB)
 • Image-to-image not natively supported in schnell variant
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


class FluxSchnellAdapter(BaseImageModelAdapter):
    """Adapter for FLUX.1-schnell (distilled rectified flow)."""

    MODEL_INFO = ImageModelInfo(
        model_id="flux_schnell",
        display_name="FLUX.1 Schnell",
        huggingface_repo="black-forest-labs/FLUX.1-schnell",
        architecture="rectified_flow_transformer",
        parameter_count="12B",
        vram_estimate_gb=6.0,
        default_resolution=(1024, 1024),
        max_resolution=(1024, 1024),
        default_steps=4,
        supports_fp16=True,
        license="Apache 2.0",
        description=(
            "FLUX.1-schnell is a distilled variant of FLUX.1 using "
            "rectified flow transformers. Achieves state-of-the-art "
            "quality in just 4 steps. Requires aggressive CPU offloading "
            "to fit in ≤8 GB VRAM. Apache 2.0 licensed."
        ),
        capabilities=[
            ImageModelCapability.TEXT_TO_IMAGE,
            ImageModelCapability.GUIDANCE_SCALE,
        ],
        version="1.0.0",
    )

    def load(self, device: str = "auto", **kwargs) -> None:
        """Load FLUX Schnell with the offloading strategy needed on small GPUs."""
        from diffusers import FluxPipeline

        device = self._resolve_device(device)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        logger.info("Loading %s on %s (this is a large model)…",
                     self.MODEL_INFO.display_name, device)

        self._pipe = FluxPipeline.from_pretrained(
            self.MODEL_INFO.huggingface_repo,
            torch_dtype=dtype,
        )

        # FLUX.1 is ~23 GB fp16 — MUST use CPU offloading for ≤8 GB
        if device == "cuda":
            self._pipe.enable_sequential_cpu_offload()
            self._pipe.enable_attention_slicing(slice_size=1)
        else:
            self._pipe.enable_model_cpu_offload()

        # VAE slicing for memory efficiency
        self._pipe.vae.enable_slicing()
        self._pipe.vae.enable_tiling()

        self._device = device
        self._loaded = True
        logger.info("✓ %s loaded (CPU offloading enabled)", self.MODEL_INFO.display_name)

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Run a FLUX inference pass and package useful diagnostic metadata."""
        if not self._loaded:
            return ImageGenerationResult(images=[], model_id=self.MODEL_INFO.model_id,
                                          error="Model not loaded")

        seed = self._resolve_seed(request.seed)
        generator = self._make_generator(seed, self._device)

        steps = request.num_inference_steps or self.MODEL_INFO.default_steps
        guidance = request.guidance_scale or 0.0  # Schnell uses CFG-free, guidance=0

        start = time.time()
        try:
            result = self._pipe(
                prompt=request.prompt,
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
                    "architecture": "rectified_flow",
                    "note": "FLUX.1 Schnell, CFG-free distilled",
                },
            )
        except Exception as e:
            return ImageGenerationResult(
                images=[], model_id=self.MODEL_INFO.model_id,
                error=f"Generation failed: {e}",
            )
