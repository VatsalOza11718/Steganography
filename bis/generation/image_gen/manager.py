"""Image Generation Manager — unified orchestration layer.

Provides seamless switching between image generation models while
handling VRAM management, error recovery, model versioning, and
performance monitoring.

Usage::

    mgr = ImageGenerationManager()
    mgr.load_model("sd15")  # or "sdxl_turbo", "pixart_sigma", etc.

    result = mgr.generate(ImageGenerationRequest(
        prompt="A photo of a sunset over mountains",
        width=512, height=512,
    ))

    for img in result.images:
        img.save("output.png")

    mgr.switch_model("pixart_sigma")   # hot-swap models
    result2 = mgr.generate(...)        # uses PixArt now
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from bis.generation.image_gen.interface import (
    BaseImageModelAdapter,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageModelCapability,
    ImageModelInfo,
)
from bis.generation.image_gen.registry import ModelRegistry

logger = logging.getLogger(__name__)


# ── Performance Tracking ─────────────────────────────────────

@dataclass
class ModelPerformanceStats:
    """Tracks generation statistics for a single model."""
    model_id: str
    total_generations: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    total_time_sec: float = 0.0
    total_images: int = 0
    oom_count: int = 0
    last_error: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Return the fraction of generations that finished successfully."""
        if self.total_generations == 0:
            return 0.0
        return self.successful_generations / self.total_generations

    @property
    def avg_time_sec(self) -> float:
        """Average runtime of successful generations for this model."""
        if self.successful_generations == 0:
            return 0.0
        return self.total_time_sec / self.successful_generations

    def to_dict(self) -> dict:
        """Return a compact JSON-friendly snapshot of performance counters."""
        return {
            "model_id": self.model_id,
            "total_generations": self.total_generations,
            "successful": self.successful_generations,
            "failed": self.failed_generations,
            "oom_count": self.oom_count,
            "success_rate": round(self.success_rate, 3),
            "avg_time_sec": round(self.avg_time_sec, 2),
            "total_images": self.total_images,
        }


@dataclass
class PerformanceMonitor:
    """Aggregates stats across all models."""
    stats: dict[str, ModelPerformanceStats] = field(default_factory=dict)
    generation_log: list[dict] = field(default_factory=list)
    _max_log_entries: int = 1000

    def record(self, result: ImageGenerationResult) -> None:
        """Update aggregate counters and append one lightweight log entry."""
        mid = result.model_id
        if mid not in self.stats:
            self.stats[mid] = ModelPerformanceStats(model_id=mid)
        s = self.stats[mid]

        s.total_generations += 1
        if result.success:
            s.successful_generations += 1
            s.total_time_sec += result.generation_time_sec
            s.total_images += len(result.images)
        else:
            s.failed_generations += 1
            s.last_error = result.error
            if result.error and "CUDA out of memory" in result.error:
                s.oom_count += 1

        entry = {
            "model_id": mid,
            "success": result.success,
            "time_sec": round(result.generation_time_sec, 3),
            "num_images": len(result.images),
            "seed": result.seed_used,
            "timestamp": time.time(),
        }
        self.generation_log.append(entry)
        if len(self.generation_log) > self._max_log_entries:
            self.generation_log = self.generation_log[-self._max_log_entries:]

    def summary(self) -> dict:
        """Return the current per-model performance summary."""
        return {mid: s.to_dict() for mid, s in self.stats.items()}

    def export_json(self, path: str | Path) -> None:
        """Write the current summary to disk for offline inspection."""
        Path(path).write_text(json.dumps(self.summary(), indent=2))


# ── Version Tracker ──────────────────────────────────────────

@dataclass
class ModelVersion:
    """Tracks loaded model version and configuration."""
    model_id: str
    version: str
    load_time: float
    config: dict[str, Any] = field(default_factory=dict)


# ── Main Manager ─────────────────────────────────────────────

class ImageGenerationManager:
    """Unified manager for switching between image generation models.

    Features:
     • Seamless hot-swapping between models
     • VRAM-aware model loading and unloading
     • Automatic OOM recovery with model fallback
     • Performance monitoring and statistics
     • Model versioning and configuration tracking
     • Consistent input/output format across all backends
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        max_vram_gb: float = 7.5,
        auto_fallback: bool = True,
        output_dir: str | Path | None = None,
    ) -> None:
        """Create the registry, runtime options, and performance monitor."""
        self._registry = registry or ModelRegistry()
        self._max_vram_gb = max_vram_gb
        self._auto_fallback = auto_fallback
        self._output_dir = Path(output_dir) if output_dir else None

        self._active_adapter: BaseImageModelAdapter | None = None
        self._active_model_id: str | None = None
        self._version_history: list[ModelVersion] = []

        self._monitor = PerformanceMonitor()

    # ── Properties ───────────────────────────────────────────

    @property
    def active_model(self) -> str | None:
        """Currently loaded model ID."""
        return self._active_model_id

    @property
    def active_info(self) -> ImageModelInfo | None:
        """Metadata for the currently loaded model."""
        if self._active_adapter:
            return self._active_adapter.info
        return None

    @property
    def is_loaded(self) -> bool:
        """Report whether an active adapter exists and is ready to serve."""
        return self._active_adapter is not None and self._active_adapter.is_loaded

    @property
    def monitor(self) -> PerformanceMonitor:
        """Expose the statistics collector associated with this manager."""
        return self._monitor

    @property
    def registry(self) -> ModelRegistry:
        """Return the adapter registry that knows which backends exist."""
        return self._registry

    # ── Model Lifecycle ──────────────────────────────────────

    def load_model(self, model_id: str, device: str = "auto", **kwargs) -> None:
        """Load a model by ID, unloading any currently active model.

        Parameters
        ----------
        model_id : str
            One of the registered model IDs (e.g. "sd15", "pixart_sigma").
        device : str
            "auto", "cuda", or "cpu".
        **kwargs
            Additional arguments forwarded to the adapter's ``load()``
            (e.g. ``drop_t5=True``, ``use_8bit_text_encoder=True``).
        """
        if self._active_model_id == model_id and self.is_loaded:
            logger.info("Model '%s' already loaded", model_id)
            return

        # Unload current model first
        self.unload()

        logger.info("Loading image model '%s'…", model_id)
        adapter = self._registry.create(model_id)

        try:
            adapter.load(device=device, **kwargs)
        except Exception as e:
            logger.error("Failed to load '%s': %s", model_id, e)
            raise RuntimeError(f"Failed to load model '{model_id}': {e}") from e

        self._active_adapter = adapter
        self._active_model_id = model_id
        self._version_history.append(ModelVersion(
            model_id=model_id,
            version=adapter.info.version,
            load_time=time.time(),
            config=kwargs,
        ))
        logger.info("✓ Active image model: %s", adapter.info.display_name)

    def unload(self) -> None:
        """Unload the active model and free resources."""
        if self._active_adapter is not None:
            logger.info("Unloading %s…", self._active_model_id)
            self._active_adapter.unload()
            self._active_adapter = None
            self._active_model_id = None

    def switch_model(self, model_id: str, device: str = "auto", **kwargs) -> None:
        """Convenience alias: unload current + load new model."""
        self.load_model(model_id, device=device, **kwargs)

    # ── Generation ───────────────────────────────────────────

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Generate image(s) using the active model.

        If ``auto_fallback`` is enabled and the generation fails due to
        OOM, automatically retries with the smallest available model.

        Parameters
        ----------
        request : ImageGenerationRequest
            Unified generation request.

        Returns
        -------
        ImageGenerationResult
            Contains images, metadata, timing, and any errors.
        """
        if not self.is_loaded:
            return ImageGenerationResult(
                images=[], model_id="none",
                error="No model loaded. Call load_model() first.",
            )

        result = self._try_generate(request)
        self._monitor.record(result)

        # Auto-fallback on OOM
        if (
            not result.success
            and self._auto_fallback
            and result.error
            and "CUDA out of memory" in result.error
        ):
            logger.warning("OOM with %s, attempting fallback…", self._active_model_id)
            result = self._oom_fallback(request)

        # Save output if output_dir is set
        if result.success and self._output_dir:
            self._save_outputs(result)

        return result

    def generate_with_model(
        self, model_id: str, request: ImageGenerationRequest,
        device: str = "auto", **load_kwargs,
    ) -> ImageGenerationResult:
        """Load a specific model, generate, return result.

        Convenience method that loads the model if it's not already
        active, generates, and returns. Does NOT unload afterwards —
        the model stays loaded for further use.
        """
        if self._active_model_id != model_id:
            self.load_model(model_id, device=device, **load_kwargs)
        return self.generate(request)

    def batch_generate(
        self, requests: list[ImageGenerationRequest],
    ) -> list[ImageGenerationResult]:
        """Generate multiple images sequentially."""
        results = []
        for i, req in enumerate(requests):
            logger.info("Batch generation %d/%d", i + 1, len(requests))
            results.append(self.generate(req))
        return results

    # ── Model Comparison ─────────────────────────────────────

    def compare_models(
        self,
        request: ImageGenerationRequest,
        model_ids: list[str] | None = None,
        device: str = "auto",
    ) -> dict[str, ImageGenerationResult]:
        """Generate the same prompt with multiple models for comparison.

        Parameters
        ----------
        request : ImageGenerationRequest
            Must have a fixed seed for fair comparison.
        model_ids : list[str] or None
            Models to compare. None = use all that fit VRAM.
        device : str
            Device for loading.

        Returns
        -------
        dict  mapping model_id → ImageGenerationResult
        """
        if model_ids is None:
            candidates = self._registry.filter_by_vram(self._max_vram_gb)
            model_ids = [m.model_id for m in candidates]

        results = {}
        original_model = self._active_model_id

        for mid in model_ids:
            logger.info("Comparing model: %s", mid)
            try:
                self.load_model(mid, device=device)
                result = self.generate(request)
                results[mid] = result
            except Exception as e:
                results[mid] = ImageGenerationResult(
                    images=[], model_id=mid, error=str(e),
                )
            finally:
                self.unload()

        # Restore original model if it was loaded
        if original_model:
            try:
                self.load_model(original_model, device=device)
            except Exception:
                pass

        return results

    # ── Information ──────────────────────────────────────────

    def list_available_models(self) -> list[ImageModelInfo]:
        """Return info for all registered models."""
        return self._registry.list_models()

    def recommend_model(
        self,
        prefer_speed: bool = False,
        prefer_quality: bool = False,
    ) -> ImageModelInfo | None:
        """Get a model recommendation based on VRAM budget."""
        return self._registry.recommend(
            max_vram_gb=self._max_vram_gb,
            prefer_speed=prefer_speed,
            prefer_quality=prefer_quality,
        )

    def status(self) -> dict:
        """Return a JSON-serializable status dict."""
        return {
            "active_model": self._active_model_id,
            "is_loaded": self.is_loaded,
            "max_vram_gb": self._max_vram_gb,
            "auto_fallback": self._auto_fallback,
            "available_models": self._registry.list_model_ids(),
            "performance": self._monitor.summary(),
            "version_history": [
                {
                    "model_id": v.model_id,
                    "version": v.version,
                    "load_time": v.load_time,
                }
                for v in self._version_history
            ],
        }

    # ── Private helpers ──────────────────────────────────────

    def _try_generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Single generation attempt with error wrapping."""
        try:
            return self._active_adapter.generate(request)
        except torch.cuda.OutOfMemoryError as e:
            self._gc()
            return ImageGenerationResult(
                images=[], model_id=self._active_model_id,
                error=f"CUDA out of memory: {e}",
            )
        except Exception as e:
            return ImageGenerationResult(
                images=[], model_id=self._active_model_id,
                error=f"Unexpected error: {e}",
            )

    def _oom_fallback(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """On OOM, switch to the smallest model and retry."""
        self.unload()
        self._gc()

        # Find smallest model
        smallest = self._registry.filter_by_vram(self._max_vram_gb)
        if not smallest:
            return ImageGenerationResult(
                images=[], model_id="none",
                error="No models fit within VRAM budget after OOM.",
            )

        smallest.sort(key=lambda m: m.vram_estimate_gb)
        fallback = smallest[0]

        logger.info("Falling back to %s (%.1f GB)", fallback.model_id, fallback.vram_estimate_gb)
        try:
            self.load_model(fallback.model_id)
            result = self._try_generate(request)
            result.metadata["fallback"] = True
            result.metadata["original_model"] = self._active_model_id
            self._monitor.record(result)
            return result
        except Exception as e:
            return ImageGenerationResult(
                images=[], model_id=fallback.model_id,
                error=f"Fallback also failed: {e}",
            )

    def _save_outputs(self, result: ImageGenerationResult) -> None:
        """Auto-save generated images to output directory."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        for i, img in enumerate(result.images):
            fname = f"{result.model_id}_{ts}_{result.seed_used}_{i}.png"
            img.save(self._output_dir / fname)

    @staticmethod
    def _gc() -> None:
        """Run host and CUDA cleanup after unloads or failed generations."""
        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
