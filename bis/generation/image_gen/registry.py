"""Model Registry — discovers and registers all available image model adapters.

The registry is the single source of truth for which models exist,
their metadata, and how to instantiate them.  It supports both
built-in adapters and user-registered custom adapters.
"""

from __future__ import annotations

import logging
from typing import Type

from bis.generation.image_gen.interface import (
    BaseImageModelAdapter,
    ImageModelInfo,
    ImageModelCapability,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry mapping model IDs → adapter classes.

    Usage::

        reg = ModelRegistry()          # auto-discovers built-ins
        info = reg.list_models()       # all available
        adapter = reg.create("sd15")   # factory method
    """

    def __init__(self, auto_discover: bool = True) -> None:
        """Create an empty registry and optionally populate built-in adapters."""
        self._adapters: dict[str, Type[BaseImageModelAdapter]] = {}
        if auto_discover:
            self._discover_builtins()

    # ── Registration ─────────────────────────────────────────

    def register(self, adapter_cls: Type[BaseImageModelAdapter]) -> None:
        """Register an adapter class (built-in or custom)."""
        mid = adapter_cls.MODEL_INFO.model_id
        if mid in self._adapters:
            logger.warning("Overwriting existing adapter for '%s'", mid)
        self._adapters[mid] = adapter_cls
        logger.debug("Registered model adapter '%s' (%s)",
                     mid, adapter_cls.MODEL_INFO.display_name)

    def unregister(self, model_id: str) -> None:
        """Remove a registered adapter."""
        self._adapters.pop(model_id, None)

    # ── Discovery ────────────────────────────────────────────

    def _discover_builtins(self) -> None:
        """Import and register all built-in adapters."""
        from bis.generation.image_gen.adapters import (
            StableDiffusion15Adapter,
            SDXLTurboAdapter,
            PixArtSigmaAdapter,
            StableDiffusion3Adapter,
            FluxSchnellAdapter,
        )
        for cls in [
            StableDiffusion15Adapter,
            SDXLTurboAdapter,
            PixArtSigmaAdapter,
            StableDiffusion3Adapter,
            FluxSchnellAdapter,
        ]:
            self.register(cls)

    # ── Query ────────────────────────────────────────────────

    def list_models(self) -> list[ImageModelInfo]:
        """Return metadata for all registered models."""
        return [cls.MODEL_INFO for cls in self._adapters.values()]

    def list_model_ids(self) -> list[str]:
        """Return just the model IDs."""
        return list(self._adapters.keys())

    def get_info(self, model_id: str) -> ImageModelInfo | None:
        """Get model info or None."""
        cls = self._adapters.get(model_id)
        return cls.MODEL_INFO if cls else None

    def has_model(self, model_id: str) -> bool:
        """Return ``True`` when ``model_id`` is present in the registry."""
        return model_id in self._adapters

    # ── Factory ──────────────────────────────────────────────

    def create(self, model_id: str) -> BaseImageModelAdapter:
        """Instantiate an adapter (unloaded) by model ID.

        Raises
        ------
        KeyError
            If *model_id* is not registered.
        """
        cls = self._adapters.get(model_id)
        if cls is None:
            available = ", ".join(self._adapters.keys())
            raise KeyError(
                f"Model '{model_id}' not in registry. "
                f"Available: {available}"
            )
        return cls()

    # ── Filtering ────────────────────────────────────────────

    def filter_by_capability(
        self, capability: ImageModelCapability,
    ) -> list[ImageModelInfo]:
        """Return models that support a given capability."""
        return [
            cls.MODEL_INFO
            for cls in self._adapters.values()
            if capability in cls.MODEL_INFO.capabilities
        ]

    def filter_by_vram(self, max_vram_gb: float) -> list[ImageModelInfo]:
        """Return models that fit within a VRAM budget."""
        return [
            cls.MODEL_INFO
            for cls in self._adapters.values()
            if cls.MODEL_INFO.vram_estimate_gb <= max_vram_gb
        ]

    def recommend(
        self,
        max_vram_gb: float = 8.0,
        prefer_speed: bool = False,
        prefer_quality: bool = False,
    ) -> ImageModelInfo | None:
        """Recommend the best model for the given constraints.

        Parameters
        ----------
        max_vram_gb : float
            Available GPU memory.
        prefer_speed : bool
            Prefer faster inference (fewer steps).
        prefer_quality : bool
            Prefer higher quality output.

        Returns
        -------
        ImageModelInfo or None
        """
        candidates = self.filter_by_vram(max_vram_gb)
        if not candidates:
            return None

        if prefer_speed:
            # Sort by default steps (lower = faster)
            candidates.sort(key=lambda m: m.default_steps)
        elif prefer_quality:
            # Prefer larger / newer architectures
            _arch_score = {"rectified_flow_transformer": 4, "mmdit": 3, "dit": 2, "unet_add": 1, "unet": 0}
            candidates.sort(
                key=lambda m: _arch_score.get(m.architecture, 0),
                reverse=True,
            )
        else:
            # Balance: sort by VRAM (smaller = better for coexistence)
            candidates.sort(key=lambda m: m.vram_estimate_gb)

        return candidates[0]

    # ── Comparison ───────────────────────────────────────────

    def compare(self, *model_ids: str) -> list[dict]:
        """Side-by-side comparison of listed models."""
        rows = []
        for mid in model_ids:
            info = self.get_info(mid)
            if info is None:
                continue
            rows.append({
                "model_id": info.model_id,
                "name": info.display_name,
                "architecture": info.architecture,
                "params": info.parameter_count,
                "vram_gb": info.vram_estimate_gb,
                "resolution": f"{info.default_resolution[0]}×{info.default_resolution[1]}",
                "steps": info.default_steps,
                "license": info.license,
                "capabilities": [c.value for c in info.capabilities],
            })
        return rows
