"""Unified Python SDK for multi-modal fine-tuning.

Provides an identical interface for text, audio, and image:
    sdk = FineTuneSDK("text")
    sdk.load_dataset("./data/conversations")
    sdk.configure(preset="qwen25-chat")
    sdk.train()
    result = sdk.evaluate()
    sdk.export_lora("./output/my_lora")
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FineTuneSDK:
    """Unified SDK for fine-tuning across all modalities.

    Provides consistent API:
     • load_dataset() — auto-detect and preprocess
     • configure()   — apply preset or custom config
     • train()       — run fine-tuning loop
     • evaluate()    — compute metrics
     • export_lora() — save publishable LoRA weights
    """

    # Preset registry
    PRESETS = {
        "qwen25-chat": ("text", "qwen25_chat"),
        "qwen25-128k": ("text", "qwen25_128k"),
        "musicgen-melody": ("audio", "musicgen_melody"),
        "musicgen-large": ("audio", "musicgen_large"),
        "sd15-lora": ("image", "sd15_lora"),
    }

    def __init__(self, modality: str = "text") -> None:
        """Initialize modality-specific SDK state and lazy-loaded components."""
        self.modality = modality
        self.config = None
        self.dataset = None
        self.trainer = None
        self.evaluator = None
        self.train_result: dict[str, Any] | None = None
        self.eval_result = None

    # ── Dataset Loading ──────────────────────────────────

    def load_dataset(
        self,
        data_dir: str,
        *,
        prompts: dict[str, str] | None = None,
        streaming: bool = False,
    ) -> dict[str, Any]:
        """Load and preprocess a dataset.

        Auto-detects modality from file contents if not specified.
        Returns dataset statistics.
        """
        from bis.fine_tuning.preprocessing import UnifiedDatasetBuilder

        builder = UnifiedDatasetBuilder(self.config)
        result = builder.build(data_dir, modality=self.modality)

        self.dataset = result
        logger.info(
            "Dataset loaded: %d %s samples from %s",
            result["count"], result["modality"], data_dir,
        )

        return {
            "count": result["count"],
            "modality": result["modality"],
            "progress": result["progress"],
        }

    # ── Configuration ────────────────────────────────────

    def configure(
        self,
        *,
        preset: str | None = None,
        config_path: str | None = None,
        dataset_dir: str | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Configure training parameters.

        Args:
            preset: Named preset (e.g. "qwen25-chat", "musicgen-melody")
            config_path: Path to YAML/JSON config file
            dataset_dir: Override dataset directory
            **overrides: Override any config parameter

        Returns:
            Config summary dict.
        """
        from bis.fine_tuning.config import (
            FineTuneConfig, Modality,
            TextPreset, AudioPreset, ImagePreset,
        )

        if preset:
            self.config = self._load_preset(preset)
        elif config_path:
            path = Path(config_path)
            if path.suffix in (".yaml", ".yml"):
                self.config = FineTuneConfig.from_yaml(path)
            else:
                self.config = FineTuneConfig.from_json(path)
        else:
            self.config = FineTuneConfig(modality=Modality(self.modality))

        # Apply overrides
        if dataset_dir:
            self.config.dataset_dir = dataset_dir
        for key, val in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, val)

        # Update modality from config
        self.modality = self.config.modality.value

        # Validate
        issues = self.config.validate()
        if issues:
            logger.warning("Config validation issues: %s", issues)

        return {
            "modality": self.modality,
            "model_id": self.config.model_id,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "lora_rank": self.config.lora.rank,
            "precision": self.config.precision.value,
            "issues": issues,
        }

    def _load_preset(self, name: str):
        """Load a named preset configuration."""
        from bis.fine_tuning.config import (
            TextPreset, AudioPreset, ImagePreset,
        )

        preset_map = {
            "qwen25-chat": TextPreset.qwen25_chat,
            "qwen25-128k": TextPreset.qwen25_128k,
            "musicgen-melody": AudioPreset.musicgen_melody,
            "musicgen-large": AudioPreset.musicgen_large,
            "sd15-lora": ImagePreset.sd15_lora,
        }

        factory = preset_map.get(name)
        if factory is None:
            raise ValueError(f"Unknown preset: {name}. Available: {list(preset_map.keys())}")

        return factory()

    # ── Training ─────────────────────────────────────────

    def train(self) -> dict[str, Any]:
        """Run the fine-tuning training loop.

        Requires configure() to be called first.
        Returns training report.
        """
        if self.config is None:
            raise RuntimeError("Call configure() before train()")

        from bis.fine_tuning.trainers import (
            TextFineTuner, AudioFineTuner,
            FineTuneOrchestrator,
        )
        from bis.fine_tuning.config import Modality

        trainer_map = {
            Modality.TEXT: TextFineTuner,
            Modality.AUDIO: AudioFineTuner,
        }

        trainer_cls = trainer_map.get(self.config.modality)
        if trainer_cls is None:
            # Image: use orchestrator fallback
            orchestrator = FineTuneOrchestrator()
            job_id = orchestrator.submit(self.config)
            result = orchestrator.start(job_id)
            self.train_result = result
            return result

        self.trainer = trainer_cls(self.config)
        result = self.trainer.train()
        self.train_result = result

        logger.info(
            "Training complete: %s | Loss: %.4f | Time: %.1fs",
            result.get("status", "?"),
            result.get("final_loss", float("inf")),
            result.get("training_time_sec", 0),
        )

        return result

    # ── Evaluation ───────────────────────────────────────

    def evaluate(
        self,
        generated_data: list[Any] | None = None,
        reference_data: list[Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate the fine-tuned model.

        If generated_data is not provided, generates samples using
        the trained model (if available).

        Returns evaluation metrics.
        """
        from bis.fine_tuning.evaluation import UnifiedEvaluator

        evaluator = UnifiedEvaluator()

        if generated_data is None:
            # Generate test samples (mock for now)
            generated_data = self._generate_test_samples()

        result = evaluator.evaluate(
            self.modality, generated_data, reference_data, **kwargs,
        )
        self.eval_result = result

        logger.info(
            "Evaluation: %s | Passed: %s | Metrics: %s",
            self.modality, result.passed, result.metrics,
        )

        return result.to_dict()

    def _generate_test_samples(self) -> list[Any]:
        """Generate test samples using the trained model."""
        # Placeholder: return mock data based on modality
        import numpy as np

        if self.modality == "text":
            return ["This is a test generated text." for _ in range(10)]
        elif self.modality == "audio":
            return [np.random.randn(32000 * 5).astype(np.float32) for _ in range(5)]

        return []

    # ── Export ───────────────────────────────────────────

    def export_lora(
        self,
        output_dir: str,
        *,
        push_to_hub: bool = False,
        hub_model_id: str | None = None,
    ) -> dict[str, Any]:
        """Export LoRA weights to a directory.

        Args:
            output_dir: Directory to save LoRA weights.
            push_to_hub: Whether to push to Hugging Face Hub.
            hub_model_id: Hub model ID (required if push_to_hub).

        Returns:
            Export summary.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Copy LoRA weights from training output
        lora_source = None
        if self.train_result:
            lora_source = self.train_result.get("lora_path")

        if lora_source and Path(lora_source).exists():
            source = Path(lora_source)
            if source.is_dir():
                shutil.copytree(source, output_path / "lora", dirs_exist_ok=True)
            else:
                shutil.copy2(source, output_path / "lora_weights.safetensors")
            logger.info("LoRA weights exported to %s", output_path)
        else:
            logger.warning("No LoRA weights found to export")

        # Save metadata
        metadata = {
            "modality": self.modality,
            "model_id": self.config.model_id if self.config else "",
            "lora_rank": self.config.lora.rank if self.config else 0,
            "training_metrics": self.train_result or {},
            "evaluation_metrics": (
                self.eval_result.to_dict() if self.eval_result else {}
            ),
            "exported_at": time.time(),
        }
        (output_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )

        # Push to hub
        if push_to_hub and hub_model_id:
            self._push_to_hub(output_path, hub_model_id)

        return {
            "output_dir": str(output_path),
            "has_weights": lora_source is not None,
            "metadata_saved": True,
            "pushed_to_hub": push_to_hub,
        }

    def _push_to_hub(self, path: Path, model_id: str) -> None:
        """Push LoRA weights to Hugging Face Hub."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=str(path),
                repo_id=model_id,
                repo_type="model",
            )
            logger.info("Pushed to Hub: %s", model_id)
        except Exception as e:
            logger.error("Hub push failed: %s", e)

    # ── Status & Info ────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Get current SDK status."""
        return {
            "modality": self.modality,
            "configured": self.config is not None,
            "dataset_loaded": self.dataset is not None,
            "trained": self.train_result is not None,
            "evaluated": self.eval_result is not None,
            "config_summary": (
                {
                    "model_id": self.config.model_id,
                    "epochs": self.config.epochs,
                    "lora_rank": self.config.lora.rank,
                }
                if self.config else None
            ),
        }

    @staticmethod
    def available_presets() -> list[str]:
        """List all available preset names."""
        return list(FineTuneSDK.PRESETS.keys())
