"""Unified fine-tuning configuration engine.

Single YAML/JSON schema covering all modalities — learning rate, batch size,
gradient accumulation, precision, checkpointing, and early-stopping criteria.

Per-model presets:
 • Qwen 2.5  — LoRA rank-64, alpha-16, 128k context
 • MusicGen  — LoRA rank-32, decoder-only
 • SD/SDXL/FLUX — LoRA rank-8, UNet attention layers

Supports:
 • DeepSpeed ZeRO-3, FSDP, 8-bit/4-bit QLoRA
 • Dynamic batching by sequence length / duration / frame count
 • Early stopping with patience and delta

The dataclasses in this file are deliberately plain so they can be loaded
from YAML/JSON, edited in code, and returned through APIs with minimal glue.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════

class Modality(str, Enum):
    """Supported fine-tuning modalities."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"


class Precision(str, Enum):
    """Training precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"       # 8-bit QLoRA
    INT4 = "int4"       # 4-bit QLoRA (NF4)


class DistributedBackend(str, Enum):
    """Distributed training backends."""
    NONE = "none"
    DEEPSPEED_ZERO2 = "deepspeed_zero2"
    DEEPSPEED_ZERO3 = "deepspeed_zero3"
    FSDP = "fsdp"


class EarlyStopMetric(str, Enum):
    """Metric to monitor for early stopping."""
    LOSS = "loss"
    PERPLEXITY = "perplexity"
    BLEU = "bleu"
    FAD = "fad"


# ═══════════════════════════════════════════════════════════
#  EARLY STOPPING
# ═══════════════════════════════════════════════════════════

@dataclass
class EarlyStoppingConfig:
    """Early stopping criteria."""
    enabled: bool = True
    metric: str = "loss"           # metric key to watch
    patience: int = 5              # epochs without improvement
    min_delta: float = 1e-4        # minimum improvement threshold
    mode: str = "min"              # "min" or "max"


# ═══════════════════════════════════════════════════════════
#  LORA CONFIG
# ═══════════════════════════════════════════════════════════

@dataclass
class LoRAHyperparams:
    """LoRA-specific hyperparameters."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
    ])
    bias: str = "none"             # "none", "all", "lora_only"
    use_rslora: bool = False       # rank-stabilized LoRA


@dataclass
class QLoRAHyperparams:
    """QLoRA-specific settings (extends LoRA with quantization)."""
    lora: LoRAHyperparams = field(default_factory=LoRAHyperparams)
    quantization_bits: int = 4     # 4 or 8
    double_quant: bool = True      # nested quantization
    quant_type: str = "nf4"        # "nf4" or "fp4"


# ═══════════════════════════════════════════════════════════
#  DYNAMIC BATCHING
# ═══════════════════════════════════════════════════════════

@dataclass
class DynamicBatchingConfig:
    """Dynamic batching configuration per modality type."""
    enabled: bool = True
    # Text: group by sequence length
    text_length_buckets: list[int] = field(
        default_factory=lambda: [256, 512, 1024, 2048, 4096, 8192]
    )
    # Audio: group by duration
    audio_duration_buckets_sec: list[float] = field(
        default_factory=lambda: [5.0, 10.0, 30.0, 60.0, 120.0]
    )
    max_tokens_per_batch: int = 32768  # token budget approach


# ═══════════════════════════════════════════════════════════
#  CHECKPOINT CONFIG
# ═══════════════════════════════════════════════════════════

@dataclass
class CheckpointConfig:
    """Checkpointing and saving configuration."""
    output_dir: str = "./fine_tune_output"
    save_every_steps: int = 500
    save_every_epochs: int = 1
    save_total_limit: int = 5      # keep only N most recent
    save_optimizer: bool = False    # saves disk space
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None


# ═══════════════════════════════════════════════════════════
#  LOGGING CONFIG
# ═══════════════════════════════════════════════════════════

@dataclass
class LoggingConfig:
    """Logging and experiment tracking."""
    log_every_steps: int = 10
    log_to_tensorboard: bool = True
    log_to_wandb: bool = False
    wandb_project: str = "bis-fine-tune"
    wandb_entity: Optional[str] = None
    tensorboard_dir: str = "./runs"
    sample_every_steps: int = 200  # generate samples for visual inspection


# ═══════════════════════════════════════════════════════════
#  EVALUATION CONFIG
# ═══════════════════════════════════════════════════════════

@dataclass
class EvaluationConfig:
    """Evaluation pipeline configuration."""
    eval_every_steps: int = 500
    eval_every_epochs: int = 1
    num_eval_samples: int = 100
    # Text
    compute_perplexity: bool = True
    compute_bleu: bool = True
    compute_rouge: bool = True
    # Audio
    compute_fad: bool = True
    compute_clap: bool = True
    fad_reference_dir: Optional[str] = None


# ═══════════════════════════════════════════════════════════
#  MAIN CONFIG
# ═══════════════════════════════════════════════════════════

@dataclass
class FineTuneConfig:
    """Unified fine-tuning configuration for all modalities.

    Can be loaded from YAML or JSON, or constructed programmatically.
    A single schema that covers text, audio, and image fine-tuning.

    Example YAML::

        modality: text
        model_id: Qwen/Qwen2.5-1.5B-Instruct
        dataset_dir: ./data/conversations
        training:
          learning_rate: 2e-5
          batch_size: 4
          epochs: 3
        lora:
          rank: 64
          alpha: 16
    """

    # ── Identity ──────────────────────────────────────────
    modality: Modality = Modality.TEXT
    model_id: str = ""                   # HuggingFace model ID
    dataset_dir: str = ""                # path to dataset
    run_name: str = ""                   # experiment name (auto-generated if empty)

    # ── Training ──────────────────────────────────────────
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    max_train_steps: int = -1            # -1 = epochs-based
    warmup_ratio: float = 0.05
    warmup_steps: int = -1               # -1 = use warmup_ratio
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    precision: Precision = Precision.BF16
    gradient_checkpointing: bool = True

    # ── LoRA ──────────────────────────────────────────────
    lora: LoRAHyperparams = field(default_factory=LoRAHyperparams)
    qlora: Optional[QLoRAHyperparams] = None  # set to enable QLoRA

    # ── Dynamic Batching ──────────────────────────────────
    dynamic_batching: DynamicBatchingConfig = field(
        default_factory=DynamicBatchingConfig
    )

    # ── Distributed ───────────────────────────────────────
    distributed: DistributedBackend = DistributedBackend.NONE
    num_gpus: int = 1

    # ── Checkpointing ─────────────────────────────────────
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # ── Logging ───────────────────────────────────────────
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ── Evaluation ────────────────────────────────────────
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # ── Early Stopping ────────────────────────────────────
    early_stopping: EarlyStoppingConfig = field(
        default_factory=EarlyStoppingConfig
    )

    # ── Text-Specific ─────────────────────────────────────
    max_seq_length: int = 4096           # context window for training
    conversation_packing: bool = True    # pack short conversations
    deduplication: bool = True
    prompt_template: str = "chatml"      # chatml, alpaca, llama

    # ── Audio-Specific ────────────────────────────────────
    audio_sample_rate: int = 32000       # target sample rate
    audio_mono: bool = True
    audio_target_lufs: float = -16.0     # loudness normalization
    audio_vad_enabled: bool = True       # voice activity detection
    audio_max_duration_sec: float = 30.0


    # ── Image-Specific (preserved from existing pipeline) ─
    image_resolution: int = 512

    # ── Serialization ─────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a JSON-compatible dictionary."""
        d = asdict(self)
        # Convert Enum objects into plain strings so the output is valid JSON
        # and easy to display in dashboards or config editors.
        d["modality"] = self.modality.value
        d["precision"] = self.precision.value
        d["distributed"] = self.distributed.value
        return d

    def to_json(self, path: str | Path) -> None:
        """Save config as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Config saved to %s", path)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FineTuneConfig:
        """Create config from a dictionary (e.g. parsed YAML/JSON)."""
        # Convert string enums
        if "modality" in d:
            d["modality"] = Modality(d["modality"])
        if "precision" in d:
            d["precision"] = Precision(d["precision"])
        if "distributed" in d:
            d["distributed"] = DistributedBackend(d["distributed"])

        # Nested dataclasses
        nested_map = {
            "lora": LoRAHyperparams,
            "qlora": QLoRAHyperparams,
            "dynamic_batching": DynamicBatchingConfig,
            "checkpoint": CheckpointConfig,
            "logging": LoggingConfig,
            "evaluation": EvaluationConfig,
            "early_stopping": EarlyStoppingConfig,
        }
        for key, cls_type in nested_map.items():
            if key in d and isinstance(d[key], dict):
                if key == "qlora" and d[key] is None:
                    continue
                if key == "qlora" and "lora" in d[key]:
                    d[key]["lora"] = LoRAHyperparams(**d[key]["lora"])
                d[key] = cls_type(**d[key])

        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str | Path) -> FineTuneConfig:
        """Load config from a JSON file."""
        text = Path(path).read_text()
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_yaml(cls, path: str | Path) -> FineTuneConfig:
        """Load config from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")
        text = Path(path).read_text()
        return cls.from_dict(yaml.safe_load(text))

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []
        if not self.model_id:
            issues.append("model_id is required")
        if not self.dataset_dir:
            issues.append("dataset_dir is required")
        if self.learning_rate <= 0:
            issues.append("learning_rate must be positive")
        if self.batch_size < 1:
            issues.append("batch_size must be ≥ 1")
        if self.epochs < 1 and self.max_train_steps < 1:
            issues.append("either epochs or max_train_steps must be ≥ 1")
        if self.lora.rank < 1:
            issues.append("LoRA rank must be ≥ 1")
        if self.modality == Modality.TEXT and self.max_seq_length < 32:
            issues.append("max_seq_length must be ≥ 32 for text")
        if self.modality == Modality.AUDIO and self.audio_sample_rate < 8000:
            issues.append("audio_sample_rate must be ≥ 8000")
        return issues


# ═══════════════════════════════════════════════════════════
#  PER-MODEL PRESETS
# ═══════════════════════════════════════════════════════════

class TextPreset:
    """Preset configurations for text fine-tuning."""

    @staticmethod
    def qwen25_chat() -> FineTuneConfig:
        """Qwen 2.5 chat fine-tuning preset."""
        return FineTuneConfig(
            modality=Modality.TEXT,
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            learning_rate=2e-5,
            batch_size=4,
            gradient_accumulation_steps=4,
            epochs=3,
            precision=Precision.BF16,
            gradient_checkpointing=True,
            max_seq_length=4096,
            conversation_packing=True,
            deduplication=True,
            prompt_template="chatml",
            lora=LoRAHyperparams(
                rank=64,
                alpha=16.0,
                dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            ),
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                metric="perplexity",
                patience=3,
                mode="min",
            ),
        )

    @staticmethod
    def qwen25_128k() -> FineTuneConfig:
        """Qwen 2.5 with extended 128K context."""
        config = TextPreset.qwen25_chat()
        config.max_seq_length = 131072
        config.batch_size = 1
        config.gradient_accumulation_steps = 16
        config.gradient_checkpointing = True
        config.qlora = QLoRAHyperparams(
            lora=LoRAHyperparams(rank=64, alpha=16.0),
            quantization_bits=4,
        )
        return config


class AudioPreset:
    """Preset configurations for audio fine-tuning."""

    @staticmethod
    def musicgen_melody() -> FineTuneConfig:
        """MusicGen melody fine-tuning preset."""
        return FineTuneConfig(
            modality=Modality.AUDIO,
            model_id="facebook/musicgen-small",
            learning_rate=1e-4,
            batch_size=2,
            gradient_accumulation_steps=8,
            epochs=5,
            precision=Precision.BF16,
            gradient_checkpointing=True,
            audio_sample_rate=32000,
            audio_mono=True,
            audio_target_lufs=-16.0,
            audio_vad_enabled=True,
            audio_max_duration_sec=30.0,
            lora=LoRAHyperparams(
                rank=32,
                alpha=16.0,
                dropout=0.1,
                target_modules=["out_proj", "v_proj", "q_proj", "k_proj"],
                bias="none",
            ),
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                metric="loss",
                patience=5,
                mode="min",
            ),
        )

    @staticmethod
    def musicgen_large() -> FineTuneConfig:
        """MusicGen large model preset."""
        config = AudioPreset.musicgen_melody()
        config.model_id = "facebook/musicgen-large"
        config.batch_size = 1
        config.gradient_accumulation_steps = 16
        config.qlora = QLoRAHyperparams(
            lora=LoRAHyperparams(rank=32, alpha=16.0),
            quantization_bits=8,
        )
        return config


class ImagePreset:
    """Preset configurations for image fine-tuning (preserved pipeline)."""

    @staticmethod
    def sd15_lora() -> FineTuneConfig:
        """Stable Diffusion 1.5 LoRA preset."""
        return FineTuneConfig(
            modality=Modality.IMAGE,
            model_id="sd15",
            learning_rate=1e-4,
            batch_size=1,
            gradient_accumulation_steps=4,
            epochs=10,
            precision=Precision.FP16,
            image_resolution=512,
            lora=LoRAHyperparams(
                rank=8,
                alpha=32.0,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"],
            ),
        )
