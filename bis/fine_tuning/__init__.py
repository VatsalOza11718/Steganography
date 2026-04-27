"""Multi-Modal Fine-Tuning Platform.

Production-grade LoRA fine-tuning for text (Qwen 2.5), audio (MusicGen),
and image (SD/SDXL/FLUX) generation models.

Unified YAML/JSON configuration, modality-specific preprocessing,
evaluation pipelines, REST API, CLI, and web dashboard.

Public API
----------
Configuration:
    FineTuneConfig — single YAML/JSON schema for all modalities
    TextPreset, AudioPreset — per-model presets

Preprocessing:
    TextPreprocessor — tokenization, packing, dedup for Qwen 2.5
    AudioPreprocessor — resampling, VAD, EnCodec extraction
    UnifiedDatasetBuilder — auto-detect + process any modality

Trainers:
    TextFineTuner — Qwen 2.5 LoRA/QLoRA
    AudioFineTuner — MusicGen decoder-only LoRA
    FineTuneOrchestrator — manages jobs, queue, lifecycle

Evaluation:
    TextEvaluator — perplexity, BLEU, ROUGE
    AudioEvaluator — FAD, CLAP score
    UnifiedEvaluator — runs appropriate evaluator per modality

SDK:
    FineTuneSDK — load_dataset, configure, train, evaluate, export_lora

This module re-exports the main building blocks so users can import the
platform from a single package entry point instead of multiple files.
"""

from bis.fine_tuning.config import (
    FineTuneConfig,
    TextPreset,
    AudioPreset,
    ImagePreset,
    Modality,
)

from bis.fine_tuning.preprocessing import (
    TextPreprocessor,
    AudioPreprocessor,
    UnifiedDatasetBuilder,
)

from bis.fine_tuning.trainers import (
    TextFineTuner,
    AudioFineTuner,
    FineTuneOrchestrator,
)

from bis.fine_tuning.evaluation import (
    TextEvaluator,
    AudioEvaluator,
    UnifiedEvaluator,
)

from bis.fine_tuning.sdk import FineTuneSDK
from bis.fine_tuning.api import register_fine_tune_routes

__all__ = [
    # Config
    "FineTuneConfig", "TextPreset", "AudioPreset",
    "ImagePreset", "Modality",
    # Preprocessing
    "TextPreprocessor", "AudioPreprocessor",
    "UnifiedDatasetBuilder",
    # Trainers
    "TextFineTuner", "AudioFineTuner",
    "FineTuneOrchestrator",
    # Evaluation
    "TextEvaluator", "AudioEvaluator", "UnifiedEvaluator",
    # SDK
    "FineTuneSDK",
    # API
    "register_fine_tune_routes",
]
