#!/usr/bin/env python3
"""Dataset preparation utilities for multi-modal fine-tuning.

Downloads and prepares datasets from HuggingFace for text and audio training.

Usage:
    python -m bis.fine_tuning.prepare_data --modality text
    python -m bis.fine_tuning.prepare_data --modality audio
    python -m bis.fine_tuning.prepare_data --modality all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

TEXT_DATASETS = {
    "alpaca": {
        "name": "tatsu-lab/alpaca",
        "split": "train[:1000]",  # Small subset for demo
        "format": "instruction",
    },
    "openassistant": {
        "name": "OpenAssistant/oasst1", 
        "split": "train[:500]",
        "format": "conversation",
    },
    "dolly": {
        "name": "databricks/databricks-dolly-15k",
        "split": "train[:500]",
        "format": "instruction",
    },
}

AUDIO_DATASETS = {
    "musiccaps": {
        "name": "google/MusicCaps",
        "split": "train[:100]",  # Small subset
        "format": "audio_text",
    },
    "gtzan": {
        "name": "marsyas/gtzan",
        "split": "train[:50]",
        "format": "audio_classification",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_text_data(output_dir: Path, dataset_name: str = "alpaca") -> None:
    """Download and prepare text dataset for fine-tuning.
    
    Creates JSONL files in ChatML format.
    """
    logger.info("Preparing text dataset: %s", dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library required: pip install datasets")
        return _create_sample_text_data(output_dir)
    
    config = TEXT_DATASETS.get(dataset_name, TEXT_DATASETS["alpaca"])
    
    try:
        logger.info("Downloading %s from HuggingFace...", config["name"])
        ds = load_dataset(config["name"], split=config["split"], trust_remote_code=True)
    except Exception as e:
        logger.warning("Failed to download %s: %s. Using sample data.", config["name"], e)
        return _create_sample_text_data(output_dir)
    
    # Convert to ChatML format
    conversations = []
    
    for item in ds:
        if config["format"] == "instruction":
            # Alpaca/Dolly format: instruction, input, output
            conv = {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": _build_instruction_prompt(item)},
                    {"role": "assistant", "content": item.get("output", item.get("response", ""))},
                ]
            }
        elif config["format"] == "conversation":
            # OpenAssistant format: messages with roles
            messages = []
            if "messages" in item:
                for msg in item["messages"]:
                    role = msg.get("role", "user")
                    content = msg.get("content", msg.get("text", ""))
                    messages.append({"role": role, "content": content})
            elif "text" in item:
                messages = [
                    {"role": "user", "content": item["text"]},
                    {"role": "assistant", "content": "I understand."},
                ]
            conv = {"messages": messages} if messages else None
        else:
            continue
            
        if conv:
            conversations.append(conv)
    
    # Write to JSONL
    output_file = output_dir / "train.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    logger.info("Saved %d conversations to %s", len(conversations), output_file)
    
    # Create metadata
    metadata = {
        "dataset": config["name"],
        "num_samples": len(conversations),
        "format": "chatml",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _build_instruction_prompt(item: dict) -> str:
    """Build instruction prompt from Alpaca/Dolly format."""
    instruction = item.get("instruction", item.get("context", ""))
    input_text = item.get("input", "")
    
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def _create_sample_text_data(output_dir: Path) -> None:
    """Create sample text data for testing when download fails."""
    logger.info("Creating sample text data...")
    
    samples = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is steganography?"},
                {"role": "assistant", "content": "Steganography is the practice of concealing information within other data, such as hiding text in images or audio files. Unlike encryption which makes data unreadable, steganography hides the existence of the secret data entirely."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "How does image steganography work?"},
                {"role": "assistant", "content": "Image steganography typically works by modifying the least significant bits (LSB) of pixel values. Since these bits have minimal impact on visual appearance, changing them allows data to be hidden without noticeable changes to the image."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is AES-256 encryption?"},
                {"role": "assistant", "content": "AES-256 is a symmetric encryption algorithm that uses a 256-bit key. It's considered highly secure and is used by governments and organizations worldwide to protect sensitive data. The '256' refers to the key length, providing 2^256 possible key combinations."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Explain the difference between encryption and steganography."},
                {"role": "assistant", "content": "Encryption transforms readable data into an unreadable format using a key. The encrypted data is visible but unintelligible without decryption. Steganography, on the other hand, hides data within other files (like images or audio) so that the existence of secret data is concealed. Both can be combined for enhanced security."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is LoRA fine-tuning?"},
                {"role": "assistant", "content": "LoRA (Low-Rank Adaptation) is an efficient fine-tuning method for large language models. Instead of updating all model parameters, LoRA adds small trainable matrices to the attention layers. This dramatically reduces memory requirements and training time while maintaining model quality."},
            ]
        },
    ] * 20  # Repeat to get 100 samples
    
    output_file = output_dir / "train.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for conv in samples:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    logger.info("Created %d sample conversations", len(samples))


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_audio_data(output_dir: Path, dataset_name: str = "musiccaps") -> None:
    """Download and prepare audio dataset for MusicGen fine-tuning."""
    logger.info("Preparing audio dataset: %s", dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library required: pip install datasets")
        return _create_sample_audio_data(output_dir)
    
    config = AUDIO_DATASETS.get(dataset_name, AUDIO_DATASETS["musiccaps"])
    
    try:
        logger.info("Downloading %s from HuggingFace...", config["name"])
        ds = load_dataset(config["name"], split=config["split"], trust_remote_code=True)
    except Exception as e:
        logger.warning("Failed to download %s: %s. Using synthesized data.", config["name"], e)
        return _create_sample_audio_data(output_dir)
    
    prompts = {}
    
    for idx, item in enumerate(ds):
        # Save audio file
        audio_path = output_dir / f"audio_{idx:04d}.wav"
        
        if "audio" in item:
            audio_data = item["audio"]
            _save_audio(audio_data, audio_path)
        
        # Get text prompt
        prompt = ""
        if "caption" in item:
            prompt = item["caption"]
        elif "text" in item:
            prompt = item["text"]
        elif "label" in item:
            prompt = str(item["label"])
            
        prompts[f"audio_{idx:04d}.wav"] = prompt
    
    # Save prompts mapping
    (output_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
    
    logger.info("Saved %d audio files with prompts", len(prompts))


def _save_audio(audio_data: dict | Any, path: Path) -> None:
    """Save audio data to WAV file."""
    import numpy as np
    
    try:
        import soundfile as sf
        
        if isinstance(audio_data, dict):
            array = np.array(audio_data["array"])
            sr = audio_data.get("sampling_rate", 32000)
        else:
            array = np.array(audio_data)
            sr = 32000
            
        # Ensure correct shape
        if array.ndim > 1:
            array = array.mean(axis=0) if array.shape[0] <= 2 else array[:, 0]
            
        sf.write(str(path), array, sr)
        
    except ImportError:
        import wave
        import struct
        
        if isinstance(audio_data, dict):
            array = np.array(audio_data["array"])
            sr = audio_data.get("sampling_rate", 32000)
        else:
            array = np.array(audio_data)
            sr = 32000
            
        # Convert to int16
        if array.dtype == np.float32 or array.dtype == np.float64:
            array = (array * 32767).astype(np.int16)
        elif array.dtype != np.int16:
            array = array.astype(np.int16)
            
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(struct.pack(f"<{len(array)}h", *array))


def _create_sample_audio_data(output_dir: Path) -> None:
    """Create synthesized audio samples for testing."""
    import numpy as np
    
    logger.info("Creating synthesized audio samples...")
    
    sr = 32000
    duration = 5.0  # 5 seconds
    prompts = {}
    
    # Generate simple tones with different characteristics
    audio_configs = [
        ("calm piano melody", 440, "sine"),
        ("upbeat electronic beat", 330, "square"),
        ("ambient drone", 220, "sine"),
        ("rhythmic percussion", 440, "noise"),
        ("gentle acoustic guitar", 392, "triangle"),
    ] * 4  # 20 samples
    
    for idx, (prompt, freq, wave_type) in enumerate(audio_configs):
        t = np.linspace(0, duration, int(sr * duration))
        
        if wave_type == "sine":
            waveform = np.sin(2 * np.pi * freq * t)
        elif wave_type == "square":
            waveform = np.sign(np.sin(2 * np.pi * freq * t))
        elif wave_type == "triangle":
            waveform = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
        elif wave_type == "noise":
            waveform = np.random.randn(len(t)) * 0.3
            envelope = np.exp(-2 * (t % 0.5))
            waveform *= envelope
        else:
            waveform = np.sin(2 * np.pi * freq * t)
        
        # Apply envelope
        envelope = np.exp(-0.3 * t)
        waveform = (waveform * envelope * 0.5).astype(np.float32)
        
        # Save
        audio_path = output_dir / f"audio_{idx:04d}.wav"
        _save_audio({"array": waveform, "sampling_rate": sr}, audio_path)
        prompts[f"audio_{idx:04d}.wav"] = prompt
    
    (output_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
    logger.info("Created %d synthesized audio samples", len(prompts))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Parse command-line options and prepare the requested datasets."""
    parser = argparse.ArgumentParser(description="Prepare datasets for fine-tuning")
    parser.add_argument(
        "--modality",
        choices=["text", "audio", "all"],
        default="all",
        help="Which modality to prepare",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--text-dataset",
        type=str,
        default="alpaca",
        choices=list(TEXT_DATASETS.keys()),
        help="Text dataset to download",
    )
    parser.add_argument(
        "--audio-dataset",
        type=str,
        default="musiccaps",
        choices=list(AUDIO_DATASETS.keys()),
        help="Audio dataset to download",
    )
    
    args = parser.parse_args()
    base_dir = Path(args.output_dir)
    
    if args.modality in ("text", "all"):
        prepare_text_data(base_dir / "text_train", args.text_dataset)
        
    if args.modality in ("audio", "all"):
        prepare_audio_data(base_dir / "audio_train", args.audio_dataset)
    
    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()
