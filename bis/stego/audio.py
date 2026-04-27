"""LSB-based audio steganography for WAV files.

Hides arbitrary binary data within audio samples using Least Significant
Bit (LSB) substitution. Supports 16-bit WAV files with any number of
channels and sample rates.

Binary format embedded in audio:
    magic(4 bytes "BISA") + data_length(4 bytes big-endian) + data(N bytes)
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Union

import numpy as np

# Magic bytes to identify BIS audio steganography
MAGIC = b"BISA"
HEADER_SIZE = len(MAGIC) + 4  # 4 magic + 4 length = 8 bytes


def _validate_wav(audio_path: str | Path) -> None:
    """Check that the audio file is a supported WAV format."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if path.suffix.lower() not in (".wav", ".wave"):
        raise ValueError(
            f"Only WAV files are supported. Got: {path.suffix}"
        )


def _bytes_to_bits(data: bytes) -> list[int]:
    """Convert bytes to a flat list of bits (MSB-first per byte)."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_bytes(bits: list[int]) -> bytes:
    """Convert a flat list of bits back to bytes."""
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte_bits = bits[i : i + 8]
        if len(byte_bits) < 8:
            byte_bits.extend([0] * (8 - len(byte_bits)))
        byte = 0
        for bit in byte_bits:
            byte = (byte << 1) | bit
        result.append(byte)
    return bytes(result)


def get_audio_capacity(audio_path: str | Path) -> int:
    """Get the maximum number of bytes that can be hidden in an audio file.

    Args:
        audio_path: Path to the WAV file.

    Returns:
        Maximum payload size in bytes (excluding header).
    """
    _validate_wav(audio_path)

    with wave.open(str(audio_path), "rb") as wav:
        n_frames = wav.getnframes()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()

    if sample_width != 2:
        raise ValueError(
            f"Only 16-bit WAV files are supported. Got {sample_width * 8}-bit."
        )

    total_samples = n_frames * n_channels
    # Each sample stores 1 bit via LSB, need 8 samples per byte
    total_bytes = total_samples // 8
    return max(0, total_bytes - HEADER_SIZE)


def hide_in_audio(
    audio_path: str | Path,
    data: bytes,
    output_path: str | Path,
) -> dict[str, any]:
    """Hide binary data within a WAV audio file using LSB steganography.

    Args:
        audio_path: Path to the cover WAV file.
        data: Binary data to hide.
        output_path: Path to save the stego WAV file.

    Returns:
        Dictionary with metadata (capacity, used, etc.).

    Raises:
        ValueError: If the audio can't hold the data.
    """
    _validate_wav(audio_path)

    with wave.open(str(audio_path), "rb") as wav:
        params = wav.getparams()
        n_frames = wav.getnframes()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        raw_audio = wav.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(
            f"Only 16-bit WAV supported. Got {sample_width * 8}-bit."
        )

    # Convert raw audio to numpy int16 array
    samples = np.frombuffer(raw_audio, dtype=np.int16).copy()
    total_samples = len(samples)

    # Prefixing the raw payload lets extraction confirm the file really
    # contains BIS audio steganography before trusting the decoded length.
    payload = MAGIC + struct.pack(">I", len(data)) + data
    bits = _bytes_to_bits(payload)

    if len(bits) > total_samples:
        capacity = get_audio_capacity(audio_path)
        raise ValueError(
            f"Data too large! Need {len(data)} bytes but audio can hold "
            f"{capacity} bytes. Use a longer audio file."
        )

    # Each sample contributes one hidden bit by replacing only its least
    # significant bit, which keeps audible distortion extremely small.
    # Use uint16 view to avoid sign issues with bit manipulation
    samples_u = samples.view(np.uint16)
    for i, bit in enumerate(bits):
        samples_u[i] = (samples_u[i] & 0xFFFE) | bit

    # Write output WAV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as out_wav:
        out_wav.setparams(params)
        out_wav.writeframes(samples.tobytes())

    return {
        "capacity": get_audio_capacity(audio_path),
        "data_size": len(data),
        "bits_used": len(bits),
        "total_samples": total_samples,
        "sample_rate": params.framerate,
        "channels": params.nchannels,
        "duration_sec": round(n_frames / params.framerate, 2),
    }


def extract_from_audio(audio_path: str | Path) -> bytes:
    """Extract hidden binary data from a stego WAV file.

    Args:
        audio_path: Path to the stego WAV file.

    Returns:
        Extracted binary data.

    Raises:
        ValueError: If no hidden data is found or data is corrupted.
    """
    _validate_wav(audio_path)

    with wave.open(str(audio_path), "rb") as wav:
        n_frames = wav.getnframes()
        sample_width = wav.getsampwidth()
        raw_audio = wav.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit WAV supported. Got {sample_width * 8}-bit.")

    samples = np.frombuffer(raw_audio, dtype=np.int16)

    # Extract header bits: magic(4 bytes = 32 bits) + length(4 bytes = 32 bits)
    header_bits_count = HEADER_SIZE * 8
    if len(samples) < header_bits_count:
        raise ValueError("Audio too short to contain hidden data.")

    header_bits = [(int(samples[i]) & 1) for i in range(header_bits_count)]
    header_bytes = _bits_to_bytes(header_bits)

    # Verify magic
    magic = header_bytes[:4]
    if magic != MAGIC:
        raise ValueError(
            "No hidden data found in this audio file (magic header mismatch)."
        )

    # Read the declared payload size after the magic header passes validation.
    data_length = struct.unpack(">I", header_bytes[4:8])[0]

    # Validate length
    total_bits_needed = (HEADER_SIZE + data_length) * 8
    if total_bits_needed > len(samples):
        raise ValueError(
            f"Corrupted data — claimed {data_length} bytes but audio is too short."
        )

    # Extract data bits
    data_bits = [
        (int(samples[i]) & 1)
        for i in range(header_bits_count, header_bits_count + data_length * 8)
    ]
    return _bits_to_bytes(data_bits)[:data_length]


def hide_text_in_audio(
    audio_path: str | Path,
    text: str,
    output_path: str | Path,
    password: str | None = None,
) -> dict[str, any]:
    """Hide a text message in a WAV file, optionally encrypted.

    Args:
        audio_path: Path to cover WAV file.
        text: Message to hide.
        output_path: Path to save stego WAV file.
        password: Optional encryption password for AES-256.

    Returns:
        Metadata dictionary.
    """
    data = text.encode("utf-8")

    if password:
        from bis.utils.crypto import encrypt_bytes
        data = encrypt_bytes(data, password)

    # Prepend a flag byte: 0x01 = encrypted, 0x00 = plain
    flag = b"\x01" if password else b"\x00"
    payload = flag + data

    return hide_in_audio(audio_path, payload, output_path)


def extract_text_from_audio(
    audio_path: str | Path,
    password: str | None = None,
) -> str:
    """Extract a hidden text message from a stego WAV file.

    Args:
        audio_path: Path to stego WAV file.
        password: Decryption password (required if message was encrypted).

    Returns:
        Extracted text string.
    """
    payload = extract_from_audio(audio_path)

    if len(payload) < 1:
        raise ValueError("Extracted data is empty.")

    is_encrypted = payload[0] == 0x01
    data = payload[1:]

    if is_encrypted:
        if not password:
            raise ValueError(
                "This message is encrypted. Please provide the decryption password."
            )
        from bis.utils.crypto import decrypt_bytes
        data = decrypt_bytes(data, password)

    return data.decode("utf-8")
