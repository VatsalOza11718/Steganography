"""Frame-based video steganography using LSB embedding.

Hides arbitrary binary data within video frames using Least Significant
Bit (LSB) substitution on pixel values. Supports common video formats
via OpenCV (mp4, avi, mkv).

Strategy:
- Extract frames from video
- Embed data bits in LSBs of blue channel pixels across frames
- Reconstruct video with modified frames
- Audio track is preserved when possible

Binary format embedded in frames:
    magic(4 bytes "BISV") + data_length(4 bytes big-endian) + data(N bytes)
"""

from __future__ import annotations

import struct
import tempfile
import shutil
from pathlib import Path
from typing import Union

import cv2
import numpy as np

# Magic bytes for BIS video steganography
MAGIC = b"BISV"
HEADER_SIZE = len(MAGIC) + 4  # 4 magic + 4 length = 8 bytes


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


def get_video_capacity(video_path: str | Path) -> int:
    """Get the maximum bytes that can be hidden in a video.

    Uses only the blue channel LSB of each pixel across all frames.

    Args:
        video_path: Path to the video file.

    Returns:
        Maximum payload size in bytes (excluding header).
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Each pixel's blue channel stores 1 bit
    total_bits = frame_count * width * height
    total_bytes = total_bits // 8
    return max(0, total_bytes - HEADER_SIZE)


def hide_in_video(
    video_path: str | Path,
    data: bytes,
    output_path: str | Path,
) -> dict[str, any]:
    """Hide binary data within a video file using LSB steganography.

    Embeds data in the LSB of the blue channel across video frames.

    Args:
        video_path: Path to the cover video file.
        data: Binary data to hide.
        output_path: Path to save the stego video file.

    Returns:
        Dictionary with metadata.

    Raises:
        ValueError: If the video can't hold the data.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    pixels_per_frame = width * height

    # The header lets extraction confirm the carrier contains BIS data and
    # determine exactly how many bits must be recovered from the frames.
    payload = MAGIC + struct.pack(">I", len(data)) + data
    bits = _bytes_to_bits(payload)

    total_capacity_bits = frame_count * pixels_per_frame
    if len(bits) > total_capacity_bits:
        cap.release()
        capacity = get_video_capacity(video_path)
        raise ValueError(
            f"Data too large! Need {len(data)} bytes but video can hold "
            f"{capacity} bytes. Use a longer video."
        )

    # Determine output codec
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()

    # Use lossless or near-lossless codecs to preserve LSB
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")  # lossless
    elif ext == ".mkv":
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")  # lossless
    else:
        # For .mp4, use FFV1 in AVI container then note to user
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        if ext == ".mp4":
            output_path = output_path.with_suffix(".avi")

    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (width, height)
    )

    bit_index = 0
    frames_written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if bit_index < len(bits):
            # OpenCV uses BGR ordering, so channel 0 is blue. We only touch
            # that channel to keep the embedding rule simple and predictable.
            blue = frame[:, :, 0].flatten()

            bits_remaining = len(bits) - bit_index
            bits_this_frame = min(bits_remaining, pixels_per_frame)

            for i in range(bits_this_frame):
                blue[i] = (blue[i] & 0xFE) | bits[bit_index + i]

            frame[:, :, 0] = blue.reshape(height, width)
            bit_index += bits_this_frame

        writer.write(frame)
        frames_written += 1

    cap.release()
    writer.release()

    return {
        "capacity": get_video_capacity(video_path),
        "data_size": len(data),
        "bits_used": len(bits),
        "frames": frames_written,
        "resolution": f"{width}x{height}",
        "fps": round(fps, 2),
        "duration_sec": round(frame_count / fps, 2) if fps > 0 else 0,
        "output_path": str(output_path),
    }


def extract_from_video(video_path: str | Path) -> bytes:
    """Extract hidden binary data from a stego video file.

    Args:
        video_path: Path to the stego video file.

    Returns:
        Extracted binary data.

    Raises:
        ValueError: If no hidden data found or data is corrupted.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pixels_per_frame = width * height

    # Read just enough bits for the header first; only then do we know how
    # many additional frame bits belong to the hidden payload.
    header_bits_count = HEADER_SIZE * 8
    all_bits: list[int] = []
    data_length = None
    total_bits_needed = header_bits_count  # Start with header

    while len(all_bits) < total_bits_needed:
        ret, frame = cap.read()
        if not ret:
            break

        blue = frame[:, :, 0].flatten()

        # Extract bits from this frame up to what we still need
        bits_needed = total_bits_needed - len(all_bits)
        bits_from_frame = min(bits_needed, pixels_per_frame)

        for i in range(bits_from_frame):
            all_bits.append(int(blue[i]) & 1)

        # Once we have the header, determine total data length
        if data_length is None and len(all_bits) >= header_bits_count:
            header_bytes = _bits_to_bytes(all_bits[:header_bits_count])
            magic = header_bytes[:4]
            if magic != MAGIC:
                cap.release()
                raise ValueError(
                    "No hidden data found in this video (magic header mismatch)."
                )
            data_length = struct.unpack(">I", header_bytes[4:8])[0]
            total_bits_needed = (HEADER_SIZE + data_length) * 8

            # Continue reading more bits from THIS SAME frame if needed
            remaining_needed = total_bits_needed - len(all_bits)
            if remaining_needed > 0:
                remaining_in_frame = pixels_per_frame - bits_from_frame
                extra = min(remaining_needed, remaining_in_frame)
                for i in range(bits_from_frame, bits_from_frame + extra):
                    all_bits.append(int(blue[i]) & 1)

    cap.release()

    if data_length is None:
        raise ValueError("Video too short to contain hidden data.")

    if len(all_bits) < total_bits_needed:
        raise ValueError(
            f"Corrupted data — claimed {data_length} bytes but not enough "
            f"data in video."
        )

    # Extract data (skip header)
    data_bits = all_bits[header_bits_count : header_bits_count + data_length * 8]
    return _bits_to_bytes(data_bits)[:data_length]


def hide_text_in_video(
    video_path: str | Path,
    text: str,
    output_path: str | Path,
    password: str | None = None,
) -> dict[str, any]:
    """Hide a text message in a video file, optionally encrypted.

    Args:
        video_path: Path to cover video file.
        text: Message to hide.
        output_path: Path to save stego video file.
        password: Optional encryption password for AES-256.

    Returns:
        Metadata dictionary.
    """
    data = text.encode("utf-8")

    if password:
        from bis.utils.crypto import encrypt_bytes
        data = encrypt_bytes(data, password)

    # Prepend flag byte: 0x01 = encrypted, 0x00 = plain
    flag = b"\x01" if password else b"\x00"
    payload = flag + data

    return hide_in_video(video_path, payload, output_path)


def extract_text_from_video(
    video_path: str | Path,
    password: str | None = None,
) -> str:
    """Extract a hidden text message from a stego video file.

    Args:
        video_path: Path to stego video file.
        password: Decryption password (required if encrypted).

    Returns:
        Extracted text string.
    """
    payload = extract_from_video(video_path)

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
