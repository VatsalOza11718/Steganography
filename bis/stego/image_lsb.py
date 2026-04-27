"""LSB (Least Significant Bit) Image Steganography.

Hides text data inside image pixel values by modifying the least significant
bits of each colour channel.  Works with PNG, JPG and BMP.

Format of the embedded payload::

    [1 byte]   encryption flag  (0x00 = plain, 0x01 = encrypted)
    [4 bytes]  data length (big-endian uint32)
    [N bytes]  data payload (plain UTF-8 or AES-256-GCM ciphertext)

Each byte is spread across 8 pixel channels (R, G, B) using the LSB.
The image must have enough capacity: ``width * height * 3 // 8`` bytes.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ── Constants ──────────────────────────────────────────────────

HEADER_SIZE = 5  # 1 (flag) + 4 (length)
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp"}


# ── Capacity ───────────────────────────────────────────────────

def image_capacity(img: Image.Image) -> int:
    """Return the maximum number of payload bytes the image can hold."""
    w, h = img.size
    channels = 3  # we only use RGB
    total_bits = w * h * channels
    return (total_bits // 8) - HEADER_SIZE


def validate_image(path: str | Path) -> Image.Image:
    """Open and validate an image for steganography."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    img = Image.open(path).convert("RGB")
    return img


# ── Core bit operations ───────────────────────────────────────

def _bytes_to_bits(data: bytes) -> list[int]:
    """Convert bytes to a flat list of bits (MSB first per byte)."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_bytes(bits: list[int]) -> bytes:
    """Convert a flat list of bits back to bytes."""
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte = (byte << 1) | bits[i + j]
            else:
                byte <<= 1
        result.append(byte)
    return bytes(result)


# ── Embed / Extract ───────────────────────────────────────────

def hide_text_in_image(
    cover_path: str | Path,
    text: str,
    output_path: str | Path,
    password: Optional[str] = None,
) -> dict:
    """Hide text inside an image using LSB steganography.

    Args:
        cover_path:  Path to the cover image (PNG/JPG/BMP).
        text:        The secret text to hide.
        output_path: Where to save the stego image (always PNG).
        password:    Optional AES-256 encryption password.

    Returns:
        Dict with metadata: capacity, data_size, encrypted.
    """
    img = validate_image(cover_path)
    capacity = image_capacity(img)

    # Prepare payload
    data = text.encode("utf-8")
    encrypted = False

    if password:
        from bis.utils.crypto import encrypt_bytes
        data = encrypt_bytes(data, password)
        encrypted = True

    if len(data) > capacity:
        raise ValueError(
            f"Data too large ({len(data)} bytes) for this image "
            f"(capacity: {capacity} bytes). Use a larger image."
        )

    # Store encryption state and payload size up front so extraction knows
    # how many bytes to read and whether decryption is required.
    flag = b"\x01" if encrypted else b"\x00"
    header = flag + struct.pack(">I", len(data))
    payload = header + data

    # Convert to bits
    bits = _bytes_to_bits(payload)

    # Embed bits into image pixels
    pixels = np.array(img, dtype=np.uint8)
    flat = pixels.flatten()

    if len(bits) > len(flat):
        raise ValueError("Not enough pixel channels for the payload.")

    for i, bit in enumerate(bits):
        flat[i] = (flat[i] & 0xFE) | bit  # clear LSB and set

    pixels = flat.reshape(pixels.shape)
    stego_img = Image.fromarray(pixels, "RGB")

    # Always save as PNG to preserve LSB data (lossy formats destroy it)
    out = Path(output_path)
    if out.suffix.lower() != ".png":
        out = out.with_suffix(".png")
    stego_img.save(str(out), format="PNG")

    return {
        "capacity": capacity,
        "data_size": len(data),
        "encrypted": encrypted,
        "output_path": str(out),
    }


def extract_text_from_image(
    stego_path: str | Path,
    password: Optional[str] = None,
) -> str:
    """Extract hidden text from a stego image.

    Args:
        stego_path: Path to the stego image.
        password:   Decryption password (if text was encrypted).

    Returns:
        The extracted text string.
    """
    img = validate_image(stego_path)
    pixels = np.array(img, dtype=np.uint8)
    flat = pixels.flatten()

    # Extract header bits (5 bytes = 40 bits)
    header_bits_count = HEADER_SIZE * 8
    if len(flat) < header_bits_count:
        raise ValueError("Image too small to contain a steganographic payload.")

    header_bits = [(flat[i] & 1) for i in range(header_bits_count)]
    header_bytes = _bits_to_bytes(header_bits)

    flag = header_bytes[0]
    data_len = struct.unpack(">I", header_bytes[1:5])[0]

    # Sanity check
    max_capacity = image_capacity(img)
    if data_len > max_capacity or data_len > 50_000_000:
        raise ValueError(
            "Invalid payload length detected. "
            "Image may not contain hidden data or is corrupted."
        )

    # Extract data bits
    total_bits_needed = (HEADER_SIZE + data_len) * 8
    if len(flat) < total_bits_needed:
        raise ValueError("Image too small for the declared payload size.")

    data_bits = [(flat[i] & 1) for i in range(header_bits_count, total_bits_needed)]
    data = _bits_to_bytes(data_bits)[:data_len]

    # Decrypt if needed
    is_encrypted = flag == 0x01
    if is_encrypted:
        if not password:
            raise ValueError(
                "This message is encrypted. Please provide the decryption password."
            )
        from bis.utils.crypto import decrypt_bytes
        data = decrypt_bytes(data, password)

    return data.decode("utf-8")
