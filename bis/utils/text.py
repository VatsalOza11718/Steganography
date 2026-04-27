"""Text <-> binary conversion utilities for steganographic encoding."""

from __future__ import annotations

import numpy as np


def text_to_bits(text: str) -> list[int]:
    """Convert a UTF-8 string to a list of bits (0/1).

    The encoding format is:
        [32-bit length header] [payload bits]
    where the length header stores the number of payload bits.
    """
    encoded = text.encode("utf-8")
    bits: list[int] = []
    for byte in encoded:
        # Emit bits most-significant-bit first so encode/decode order stays
        # deterministic across all BIS hiding backends.
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    # Prepend 32-bit length header
    length = len(bits)
    header = []
    for i in range(31, -1, -1):
        header.append((length >> i) & 1)

    return header + bits


def bits_to_text(bits: list[int] | np.ndarray) -> str:
    """Convert a list of bits back to a UTF-8 string.

    Expects the format produced by `text_to_bits`:
        [32-bit length header] [payload bits]
    """
    if isinstance(bits, np.ndarray):
        bits = bits.tolist()

    if len(bits) < 32:
        raise ValueError("Bit sequence too short to contain a valid header.")

    # Read the 32-bit header first so we know exactly how many payload bits
    # belong to the message and which trailing bits are just padding.
    length = 0
    for i in range(32):
        length = (length << 1) | int(bits[i])

    payload = bits[32: 32 + length]

    if len(payload) < length:
        raise ValueError(
            f"Expected {length} payload bits but only got {len(payload)}."
        )

    # Convert bit groups back into bytes before decoding the UTF-8 text.
    if len(payload) % 8 != 0:
        # Pad with zeros to complete the last byte
        payload = payload + [0] * (8 - len(payload) % 8)

    byte_array = bytearray()
    for i in range(0, len(payload), 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | int(payload[i + j])
        byte_array.append(byte_val)

    return byte_array.decode("utf-8", errors="replace")


def bits_to_bytearray(bits: list[int]) -> bytearray:
    """Convert a list of bits to a bytearray."""
    if len(bits) % 8 != 0:
        bits = bits + [0] * (8 - len(bits) % 8)

    result = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | int(bits[i + j])
        result.append(byte_val)
    return result


def text_to_binary_tensor(
    text: str, data_depth: int, height: int, width: int
) -> np.ndarray:
    """Convert text to a spatial binary tensor of shape (data_depth, height, width).

    The bits are tiled across the spatial dimensions for redundancy.
    """
    bits = text_to_bits(text)
    total_capacity = data_depth * height * width

    if len(bits) > total_capacity:
        raise ValueError(
            f"Text requires {len(bits)} bits but tensor capacity is "
            f"{total_capacity} bits ({data_depth}×{height}×{width})."
        )

    # The tensor is fully populated because downstream models expect a fixed
    # rectangular input shape rather than a variable-length bit stream.
    padded = bits + [0] * (total_capacity - len(bits))
    tensor = np.array(padded, dtype=np.float32).reshape(data_depth, height, width)
    return tensor


def binary_tensor_to_text(tensor: np.ndarray) -> str:
    """Extract text from a spatial binary tensor.

    Inverse of `text_to_binary_tensor`.
    """
    bits = tensor.flatten().tolist()
    # Neural outputs may be probabilities rather than exact zeros/ones, so
    # we threshold them back to discrete bits before decoding text.
    bits = [1 if b > 0.5 else 0 for b in bits]
    return bits_to_text(bits)
