"""Zero-width character text steganography.

Hides arbitrary binary data within cover text using invisible Unicode
zero-width characters. The stego text looks identical to the original
when displayed, but contains hidden data between visible characters.

Encoding scheme:
    \u200b (zero-width space)      = bit 0
    \u200c (zero-width non-joiner) = bit 1
    \u200d (zero-width joiner)     = delimiter (marks start/end of payload)

Binary format embedded in text:
    delimiter + magic("BIST" as bits) + data_length(4 bytes big-endian as bits)
    + data(N bytes as bits) + delimiter
"""

from __future__ import annotations

import struct
from pathlib import Path

# Zero-width characters used for encoding
ZW_ZERO = "\u200b"    # zero-width space       → bit 0
ZW_ONE = "\u200c"     # zero-width non-joiner  → bit 1
ZW_DELIM = "\u200d"   # zero-width joiner      → payload delimiter

MAGIC = b"BIST"
HEADER_SIZE = len(MAGIC) + 4  # 4 magic + 4 length = 8 bytes

# All zero-width characters used (for stripping / detection)
ZW_CHARS = {ZW_ZERO, ZW_ONE, ZW_DELIM}


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


def _bits_to_zwc(bits: list[int]) -> str:
    """Convert a list of bits to zero-width character string."""
    return "".join(ZW_ZERO if b == 0 else ZW_ONE for b in bits)


def _zwc_to_bits(zwc: str) -> list[int]:
    """Convert a zero-width character string to a list of bits."""
    bits = []
    for ch in zwc:
        if ch == ZW_ZERO:
            bits.append(0)
        elif ch == ZW_ONE:
            bits.append(1)
        # Skip delimiters or other characters
    return bits


def get_text_capacity(cover_text: str) -> int:
    """Get the maximum number of bytes that can be hidden in cover text.

    Each visible character can host zero-width chars after it.
    We use all gaps between characters + before first + after last.
    Total insertion points = len(cover_text) + 1
    But we need 2 delimiter chars, so usable points = len(cover_text) - 1
    Each insertion point can carry multiple ZW chars.

    In our scheme, we insert the entire ZW payload between the first
    two visible characters, so capacity is limited only by the payload
    size and the requirement for at least 2 visible characters.
    """
    # We need at least 1 visible character to embed after
    clean = _strip_zwc(cover_text)
    if len(clean) < 1:
        return 0
    # Theoretical: unlimited embedding in a single insertion point
    # Practical: we embed all bits in one block after first visible char
    # Return a generous limit (64KB should be enough for text stego)
    return 65536 - HEADER_SIZE


def _strip_zwc(text: str) -> str:
    """Remove all zero-width characters from text."""
    return "".join(ch for ch in text if ch not in ZW_CHARS)


def hide_in_text(
    cover_text: str,
    data: bytes,
) -> str:
    """Hide binary data within cover text using zero-width characters.

    The payload is inserted after the first visible character of the
    cover text, wrapped in delimiters.

    Args:
        cover_text: The visible text to use as a carrier.
        data: Binary data to hide.

    Returns:
        The stego text with hidden data embedded.
    """
    clean = _strip_zwc(cover_text)
    if len(clean) < 1:
        raise ValueError("Cover text must contain at least 1 visible character.")

    # Prefix the payload with a magic marker and declared byte length so the
    # extractor can validate that it is reading BIS data, not random ZW chars.
    payload = MAGIC + struct.pack(">I", len(data)) + data
    bits = _bytes_to_bits(payload)
    zwc_payload = ZW_DELIM + _bits_to_zwc(bits) + ZW_DELIM

    # The visible text stays unchanged because the payload is inserted with
    # zero-width characters that do not render on screen.
    stego_text = clean[0] + zwc_payload + clean[1:]

    return stego_text


def extract_from_text(stego_text: str) -> bytes:
    """Extract hidden binary data from stego text.

    Finds the ZW delimiter markers and decodes the payload between them.

    Args:
        stego_text: Text that may contain hidden zero-width data.

    Returns:
        The hidden binary data.
    """
    # Find the delimited ZW payload
    delim_positions = [i for i, ch in enumerate(stego_text) if ch == ZW_DELIM]
    if len(delim_positions) < 2:
        raise ValueError(
            "No hidden data found in this text (missing delimiters)."
        )

    # Extract ZW chars between the first two delimiters
    start = delim_positions[0] + 1
    end = delim_positions[1]
    zwc_payload = stego_text[start:end]

    # Convert ZW chars to bits
    bits = _zwc_to_bits(zwc_payload)

    # Parse header
    header_bits_count = HEADER_SIZE * 8
    if len(bits) < header_bits_count:
        raise ValueError("Hidden data is corrupted (too short for header).")

    header_bytes = _bits_to_bytes(bits[:header_bits_count])
    magic = header_bytes[:4]
    if magic != MAGIC:
        raise ValueError(
            "No hidden data found in this text (magic header mismatch)."
        )

    data_length = struct.unpack(">I", header_bytes[4:8])[0]
    total_bits_needed = (HEADER_SIZE + data_length) * 8
    if len(bits) < total_bits_needed:
        raise ValueError(
            f"Corrupted data — claimed {data_length} bytes but payload is too short."
        )

    data_bits = bits[header_bits_count : header_bits_count + data_length * 8]
    return _bits_to_bytes(data_bits)[:data_length]


def hide_text_in_text(
    cover_text: str,
    secret_text: str,
    password: str | None = None,
) -> str:
    """Hide a secret text message within cover text, optionally encrypted.

    Args:
        cover_text: The visible carrier text.
        secret_text: The secret message to hide.
        password: Optional AES-256 encryption password.

    Returns:
        The stego text with the secret embedded as zero-width characters.
    """
    data = secret_text.encode("utf-8")
    if password:
        from bis.utils.crypto import encrypt_bytes
        data = encrypt_bytes(data, password)
    # A leading flag byte tells extraction whether it must decrypt first.
    flag = b"\x01" if password else b"\x00"
    payload = flag + data
    return hide_in_text(cover_text, payload)


def extract_text_from_text(
    stego_text: str,
    password: str | None = None,
) -> str:
    """Extract a hidden text message from stego text.

    Args:
        stego_text: Text containing hidden zero-width data.
        password: Decryption password (if the message was encrypted).

    Returns:
        The hidden text message.
    """
    payload = extract_from_text(stego_text)
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


def get_visible_text(stego_text: str) -> str:
    """Get the visible text from a stego text (strip hidden data)."""
    return _strip_zwc(stego_text)
