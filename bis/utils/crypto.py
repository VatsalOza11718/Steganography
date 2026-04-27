"""AES-256-GCM encryption/decryption for end-to-end steganography security.

Provides password-based encryption using:
- PBKDF2 key derivation with random salt
- AES-256-GCM authenticated encryption
- Compact binary format: salt(16) + nonce(12) + tag(16) + ciphertext
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import Union

from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256, HMAC


# ─── Constants ────────────────────────────────────────────
SALT_SIZE = 16
NONCE_SIZE = 12
TAG_SIZE = 16
KEY_SIZE = 32  # AES-256
KDF_ITERATIONS = 100_000


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 256-bit key from password using PBKDF2-SHA256.

    Args:
        password: User-provided password string.
        salt: Random salt bytes.

    Returns:
        32-byte derived key.
    """
    # PBKDF2 intentionally stretches a human password into a strong binary key
    # so the encryption step does not rely on low-entropy input directly.
    return PBKDF2(
        password.encode("utf-8"),
        salt,
        dkLen=KEY_SIZE,
        count=KDF_ITERATIONS,
        prf=lambda p, s: HMAC.new(p, s, SHA256).digest(),
    )


def encrypt_message(plaintext: str, password: str) -> bytes:
    """Encrypt a plaintext message with AES-256-GCM.

    Format: salt(16) || nonce(12) || tag(16) || ciphertext(N)

    Args:
        plaintext: Message to encrypt.
        password: Encryption password.

    Returns:
        Encrypted binary data.

    Raises:
        ValueError: If password is empty.
    """
    if not password:
        raise ValueError("Password cannot be empty for encryption.")

    # A fresh salt and nonce ensure that encrypting the same message twice
    # does not produce the same output bytes.
    salt = os.urandom(SALT_SIZE)
    key = _derive_key(password, salt)

    cipher = AES.new(key, AES.MODE_GCM, nonce=os.urandom(NONCE_SIZE))
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode("utf-8"))

    # The binary blob is self-contained, so the decryptor can recover every
    # parameter it needs directly from the stored payload.
    return salt + cipher.nonce + tag + ciphertext


def decrypt_message(data: bytes, password: str) -> str:
    """Decrypt AES-256-GCM encrypted data.

    Args:
        data: Encrypted binary (salt + nonce + tag + ciphertext).
        password: Decryption password.

    Returns:
        Decrypted plaintext string.

    Raises:
        ValueError: If data is too short or password is wrong.
    """
    if not password:
        raise ValueError("Password cannot be empty for decryption.")

    min_size = SALT_SIZE + NONCE_SIZE + TAG_SIZE
    if len(data) < min_size:
        raise ValueError(
            f"Encrypted data too short ({len(data)} bytes, minimum {min_size})."
        )

    # Slice the packed payload back into its cryptographic components.
    salt = data[:SALT_SIZE]
    nonce = data[SALT_SIZE : SALT_SIZE + NONCE_SIZE]
    tag = data[SALT_SIZE + NONCE_SIZE : SALT_SIZE + NONCE_SIZE + TAG_SIZE]
    ciphertext = data[SALT_SIZE + NONCE_SIZE + TAG_SIZE :]

    key = _derive_key(password, salt)

    try:
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        return plaintext.decode("utf-8")
    except (ValueError, KeyError) as e:
        raise ValueError(
            "Decryption failed — wrong password or corrupted data."
        ) from e


def encrypt_bytes(data: bytes, password: str) -> bytes:
    """Encrypt raw bytes with AES-256-GCM.

    Args:
        data: Raw bytes to encrypt.
        password: Encryption password.

    Returns:
        Encrypted binary data.
    """
    if not password:
        raise ValueError("Password cannot be empty for encryption.")

    salt = os.urandom(SALT_SIZE)
    key = _derive_key(password, salt)

    cipher = AES.new(key, AES.MODE_GCM, nonce=os.urandom(NONCE_SIZE))
    ciphertext, tag = cipher.encrypt_and_digest(data)

    return salt + cipher.nonce + tag + ciphertext


def decrypt_bytes(data: bytes, password: str) -> bytes:
    """Decrypt AES-256-GCM encrypted raw bytes.

    Args:
        data: Encrypted binary data.
        password: Decryption password.

    Returns:
        Decrypted raw bytes.
    """
    if not password:
        raise ValueError("Password cannot be empty for decryption.")

    min_size = SALT_SIZE + NONCE_SIZE + TAG_SIZE
    if len(data) < min_size:
        raise ValueError(
            f"Encrypted data too short ({len(data)} bytes, minimum {min_size})."
        )

    salt = data[:SALT_SIZE]
    nonce = data[SALT_SIZE : SALT_SIZE + NONCE_SIZE]
    tag = data[SALT_SIZE + NONCE_SIZE : SALT_SIZE + NONCE_SIZE + TAG_SIZE]
    ciphertext = data[SALT_SIZE + NONCE_SIZE + TAG_SIZE :]

    key = _derive_key(password, salt)

    try:
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        return cipher.decrypt_and_verify(ciphertext, tag)
    except (ValueError, KeyError) as e:
        raise ValueError(
            "Decryption failed — wrong password or corrupted data."
        ) from e
