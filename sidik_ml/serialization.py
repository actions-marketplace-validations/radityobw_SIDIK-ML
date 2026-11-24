"""Integrity verification and safe offline deserialization for trained Non-ANN models."""

import hashlib
import hmac
import os
from typing import Any, List, Optional


def compute_file_sha256(file_path: str) -> str:
    """Computes SHA-256 hex digest of an artifact file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_artifact_sha256(file_path: str, expected_sha256: str) -> bool:
    """Verifies an artifact file matches its recorded SHA-256 digest in constant time."""
    if not os.path.isfile(file_path):
        return False
    actual = compute_file_sha256(file_path)
    return hmac.compare_digest(actual.lower(), expected_sha256.strip().lower())


def load_verified_skops_model(
    model_path: str,
    expected_sha256: str,
    trusted_types: Optional[List[str]] = None,
) -> Any:
    """Loads a skops-serialized scikit-learn model after verifying its SHA-256 checksum.

    Never invokes raw pickle or joblib deserialization.
    """
    if not verify_artifact_sha256(model_path, expected_sha256):
        raise ValueError(f"Checksum mismatch or missing model artifact: {model_path}")

    try:
        import skops.io as sio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "skops is required for safe offline model loading. Install with: pip install 'sidik-ml[train]'"
        ) from exc

    allowed = trusted_types if trusted_types is not None else sio.get_untrusted_types(file=model_path)
    return sio.load(model_path, trusted=allowed)
