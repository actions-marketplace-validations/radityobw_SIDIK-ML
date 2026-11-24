"""Integrity verification and safe offline serialization/deserialization for trained Non-ANN models."""

import hashlib
import hmac
import json
import os
from typing import Any, Dict, List, Optional


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


def save_verified_skops_model(
    model: Any,
    model_path: str,
    manifest_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serializes a scikit-learn model to .skops format and writes a SHA-256 verification manifest."""
    try:
        import skops.io as sio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "skops is required for safe offline model serialization. Install with: pip install 'sidik-ml[train]'"
        ) from exc

    parent = os.path.dirname(os.path.abspath(model_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    sio.dump(model, model_path)
    digest = compute_file_sha256(model_path)

    manifest = {
        "model_file": os.path.basename(model_path),
        "sha256": digest,
        "format": "skops",
        "metadata": dict(metadata or {}),
    }
    m_path = manifest_path or f"{model_path}.manifest.json"
    with open(m_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    return manifest


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
