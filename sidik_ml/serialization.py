"""Integrity verification and safe offline serialization/deserialization for trained Non-ANN models."""

import hashlib
import hmac
import json
import os
import sys
import urllib.request
import warnings
from typing import Any, Dict, List, Optional, Tuple

import sklearn
import skops


SAFE_SKOPS_TYPE_PREFIXES = ("sklearn.", "numpy.", "scipy.", "builtins.", "collections.")
MAX_MODEL_DOWNLOAD_BYTES = 64 * 1024 * 1024  # 64 MB artifact guard


def get_ml_cache_dir() -> str:
    """Returns a user-writable cache directory for verified SIDIK-ML artifacts."""
    override = os.environ.get("SIDIK_ML_CACHE_DIR", "").strip()
    if override:
        return os.path.abspath(override)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "sidik-ml")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Caches/sidik-ml")
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(xdg, "sidik-ml")


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

    meta = dict(metadata or {})
    meta.setdefault("sklearn_version", sklearn.__version__)
    meta.setdefault("skops_version", getattr(skops, "__version__", "unknown"))

    manifest = {
        "model_file": os.path.basename(model_path),
        "sha256": digest,
        "format": "skops",
        "metadata": meta,
    }
    m_path = manifest_path or f"{model_path}.manifest.json"
    with open(m_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    return manifest


def _warn_on_sklearn_version_skew(model_path: str) -> None:
    """Emits a RuntimeWarning when the manifest's scikit-learn major.minor differs from runtime."""
    manifest_path = f"{model_path}.manifest.json"
    if not os.path.isfile(manifest_path):
        return
    try:
        with open(manifest_path, "r", encoding="utf-8") as mf:
            data = json.load(mf)
        recorded = str((data.get("metadata") or {}).get("sklearn_version", "")).strip()
        if not recorded:
            return
        rec_mm = tuple(recorded.split(".")[:2])
        cur_mm = tuple(sklearn.__version__.split(".")[:2])
        if rec_mm != cur_mm:
            warnings.warn(
                f"Model {os.path.basename(model_path)} was saved with scikit-learn {recorded}, "
                f"but runtime has {sklearn.__version__}. Tree/estimator internals may differ across versions.",
                RuntimeWarning,
                stacklevel=3,
            )
    except Exception:
        pass


def fetch_verified_model(
    url: str,
    expected_sha256: str,
    output_path: Optional[str] = None,
    timeout: int = 120,
) -> Tuple[str, str]:
    """Downloads an HTTPS .skops model artifact, verifies SHA-256 before saving, and writes manifest."""
    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS model URL: {url}")
    if not expected_sha256 or not expected_sha256.strip():
        raise ValueError("expected_sha256 is required to verify downloaded model artifact.")

    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 (validated https URL)
        raw = response.read(MAX_MODEL_DOWNLOAD_BYTES + 1)
    if len(raw) > MAX_MODEL_DOWNLOAD_BYTES:
        raise ValueError(f"Downloaded model exceeds {MAX_MODEL_DOWNLOAD_BYTES} bytes safety limit.")

    actual = hashlib.sha256(raw).hexdigest()
    expected = expected_sha256.strip().lower()
    if not hmac.compare_digest(actual.lower(), expected):
        raise ValueError(f"SHA-256 verification failed: expected {expected}, got {actual}")

    if output_path:
        dest_path = os.path.abspath(output_path)
    else:
        fname = os.path.basename(url.split("?", 1)[0]) or "sidik_ml_model.skops"
        if not fname.endswith(".skops"):
            fname += ".skops"
        dest_path = os.path.join(get_ml_cache_dir(), "models", fname)

    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(raw)

    manifest = {
        "model_file": os.path.basename(dest_path),
        "sha256": actual,
        "format": "skops",
        "metadata": {
            "source_url": url,
            "sklearn_version": sklearn.__version__,
            "skops_version": getattr(skops, "__version__", "unknown"),
        },
    }
    with open(f"{dest_path}.manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    return dest_path, actual


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

    _warn_on_sklearn_version_skew(model_path)

    try:
        import skops.io as sio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "skops is required for safe offline model loading. Install with: pip install 'sidik-ml'"
        ) from exc

    if trusted_types is not None:
        allowed = trusted_types
    else:
        untrusted = sio.get_untrusted_types(file=model_path)
        rejected = [
            t for t in untrusted
            if not any(t.startswith(prefix) for prefix in SAFE_SKOPS_TYPE_PREFIXES)
        ]
        if rejected:
            raise ValueError(f"Refusing to load untrusted skops type(s): {rejected}")
        allowed = untrusted
    try:
        return sio.load(model_path, trusted=allowed)
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to deserialize {model_path} with scikit-learn {sklearn.__version__}: {exc}. "
            "If this artifact was trained on an incompatible scikit-learn version, re-train with 'sidik-ml train'."
        ) from exc
