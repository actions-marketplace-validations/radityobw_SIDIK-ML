"""Integrity verification and safe offline serialization/deserialization for trained Non-ANN models."""

import hashlib
import hmac
import json
import math
import os
import struct
import sys
import urllib.request
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sklearn
import skops

from sidik_ml.features import FEATURE_NAMES

SAFE_SKOPS_TYPE_PREFIXES = ("sklearn.", "numpy.", "scipy.", "builtins.", "collections.")
PORTABLE_GBDT_FORMAT = "sidik_portable_gbdt_v1"
PORTABLE_LINEAR_FORMAT = "sidik_portable_linear_v1"
PORTABLE_TREE_ENSEMBLE_FORMAT = "sidik_portable_tree_ensemble_v1"
MAX_MODEL_DOWNLOAD_BYTES = 64 * 1024 * 1024  # 64 MB artifact guard

BUNDLED_MODEL_FILES: Dict[str, str] = {
    "gradient_boosting": "default_gradient_boosting.json",
    "random_forest": "default_random_forest.json",
    "extra_trees": "default_extra_trees.json",
    "logistic_regression": "default_logistic_regression.json",
    "decision_tree": "default_decision_tree.json",
}

BUNDLED_MODEL_ALIASES: Dict[str, str] = {
    "default": "gradient_boosting",
    "gb": "gradient_boosting",
    "gbdt": "gradient_boosting",
    "gradient_boosting": "gradient_boosting",
    "gradient-boosting": "gradient_boosting",
    "rf": "random_forest",
    "random_forest": "random_forest",
    "random-forest": "random_forest",
    "et": "extra_trees",
    "extra_trees": "extra_trees",
    "extra-trees": "extra_trees",
    "lr": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "logistic-regression": "logistic_regression",
    "dt": "decision_tree",
    "decision_tree": "decision_tree",
    "decision-tree": "decision_tree",
}

BUNDLED_MODEL_PROFILES: Dict[str, str] = {
    "gradient_boosting": "Default (CV Winner, fast balanced triage)",
    "random_forest": "Peak held-out Recall/F1 ensemble",
    "extra_trees": "Low-FPR randomized tree ensemble",
    "logistic_regression": "Zero-FPR linear baseline, ultra-lightweight",
    "decision_tree": "Ultra-fast single interpretable tree",
}


def _to_float32(val: float) -> float:
    """Casts a Python float to IEEE-754 float32 precision matching scikit-learn tree internals."""
    return struct.unpack("f", struct.pack("f", float(val)))[0]


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


class PortableGradientBoostingClassifier:
    """Version-independent binary Gradient Boosting tree evaluator with pure-Python inference."""

    def __init__(
        self,
        model_name: str,
        feature_names: Sequence[str],
        learning_rate: float,
        init_score: float,
        feature_importances: Sequence[float],
        trees: Sequence[Dict[str, Sequence[Any]]],
    ) -> None:
        self.model_name = model_name
        self.feature_names = list(feature_names)
        self.n_features_in_ = len(self.feature_names)
        self.classes_ = [0, 1]
        self.learning_rate = float(learning_rate)
        self.init_score = float(init_score)
        self.feature_importances_ = [float(v) for v in feature_importances]
        self.trees = [
            {
                "f": [int(x) for x in tr["f"]],
                "t": [float(x) for x in tr["t"]],
                "l": [int(x) for x in tr["l"]],
                "r": [int(x) for x in tr["r"]],
                "v": [float(x) for x in tr["v"]],
            }
            for tr in trees
        ]

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        """Predicts [P(false_positive), P(true_secret)] for each feature vector."""
        probs: List[List[float]] = []
        lr = self.learning_rate
        base = self.init_score
        for row in X:
            if len(row) != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} features, got {len(row)}.")
            rf = [_to_float32(v) for v in row]
            score = base
            for tr in self.trees:
                idx = 0
                f = tr["f"]
                th = tr["t"]
                left = tr["l"]
                right = tr["r"]
                val = tr["v"]
                while f[idx] >= 0:
                    idx = left[idx] if rf[f[idx]] <= th[idx] else right[idx]
                score += lr * val[idx]
            clamped = max(-500.0, min(500.0, score))
            p1 = 1.0 / (1.0 + math.exp(-clamped))
            probs.append([1.0 - p1, p1])
        return probs

    def predict(self, X: Sequence[Sequence[float]]) -> List[int]:
        """Predicts binary labels (1 = true secret, 0 = false positive)."""
        return [1 if p[1] >= 0.5 else 0 for p in self.predict_proba(X)]


class PortableLogisticRegressionClassifier:
    """Version-independent binary Logistic Regression evaluator with StandardScaler stats."""

    def __init__(
        self,
        model_name: str,
        feature_names: Sequence[str],
        scaler_mean: Sequence[float],
        scaler_scale: Sequence[float],
        coef: Sequence[float],
        intercept: float,
    ) -> None:
        self.model_name = model_name
        self.feature_names = list(feature_names)
        self.n_features_in_ = len(self.feature_names)
        self.classes_ = [0, 1]
        self.scaler_mean = [float(v) for v in scaler_mean]
        self.scaler_scale = [float(v) for v in scaler_scale]
        self._coef_vec = [float(v) for v in coef]
        self.coef_ = [list(self._coef_vec)]
        self.intercept_ = [float(intercept)]

    def scale_feature_vector(self, row: Sequence[float]) -> List[float]:
        """Applies stored StandardScaler normalization to one feature vector."""
        if len(row) != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {len(row)}.")
        return [
            (float(v) - m) / s if s != 0.0 else (float(v) - m)
            for v, m, s in zip(row, self.scaler_mean, self.scaler_scale)
        ]

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        """Predicts [P(false_positive), P(true_secret)] for each feature vector."""
        probs: List[List[float]] = []
        b0 = self.intercept_[0]
        coefs = self._coef_vec
        for row in X:
            scaled = self.scale_feature_vector(row)
            score = b0 + sum(c * xv for c, xv in zip(coefs, scaled))
            clamped = max(-500.0, min(500.0, score))
            p1 = 1.0 / (1.0 + math.exp(-clamped))
            probs.append([1.0 - p1, p1])
        return probs

    def predict(self, X: Sequence[Sequence[float]]) -> List[int]:
        """Predicts binary labels (1 = true secret, 0 = false positive)."""
        return [1 if p[1] >= 0.5 else 0 for p in self.predict_proba(X)]


class PortableTreeEnsembleClassifier:
    """Version-independent binary DecisionTree / RandomForest / ExtraTrees pure-Python evaluator."""

    def __init__(
        self,
        model_name: str,
        feature_names: Sequence[str],
        feature_importances: Sequence[float],
        trees: Sequence[Dict[str, Sequence[Any]]],
    ) -> None:
        self.model_name = model_name
        self.feature_names = list(feature_names)
        self.n_features_in_ = len(self.feature_names)
        self.classes_ = [0, 1]
        self.feature_importances_ = [float(v) for v in feature_importances]
        self.trees = [
            {
                "f": [int(x) for x in tr["f"]],
                "t": [float(x) for x in tr["t"]],
                "l": [int(x) for x in tr["l"]],
                "r": [int(x) for x in tr["r"]],
                "p": [float(x) for x in tr["p"]],
            }
            for tr in trees
        ]

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        """Predicts [P(false_positive), P(true_secret)] by averaging tree leaf probabilities."""
        probs: List[List[float]] = []
        n_trees = float(len(self.trees))
        for row in X:
            if len(row) != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} features, got {len(row)}.")
            rf = [_to_float32(v) for v in row]
            acc_p1 = 0.0
            for tr in self.trees:
                idx = 0
                f = tr["f"]
                th = tr["t"]
                left = tr["l"]
                right = tr["r"]
                leaf_p = tr["p"]
                while f[idx] >= 0:
                    idx = left[idx] if rf[f[idx]] <= th[idx] else right[idx]
                acc_p1 += leaf_p[idx]
            p1 = max(0.0, min(1.0, acc_p1 / n_trees))
            probs.append([1.0 - p1, p1])
        return probs

    def predict(self, X: Sequence[Sequence[float]]) -> List[int]:
        """Predicts binary labels (1 = true secret, 0 = false positive)."""
        return [1 if p[1] >= 0.5 else 0 for p in self.predict_proba(X)]


def export_portable_gbdt_model(
    model: Any,
    output_path: str,
    manifest_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Exports a fitted binary GradientBoostingClassifier into portable verified JSON + manifest."""
    if not hasattr(model, "estimators_") or not hasattr(model, "_raw_predict_init"):
        raise ValueError("export_portable_gbdt_model requires a fitted scikit-learn GradientBoostingClassifier.")
    n_feat = int(getattr(model, "n_features_in_", len(FEATURE_NAMES)))
    if n_feat != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} input features, got {n_feat}.")

    raw_init = float(model._raw_predict_init([[0.0] * len(FEATURE_NAMES)])[0][0])
    trees: List[Dict[str, Any]] = []
    for stage in model.estimators_:
        t = stage[0].tree_
        trees.append({
            "f": [int(x) for x in t.feature.tolist()],
            "t": [round(float(x), 12) for x in t.threshold.tolist()],
            "l": [int(x) for x in t.children_left.tolist()],
            "r": [int(x) for x in t.children_right.tolist()],
            "v": [round(float(x[0][0]), 12) for x in t.value.tolist()],
        })

    payload = {
        "model_format": PORTABLE_GBDT_FORMAT,
        "model_name": str((metadata or {}).get("model_name", "gradient_boosting")),
        "feature_names": list(FEATURE_NAMES),
        "learning_rate": float(model.learning_rate),
        "init_score": raw_init,
        "feature_importances": [round(float(x), 12) for x in model.feature_importances_.tolist()],
        "trees": trees,
    }

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    digest = compute_file_sha256(output_path)
    meta = dict(metadata or {})
    meta.setdefault("n_trees", len(trees))
    meta.setdefault("n_features", len(FEATURE_NAMES))
    manifest = {
        "model_file": os.path.basename(output_path),
        "sha256": digest,
        "format": PORTABLE_GBDT_FORMAT,
        "metadata": meta,
    }
    m_path = manifest_path or f"{output_path}.manifest.json"
    with open(m_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    return manifest


def load_verified_portable_model(model_path: str, expected_sha256: str) -> Any:
    """Loads a portable JSON Non-ANN model after verifying its SHA-256 digest."""
    if not verify_artifact_sha256(model_path, expected_sha256):
        raise ValueError(f"Checksum mismatch or missing portable model artifact: {model_path}")
    with open(model_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid portable model payload in {model_path}")
    fmt = payload.get("model_format")
    if payload.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError(f"Portable model feature schema mismatch in {model_path}")

    if fmt == PORTABLE_GBDT_FORMAT:
        trees = payload.get("trees")
        if not isinstance(trees, list) or not trees:
            raise ValueError(f"Portable GBDT model contains no decision trees: {model_path}")
        return PortableGradientBoostingClassifier(
            model_name=str(payload.get("model_name", "gradient_boosting")),
            feature_names=payload["feature_names"],
            learning_rate=float(payload["learning_rate"]),
            init_score=float(payload["init_score"]),
            feature_importances=payload["feature_importances"],
            trees=trees,
        )

    if fmt == PORTABLE_LINEAR_FORMAT:
        return PortableLogisticRegressionClassifier(
            model_name=str(payload.get("model_name", "logistic_regression")),
            feature_names=payload["feature_names"],
            scaler_mean=payload["scaler_mean"],
            scaler_scale=payload["scaler_scale"],
            coef=payload["coef"],
            intercept=float(payload["intercept"]),
        )

    if fmt == PORTABLE_TREE_ENSEMBLE_FORMAT:
        trees = payload.get("trees")
        if not isinstance(trees, list) or not trees:
            raise ValueError(f"Portable tree-ensemble model contains no trees: {model_path}")
        return PortableTreeEnsembleClassifier(
            model_name=str(payload.get("model_name", "random_forest")),
            feature_names=payload["feature_names"],
            feature_importances=payload["feature_importances"],
            trees=trees,
        )

    raise ValueError(f"Unsupported portable model format '{fmt}' in {model_path}")


def export_portable_linear_model(
    model: Any,
    output_path: str,
    manifest_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Exports a fitted LogisticRegression (or Pipeline[StandardScaler, LogisticRegression]) to verified JSON."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        scaler = model.named_steps.get("scaler")
        clf = model.steps[-1][1]
    else:
        scaler = None
        clf = model

    if not isinstance(clf, LogisticRegression) or not hasattr(clf, "coef_"):
        raise ValueError("export_portable_linear_model requires a fitted LogisticRegression or Pipeline.")
    n_feat = int(getattr(clf, "n_features_in_", len(FEATURE_NAMES)))
    if n_feat != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} input features, got {n_feat}.")

    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean_vec = [float(x) for x in scaler.mean_.tolist()]
        scale_vec = [float(x) for x in scaler.scale_.tolist()]
    else:
        mean_vec = [0.0] * len(FEATURE_NAMES)
        scale_vec = [1.0] * len(FEATURE_NAMES)

    raw_coef = clf.coef_[0] if len(clf.coef_.shape) > 1 else clf.coef_
    intercept_val = float(clf.intercept_[0] if hasattr(clf.intercept_, "__len__") else clf.intercept_)
    payload = {
        "model_format": PORTABLE_LINEAR_FORMAT,
        "model_name": str((metadata or {}).get("model_name", "logistic_regression")),
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": mean_vec,
        "scaler_scale": scale_vec,
        "coef": [float(x) for x in raw_coef.tolist()],
        "intercept": intercept_val,
    }

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    digest = compute_file_sha256(output_path)
    meta = dict(metadata or {})
    meta.setdefault("n_features", len(FEATURE_NAMES))
    manifest = {
        "model_file": os.path.basename(output_path),
        "sha256": digest,
        "format": PORTABLE_LINEAR_FORMAT,
        "metadata": meta,
    }
    m_path = manifest_path or f"{output_path}.manifest.json"
    with open(m_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    return manifest


def export_portable_tree_ensemble_model(
    model: Any,
    output_path: str,
    manifest_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Exports a fitted DecisionTree, RandomForest, or ExtraTrees classifier to verified JSON."""
    if hasattr(model, "_raw_predict_init"):
        raise ValueError("Use export_portable_gbdt_model for GradientBoostingClassifier.")
    if hasattr(model, "tree_"):
        estimators = [model]
        default_name = "decision_tree"
    elif hasattr(model, "estimators_") and model.estimators_:
        estimators = list(model.estimators_)
        default_name = "random_forest" if "RandomForest" in type(model).__name__ else "extra_trees"
    else:
        raise ValueError("export_portable_tree_ensemble_model requires DecisionTree, RandomForest, or ExtraTrees.")

    n_feat = int(getattr(model, "n_features_in_", len(FEATURE_NAMES)))
    if n_feat != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} input features, got {n_feat}.")

    trees: List[Dict[str, Any]] = []
    for est in estimators:
        t = est.tree_
        probs_leaf: List[float] = []
        for row_v in t.value.tolist():
            counts = row_v[0]
            if len(counts) < 2:
                raise ValueError("Portable tree export requires a binary classifier with 2 classes.")
            c0 = float(counts[0])
            c1 = float(counts[1])
            tot = c0 + c1
            probs_leaf.append(c1 / tot if tot > 0.0 else 0.5)
        trees.append({
            "f": [int(x) for x in t.feature.tolist()],
            "t": [float(x) for x in t.threshold.tolist()],
            "l": [int(x) for x in t.children_left.tolist()],
            "r": [int(x) for x in t.children_right.tolist()],
            "p": probs_leaf,
        })

    model_name = str((metadata or {}).get("model_name", default_name))
    payload = {
        "model_format": PORTABLE_TREE_ENSEMBLE_FORMAT,
        "model_name": model_name,
        "feature_names": list(FEATURE_NAMES),
        "feature_importances": [float(x) for x in model.feature_importances_.tolist()],
        "trees": trees,
    }

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    digest = compute_file_sha256(output_path)
    meta = dict(metadata or {})
    meta.setdefault("n_trees", len(trees))
    meta.setdefault("n_features", len(FEATURE_NAMES))
    manifest = {
        "model_file": os.path.basename(output_path),
        "sha256": digest,
        "format": PORTABLE_TREE_ENSEMBLE_FORMAT,
        "metadata": meta,
    }
    m_path = manifest_path or f"{output_path}.manifest.json"
    with open(m_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    return manifest


def export_portable_model(
    model: Any,
    output_path: str,
    manifest_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Exports any fitted supported Non-ANN classifier into a verified portable JSON artifact."""
    from sklearn.pipeline import Pipeline

    est = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if hasattr(model, "_raw_predict_init") and hasattr(model, "estimators_"):
        return export_portable_gbdt_model(model, output_path, manifest_path=manifest_path, metadata=metadata)
    if hasattr(est, "coef_"):
        return export_portable_linear_model(model, output_path, manifest_path=manifest_path, metadata=metadata)
    if hasattr(model, "tree_") or hasattr(model, "estimators_"):
        return export_portable_tree_ensemble_model(model, output_path, manifest_path=manifest_path, metadata=metadata)
    raise ValueError(f"Unsupported Non-ANN estimator type for portable JSON export: {type(model).__name__}")


def load_verified_model(
    model_path: str,
    expected_sha256: str,
    trusted_types: Optional[List[str]] = None,
) -> Any:
    """Loads either a verified portable JSON Non-ANN model or a verified .skops artifact."""
    if model_path.lower().endswith(".json"):
        return load_verified_portable_model(model_path, expected_sha256)
    return load_verified_skops_model(model_path, expected_sha256, trusted_types=trusted_types)


def resolve_bundled_model_name(name: str) -> Optional[str]:
    """Resolves canonical or alias preset name (including 'preset:<name>') to a bundled model key."""
    cleaned = name.strip().lower()
    if cleaned.startswith("preset:"):
        cleaned = cleaned[len("preset:"):].strip()
    return BUNDLED_MODEL_ALIASES.get(cleaned)


def get_bundled_model_path(model_name: str = "gradient_boosting") -> Optional[str]:
    """Returns path to a bundled SHA-256-verified portable Non-ANN model if present on disk."""
    resolved = resolve_bundled_model_name(model_name)
    if not resolved:
        return None
    filename = BUNDLED_MODEL_FILES.get(resolved)
    if not filename:
        return None
    candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", filename)
    if os.path.isfile(candidate) and os.path.isfile(f"{candidate}.manifest.json"):
        return candidate
    return None


def get_bundled_default_model_path() -> Optional[str]:
    """Returns path to the bundled SHA-256-verified portable Gradient Boosting model if present."""
    return get_bundled_model_path("gradient_boosting")


def list_bundled_models() -> Dict[str, Dict[str, Any]]:
    """Lists all available bundled pre-trained Non-ANN models with sizes, checksums, and profiles."""
    catalog: Dict[str, Dict[str, Any]] = {}
    for canonical, filename in BUNDLED_MODEL_FILES.items():
        path = get_bundled_model_path(canonical)
        if not path:
            continue
        manifest_path = f"{path}.manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as mf:
            manifest = json.load(mf)
        digest = str(manifest.get("sha256", "")).strip()
        catalog[canonical] = {
            "model_name": canonical,
            "model_file": filename,
            "path": path,
            "sha256": digest,
            "format": manifest.get("format", ""),
            "size_kb": round(os.path.getsize(path) / 1024.0, 2),
            "is_default": canonical == "gradient_boosting",
            "profile": BUNDLED_MODEL_PROFILES.get(canonical, ""),
            "metadata": manifest.get("metadata", {}),
        }
    return catalog


def load_bundled_model(model_name: str = "gradient_boosting") -> Any:
    """Loads and SHA-256-verifies a bundled pre-trained Non-ANN model by preset name or alias."""
    resolved = resolve_bundled_model_name(model_name)
    path = get_bundled_model_path(model_name)
    if not resolved or not path:
        raise ValueError(
            f"Bundled Non-ANN model '{model_name}' not found. Available: {', '.join(BUNDLED_MODEL_FILES.keys())}"
        )
    with open(f"{path}.manifest.json", "r", encoding="utf-8") as mf:
        digest = str(json.load(mf).get("sha256", "")).strip()
    return load_verified_portable_model(path, digest)
