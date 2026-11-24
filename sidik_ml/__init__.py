"""SIDIK-ML: Explainable Non-Neural ML Extension for SIDIK Static Security Analysis."""

from sidik_ml.schemas import MLPrediction, HybridFinding, TriagePolicy
from sidik_ml.features import FEATURE_SCHEMA_VERSION, FEATURE_NAMES, extract_feature_dict, extract_feature_vector
from sidik_ml.serialization import compute_file_sha256, verify_artifact_sha256, load_verified_skops_model
from sidik_ml.integration import HybridTriagePipeline

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "MLPrediction",
    "HybridFinding",
    "TriagePolicy",
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_NAMES",
    "extract_feature_dict",
    "extract_feature_vector",
    "compute_file_sha256",
    "verify_artifact_sha256",
    "load_verified_skops_model",
    "HybridTriagePipeline",
]
