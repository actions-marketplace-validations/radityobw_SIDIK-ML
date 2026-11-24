"""Data contracts and safe triage policies for SIDIK-ML."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple


@dataclass
class TriagePolicy:
    """Configures how ML predictions annotate or filter deterministic SIDIK findings.

    By default, suppress_false_positives is False (advisory ranking only) and
    CRITICAL severity findings are never suppressed silently.
    """
    decision_threshold: float = 0.5
    suppress_false_positives: bool = False
    never_suppress_severities: Tuple[str, ...] = ("CRITICAL",)
    preserve_deterministic_evidence: bool = True


@dataclass
class MLPrediction:
    """Explainable Non-ANN classifier output for a single SecretCandidate."""
    candidate_id: str
    predicted_is_secret: bool
    secret_probability: float
    model_name: str
    model_version: str
    feature_schema_version: str
    top_feature_contributions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HybridFinding:
    """Combined deterministic SIDIK evidence and optional SIDIK-ML triage signal."""
    candidate_id: str
    rule_id: str
    category: str
    severity: str
    file_path: str
    line_number: int
    masked_value: str
    passed_deterministic_filter: bool
    deterministic_confidence: str
    ml_prediction: MLPrediction
    final_action: str  # "KEEP_ALERT", "REVIEW_PRIORITY", or "SUPPRESSED_FP"
    suppressed_by_ml: bool = False
    deterministic_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ml_prediction"] = self.ml_prediction.to_dict()
        return d
