"""Hybrid integration pipeline pairing deterministic SIDIK scanning with SIDIK-ML triage."""

from typing import Dict, List, Optional, Protocol, Sequence
from sidik.core import SecurityScanner
from sidik.models import SecretCandidate
from sidik_ml.explainability import _unwrap_estimator, explain_single_candidate
from sidik_ml.features import FEATURE_SCHEMA_VERSION, FEATURE_NAMES, extract_feature_vector
from sidik_ml.schemas import HybridFinding, MLPrediction, TriagePolicy


class ClassifierProtocol(Protocol):
    """Minimal scikit-learn-compatible classifier interface for offline Non-ANN inference."""

    def predict_proba(self, X: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
        ...


class HybridTriagePipeline:
    """Executes SIDIK candidate extraction followed by Non-ANN ML triage and fail-safe policy enforcement."""

    def __init__(
        self,
        scanner: Optional[SecurityScanner] = None,
        classifier: Optional[ClassifierProtocol] = None,
        model_name: str = "deterministic_pass_through",
        model_version: str = "0.5.1",
        policy: Optional[TriagePolicy] = None,
    ):
        self.scanner = scanner or SecurityScanner(enable_dependency_scan=False, measure_memory=False)
        self.classifier = classifier
        self.model_name = model_name
        self.model_version = model_version
        self.policy = policy or TriagePolicy()

    def triage_candidates(self, candidates: List[SecretCandidate]) -> List[HybridFinding]:
        """Applies feature extraction, optional ML inference, and TriagePolicy over SecretCandidate list."""
        results: List[HybridFinding] = []
        if not candidates:
            return results

        vectors = [extract_feature_vector(c) for c in candidates]
        if self.classifier is not None:
            probs = self.classifier.predict_proba(vectors)
            secret_probs = [float(p[1]) if len(p) > 1 else float(p[0]) for p in probs]
        else:
            secret_probs = [
                1.0 if c.deterministic_decision.get("passed_deterministic_filter") else 0.0
                for c in candidates
            ]

        for cand, vec, prob in zip(candidates, vectors, secret_probs):
            pred_secret = prob >= self.policy.decision_threshold
            sev = cand.severity.value if hasattr(cand.severity, "value") else str(cand.severity)
            passed_det = bool(cand.deterministic_decision.get("passed_deterministic_filter"))
            det_conf = str(cand.deterministic_decision.get("confidence", "MEDIUM"))

            suppressed = False
            action = "KEEP_ALERT" if pred_secret else "REVIEW_PRIORITY"
            if (
                self.policy.suppress_false_positives
                and not pred_secret
                and sev.upper() not in self.policy.never_suppress_severities
            ):
                suppressed = True
                action = "SUPPRESSED_FP"

            if self.classifier is not None:
                est = _unwrap_estimator(self.classifier)
                if hasattr(est, "coef_") or hasattr(est, "feature_importances_"):
                    top_feats = explain_single_candidate(
                        self.classifier,
                        vec,
                        feature_names=FEATURE_NAMES,
                        top_k=5,
                    )
                else:
                    top_feats = {
                        "shannon_entropy": vec[FEATURE_NAMES.index("shannon_entropy")],
                        "is_test_path": vec[FEATURE_NAMES.index("is_test_path")],
                        "is_placeholder_or_mock": vec[FEATURE_NAMES.index("is_placeholder_or_mock")],
                        "has_positive_keyword": vec[FEATURE_NAMES.index("has_positive_keyword")],
                    }
            else:
                top_feats = {
                    "shannon_entropy": vec[FEATURE_NAMES.index("shannon_entropy")],
                    "is_test_path": vec[FEATURE_NAMES.index("is_test_path")],
                    "is_placeholder_or_mock": vec[FEATURE_NAMES.index("is_placeholder_or_mock")],
                    "has_positive_keyword": vec[FEATURE_NAMES.index("has_positive_keyword")],
                }

            ml_pred = MLPrediction(
                candidate_id=cand.candidate_id,
                predicted_is_secret=pred_secret,
                secret_probability=round(prob, 4),
                model_name=self.model_name,
                model_version=self.model_version,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                top_feature_contributions=top_feats,
            )

            det_ev: Dict[str, object] = {}
            if self.policy.preserve_deterministic_evidence:
                det_ev = {
                    "rule_name": cand.rule_name,
                    "pattern_evidence": cand.pattern_evidence,
                    "lexical_context": cand.lexical_context,
                }

            results.append(HybridFinding(
                candidate_id=cand.candidate_id,
                rule_id=cand.rule_id,
                category=cand.category,
                severity=sev,
                file_path=cand.file_path,
                line_number=cand.line_number,
                masked_value=cand.masked_value,
                passed_deterministic_filter=passed_det,
                deterministic_confidence=det_conf,
                ml_prediction=ml_pred,
                final_action=action,
                suppressed_by_ml=suppressed,
                deterministic_evidence=det_ev,
            ))

        return results

    def scan_path(self, target_path: str, include_filtered: bool = True) -> List[HybridFinding]:
        """Scans target_path via SIDIK and triages extracted SecretCandidate items."""
        candidates = self.scanner.extract_secret_candidates(target_path, include_filtered=include_filtered)
        return self.triage_candidates(candidates)
