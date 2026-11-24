"""Decoupled SIDIK plugin entry-point adapter for SIDIK-ML Non-ANN triage."""

import json
import os
from typing import Any, Dict, Optional, Tuple

from sidik.models import ScanReport
from sidik_ml.integration import HybridTriagePipeline
from sidik_ml.schemas import TriagePolicy
from sidik_ml.serialization import get_ml_cache_dir, load_verified_skops_model


def resolve_model_and_sha256(model_path: str, sha256: Optional[str] = None) -> Tuple[str, str]:
    """Resolves a .skops model path and verifies presence of its SHA-256 checksum."""
    if not model_path or not os.path.isfile(model_path):
        raise ValueError(f"Model artifact not found: {model_path}")
    if sha256 and sha256.strip():
        return model_path, sha256.strip()

    manifest_path = f"{model_path}.manifest.json"
    if not os.path.isfile(manifest_path):
        raise ValueError(
            f"SHA-256 digest is required: provide --ml-sha256/--sha256 or place a valid manifest at {manifest_path}."
        )
    with open(manifest_path, "r", encoding="utf-8") as mf:
        data = json.load(mf)
    digest = str(data.get("sha256", "")).strip()
    if not digest:
        raise ValueError(f"Manifest does not contain a valid 'sha256' field: {manifest_path}")
    return model_path, digest


def discover_default_model() -> Optional[Tuple[str, str]]:
    """Discovers an optional local .skops model artifact and its manifest SHA-256."""
    candidates = []
    env_path = os.environ.get("SIDIK_ML_MODEL_PATH")
    if env_path:
        candidates.append(env_path.strip())

    models_cache_dir = os.path.join(get_ml_cache_dir(), "models")
    if os.path.isdir(models_cache_dir):
        for fname in sorted(os.listdir(models_cache_dir)):
            if fname.endswith(".skops"):
                candidates.append(os.path.join(models_cache_dir, fname))

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates.append(os.path.join(repo_root, "artifacts", "gradient_boosting.skops"))

    for candidate_path in candidates:
        if candidate_path and os.path.isfile(candidate_path) and os.path.isfile(f"{candidate_path}.manifest.json"):
            try:
                return resolve_model_and_sha256(candidate_path)
            except Exception:
                continue
    return None


class SidikMLPlugin:
    """SIDIK ``sidik.plugins`` entry-point adapter running Non-ANN ML triage over secret findings."""

    name = "sidik-ml"
    version = "0.5.4"
    description = "Explainable Non-ANN Machine Learning Triage Plugin for SIDIK"

    def __init__(self) -> None:
        self._model_cache: Dict[Tuple[str, str], Any] = {}

    def discover_default_model(self) -> Optional[Tuple[str, str]]:
        """Exposes local verified model auto-discovery to SIDIK interactive CLI and GUI."""
        return discover_default_model()

    def _get_classifier(self, model_path: Optional[str], sha256: Optional[str]) -> Tuple[Optional[Any], str]:
        if not model_path:
            return None, "deterministic_pass_through"
        resolved_path, resolved_sha = resolve_model_and_sha256(model_path, sha256)
        cache_key = (os.path.abspath(resolved_path), resolved_sha.lower())
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = load_verified_skops_model(resolved_path, resolved_sha)
        return self._model_cache[cache_key], os.path.basename(resolved_path)

    def triage_report(
        self,
        scanner: Any,
        target_path: str,
        report: ScanReport,
        options: Optional[Dict[str, Any]] = None,
    ) -> ScanReport:
        """Enriches secret findings in ``report`` with ML triage metadata while keeping SCA intact."""
        opts = options or {}
        model_path = opts.get("model_path")
        sha256 = opts.get("sha256")
        threshold = float(opts.get("threshold", 0.5))
        suppress_fp = bool(opts.get("suppress_false_positives", False))
        candidate_wide = bool(opts.get("candidate_wide", False))

        classifier, model_label = self._get_classifier(model_path, sha256)
        policy = TriagePolicy(
            decision_threshold=threshold,
            suppress_false_positives=suppress_fp,
        )
        pipeline = HybridTriagePipeline(
            scanner=scanner,
            classifier=classifier,
            model_name=model_label,
            model_version=self.version,
            policy=policy,
        )

        preextracted = opts.get("_preextracted_candidates")
        if isinstance(preextracted, list):
            if candidate_wide:
                candidates = preextracted
            else:
                candidates = [
                    c for c in preextracted
                    if bool(c.deterministic_decision.get("passed_deterministic_filter"))
                ]
        else:
            candidates = scanner.extract_secret_candidates(target_path, include_filtered=candidate_wide)
        hybrid_findings = pipeline.triage_candidates(candidates)
        by_id = {hf.candidate_id: hf for hf in hybrid_findings}

        kept_findings = []
        for finding in report.findings:
            if finding.finding_type != "secret":
                kept_findings.append(finding)
                continue

            hf = by_id.get(finding.id)
            if hf is not None:
                finding.metadata["ml_triage"] = {
                    "action": hf.final_action,
                    "predicted_is_secret": hf.ml_prediction.predicted_is_secret,
                    "secret_probability": hf.ml_prediction.secret_probability,
                    "model_name": hf.ml_prediction.model_name,
                    "model_version": hf.ml_prediction.model_version,
                    "suppressed_by_ml": hf.suppressed_by_ml,
                    "top_feature_contributions": hf.ml_prediction.top_feature_contributions,
                }
                if hf.suppressed_by_ml:
                    continue
            kept_findings.append(finding)

        report.findings = kept_findings
        report.summary.total_secrets_found = sum(1 for f in kept_findings if f.finding_type == "secret")
        report.summary.total_vulnerabilities_found = sum(1 for f in kept_findings if f.finding_type == "dependency")
        return report


def create_plugin() -> SidikMLPlugin:
    """Entry-point factory returning a SidikMLPlugin instance."""
    return SidikMLPlugin()
