"""Integration tests validating SIDIK v1.1.0 -> SIDIK-ML v0.1.0 contract."""

import json
import os
import tempfile
import unittest

from sidik_ml import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    HybridTriagePipeline,
    TriagePolicy,
    compute_file_sha256,
    extract_feature_dict,
    extract_feature_vector,
    load_verified_skops_model,
    verify_artifact_sha256,
)


class _StubNonNeuralClassifier:
    """Simple test double mimicking scikit-learn predict_proba for contract testing."""

    def predict_proba(self, X):
        out = []
        is_test_idx = FEATURE_NAMES.index("is_test_path")
        for row in X:
            # Assign lower secret probability to test-path findings
            p_sec = 0.20 if row[is_test_idx] >= 1.0 else 0.92
            out.append([1.0 - p_sec, p_sec])
        return out


class TestSidikMLContract(unittest.TestCase):
    """Verifies candidate ingestion, leak-free feature extraction, safe policy, and SHA-256 checks."""

    def test_hybrid_pipeline_preserves_critical_even_when_suppress_enabled(self):
        raw_secret = "AKIAJ2M4K8P9Q3R7W5T1"
        with tempfile.TemporaryDirectory() as tmp:
            test_sub = os.path.join(tmp, "tests")
            os.makedirs(test_sub)
            sample = os.path.join(test_sub, "fixture.py")
            with open(sample, "w", encoding="utf-8") as f:
                f.write(f"aws_key = '{raw_secret}'\n")
                f.write("jwt_sample = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c'\n")

            pipeline = HybridTriagePipeline(
                classifier=_StubNonNeuralClassifier(),
                model_name="stub_non_neural",
                policy=TriagePolicy(suppress_false_positives=True, decision_threshold=0.5),
            )
            results = pipeline.scan_path(tmp, include_filtered=True)
            self.assertEqual(len(results), 2)

            crit_finding = next(r for r in results if r.severity == "CRITICAL")
            med_finding = next(r for r in results if r.severity == "MEDIUM")

            # CRITICAL must never be silently suppressed even with low ML probability
            self.assertFalse(crit_finding.suppressed_by_ml)
            self.assertNotEqual(crit_finding.final_action, "SUPPRESSED_FP")

            # Non-CRITICAL can be suppressed only when opt-in policy allows it
            self.assertTrue(med_finding.suppressed_by_ml)
            self.assertEqual(med_finding.final_action, "SUPPRESSED_FP")

            # Raw secret must never appear anywhere in serialized output
            payload = json.dumps([r.to_dict() for r in results])
            self.assertNotIn(raw_secret, payload)

    def test_hybrid_pipeline_real_model_explainability(self):
        from sidik_ml.dataset import LabeledCandidateRecord, records_to_matrix
        from sidik_ml.models import create_non_ann_model

        recs = []
        for idx in range(6):
            for lbl in (1, 0):
                feats = {n: 0.0 for n in FEATURE_NAMES}
                feats["shannon_entropy"] = 4.4 if lbl == 1 else 2.0
                feats["is_test_path"] = 0.0 if lbl == 1 else 1.0
                recs.append(LabeledCandidateRecord(
                    candidate_id=f"c_{idx}_{lbl}",
                    repo_id=f"r_{idx}",
                    file_path="app.py",
                    value_sha256=f"h_{idx}_{lbl}",
                    masked_context="x",
                    label=lbl,
                    features=feats,
                ))
        X, y, _ = records_to_matrix(recs)
        rf = create_non_ann_model("random_forest", random_state=42)
        rf.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp:
            sample = os.path.join(tmp, "app.py")
            with open(sample, "w", encoding="utf-8") as f:
                f.write("AWS_ACCESS_KEY_ID = 'AKIAJ2M4K8P9Q3R7W5T1'\n")
            pipeline = HybridTriagePipeline(classifier=rf, model_name="random_forest")
            results = pipeline.scan_path(tmp, include_filtered=True)
            self.assertEqual(len(results), 1)
            self.assertEqual(len(results[0].ml_prediction.top_feature_contributions), 5)

    def test_feature_vector_alignment_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "app.py")
            with open(fp, "w", encoding="utf-8") as f:
                f.write("OPENAI_API_KEY = 'sk-proj-uW8vB2xY9zQpLmK3jH5gD7sA1rT0oPqVwXyZ'\n")

            pipeline = HybridTriagePipeline()
            cands = pipeline.scanner.extract_secret_candidates(tmp)
            self.assertEqual(len(cands), 1)
            fdict = extract_feature_dict(cands[0])
            fvec = extract_feature_vector(cands[0])
            self.assertEqual(len(fvec), len(FEATURE_NAMES))
            self.assertEqual(FEATURE_SCHEMA_VERSION, "1.0.0")
            self.assertAlmostEqual(fvec[0], fdict[FEATURE_NAMES[0]])

    def test_sha256_verification_rejects_tampered_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            art = os.path.join(tmp, "model.skops")
            with open(art, "wb") as f:
                f.write(b"dummy artifact bytes")
            digest = compute_file_sha256(art)
            self.assertTrue(verify_artifact_sha256(art, digest))
            self.assertFalse(verify_artifact_sha256(art, "0" * 64))
            with self.assertRaises(ValueError):
                load_verified_skops_model(art, "0" * 64)


if __name__ == "__main__":
    unittest.main()
