"""Tests for Non-ANN model guard, training/comparison, statistical tests, explainability, and skops I/O."""

import os
import tempfile
import tracemalloc
import unittest

from sidik_ml.dataset import LabeledCandidateRecord, records_to_matrix
from sidik_ml.evaluation import (
    bootstrap_metric_ci,
    compute_classification_metrics,
    mcnemar_paired_test,
    profile_inference_overhead,
    run_feature_group_ablation,
)
from sidik_ml.explainability import explain_single_candidate, extract_global_explanations
from sidik_ml.features import FEATURE_NAMES
from sidik_ml.models import SUPPORTED_MODELS, create_non_ann_model, validate_non_ann_model_name
from sidik_ml.serialization import load_verified_skops_model, save_verified_skops_model
from sidik_ml.training import train_and_compare_models


def _build_contract_fixture_records() -> list[LabeledCandidateRecord]:
    """Builds synthetic contract-test records solely to exercise ML code paths (not research results)."""
    records = []
    for repo_idx in range(10):
        repo = f"contract_repo_{repo_idx}"
        for is_pos in (1, 0):
            feats = {name: 0.0 for name in FEATURE_NAMES}
            feats["str_length"] = 32.0 if is_pos else 16.0
            feats["shannon_entropy"] = 4.3 + (0.05 * (repo_idx % 3)) if is_pos else 2.2 + (0.05 * (repo_idx % 3))
            feats["meets_entropy_threshold"] = 1.0 if is_pos else 0.0
            feats["has_positive_keyword"] = 1.0 if is_pos else 0.0
            feats["is_test_path"] = 0.0 if is_pos else 1.0
            feats["is_placeholder_or_mock"] = 0.0 if is_pos else 1.0
            feats["passed_deterministic_filter"] = 1.0
            records.append(LabeledCandidateRecord(
                candidate_id=f"cid_{repo_idx}_{is_pos}",
                repo_id=repo,
                file_path="src/cfg.py" if is_pos else "tests/test_cfg.py",
                value_sha256=f"sha_{repo_idx}_{is_pos}",
                masked_context=f"ctx_{repo_idx}_{is_pos}",
                label=is_pos,
                features=feats,
                deterministic_passed=True,
                severity="HIGH" if is_pos else "MEDIUM",
            ))
    return records


class TestModelsAndTraining(unittest.TestCase):
    """Validates Non-ANN factory, skops roundtrip, evaluation metrics, ablation, and explainability."""

    def test_neural_exclusion_guard(self):
        for bad in ("ann", "mlp_classifier", "cnn", "rnn", "lstm", "transformer", "codebert", "deep_net"):
            with self.assertRaises(ValueError):
                validate_non_ann_model_name(bad)
        for good in SUPPORTED_MODELS:
            self.assertEqual(validate_non_ann_model_name(good), good)

    def test_skops_save_load_verification(self):
        recs = _build_contract_fixture_records()
        X, y, _ = records_to_matrix(recs)
        clf = create_non_ann_model("random_forest", random_state=42)
        clf.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp:
            m_path = os.path.join(tmp, "rf.skops")
            manifest = save_verified_skops_model(clf, m_path, metadata={"model": "random_forest"})
            loaded = load_verified_skops_model(m_path, manifest["sha256"])
            preds = loaded.predict(X[:2])
            self.assertEqual(len(preds), 2)

            with self.assertRaises(ValueError):
                load_verified_skops_model(m_path, "f" * 64)

    def test_training_comparison_explainability_and_ablation(self):
        recs = _build_contract_fixture_records()
        with tempfile.TemporaryDirectory() as tmp:
            report = train_and_compare_models(
                recs,
                model_names=("logistic_regression", "decision_tree", "random_forest"),
                test_size=0.3,
                random_state=42,
                output_dir=tmp,
            )
            self.assertIn("provenance", report)
            self.assertIn("models", report)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "experiment_report.json")))
            self.assertIsNotNone(report["saved_artifact"])

            X, y, cols = records_to_matrix(recs)
            dt = create_non_ann_model("decision_tree", random_state=42)
            dt.fit(X, y)
            g_exp = extract_global_explanations(dt, X, y, feature_names=cols)
            self.assertIsNotNone(g_exp["decision_tree_rules"])
            s_exp = explain_single_candidate(dt, X[0], feature_names=cols, top_k=3)
            self.assertEqual(len(s_exp), 3)

            ablation = run_feature_group_ablation("logistic_regression", recs[:12], recs[12:])
            self.assertIn("all_features", ablation)
            self.assertIn("without_lexical_context", ablation)

    def test_metrics_stats_and_non_intrusive_profiling(self):
        y_t = [1, 1, 0, 0, 1, 0]
        y_a = [1, 0, 1, 0, 1, 0]
        y_b = [1, 1, 0, 0, 1, 0]
        m = compute_classification_metrics(y_t, y_b)
        self.assertEqual(m["f1_score"], 1.0)
        self.assertEqual(m["mcc"], 1.0)

        ci = bootstrap_metric_ci(y_t, y_b, n_bootstrap=50, random_state=42)
        self.assertLessEqual(ci["ci_low"], ci["ci_high"])

        mc = mcnemar_paired_test(y_t, y_a, y_b)
        self.assertIn("p_value", mc)

        recs = _build_contract_fixture_records()
        X, y, _ = records_to_matrix(recs)
        lr = create_non_ann_model("logistic_regression", random_state=42)
        lr.fit(X, y)

        tracemalloc.start()
        try:
            prof = profile_inference_overhead(lr, X)
            self.assertGreaterEqual(prof["total_latency_ms"], 0.0)
            self.assertTrue(tracemalloc.is_tracing())
        finally:
            if tracemalloc.is_tracing():
                tracemalloc.stop()


if __name__ == "__main__":
    unittest.main()
