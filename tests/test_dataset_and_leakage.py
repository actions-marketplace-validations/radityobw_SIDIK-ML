"""Tests for dataset deduplication, repository-grouped splitting, and leakage prevention."""

import os
import tempfile
import unittest

from sidik_ml.dataset import (
    LabeledCandidateRecord,
    assert_no_leakage,
    deduplicate_records,
    iter_repository_group_folds,
    load_jsonl_dataset,
    repository_grouped_split,
    resolve_cross_repo_value_duplicates,
    save_jsonl_dataset,
    stratified_repository_split,
)
from sidik_ml.features import FEATURE_NAMES


def _make_record(cid: str, repo: str, sha: str, ctx: str, label: int) -> LabeledCandidateRecord:
    feats = {name: 0.0 for name in FEATURE_NAMES}
    feats["shannon_entropy"] = 4.2 if label == 1 else 2.1
    feats["is_test_path"] = 0.0 if label == 1 else 1.0
    return LabeledCandidateRecord(
        candidate_id=cid,
        repo_id=repo,
        file_path="app.py" if label == 1 else "tests/mock.py",
        value_sha256=sha,
        masked_context=ctx,
        label=label,
        features=feats,
        deterministic_passed=True,
        severity="HIGH" if label == 1 else "MEDIUM",
    )


class TestDatasetAndLeakage(unittest.TestCase):
    """Validates deduplication, JSONL persistence, and zero repo/hash overlap across splits."""

    def test_deduplication_and_jsonl_roundtrip(self):
        r1 = _make_record("c1", "repoA", "aaa111", "aws_key = 'AKIA****1234'", 1)
        r2 = _make_record("c2", "repoB", "aaa111", "  AWS_KEY = 'AKIA****1234' ", 1)
        r3 = _make_record("c3", "repoB", "bbb222", "dummy = 'test'", 0)

        deduped, removed = deduplicate_records([r1, r2, r3])
        self.assertEqual(removed, 1)
        self.assertEqual(len(deduped), 2)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "records.jsonl")
            save_jsonl_dataset(deduped, path)
            loaded = load_jsonl_dataset(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].candidate_id, "c1")

    def test_assert_no_leakage_catches_repo_and_hash_overlap(self):
        tr = [_make_record("c1", "repoA", "h1", "a", 1)]
        te_repo_leak = [_make_record("c2", "repoA", "h2", "b", 0)]
        with self.assertRaises(ValueError):
            assert_no_leakage(tr, te_repo_leak)

        te_hash_leak = [_make_record("c3", "repoB", "h1", "c", 1)]
        with self.assertRaises(ValueError):
            assert_no_leakage(tr, te_hash_leak)

    def test_repository_grouped_split_and_kfold(self):
        records = []
        for r_idx in range(6):
            repo = f"repo_{r_idx}"
            records.append(_make_record(f"{repo}_pos", repo, f"hash_pos_{r_idx}", f"prod = {r_idx}", 1))
            records.append(_make_record(f"{repo}_neg", repo, f"hash_neg_{r_idx}", f"test = {r_idx}", 0))

        train_recs, test_recs = repository_grouped_split(records, test_size=0.33, random_state=42)
        self.assertGreater(len(train_recs), 0)
        self.assertGreater(len(test_recs), 0)
        assert_no_leakage(train_recs, test_recs)

        folds = list(iter_repository_group_folds(records, n_splits=3))
        self.assertEqual(len(folds), 3)
        for _, f_tr, f_va in folds:
            assert_no_leakage(f_tr, f_va)

    def test_cross_repo_value_dedup_and_stratified_split(self):
        records = [
            _make_record("c1", "repoA", "same_hash", "k = 'v1'", 1),
            _make_record("c2", "repoB", "same_hash", "k = 'v1'", 1),
            _make_record("c3", "repoA", "conflict_h", "k = 'v2'", 1),
            _make_record("c4", "repoC", "conflict_h", "k = 'v2'", 0),
        ]
        resolved, stats = resolve_cross_repo_value_duplicates(records)
        self.assertEqual(stats["cross_repo_value_deduped"], 1)
        self.assertEqual(stats["conflicting_label_dropped"], 2)
        self.assertEqual(len(resolved), 1)

        rich = []
        for idx in range(10):
            r = f"s_repo_{idx}"
            rich.append(_make_record(f"p_{idx}", r, f"hp_{idx}", f"prod={idx}", 1))
            rich.append(_make_record(f"n_{idx}", r, f"hn_{idx}", f"test={idx}", 0))
        tr, te = stratified_repository_split(rich, n_splits=5, test_fold=0)
        assert_no_leakage(tr, te)
        self.assertTrue(any(r.label == 1 for r in te) and any(r.label == 0 for r in te))


if __name__ == "__main__":
    unittest.main()
