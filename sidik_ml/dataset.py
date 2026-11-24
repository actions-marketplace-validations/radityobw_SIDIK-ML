"""Dataset loading, deduplication, and repository-grouped leakage-free splitting."""

from dataclasses import asdict, dataclass
import json
import os
import re
from typing import Dict, Iterator, List, Tuple

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from sidik_ml.features import FEATURE_NAMES


@dataclass
class LabeledCandidateRecord:
    """Represents a labeled secret candidate ready for leakage-free classical ML workflows."""
    candidate_id: str
    repo_id: str
    file_path: str
    value_sha256: str
    masked_context: str
    label: int
    features: Dict[str, float]
    deterministic_passed: bool = True
    severity: str = "MEDIUM"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "LabeledCandidateRecord":
        raw_feats = data.get("features")
        feat_dict: Dict[str, float] = {}
        if isinstance(raw_feats, dict):
            feat_dict = {str(k): float(v) for k, v in raw_feats.items()}
        raw_label = data.get("label", 0)
        label_val = int(raw_label) if isinstance(raw_label, (int, float, str, bool)) else 0
        return cls(
            candidate_id=str(data.get("candidate_id", "")),
            repo_id=str(data.get("repo_id", "")),
            file_path=str(data.get("file_path", "")),
            value_sha256=str(data.get("value_sha256", "")),
            masked_context=str(data.get("masked_context", "")),
            label=label_val,
            features=feat_dict,
            deterministic_passed=bool(data.get("deterministic_passed", True)),
            severity=str(data.get("severity", "MEDIUM")),
        )


def _normalize_context(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_jsonl_dataset(path: str) -> List[LabeledCandidateRecord]:
    """Loads labeled candidate records from a UTF-8 JSONL file."""
    records: List[LabeledCandidateRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            records.append(LabeledCandidateRecord.from_dict(obj))
    return records


def save_jsonl_dataset(records: List[LabeledCandidateRecord], path: str) -> None:
    """Saves labeled candidate records to a UTF-8 JSONL file."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def deduplicate_records(
    records: List[LabeledCandidateRecord],
) -> Tuple[List[LabeledCandidateRecord], int]:
    """Removes exact and context-equivalent duplicate samples prior to dataset splitting."""
    seen: set[Tuple[str, str]] = set()
    unique: List[LabeledCandidateRecord] = []
    removed = 0
    for rec in records:
        key = (rec.value_sha256.strip().lower(), _normalize_context(rec.masked_context))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append(rec)
    return unique, removed


def resolve_cross_repo_value_duplicates(
    records: List[LabeledCandidateRecord],
) -> Tuple[List[LabeledCandidateRecord], Dict[str, int]]:
    """Enforces zero cross-repository value_sha256 overlap and drops conflicting-label hashes."""
    ordered = sorted(records, key=lambda r: (r.repo_id, r.file_path, r.candidate_id))
    grouped: Dict[str, List[LabeledCandidateRecord]] = {}
    nohash_records: List[LabeledCandidateRecord] = []

    for rec in ordered:
        h = rec.value_sha256.strip().lower()
        if not h:
            nohash_records.append(rec)
            continue
        grouped.setdefault(h, []).append(rec)

    resolved: List[LabeledCandidateRecord] = list(nohash_records)
    cross_repo_deduped = 0
    conflicting_dropped = 0

    for h in sorted(grouped):
        grp = grouped[h]
        distinct_labels = {int(r.label) for r in grp}
        if len(distinct_labels) > 1:
            conflicting_dropped += len(grp)
            continue
        resolved.append(grp[0])
        cross_repo_deduped += len(grp) - 1

    stats = {
        "input_records": len(records),
        "output_records": len(resolved),
        "cross_repo_value_deduped": cross_repo_deduped,
        "conflicting_label_dropped": conflicting_dropped,
    }
    return resolved, stats


def assert_no_leakage(
    train_records: List[LabeledCandidateRecord],
    test_records: List[LabeledCandidateRecord],
) -> None:
    """Raises ValueError if repository IDs or candidate value hashes overlap between train and test."""
    train_repos = {r.repo_id for r in train_records if r.repo_id}
    test_repos = {r.repo_id for r in test_records if r.repo_id}
    overlap_repos = train_repos.intersection(test_repos)
    if overlap_repos:
        raise ValueError(f"Repository leakage detected across splits: {sorted(overlap_repos)[:5]}")

    train_hashes = {r.value_sha256 for r in train_records if r.value_sha256}
    test_hashes = {r.value_sha256 for r in test_records if r.value_sha256}
    overlap_hashes = train_hashes.intersection(test_hashes)
    if overlap_hashes:
        raise ValueError(
            f"Candidate string hash leakage detected across splits ({len(overlap_hashes)} overlapping hashes)"
        )


def repository_grouped_split(
    records: List[LabeledCandidateRecord],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[LabeledCandidateRecord], List[LabeledCandidateRecord]]:
    """Splits records into train and held-out test sets strictly grouped by repo_id."""
    if len(records) < 2:
        raise ValueError("At least 2 records are required for a grouped train/test split.")
    groups = [r.repo_id or r.candidate_id for r in records]
    unique_groups = set(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least 2 distinct repo_id groups are required for repository-grouped splitting.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    labels = [r.label for r in records]
    train_idx, test_idx = next(splitter.split(records, labels, groups=groups))

    train_records = [records[int(i)] for i in train_idx]
    test_records = [records[int(i)] for i in test_idx]
    assert_no_leakage(train_records, test_records)
    return train_records, test_records


def stratified_repository_split(
    records: List[LabeledCandidateRecord],
    n_splits: int = 5,
    test_fold: int = 0,
) -> Tuple[List[LabeledCandidateRecord], List[LabeledCandidateRecord]]:
    """Splits records into stratified train and held-out test folds grouped strictly by repo_id."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if not (0 <= test_fold < n_splits):
        raise ValueError(f"test_fold must satisfy 0 <= test_fold < {n_splits}.")
    groups = [r.repo_id or r.candidate_id for r in records]
    unique_groups = set(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"At least {n_splits} distinct repo_id groups are required for {n_splits}-fold stratified repo split."
        )

    labels = [r.label for r in records]
    cv = StratifiedGroupKFold(n_splits=n_splits)
    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(records, labels, groups=groups)):
        if fold_idx == test_fold:
            train_records = [records[int(i)] for i in tr_idx]
            test_records = [records[int(i)] for i in te_idx]
            assert_no_leakage(train_records, test_records)
            return train_records, test_records

    raise ValueError(f"Unable to select fold {test_fold} from {n_splits}-fold StratifiedGroupKFold.")


def iter_repository_group_folds(
    records: List[LabeledCandidateRecord],
    n_splits: int = 5,
) -> Iterator[Tuple[int, List[LabeledCandidateRecord], List[LabeledCandidateRecord]]]:
    """Yields leakage-verified StratifiedGroupKFold splits grouped by repo_id."""
    groups = [r.repo_id or r.candidate_id for r in records]
    labels = [r.label for r in records]
    cv = StratifiedGroupKFold(n_splits=n_splits)
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(records, labels, groups=groups)):
        tr_recs = [records[int(i)] for i in tr_idx]
        va_recs = [records[int(i)] for i in va_idx]
        assert_no_leakage(tr_recs, va_recs)
        yield fold_idx, tr_recs, va_recs


def records_to_matrix(
    records: List[LabeledCandidateRecord],
    selected_features: List[str] | None = None,
) -> Tuple[List[List[float]], List[int], List[str]]:
    """Converts labeled records into aligned feature matrix X, labels y, and feature name list."""
    cols = list(selected_features) if selected_features is not None else list(FEATURE_NAMES)
    X: List[List[float]] = []
    y: List[int] = []
    for rec in records:
        X.append([float(rec.features.get(col, 0.0)) for col in cols])
        y.append(int(rec.label))
    return X, y, cols
