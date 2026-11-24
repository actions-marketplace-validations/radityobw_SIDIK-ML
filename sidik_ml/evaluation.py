"""Metrics, statistical tests, overhead profiling, and feature-group ablation for SIDIK-ML."""

import math
import random
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Sequence

from scipy.stats import binomtest, chi2

from sidik_ml.dataset import LabeledCandidateRecord, records_to_matrix
from sidik_ml.features import FEATURE_NAMES
from sidik_ml.models import create_non_ann_model

FEATURE_GROUPS: Dict[str, List[str]] = {
    "string_stats": [
        "str_length", "str_unique_chars", "str_unique_ratio",
        "digit_ratio", "lower_ratio", "upper_ratio", "symbol_ratio",
        "is_hex_like", "is_base64_like", "is_uuid_like", "is_git_sha_like",
        "has_provider_prefix",
    ],
    "entropy": [
        "shannon_entropy", "min_rule_entropy", "entropy_margin", "meets_entropy_threshold",
    ],
    "pattern_evidence": [
        "is_generic_rule",
    ],
    "lexical_context": [
        "is_comment", "is_placeholder_or_mock", "has_positive_keyword",
    ],
    "code_context": [
        "line_length", "is_assignment", "is_dict_entry",
        "is_function_call", "is_env_reference", "prev_line_is_comment",
    ],
    "file_context": [
        "is_test_path", "is_example_or_doc_path", "path_depth",
    ],
    "deterministic_signal": [
        "passed_deterministic_filter",
    ],
}


def compute_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    """Computes full confusion matrix and standard security classification metrics including MCC."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length.")

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0.0

    return {
        "total_samples": total,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "specificity": round(specificity, 4),
        "mcc": round(mcc, 4),
    }


def bootstrap_metric_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    metric_name: str = "f1_score",
    n_bootstrap: int = 500,
    ci: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """Computes non-parametric bootstrap confidence intervals for a chosen metric."""
    n = len(y_true)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = random.Random(random_state)
    scores: List[float] = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        yt_b = [y_true[i] for i in idxs]
        yp_b = [y_pred[i] for i in idxs]
        m = compute_classification_metrics(yt_b, yp_b)
        scores.append(float(m[metric_name]))

    scores.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = max(0, int(math.floor(alpha * len(scores))))
    hi_idx = min(len(scores) - 1, int(math.floor((1.0 - alpha) * len(scores))))
    mean_val = sum(scores) / len(scores)
    return {
        "mean": round(mean_val, 4),
        "ci_low": round(scores[lo_idx], 4),
        "ci_high": round(scores[hi_idx], 4),
    }


def mcnemar_paired_test(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
) -> Dict[str, Any]:
    """Runs McNemar's paired test comparing prediction correctness of model A vs model B."""
    if not (len(y_true) == len(y_pred_a) == len(y_pred_b)):
        raise ValueError("All prediction sequences must match y_true length.")

    b_only_a = 0  # A correct, B wrong
    c_only_b = 0  # A wrong, B correct
    for yt, pa, pb in zip(y_true, y_pred_a, y_pred_b):
        ca = (pa == yt)
        cb = (pb == yt)
        if ca and not cb:
            b_only_a += 1
        elif cb and not ca:
            c_only_b += 1

    discordant = b_only_a + c_only_b
    if discordant == 0:
        return {
            "method": "exact_identical",
            "b_a_correct_b_wrong": 0,
            "c_a_wrong_b_correct": 0,
            "statistic": 0.0,
            "p_value": 1.0,
        }

    if discordant < 25:
        res = binomtest(k=min(b_only_a, c_only_b), n=discordant, p=0.5, alternative="two-sided")
        return {
            "method": "exact_binomial",
            "b_a_correct_b_wrong": b_only_a,
            "c_a_wrong_b_correct": c_only_b,
            "statistic": float(min(b_only_a, c_only_b)),
            "p_value": round(float(res.pvalue), 6),
        }

    stat = ((abs(b_only_a - c_only_b) - 1.0) ** 2) / float(discordant)
    p_val = float(chi2.sf(stat, df=1))
    return {
        "method": "chi2_continuity_corrected",
        "b_a_correct_b_wrong": b_only_a,
        "c_a_wrong_b_correct": c_only_b,
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
    }


def profile_inference_overhead(model: Any, X: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Measures batch inference latency and peak memory without touching pre-existing tracemalloc."""
    if not X:
        return {"total_latency_ms": 0.0, "per_sample_latency_ms": 0.0, "peak_memory_kb": 0.0}

    started = False
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        started = True

    t0 = time.perf_counter()
    try:
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(X)
        else:
            _ = model.predict(X)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if started:
            tracemalloc.stop()

    return {
        "total_latency_ms": round(elapsed_ms, 4),
        "per_sample_latency_ms": round(elapsed_ms / len(X), 6),
        "peak_memory_kb": round(peak_bytes / 1024.0, 2),
    }


def run_feature_group_ablation(
    model_name: str,
    train_records: List[LabeledCandidateRecord],
    test_records: List[LabeledCandidateRecord],
    random_state: int = 42,
    groups: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluates leave-one-group-out ablation across structured static feature groups."""
    active_groups = groups or FEATURE_GROUPS
    results: Dict[str, Dict[str, Any]] = {}

    X_tr_full, y_tr, _ = records_to_matrix(train_records, FEATURE_NAMES)
    X_te_full, y_te, _ = records_to_matrix(test_records, FEATURE_NAMES)
    clf_full = create_non_ann_model(model_name, random_state=random_state)
    clf_full.fit(X_tr_full, y_tr)
    y_pred_full = [int(v) for v in clf_full.predict(X_te_full)]
    results["all_features"] = {
        "feature_count": len(FEATURE_NAMES),
        "metrics": compute_classification_metrics(y_te, y_pred_full),
    }

    for grp_name, grp_cols in active_groups.items():
        kept_cols = [c for c in FEATURE_NAMES if c not in grp_cols]
        if not kept_cols:
            continue
        X_tr_ab, _, _ = records_to_matrix(train_records, kept_cols)
        X_te_ab, _, _ = records_to_matrix(test_records, kept_cols)
        clf_ab = create_non_ann_model(model_name, random_state=random_state)
        clf_ab.fit(X_tr_ab, y_tr)
        y_pred_ab = [int(v) for v in clf_ab.predict(X_te_ab)]
        results[f"without_{grp_name}"] = {
            "feature_count": len(kept_cols),
            "removed_features": list(grp_cols),
            "metrics": compute_classification_metrics(y_te, y_pred_ab),
        }

    return results
