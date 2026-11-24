"""Reproducible Non-ANN training, cross-model comparison, and provenance recording."""

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import sklearn
import skops

from sidik import __version__ as SIDIK_VERSION
from sidik.models import CANDIDATE_SCHEMA_VERSION
from sidik_ml.dataset import (
    LabeledCandidateRecord,
    deduplicate_records,
    iter_repository_group_folds,
    records_to_matrix,
    repository_grouped_split,
    resolve_cross_repo_value_duplicates,
    stratified_repository_split,
)
from sidik_ml.evaluation import (
    bootstrap_metric_ci,
    compute_classification_metrics,
    mcnemar_paired_test,
    profile_inference_overhead,
)
from sidik_ml.explainability import extract_global_explanations
from sidik_ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from sidik_ml.models import SUPPORTED_MODELS, create_non_ann_model
from sidik_ml.serialization import compute_file_sha256, save_verified_skops_model


def _git_head_sha(repo_dir: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def collect_provenance(
    dataset_path: Optional[str],
    model_names: Sequence[str],
    random_state: int,
    split_strategy: str,
) -> Dict[str, Any]:
    """Records software, environment, schema, and dataset provenance metadata."""
    ml_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sidik_version": SIDIK_VERSION,
        "sidik_ml_git_commit": _git_head_sha(ml_repo_dir),
        "python_version": platform.python_version(),
        "os_platform": f"{platform.system()} {platform.release()}",
        "dependency_versions": {
            "scikit-learn": sklearn.__version__,
            "skops": getattr(skops, "__version__", "installed"),
        },
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_path": dataset_path,
        "dataset_sha256": compute_file_sha256(dataset_path) if dataset_path and os.path.isfile(dataset_path) else None,
        "evaluated_models": list(model_names),
        "random_state": random_state,
        "split_strategy": split_strategy,
    }


def train_and_compare_models(
    records: List[LabeledCandidateRecord],
    model_names: Sequence[str] = SUPPORTED_MODELS,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: Optional[str] = None,
    dataset_path: Optional[str] = None,
    use_stratified_split: bool = False,
    n_cv_folds: int = 5,
    n_bootstrap: int = 300,
) -> Dict[str, Any]:
    """Runs deduplication, repository-grouped split, baseline vs Non-ANN training, and provenance capture."""
    deduped_context, removed_dups = deduplicate_records(records)
    deduped, cross_repo_stats = resolve_cross_repo_value_duplicates(deduped_context)

    if use_stratified_split:
        train_recs, test_recs = stratified_repository_split(deduped, n_splits=n_cv_folds, test_fold=0)
        split_label = f"StratifiedGroupKFold(n_splits={n_cv_folds}, test_fold=0, group=repo_id)"
    else:
        train_recs, test_recs = repository_grouped_split(
            deduped,
            test_size=test_size,
            random_state=random_state,
        )
        split_label = f"GroupShuffleSplit(test_size={test_size}, group=repo_id)"

    X_train, y_train, cols = records_to_matrix(train_recs, FEATURE_NAMES)
    X_test, y_test, _ = records_to_matrix(test_recs, FEATURE_NAMES)
    y_det_test = [1 if r.deterministic_passed else 0 for r in test_recs]

    det_metrics = compute_classification_metrics(y_test, y_det_test)
    model_results: Dict[str, Any] = {}
    best_model_name = ""
    best_model_obj: Any = None
    best_score = (-1.0, -2.0)

    can_run_cv = n_cv_folds >= 2 and len({r.repo_id or r.candidate_id for r in train_recs}) >= n_cv_folds and len(set(y_train)) > 1
    cv_folds = list(iter_repository_group_folds(train_recs, n_splits=n_cv_folds)) if can_run_cv else []

    for name in model_names:
        cv_f1s: List[float] = []
        cv_mccs: List[float] = []
        cv_precs: List[float] = []
        cv_recs: List[float] = []
        for _, f_tr, f_va in cv_folds:
            X_f_tr, y_f_tr, _ = records_to_matrix(f_tr, FEATURE_NAMES)
            X_f_va, y_f_va, _ = records_to_matrix(f_va, FEATURE_NAMES)
            if len(set(y_f_tr)) < 2:
                continue
            cv_clf = create_non_ann_model(name, random_state=random_state)
            cv_clf.fit(X_f_tr, y_f_tr)
            y_f_pred = [int(v) for v in cv_clf.predict(X_f_va)]
            fm = compute_classification_metrics(y_f_va, y_f_pred)
            cv_f1s.append(float(fm["f1_score"]))
            cv_mccs.append(float(fm["mcc"]))
            cv_precs.append(float(fm["precision"]))
            cv_recs.append(float(fm["recall"]))

        clf = create_non_ann_model(name, random_state=random_state)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        train_ms = (time.perf_counter() - t0) * 1000.0

        y_pred = [int(v) for v in clf.predict(X_test)]
        metrics = compute_classification_metrics(y_test, y_pred)
        f1_ci = bootstrap_metric_ci(y_test, y_pred, metric_name="f1_score", n_bootstrap=n_bootstrap, random_state=random_state)
        mcnemar_vs_det = mcnemar_paired_test(y_test, y_det_test, y_pred)
        overhead = profile_inference_overhead(clf, X_test)
        explain = extract_global_explanations(clf, X_test, y_test, feature_names=cols, random_state=random_state)

        cv_summary: Optional[Dict[str, Any]] = None
        if cv_f1s:
            cv_summary = {
                "folds_evaluated": len(cv_f1s),
                "f1_mean": round(statistics.mean(cv_f1s), 4),
                "f1_std": round(statistics.stdev(cv_f1s) if len(cv_f1s) > 1 else 0.0, 4),
                "mcc_mean": round(statistics.mean(cv_mccs), 4),
                "mcc_std": round(statistics.stdev(cv_mccs) if len(cv_mccs) > 1 else 0.0, 4),
                "precision_mean": round(statistics.mean(cv_precs), 4),
                "recall_mean": round(statistics.mean(cv_recs), 4),
            }

        model_results[name] = {
            "cv_summary": cv_summary,
            "metrics": metrics,
            "f1_bootstrap_95ci": f1_ci,
            "mcnemar_vs_deterministic_sidik": mcnemar_vs_det,
            "train_latency_ms": round(train_ms, 4),
            "inference_overhead": overhead,
            "explainability": explain,
        }

        if use_stratified_split and cv_summary is not None:
            cand_score = (float(cv_summary["f1_mean"]), float(cv_summary["mcc_mean"]))
        else:
            cand_score = (float(metrics["f1_score"]), float(metrics["mcc"]))

        if cand_score > best_score:
            best_score = cand_score
            best_model_name = name
            best_model_obj = clf

    provenance = collect_provenance(
        dataset_path=dataset_path,
        model_names=model_names,
        random_state=random_state,
        split_strategy=split_label,
    )
    provenance["dataset_stats"] = {
        "raw_records": len(records),
        "context_duplicates_removed": removed_dups,
        "cross_repo_value_deduped": cross_repo_stats["cross_repo_value_deduped"],
        "conflicting_label_dropped": cross_repo_stats["conflicting_label_dropped"],
        "deduplicated_records": len(deduped),
        "train_samples": len(train_recs),
        "test_samples": len(test_recs),
        "train_repos": len({r.repo_id for r in train_recs}),
        "test_repos": len({r.repo_id for r in test_recs}),
    }

    saved_artifact: Optional[Dict[str, Any]] = None
    if output_dir and best_model_obj is not None:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{best_model_name}.skops")
        saved_artifact = save_verified_skops_model(
            best_model_obj,
            model_path=model_path,
            metadata={
                "model_name": best_model_name,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "random_state": random_state,
            },
        )
        summary_out = {
            "provenance": provenance,
            "deterministic_baseline_test": det_metrics,
            "models": model_results,
            "best_model": best_model_name,
            "saved_artifact": saved_artifact,
        }
        with open(os.path.join(output_dir, "experiment_report.json"), "w", encoding="utf-8") as f:
            json.dump(summary_out, f, indent=2)

    return {
        "provenance": provenance,
        "deterministic_baseline_test": det_metrics,
        "models": model_results,
        "best_model": best_model_name,
        "saved_artifact": saved_artifact,
    }
