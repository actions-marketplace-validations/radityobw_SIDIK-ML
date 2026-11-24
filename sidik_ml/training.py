"""Reproducible Non-ANN training, cross-model comparison, and provenance recording."""

import json
import os
import platform
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
    records_to_matrix,
    repository_grouped_split,
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
) -> Dict[str, Any]:
    """Runs deduplication, repository-grouped split, baseline vs Non-ANN training, and provenance capture."""
    deduped, removed_dups = deduplicate_records(records)
    train_recs, test_recs = repository_grouped_split(
        deduped,
        test_size=test_size,
        random_state=random_state,
    )

    X_train, y_train, cols = records_to_matrix(train_recs, FEATURE_NAMES)
    X_test, y_test, _ = records_to_matrix(test_recs, FEATURE_NAMES)
    y_det_test = [1 if r.deterministic_passed else 0 for r in test_recs]

    det_metrics = compute_classification_metrics(y_test, y_det_test)
    model_results: Dict[str, Any] = {}
    best_model_name = ""
    best_model_obj: Any = None
    best_f1 = -1.0
    best_mcc = -2.0

    for name in model_names:
        clf = create_non_ann_model(name, random_state=random_state)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        train_ms = (time.perf_counter() - t0) * 1000.0

        y_pred = [int(v) for v in clf.predict(X_test)]
        metrics = compute_classification_metrics(y_test, y_pred)
        f1_ci = bootstrap_metric_ci(y_test, y_pred, metric_name="f1_score", n_bootstrap=300, random_state=random_state)
        mcnemar_vs_det = mcnemar_paired_test(y_test, y_det_test, y_pred)
        overhead = profile_inference_overhead(clf, X_test)
        explain = extract_global_explanations(clf, X_test, y_test, feature_names=cols, random_state=random_state)

        model_results[name] = {
            "metrics": metrics,
            "f1_bootstrap_95ci": f1_ci,
            "mcnemar_vs_deterministic_sidik": mcnemar_vs_det,
            "train_latency_ms": round(train_ms, 4),
            "inference_overhead": overhead,
            "explainability": explain,
        }

        cur_f1 = float(metrics["f1_score"])
        cur_mcc = float(metrics["mcc"])
        if (cur_f1, cur_mcc) > (best_f1, best_mcc):
            best_f1 = cur_f1
            best_mcc = cur_mcc
            best_model_name = name
            best_model_obj = clf

    provenance = collect_provenance(
        dataset_path=dataset_path,
        model_names=model_names,
        random_state=random_state,
        split_strategy=f"GroupShuffleSplit(test_size={test_size}, group=repo_id)",
    )
    provenance["dataset_stats"] = {
        "raw_records": len(records),
        "deduplicated_records": len(deduped),
        "duplicates_removed": removed_dups,
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
