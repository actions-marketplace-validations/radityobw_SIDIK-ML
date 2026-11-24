"""SIDIK-ML: Explainable Non-Neural ML Extension for SIDIK Static Security Analysis."""

from sidik_ml.schemas import MLPrediction, HybridFinding, TriagePolicy
from sidik_ml.features import FEATURE_SCHEMA_VERSION, FEATURE_NAMES, extract_feature_dict, extract_feature_vector
from sidik_ml.models import SUPPORTED_MODELS, FORBIDDEN_NEURAL_KEYWORDS, validate_non_ann_model_name, create_non_ann_model
from sidik_ml.dataset import (
    LabeledCandidateRecord,
    load_jsonl_dataset,
    save_jsonl_dataset,
    deduplicate_records,
    resolve_cross_repo_value_duplicates,
    assert_no_leakage,
    repository_grouped_split,
    stratified_repository_split,
    iter_repository_group_folds,
    records_to_matrix,
)
from sidik_ml.evaluation import (
    FEATURE_GROUPS,
    compute_classification_metrics,
    bootstrap_metric_ci,
    mcnemar_paired_test,
    profile_inference_overhead,
    run_feature_group_ablation,
)
from sidik_ml.explainability import extract_global_explanations, explain_single_candidate
from sidik_ml.serialization import (
    SAFE_SKOPS_TYPE_PREFIXES,
    compute_file_sha256,
    verify_artifact_sha256,
    save_verified_skops_model,
    load_verified_skops_model,
)
from sidik_ml.training import collect_provenance, train_and_compare_models
from sidik_ml.integration import HybridTriagePipeline
from sidik_ml.plugin import SidikMLPlugin, create_plugin, resolve_model_and_sha256

__version__ = "0.5.0"

__all__ = [
    "__version__",
    "SidikMLPlugin",
    "create_plugin",
    "resolve_model_and_sha256",
    "MLPrediction",
    "HybridFinding",
    "TriagePolicy",
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_NAMES",
    "FEATURE_GROUPS",
    "extract_feature_dict",
    "extract_feature_vector",
    "SUPPORTED_MODELS",
    "FORBIDDEN_NEURAL_KEYWORDS",
    "validate_non_ann_model_name",
    "create_non_ann_model",
    "LabeledCandidateRecord",
    "load_jsonl_dataset",
    "save_jsonl_dataset",
    "deduplicate_records",
    "resolve_cross_repo_value_duplicates",
    "assert_no_leakage",
    "repository_grouped_split",
    "stratified_repository_split",
    "iter_repository_group_folds",
    "records_to_matrix",
    "compute_classification_metrics",
    "bootstrap_metric_ci",
    "mcnemar_paired_test",
    "profile_inference_overhead",
    "run_feature_group_ablation",
    "extract_global_explanations",
    "explain_single_candidate",
    "SAFE_SKOPS_TYPE_PREFIXES",
    "compute_file_sha256",
    "verify_artifact_sha256",
    "save_verified_skops_model",
    "load_verified_skops_model",
    "collect_provenance",
    "train_and_compare_models",
    "HybridTriagePipeline",
]
