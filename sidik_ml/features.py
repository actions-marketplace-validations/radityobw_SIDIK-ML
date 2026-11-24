"""Deterministic, leakage-free static feature extraction from SIDIK SecretCandidate records."""

from typing import Dict, List
from sidik.models import SecretCandidate

FEATURE_SCHEMA_VERSION = "1.0.0"

# Ordered numeric feature set strictly derived from scan-time observable properties
FEATURE_NAMES: List[str] = [
    "str_length",
    "str_unique_chars",
    "str_unique_ratio",
    "digit_ratio",
    "lower_ratio",
    "upper_ratio",
    "symbol_ratio",
    "is_hex_like",
    "is_base64_like",
    "is_uuid_like",
    "is_git_sha_like",
    "has_provider_prefix",
    "shannon_entropy",
    "min_rule_entropy",
    "entropy_margin",
    "meets_entropy_threshold",
    "is_generic_rule",
    "is_comment",
    "is_placeholder_or_mock",
    "has_positive_keyword",
    "line_length",
    "is_assignment",
    "is_dict_entry",
    "is_function_call",
    "is_env_reference",
    "prev_line_is_comment",
    "is_test_path",
    "is_example_or_doc_path",
    "path_depth",
    "passed_deterministic_filter",
]


def extract_feature_dict(candidate: SecretCandidate) -> Dict[str, float]:
    """Converts a SecretCandidate into a fixed numeric feature dictionary without target leakage."""
    s_stats = candidate.string_stats or {}
    lex = candidate.lexical_context or {}
    pat = candidate.pattern_evidence or {}
    code = candidate.code_context or {}
    fctx = candidate.file_context or {}
    det = candidate.deterministic_decision or {}

    min_ent = float(pat.get("min_entropy", 0.0))
    ent = float(candidate.entropy)

    return {
        "str_length": float(s_stats.get("length", 0)),
        "str_unique_chars": float(s_stats.get("unique_chars", 0)),
        "str_unique_ratio": float(s_stats.get("unique_ratio", 0.0)),
        "digit_ratio": float(s_stats.get("digit_ratio", 0.0)),
        "lower_ratio": float(s_stats.get("lower_ratio", 0.0)),
        "upper_ratio": float(s_stats.get("upper_ratio", 0.0)),
        "symbol_ratio": float(s_stats.get("symbol_ratio", 0.0)),
        "is_hex_like": 1.0 if s_stats.get("is_hex_like") else 0.0,
        "is_base64_like": 1.0 if s_stats.get("is_base64_like") else 0.0,
        "is_uuid_like": 1.0 if s_stats.get("is_uuid_like") else 0.0,
        "is_git_sha_like": 1.0 if s_stats.get("is_git_sha_like") else 0.0,
        "has_provider_prefix": 1.0 if s_stats.get("matched_provider_prefix") else 0.0,
        "shannon_entropy": ent,
        "min_rule_entropy": min_ent,
        "entropy_margin": round(ent - min_ent, 4),
        "meets_entropy_threshold": 1.0 if pat.get("meets_entropy_threshold") else 0.0,
        "is_generic_rule": 1.0 if pat.get("is_generic_rule") else 0.0,
        "is_comment": 1.0 if lex.get("is_comment") else 0.0,
        "is_placeholder_or_mock": 1.0 if lex.get("is_placeholder_or_mock") else 0.0,
        "has_positive_keyword": 1.0 if lex.get("has_positive_keyword") else 0.0,
        "line_length": float(code.get("line_length", 0)),
        "is_assignment": 1.0 if code.get("is_assignment") else 0.0,
        "is_dict_entry": 1.0 if code.get("is_dict_entry") else 0.0,
        "is_function_call": 1.0 if code.get("is_function_call") else 0.0,
        "is_env_reference": 1.0 if code.get("is_env_reference") else 0.0,
        "prev_line_is_comment": 1.0 if code.get("prev_line_is_comment") else 0.0,
        "is_test_path": 1.0 if fctx.get("is_test_path") else 0.0,
        "is_example_or_doc_path": 1.0 if fctx.get("is_example_or_doc_path") else 0.0,
        "path_depth": float(fctx.get("path_depth", 0)),
        "passed_deterministic_filter": 1.0 if det.get("passed_deterministic_filter") else 0.0,
    }


def extract_feature_vector(candidate: SecretCandidate) -> List[float]:
    """Returns ordered numeric features matching FEATURE_NAMES."""
    fdict = extract_feature_dict(candidate)
    return [fdict[name] for name in FEATURE_NAMES]
