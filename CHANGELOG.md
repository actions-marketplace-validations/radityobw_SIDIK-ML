# Changelog

## [0.1.0] - 2026-11-24
- Initial modular scaffold (`sidik_ml`) defining clean contract integration with `sidik-security>=1.1.0`.
- Added versioned schemas (`MLPrediction`, `HybridFinding`, `TriagePolicy` with fail-safe `never_suppress_severities = ("CRITICAL",)`).
- Added deterministic, leakage-free static feature extractor (`extract_feature_dict`) over `SecretCandidate`.
- Added safe offline model deserialization guard (`skops` + SHA-256 verification; no raw `pickle`/`joblib`).
- Added `HybridTriagePipeline` and end-to-end contract tests (`tests/test_contract.py`).
