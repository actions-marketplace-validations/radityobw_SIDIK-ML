# Changelog

## [0.4.0] - 2026-11-24
- Added `sidik_ml.plugin` (`SidikMLPlugin`, `create_plugin()`, `resolve_model_and_sha256()`, `discover_default_model()`) and registered entry point `sidik.plugins` (`sidik-ml`) in `pyproject.toml`.
- Added thin launchers `sidik-ml interactive` (TUI) and `sidik-ml gui` / `sidik-ml-gui` delegating directly to `SIDIK`'s shared UI (`v1.2.0`) with zero UI duplication.
- Extended `sidik-ml scan` with `--include-deps` (unified OSV SCA), `--format` (`compact`, `console`, `json`, `markdown`, `csv`, `html`, `sarif`, `github`), `--output`, `--lang`, and automatic `.skops.manifest.json` SHA-256 resolution.
- Added end-to-end plugin discovery, unified SAST+SCA, and GUI integration tests in `tests/test_plugin_integration.py`.

## [0.3.0] - 2026-11-24
- Added `resolve_cross_repo_value_duplicates()` in `sidik_ml.dataset` to eliminate cross-repository `value_sha256` leakage and conflicting-label hash collisions before splitting.
- Added `stratified_repository_split()` in `sidik_ml.dataset` and `use_stratified_split` / `cv_summary` selection via `StratifiedGroupKFold` in `sidik_ml.training`.
- Enforced strict `SAFE_SKOPS_TYPE_PREFIXES` module allowlisting inside `load_verified_skops_model()` (`sidik_ml.serialization`).
- Grounded `HybridTriagePipeline` per-candidate explanations (`top_feature_contributions`) in the active classifier via `explain_single_candidate()`.
- Added `sidik-ml train` CLI subcommand for end-to-end reproducible Non-ANN model training and `.skops` export.

## [0.2.0] - 2026-11-24
- Added complete *Non-ANN* model factory (`sidik_ml.models`: Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting) with strict neural exclusion guard.
- Added dataset curation, deduplication, and repository-grouped splitting utilities (`sidik_ml.dataset`) enforcing zero overlap of `repo_id` and `value_sha256` between train and test splits.
- Added metrics, McNemar paired significance testing, bootstrap 95% confidence intervals, batch overhead profiling, and 6-group feature ablation (`sidik_ml.evaluation`).
- Added model-native, permutation-importance, decision-rule, and per-candidate explanation helpers (`sidik_ml.explainability`).
- Added `save_verified_skops_model()`, reproducible training/comparison pipeline (`sidik_ml.training`), `sidik-ml` CLI entrypoint (`sidik_ml.cli`), GitHub Actions CI, and unit test suites.

## [0.1.0] - 2026-11-24
- Initial modular scaffold (`sidik_ml`) defining clean contract integration with `sidik-security>=1.1.0`.
