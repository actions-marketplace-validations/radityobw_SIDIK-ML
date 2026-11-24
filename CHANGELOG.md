# Changelog

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
