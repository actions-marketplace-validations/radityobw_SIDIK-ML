# Changelog

## [0.5.9] - 2026-11-25
- **CLI Symmetry for Polyglot SAST & Multi-Ecosystem OSV SCA (`sidik_ml/cli.py`):** Added `sidik-ml update-db --ecosystem <list|all>` (and `--update-db`) plus `sidik-ml db-info` (and `--db-info`) subcommand/flag passthroughs delegating to `SIDIK >=v1.4.0`, and updated `--include-deps`, `--extensions`, and CLI examples for polyglot scanning across 8 SCA ecosystems (`PyPI`, `npm`, `Go`, `crates.io`, `Maven`, `Packagist`, `RubyGems`, `NuGet`).
- **GitHub Action, Container & Workflow Alignment (`action.yml`, `Dockerfile`, `.github/workflows/`):** Added optional `ecosystems` input to `action.yml`, updated Marketplace description for polyglot SAST + 8-ecosystem SCA, and aligned CI/Docker/Action/release installs to prefer `sidik>=1.4.0,<2.0.0` (with PyPI metadata keeping `sidik>=1.3.4,<2.0.0` compatibility) plus all-5 bundled model assertions in `release.yml`.
- **Documentation & Polyglot Integration Tests (`README.md`, `tests/test_plugin_integration.py`):** Documented polyglot (`30+` extensions) & 8-ecosystem SCA integration in Indonesian and English sections, and added end-to-end tests covering TypeScript/Go secrets, npm/Go/Maven advisories, `db-info`, and `update-db` forwarding.

## [0.5.8] - 2026-11-25
- **Complete 5-Model Bundled Portable Non-ANN Suite (`sidik_ml/data/`):** Pre-trained and bundled the remaining four SHA-256-verified portable Non-ANN models alongside `gradient_boosting` (`default_random_forest.json`, `default_extra_trees.json`, `default_logistic_regression.json`, and `default_decision_tree.json`), all trained strictly repository-grouped on Apache-2.0 `Samsung/CredData` with IEEE-754 `float32` branch matching (`max_abs_diff < 1e-11` vs scikit-learn).
- **Portable Linear & Tree-Ensemble Serialization (`sidik_ml/serialization.py`, `sidik_ml/explainability.py`):** Added `PortableLogisticRegressionClassifier`, `PortableTreeEnsembleClassifier`, `export_portable_linear_model()`, `export_portable_tree_ensemble_model()`, unified `export_portable_model()`, `resolve_bundled_model_name()`, `get_bundled_model_path()`, `list_bundled_models()`, and `load_bundled_model()`, plus pure-Python linear explainability support.
- **Preset Selection & Discovery Across CLI, Plugin, TUI, and GUI (`sidik_ml/cli.py`, `sidik_ml/plugin.py`):** Added `sidik-ml models [--json]` subcommand and enabled `--model <preset>` (e.g., `random_forest`, `extra_trees`, `logistic_regression`, `decision_tree`, `gradient_boosting`, or short aliases `rf`, `et`, `lr`, `dt`, `gb`) across CLI scan, Interactive TUI, Desktop GUI, and Core `sidik --enable-ml --ml-model <preset>`.

## [0.5.7] - 2026-11-25
- **Bundled Out-of-the-Box Portable Non-ANN Model (`sidik_ml/data/default_gradient_boosting.json`):** Embedded the SHA-256-verified *Gradient Boosting* ensemble (`~46 KB`, 100 trees, trained strictly repository-grouped on Apache-2.0 `Samsung/CredData`, reproducing scikit-learn probabilities within `max_abs_diff < 1e-12` across all Python/scikit-learn versions). Global `pip install sidik-ml` users now get live ML probability triage automatically across CLI, Interactive TUI, Desktop GUI, and `sidik --enable-ml` without needing a separate model download.
- **Portable Tree Serialization & Unified Verification API (`sidik_ml/serialization.py`, `sidik_ml/plugin.py`):** Added `PortableGradientBoostingClassifier`, `export_portable_gbdt_model()`, `load_verified_portable_model()`, `load_verified_model()` (dispatching `.json` portable artifacts or `.skops` artifacts under SHA-256 verification), and `get_bundled_default_model_path()`.

## [0.5.6] - 2026-11-25
- **Single-Source-of-Truth Versioning (`sidik_ml/_version.py`):** Configured `[tool.setuptools.dynamic]` in [`pyproject.toml`](pyproject.toml) so `sidik_ml/__init__.py`, `plugin.py`, `integration.py`, package metadata, and integration tests all derive from one authoritative `__version__`.
- **Unpinned PyPI Core Resolution & Live README Badges:** Updated [`Dockerfile`](Dockerfile), [`action.yml`](action.yml), [`.github/workflows/release.yml`](.github/workflows/release.yml), and [`.github/workflows/sidik-ml-ci.yml`](.github/workflows/sidik-ml-ci.yml) to resolve `sidik>=1.3.4,<2.0.0` directly from PyPI (falling back to `@main`), and replaced static version/cross-repo badges in [`README.md`](README.md) with live PyPI, GitHub Release, and CI Shields.io badges while unifying report format references (`8`).
- **Automated Release Metadata Sync Tests:** Added version, `Author`, and `CHANGELOG.md` synchronization assertions inside [`tests/test_plugin_integration.py`](tests/test_plugin_integration.py).

## [0.5.5] - 2026-11-25
- Corrected package, GitHub Action, container, and license author/copyright metadata across [`pyproject.toml`](pyproject.toml), [`action.yml`](action.yml), [`LICENSE`](LICENSE), [`Dockerfile`](Dockerfile), and [`.github/workflows/release.yml`](.github/workflows/release.yml) to list **Radityo Budi Waskito** as the sole author.
- Synced default Core `SIDIK` fallback/container reference to `v1.3.6` (keeping PyPI runtime compatibility at `sidik>=1.3.4,<2.0.0`).

## [0.5.4] - 2026-11-25
- Updated minimum Core dependency to **`sidik>=1.3.4,<2.0.0`** (with `action.yml`, `Dockerfile`, CI, and `release.yml` synced to `v1.3.4` / `v0.5.4`) and merged [`action.yml`](action.yml) into a single scan pass (`--format github --sarif-output ...`).
- **Full CLI & CI/CD Parity (`sidik_ml/cli.py`):** Defaults to Amber-Orange `console` output (SAST + SCA), `--fail-on any` exit code enforcement (`1` on findings, `2` on missing target path), explicit `stderr` pass-through notice when no `.skops` model is loaded, `[tool.sidik]` config loading, `--severity`, `--mode`, `--baseline`, `--write-baseline`, `--sarif-output`, `--extensions`, `--exclude`, `--no-color`, `--quiet`, `--no-step-summary`, and `$GITHUB_STEP_SUMMARY` integration.
- **Single-Pass Plugin Execution (`sidik_ml/plugin.py`):** Reuses `_preextracted_candidates` provided by `SecurityScanner` (`sidik>=1.3.4`) to avoid scanning files twice.
- **Model Verification, Cache & Version-Skew Guard (`sidik_ml/serialization.py`):** Added `get_ml_cache_dir()`, `fetch_verified_model()` (plus `sidik-ml fetch-model` subcommand) downloading HTTPS artifacts into user cache after SHA-256 verification, and `sklearn_version` recording in `.manifest.json` with `RuntimeWarning` skew detection.
- **Packaging & Typing:** Added `gui` optional extra (`sidik[gui]>=1.3.4,<2.0.0`), PEP 561 `sidik_ml/py.typed` in wheel `package-data`, PyPI `classifiers`, and absolute GitHub links in `README.md`.

## [0.5.3] - 2026-11-24
- Updated Core dependency from `sidik-security` to **`sidik>=1.3.3,<2.0.0`** matching the PyPI `sidik` package name, and synced `action.yml`, `Dockerfile`, CI, and `release.yml` to `v1.3.3` / `v0.5.3`.

## [0.5.2] - 2026-11-24
- Synced minimum Core dependency to `sidik-security>=1.3.2,<2.0.0` (including Marketplace-validated `SIDIK` `action.yml`, CI, and `Dockerfile` `SIDIK_CORE_REF=v1.3.2`).
- Added [`Dockerfile`](Dockerfile), [`.dockerignore`](.dockerignore), and [`.github/workflows/release.yml`](.github/workflows/release.yml) for automated GitHub Release assets, `ghcr.io/radityobw/sidik-ml`, and gated PyPI Trusted Publishing.

## [0.5.1] - 2026-11-24
- Bumped package version to `0.5.1` and required `sidik-security>=1.3.1,<2.0.0` (updating CI to install `SIDIK@v1.3.1` with Amber-Orange `#ea580c` / `#f59e0b` HTML & GUI support).
- Added composite GitHub Action [`action.yml`](action.yml) (`SIDIK-ML Hybrid Security Scanner`) supporting Non-ANN ML triage, SARIF v2.1.0, and inline PR annotations.
- Scrubbed private research-paper repository references from public source and documentation (`SIDIK` & `SIDIK-ML`).
- Added symmetric CLI shorthands (`-i`, `--gui`, `.`, `-t`), `python -m sidik_ml` module entry point, and `run_sidik_ml.bat` / `run_sidik_ml_cli.bat` / `run_sidik_ml.sh` launchers.
- Fixed Pyrefly `tk.Tk` root typing in `tests/test_plugin_integration.py` (`18/18` passing).

## [0.5.0] - 2026-11-24
- Synced package version (`sidik_ml.__version__`, `SidikMLPlugin.version`, `HybridTriagePipeline.model_version` default) to `0.5.0` and updated minimum dependency to `sidik-security>=1.3.0,<2.0.0`.
- Added `--version` / `-v` flag to `sidik-ml` CLI displaying `SIDIK-ML v0.5.0` alongside the active Core SIDIK version.
- Paired with `SIDIK v1.3.0` to deliver an unmistakable *Amber Orange* (`#f59e0b` / `#ea580c`) identity across `sidik-ml gui`, `sidik-ml-gui`, yellow/amber `sidik-ml>` TUI console (`sidik-ml interactive`), and ML-triaged Console/Markdown/HTML reports (`NON-ANN ML TRIAGE`).
- Expanded `tests/test_plugin_integration.py` to assert `sidik-ml --version`, `SecurityAnalyzerGUI` theme differentiation, and amber-orange HTML report rendering, and updated CI to install `SIDIK@v1.3.0`.

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
