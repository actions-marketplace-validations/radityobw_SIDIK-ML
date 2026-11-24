# Contributing to SIDIK-ML

Thank you for contributing to **SIDIK-ML** (*Explainable Non-Neural Machine Learning Extension for SIDIK*). This guide covers architectural constraints, local development, pre-flight verification, and the Single-Source-of-Truth (SSOT) release workflow.

---

## 1. Scope, Non-ANN Mandate, & Attribution Rules

* **Strictly Non-Neural (Non-ANN) Classical ML Only:** Primary models are restricted to classical interpretable algorithms (`logistic_regression`, `decision_tree`, `random_forest`, `extra_trees`, `gradient_boosting`) enforced by `validate_non_ann_model_name()`. Do not introduce ANN, MLP, CNN, RNN, LSTM, or Transformer dependencies.
* **Zero Data Leakage & Safe Serialization:** Features must be extracted solely from scan-time `SecretCandidate` attributes without raw secret storage, splits must remain repository-grouped (`StratifiedGroupKFold`), and `.skops` artifacts must enforce `SAFE_SKOPS_TYPE_PREFIXES` + SHA-256 verification.
* **Fail-Safe Protection:** Keep advisory pass-through behavior safe when no `.skops` model is loaded and preserve `CRITICAL` severity shielding (`never_suppress_severities=("CRITICAL",)`).
* **Author & Copyright Attribution:** Package metadata in [`pyproject.toml`](pyproject.toml), [`action.yml`](action.yml), [`LICENSE`](LICENSE), and [`Dockerfile`](Dockerfile) belongs solely to **Radityo Budi Waskito** (`test_entry_point_discovery_finds_sidik_ml` asserts `Author == "Radityo Budi Waskito"`).

---

## 2. Local Development Setup

```bash
git clone https://github.com/radityobw/SIDIK-ML.git
cd SIDIK-ML
python -m venv .venv

# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux / macOS:      source .venv/bin/activate

pip install --no-build-isolation --no-deps --config-settings editable_mode=compat -e ".[dev,gui]"
```

Run the test suite and verify plugin registration:

```bash
pytest tests -v
sidik --list-plugins
sidik-ml --version
```

---

## 3. Coding, Packaging, & Documentation Standards

1. **No Em-Dashes (`\u2014`):** Use standard hyphens (`-`), colons (`:`), or parentheses instead of `\u2014` across code, Markdown, and workflows.
2. **PEP 639 Licensing (`pyproject.toml`):** Keep `license = "MIT"` and `license-files = ["LICENSE"]`. Never add legacy `License :: OSI Approved :: ...` classifiers.
3. **PyPI-Safe `README.md` Links:** All image and file links in `README.md` must use absolute `https://raw.githubusercontent.com/...` or `https://github.com/...` URLs. Keep version badges dynamic via Shields.io (`img.shields.io/pypi/v/sidik-ml`, `img.shields.io/github/v/release/...`).
4. **GitHub Marketplace Action Limit (`action.yml`):** Keep `description` strictly under **125 characters**.
5. **Unpinned Compatible Core Dependency:** `Dockerfile`, `action.yml`, and CI workflows resolve `sidik>=1.3.4,<2.0.0` from PyPI (with `@main` Git fallback) so `SIDIK` patch releases do not break or stale-pin `SIDIK-ML`. Only bump the lower bound when `SIDIK-ML` requires a new Core API.

---

## 4. SSOT Release Checklist (`2` Files Only)

`SIDIK-ML` uses `[tool.setuptools.dynamic]` reading from [`sidik_ml/_version.py`](sidik_ml/_version.py). For a routine version bump (`A.B.C`):

1. Update `__version__ = "A.B.C"` in [`sidik_ml/_version.py`](sidik_ml/_version.py).
   * Automatically syncs `sidik_ml.__version__`, `SidikMLPlugin.version`, `HybridTriagePipeline` default `model_version`, package metadata, and integration tests.
2. Prepend a `## [A.B.C] - YYYY-MM-DD` entry in [`CHANGELOG.md`](CHANGELOG.md) (verified automatically by `tests/test_plugin_integration.py`).

### Pre-Flight Verification Gate

```bash
# Clean any stale repo-root .egg-info before editable reinstall
pip install --no-build-isolation --no-deps --config-settings editable_mode=compat -e .
pytest tests -v
python -m build
python -m twine check dist/*
```

Verify the built `.whl` contains `sidik_ml/_version.py`, `sidik_ml/py.typed`, `Version: A.B.C`, and `Author: Radityo Budi Waskito` in `.dist-info/METADATA`.

---

## 5. Publishing Order & CI/CD Automation

Always verify the required minimum **`sidik`** version is already live on PyPI **before** releasing `SIDIK-ML`:

1. Tag and push:
   ```bash
   git add -A
   git commit -m "Release vA.B.C: <summary>"
   git tag -a vA.B.C -m "Release vA.B.C: <summary>"
   git push origin main vA.B.C
   ```
2. Create a GitHub Release on `vA.B.C` (or run `gh release create vA.B.C --generate-notes`).
3. Publishing triggers [`.github/workflows/release.yml`](.github/workflows/release.yml), which automatically:
   * Runs `pytest`, builds wheel/sdist + `SHA256SUMS.txt`, runs `twine check`, and attaches release assets.
   * Builds and pushes `ghcr.io/radityobw/sidik-ml:latest` and `:A.B.C`.
   * Publishes to PyPI via OIDC Trusted Publishing when repository variable `ENABLE_PYPI_PUBLISH` is set to `true`.
