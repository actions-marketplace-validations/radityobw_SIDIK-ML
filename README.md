# SIDIK-ML

**Ekstensi *Machine Learning* *Non-Neural* yang Dapat Dijelaskan (*Explainable Non-ANN ML*) untuk Reduksi *False Positive* pada Deteksi *Hardcoded Secret* SIDIK**

`SIDIK-ML` adalah paket perangkat lunak Python independen (`v0.4.0`) yang mengonsumsi [`sidik-security`](https://github.com/radityobw/SIDIK) (`>=1.2.0,<2.0.0`) melalui kontrak data berversi `SecretCandidate` (`schema_version = "1.0.0"`) sekaligus mendaftarkan *entry point* `sidik.plugins`.

Dengan arsitektur *plugin entry-point* dan *thin launcher*, `SIDIK-ML` kini otomatis memperkaya **CLI**, **CLI Interaktif (TUI)**, **Desktop GUI (`sidik-gui` / `sidik-ml-gui`)**, **OSV SCA**, dan **7 format laporan** milik `SIDIK` tanpa duplikasi kode antarmuka.

---

## Batasan Ruang Lingkup & Prinsip Desain

1. **Klasifikasi Klasik / *Non-ANN* Eksklusif:** Hanya mengizinkan model *Non-ANN* terstruktur (`logistic_regression`, `decision_tree`, `random_forest`, `extra_trees`, `gradient_boosting`) dengan validasi penolak arsitektur *neural* (`validate_non_ann_model_name()`).
2. **Pencegahan *Data Leakage* & Privasi:** Ekstraksi `30` fitur statis murni bebas *target leakage* (`extract_feature_dict`), deduplikasi konteks dan *cross-repository value hash* (`resolve_cross_repo_value_duplicates()`), serta `assert_no_leakage()` yang menjamin nol tumpang-tindih `repo_id` maupun `value_sha256` lintas *split*. Nilai rahasia mentah tidak pernah disimpan.
3. **Kebijakan *Triage* Aman (`TriagePolicy`):** Secara *default* (`suppress_false_positives=False`), seluruh temuan deterministik `SIDIK` tetap dipertahankan bersama skor probabilitas ML dan penjelasan fitur berbasis bobot model aktif (`explain_single_candidate()`); temuan `CRITICAL` dilindungi agar tidak dibungkam diam-diam (`never_suppress_severities=("CRITICAL",)`).
4. **Deserialisasi Luring Aman:** Artefak model memakai format `skops` disertai verifikasi *checksum SHA-256* dan pemeriksaan *allowlist* prefiks tipe aman (`SAFE_SKOPS_TYPE_PREFIXES`) sebelum pemuatan (tanpa `pickle`/`joblib` mentah).

---

## Struktur Paket

```text
SIDIK-ML/
├── sidik_ml/
│   ├── __init__.py
│   ├── cli.py             # Antarmuka CLI, peluncur interactive (TUI) & gui, ekstrak fitur, train
│   ├── plugin.py          # Adapter entry point sidik.plugins (SidikMLPlugin & auto-manifest SHA-256)
│   ├── schemas.py         # MLPrediction, HybridFinding, TriagePolicy
│   ├── features.py        # 30 fitur statis bebas target leakage dari SecretCandidate
│   ├── models.py          # 5 model Non-ANN & neural exclusion guard
│   ├── dataset.py         # JSONL I/O, deduplikasi, GroupShuffleSplit / StratifiedGroupKFold
│   ├── evaluation.py      # F1, MCC, bootstrap CI, McNemar test, overhead, feature ablation
│   ├── explainability.py  # Bobot model, permutation importance, tree rules, penjelasan kandidat
│   ├── serialization.py   # Save/load .skops aman terverifikasi SHA-256
│   ├── training.py        # Pipeline pelatihan reproduksibel & pencatat provenans JSON
│   └── integration.py     # HybridTriagePipeline (SIDIK + opsional ML classifier)
├── tests/
│   ├── test_contract.py
│   ├── test_dataset_and_leakage.py
│   ├── test_models_and_training.py
│   └── test_plugin_integration.py
├── pyproject.toml
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## Instalasi & Aktivasi Lingkungan (*Virtual Environment*)

`SIDIK-ML` menggunakan mesin inti, SCA OSV, CLI interaktif, serta GUI milik `SIDIK` (`sidik-security>=1.2.0`), sehingga keduanya harus dipasang di **lingkungan Python (`.venv`) yang sama**:

```bash
# Dari direktori SIDIK-ML/ dengan venv SIDIK aktif
..\SIDIK\.venv\Scripts\Activate.ps1       # Windows PowerShell
# source ../SIDIK/.venv/bin/activate      # Linux / macOS

# Pasang SIDIK (bersama GUI) dan SIDIK-ML secara editable
pip install -e "..\SIDIK[gui]"
pip install -e .
```

> **Catatan Windows PowerShell:** Apabila Anda membuka langsung folder `SIDIK\.venv\Scripts` tanpa menjalankan `Activate.ps1`, PowerShell tidak memuat perintah dari folder aktif secara otomatis. Gunakan awalan `.\` (misalnya `.\sidik-ml gui`, `.\sidik-ml interactive`, atau `.\sidik-ml-gui`), atau aktifkan *virtual environment* terlebih dahulu dengan `.\Activate.ps1`. Jika skrip aktivasi terblokir kebijakan eksekusi PowerShell, jalankan perintah lewat `python -m sidik_ml.cli ...`.

---

## Pemakaian Singkat

Melalui *entry point* `sidik.plugins`, perintah `sidik-ml` dan `sidik` berbagi antarmuka CLI interaktif (TUI), *Desktop GUI*, serta 7 format laporan yang sama:

```bash
# Pindai path dengan kebijakan triage aman (tanpa supresi default)
sidik-ml scan path/to/project --json

# Pindai terpadu (Secret + OSV SCA) dengan ekspor laporan HTML / SARIF / Markdown
sidik-ml scan path/to/project --include-deps --model model.skops --format html -o hasil_ml.html

# Luncurkan Interactive CLI Console (TUI) atau Desktop GUI dengan plugin SIDIK-ML aktif
sidik-ml interactive
sidik-ml gui
sidik-ml-gui

# Atau gunakan langsung dari perintah SIDIK standar
sidik --target path/to/project --enable-ml --ml-model model.skops

# Ekspor fitur bebas leakage dari kandidat untuk anotasi / eksperimen
sidik-ml extract-features path/to/project --output candidates.jsonl --repo-id my_repo

# Verifikasi integritas checksum SHA-256 artefak model .skops
sidik-ml verify-model model.skops --sha256 <hex_digest>

# Latih & bandingkan 5 model Non-ANN menggunakan StratifiedGroupKFold
sidik-ml train dataset.jsonl --output-dir artifacts/ --stratified
```
