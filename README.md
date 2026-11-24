# SIDIK-ML

**Ekstensi *Machine Learning* *Non-Neural* yang Dapat Dijelaskan (*Explainable Non-ANN ML*) untuk Reduksi *False Positive* pada Deteksi *Hardcoded Secret* SIDIK**

`SIDIK-ML` adalah paket perangkat lunak Python independen (`v0.2.0`) yang mengonsumsi [`sidik-security`](https://github.com/radityobw/SIDIK) (`>=1.1.0,<2.0.0`) melalui kontrak data berversi `SecretCandidate` (`schema_version = "1.0.0"`).

---

## Batasan Ruang Lingkup & Prinsip Desain

1. **Klasifikasi Klasik / *Non-ANN* Eksklusif:** Hanya mengizinkan model *Non-ANN* terstruktur (`logistic_regression`, `decision_tree`, `random_forest`, `extra_trees`, `gradient_boosting`) dengan validasi penolak arsitektur *neural* (`validate_non_ann_model_name()`).
2. **Pencegahan *Data Leakage* & Privasi:** Ekstraksi `30` fitur statis murni bebas *target leakage* (`extract_feature_dict`), deduplikasi sebelum pemisahan, dan `assert_no_leakage()` yang menjamin nol tumpang-tindih `repo_id` maupun `value_sha256`. Nilai rahasia mentah tidak pernah disimpan.
3. **Kebijakan *Triage* Aman (`TriagePolicy`):** Secara *default* (`suppress_false_positives=False`), seluruh temuan deterministik `SIDIK` tetap dipertahankan bersama skor probabilitas ML dan penjelasan fitur; temuan `CRITICAL` dilindungi agar tidak dibungkam diam-diam (`never_suppress_severities=("CRITICAL",)`).
4. **Deserialisasi Luring Aman:** Artefak model memakai format `skops` disertai verifikasi *checksum SHA-256* sebelum pemuatan (tanpa `pickle`/`joblib` mentah).

---

## Struktur Paket

```text
SIDIK-ML/
├── sidik_ml/
│   ├── __init__.py
│   ├── cli.py             # Antarmuka CLI (`sidik-ml scan`, `extract-features`, `verify-model`)
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
│   └── test_models_and_training.py
├── pyproject.toml
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## Pemakaian Singkat

```bash
# Pindai path dengan kebijakan triage aman (tanpa supresi default)
sidik-ml scan path/to/project --json

# Ekspor fitur bebas leakage dari kandidat untuk anotasi / eksperimen
sidik-ml extract-features path/to/project --output candidates.jsonl --repo-id my_repo

# Verifikasi integritas checksum SHA-256 artefak model .skops
sidik-ml verify-model model.skops --sha256 <hex_digest>
```
