# SIDIK-ML

**Ekstensi *Machine Learning* *Non-Neural* yang Dapat Dijelaskan (*Explainable Non-ANN ML*) untuk Reduksi *False Positive* pada Deteksi *Hardcoded Secret* SIDIK**

`SIDIK-ML` adalah paket perangkat lunak Python independen yang mengonsumsi [`sidik-security`](https://github.com/radityobw/SIDIK) (`>=1.1.0,<2.0.0`) melalui kontrak data berversi `SecretCandidate` (`schema_version = "1.0.0"`).

---

## Batasan Ruang Lingkup & Prinsip Desain

1. **Klasifikasi Klasik / *Non-ANN* Eksklusif:** Hanya menggunakan algoritma *machine learning* klasik terstruktur (*Logistic Regression*, *Decision Tree*, *Random Forest*, *Extra Trees*, *Gradient Boosting*). Tidak menggunakan jaringan saraf tiruan (ANN/DNN/CNN/RNN/LSTM/*Transformers*/*CodeBERT*).
2. **Pemisahan Kontrak Bersih:** `SIDIK` tetap menjadi penganalisis deterministik mandiri tanpa ketergantungan ML. `SIDIK-ML` mengonsumsi `SecretCandidate` tanpa menyimpan nilai rahasia mentah.
3. **Kebijakan *Triage* Aman (`TriagePolicy`):** Secara *default* (`suppress_false_positives=False`), seluruh temuan deterministik `SIDIK` tetap dipertahankan bersama skor keyakinan ML dan ringkasan fitur; temuan berisiko `CRITICAL` dilindungi agar tidak dibungkam diam-diam (`never_suppress_severities=("CRITICAL",)`).
4. **Deserialisasi Luring Aman:** Artefak model memakai `skops` disertai verifikasi *checksum SHA-256* sebelum pemuatan (menolak deserialisasi `pickle`/`joblib` mentah yang tidak terverifikasi).

---

## Struktur Paket

```text
SIDIK-ML/
├── sidik_ml/
│   ├── __init__.py
│   ├── schemas.py         # MLPrediction, HybridFinding, TriagePolicy
│   ├── features.py        # Ekstraktor fitur statis bebas target leakage dari SecretCandidate
│   ├── serialization.py   # Manifest SHA-256 & pemuatan model skops aman
│   └── integration.py     # HybridTriagePipeline (SIDIK + opsional ML classifier)
├── tests/
│   └── test_contract.py   # Uji integrasi kontrak SIDIK v1.1 <-> SIDIK-ML
├── pyproject.toml
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## Status Pengembangan Saat Ini (*Phase 3 Scaffold*)
* Paket fondasi dan uji kontrak antarmuka terhadap `sidik-security v1.1.0` telah tersedia dan teruji.
* Model ML terlatih belum disertakan sebelum akuisisi dataset terpeer-review (mis. *SecretBench*) serta eksperimen terkelompok lintas-*repository* selesai di [`SIDIK-ML-paper`](https://github.com/radityobw/SIDIK-ML-paper).
