<div align="center">

<img src="https://raw.githubusercontent.com/radityobw/SIDIK/main/logo/SIDIK-nobg-dark.png" alt="SIDIK-ML Logo" width="300"/>

# SIDIK-ML: *Explainable Non-Neural ML Extension*

**Ekstensi *Machine Learning* Klasik *Non-Neural* Terjelaskan untuk Reduksi *False Positive* Deteksi *Hardcoded Secret* SIDIK**  
*Explainable Classical Non-ANN Machine Learning Extension for False Positive Reduction in SIDIK Static Secret Detection*

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3b82f6.svg)](#)
[![Version v0.5.4](https://img.shields.io/badge/Version-v0.5.4-f59e0b.svg)](#)
[![Paradigm Non-ANN Only](https://img.shields.io/badge/Paradigm-Non--ANN%20Only-ea580c.svg)](#)
[![Random Forest MCC 0.7692](https://img.shields.io/badge/RF%20F1-0.9636%20%7C%20MCC%200.7692-10b981.svg)](#)
[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-19%2F19%20Passing-22c55e.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-a855f7.svg)](#)

</div>

---

# 🇮🇩 Bahasa Indonesia

## 📌 Ikhtisar Proyek & Filosofi Desain

**SIDIK-ML** (`v0.5.4`) adalah paket perangkat lunak Python independen yang memperluas kemampuan pemindai deterministik [`SIDIK`](https://github.com/radityobw/SIDIK) (`>=v1.3.4`) dengan lapisan penyaringan *false positive* berbasis **klasifikasi klasik *Non-Artificial Neural Network* (*Non-ANN*) yang terjelaskan**.

Alih-alih mengganti mesin aturan deterministik `SIDIK` atau menyisipkan model *deep neural network* berbobot ratusan megabita, `SIDIK-ML` bekerja sebagai *plugin* opsional melalui kontrak data aman `SecretCandidate` (`schema_version = "1.0.0"`) dan *entry point* standar `sidik.plugins`. Ketika dijalankan melalui `sidik-ml gui`, `sidik-ml interactive`, maupun `--enable-ml`, antarmuka menampilkan tema khusus **Amber Orange (`#f59e0b` / `#ea580c`)** lengkap dengan lencana `NON-ANN ML`, *prompt* `sidik-ml>`, serta dasbor HTML bernuansa kuning-oranye agar langsung terbedakan dari mode deterministik standar **Ocean Blue (`#3b82f6`)** milik `SIDIK`.

---

## ✨ Fitur dan Kapabilitas Utama

1. **Lima Model Klasik *Non-ANN* dengan Penjaga Penolak Arsitektur Neural:**
   - Mendukung eksklusif *Logistic Regression*, *Decision Tree*, *Random Forest*, *Extra Trees*, dan *Gradient Boosting* via `create_non_ann_model()`.
   - Fungsi `validate_non_ann_model_name()` secara eksplisit menolak model neural (`ann`, `mlp`, `cnn`, `rnn`, `lstm`, `transformer`, `codebert`) demi menjamin transparansi keputusan dan bobot ringan.
2. **Rekayasa `30` Fitur Statis Bebas *Target Leakage* (`FEATURE_SCHEMA_VERSION = "1.0.0"`):**
   - Mengekstrak tujuh dimensi fitur murni dari konteks pemindaian (`string_stats`, `entropy`, `pattern_evidence`, `lexical_context`, `code_context`, `file_context`, serta `deterministic_signal`) tanpa pernah menyimpan nilai kredensial mentah (hanya `masked_value` dan `value_sha256`).
3. **Validasi Silang Anti-Kebocoran Data (*Zero Data Leakage*):**
   - Menyediakan `deduplicate_records()`, `resolve_cross_repo_value_duplicates()` berbasis *SHA-256*, `stratified_repository_split()` (`StratifiedGroupKFold`), serta `assert_no_leakage()` yang memastikan nol tumpang-tindih `repo_id` maupun `value_sha256` antara himpunan latih dan uji.
4. **Kebijakan *Triage* Hibrida *Fail-Safe* (`TriagePolicy`):**
   - Secara *default* beroperasi pada mode konsultatif (`suppress_false_positives=False`) sehingga seluruh temuan deterministik `SIDIK` tetap ditampilkan bersama probabilitas rahasia dan kontribusi fitur pendorongnya (`explain_single_candidate()`).
   - Saat supresi diaktifkan, `never_suppress_severities=("CRITICAL",)` melindungi temuan kritis agar tidak pernah dibungkam diam-diam oleh model ML.
5. **Integrasi Penuh *Plugin* (`sidik.plugins`) & *Thin Launcher* TUI/GUI/SCA:**
   - Terdaftar ke grup `sidik.plugins` melalui [plugin.py](https://github.com/radityobw/SIDIK-ML/blob/main/sidik_ml/plugin.py), sehingga perintah `sidik-ml interactive`, `sidik-ml gui`, dan `sidik-ml-gui` membuka antarmuka TUI serta *Desktop GUI* `SIDIK` dengan kontrol *ML Triage* aktif.
   - `sidik-ml scan` mendukung paritas CI/CD (`--fail-on`, `--severity`, `--mode`, `--baseline`, `--sarif-output`), `--include-deps` (OSV SCA), `fetch-model`, serta 8 format laporan (`console`, `compact`, `sarif`, `github`, `html`, `markdown`, `json`, `csv`).
6. **Serialisasi Luring Terverifikasi *SHA-256* (`skops`):**
   - Artefak `.skops` selalu diverifikasi *checksum SHA-256*-nya (`verify_artifact_sha256` atau otomatis dari `<model>.manifest.json`) dan diperiksa terhadap daftar putih tipe aman (`SAFE_SKOPS_TYPE_PREFIXES`) tanpa `pickle`/`joblib` mentah.

---

## 🏗️ Arsitektur Integrasi Hibrida

```text
Kode Sumber & Manifes Target
           │
           ▼
   SIDIK v1.3.4 Core (SAST 12 Aturan Regex + Entropi + Konteks + Offline OSV SCA)
           │
           ├──[Kontrak SecretCandidate v1.0.0 (masked_value + value_sha256)]──┐
           │                                                                   ▼
           │                                                    SIDIK-ML v0.5.4 Plugin
           │                                             (30 Fitur Statis -> skops + SHA-256
           │                                              -> Probabilitas + Explainability)
           │                                                                   │
           ◄──[sisipan metadata ml_triage & TriagePolicy Fail-Safe CRITICAL]───┘
           │
           ▼
  CLI Interaktif (TUI) │ Desktop GUI │ 7 Format Laporan (HTML, SARIF v2.1.0, MD, JSON, CSV)
```

---

## 📊 Ringkasan Hasil Empiris Terverifikasi

Seluruh angka di bawah tercatat pada eksperimen reproduksibel *SIDIK-ML* dengan `seed = 42`, *5-fold StratifiedGroupKFold*, `1.000` *bootstrap resamples*, dan uji *McNemar* pada dataset terbuka *Samsung/CredData* (`258` latih di `86 repo`, `64` *held-out test* di `23 unseen repo`), `25` repositori Python eksternal, serta `1.094` sampel *zero-shot CredDetX*:

### 1. *Baseline* `SIDIK v1.1.0` vs Lima Model *Non-ANN* (*CredData Held-Out*, `N = 64`)

| Model / Kondisi | *5-Fold CV F1* | *Precision* Uji | *Recall* Uji | *F1-Score* Uji | MCC Uji | FPR Uji | *McNemar p* vs SIDIK |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kondisi A: `SIDIK v1.1.0`** | - | `0,8103` | `0,8868` | `0,8468` | `-0,1465` | `1,0000` | - |
| *Logistic Regression* | `0,8550 +- 0,0444` | **`1,0000`** | `0,8302` | `0,9072` | `0,6757` | **`0,0000`** | `0,15159` |
| *Decision Tree* | `0,8146 +- 0,1020` | `0,9000` | `0,8491` | `0,8738` | `0,3600` | `0,4545` | `0,45450` |
| *Random Forest* | `0,8949 +- 0,0236` | `0,9298` | **`1,0000`** | **`0,9636`** | **`0,7692`** | `0,3636` | **`0,000244`** |
| *Extra Trees* | `0,8957 +- 0,0408` | `0,9400` | `0,8868` | `0,9126` | `0,5604` | `0,2727` | `0,11532` |
| **`Gradient Boosting` (Pilihan CV)** | **`0,9028 +- 0,0249`** | `0,9038` | `0,8868` | `0,8952` | `+0,4178` | `0,4545` | `0,21011` |

* **Integrasi Hibrida & Pengaman *Fail-Safe* `CRITICAL`:** Pada Kondisi C1 (*Post-SIDIK Triage*), *Gradient Boosting* menekan FP sebesar `36,36%` dan memulihkan MCC ke `+0,1597` sembari melindungi `3` temuan `CRITICAL`; pada Kondisi C2 (*Candidate-Wide*), *Recall* mencapai `0,9057`, *F1* `0,8889`, dan MCC `+0,2923`.
* **Generalisasi Eksternal (`E1` & `E2`):** Pada 25 *repository* Python publik (`1.193` berkas `.py`), model menandai **`7 / 20` (`35,0%`) temuan deterministik sebagai *false positive*** dengan tambahan latensi hanya **`+3,6 ms` (`+0,15%`)** dan melindungi `4` temuan `CRITICAL` (`REVIEW_PRIORITY`). Pada uji *zero-shot CredDetX* (`1.094` kandidat) yang mengalami *domain shift*, konfigurasi hibrida `C1` menjaga stabilitas **`F1 = 0,7969` dan `MCC = +0,2075`**, mengungguli `SIDIK v1.1.0` murni (`MCC = +0,1999`) maupun model tunggal (`MCC = -0,2052`).
* ***Overhead* Inferensi Luring:** Inferensi *batch* `Gradient Boosting` hanya `1,00 ms` (`0,0156 ms/kandidat`), puncak memori `12,76 KB`, dan ukuran artefak `2.533,63 KB` (`.skops`).

---

## 📁 Struktur Paket

```text
SIDIK-ML/
├── .github/workflows/sidik-ml-ci.yml  # CI otomatis pytest, plugin discovery, & CLI
├── action.yml                         # Composite GitHub Action (SARIF & PR annotations)
├── sidik_ml/
│   ├── __init__.py                    # Ekspor publik paket SIDIK-ML (v0.5.4)
│   ├── __main__.py                    # Titik masuk modul (python -m sidik_ml)
│   ├── cli.py                         # CLI scan, train, ekstrak fitur, TUI & GUI launchers
│   ├── plugin.py                      # Adapter entry point sidik.plugins (SidikMLPlugin)
│   ├── schemas.py                     # MLPrediction, HybridFinding, TriagePolicy
│   ├── features.py                    # 30 fitur statis bebas target leakage
│   ├── models.py                      # 5 model Non-ANN & neural exclusion guard
│   ├── dataset.py                     # JSONL I/O, deduplikasi SHA-256, StratifiedGroupKFold
│   ├── evaluation.py                  # F1, MCC, bootstrap CI, McNemar test, ablasi fitur
│   ├── explainability.py              # Bobot model, permutation importance, penjelasan kandidat
│   ├── serialization.py               # Save/load .skops aman terverifikasi SHA-256
│   ├── training.py                    # Pipeline pelatihan reproduksibel & provenans JSON
│   └── integration.py                 # HybridTriagePipeline (SIDIK + opsional ML classifier)
├── tests/                             # 19 unit & integration tests (pytest)
├── pyproject.toml                     # Konfigurasi paket, dependencies, & entry points
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## 🚀 Panduan Instalasi & Pemakaian Cepat

### 1. Instalasi Global via PyPI (Pengguna Umum & CI/CD)

`SIDIK-ML` secara otomatis memasang dependensi inti [`SIDIK`](https://github.com/radityobw/SIDIK) (`sidik>=1.3.4`), sehingga pengembang dapat langsung memasang satu perintah dari PyPI:

```bash
# Pemindai CLI & plugin Non-ANN ML
pip install sidik-ml

# Lengkap dengan antarmuka Desktop GUI (Pillow + Matplotlib)
pip install "sidik-ml[gui]"
```

Untuk pengembangan lokal dari *source repository*:
```bash
..\SIDIK\.venv\Scripts\Activate.ps1       # Windows PowerShell
# source ../SIDIK/.venv/bin/activate      # Linux / macOS

pip install -e "..\SIDIK[gui]"
pip install -e .
```

> **Catatan Windows PowerShell:** Apabila Anda membuka langsung folder `SIDIK\.venv\Scripts` tanpa menjalankan `Activate.ps1`, PowerShell tidak memuat perintah dari folder aktif secara otomatis. Gunakan awalan `.\` (misalnya `.\sidik-ml -i`, `.\sidik-ml --gui`, `.\sidik -i`, atau `.\sidik --gui`), atau aktifkan *virtual environment* terlebih dahulu dengan `.\Activate.ps1`. Jika skrip aktivasi terblokir kebijakan eksekusi PowerShell, jalankan lewat `python -m sidik_ml ...` atau gunakan peluncur `run_sidik_ml.bat` / `run_sidik_ml_cli.bat`.

### 2. Perintah Harian (Selaras `sidik` dan `sidik-ml`)

Melalui *entry point* `sidik.plugins`, perintah `sidik-ml` dan `sidik` kini mendukung pola argumen yang simetris untuk TUI, *Desktop GUI*, pemindaian *one-shot*, modul `python -m`, serta 7 format laporan:

```bash
# 1. Periksa versi atau luncurkan TUI / Desktop GUI (simetris dengan sidik -i / sidik --gui)
sidik-ml --version
sidik-ml -i                  # atau: sidik-ml interactive / python -m sidik_ml -i / run_sidik_ml_cli.bat
sidik-ml --gui               # atau: sidik-ml gui / sidik-ml-gui / python -m sidik_ml --gui / run_sidik_ml.bat

# 2. Pindai langsung (konsol Amber-Orange + exit code CI/CD --fail-on)
sidik-ml . --fail-on high
sidik-ml --target path/to/project --include-deps --model model.skops --format html -o hasil_ml.html
sidik-ml scan path/to/project --include-deps --model model.skops --format html -o hasil_ml.html

# 3. Pindai dengan output ringkas / JSON (tanpa supresi default)
sidik-ml scan path/to/project --format compact --fail-on none
sidik-ml scan path/to/project --json --fail-on none

# 4. Atau gunakan langsung dari perintah standar SIDIK
sidik --list-plugins
sidik --target path/to/project --enable-ml --ml-model model.skops

# 5. Unduh model .skops terverifikasi SHA-256 ke cache pengguna
sidik-ml fetch-model https://example.com/gradient_boosting.skops --sha256 <hex_digest>

# 6. Ekspor fitur bebas leakage dari kandidat untuk anotasi / eksperimen
sidik-ml extract-features path/to/project --output candidates.jsonl --repo-id my_repo

# 7. Verifikasi integritas checksum SHA-256 artefak model .skops
sidik-ml verify-model model.skops --sha256 <hex_digest>

# 8. Latih & bandingkan 5 model Non-ANN menggunakan StratifiedGroupKFold
sidik-ml train dataset.jsonl --output-dir artifacts/ --stratified

# 9. Jalankan seluruh 19 unit & integration tests
pytest tests -v

# 10. Atau jalankan lewat GitHub Container Registry (ghcr.io) tanpa setup Python
docker run --rm -v "${PWD}:/workspace" ghcr.io/radityobw/sidik-ml:latest /workspace --include-deps
```

---

## 🗺️ Ekosistem Perangkat Lunak Publik (*Public Software Ecosystem*)

| Repositori | Peran Utama | Versi Terkini | Tautan |
| :--- | :--- | :---: | :--- |
| **SIDIK** | Mesin pemindai deterministik SAST & OSV SCA + host *plugin* UI | `v1.3.4` | [`github.com/radityobw/SIDIK`](https://github.com/radityobw/SIDIK) |
| **SIDIK-ML** | Perangkat lunak ekstensi *Explainable Non-ANN ML* & *plugin* | `v0.5.4` | [`github.com/radityobw/SIDIK-ML`](https://github.com/radityobw/SIDIK-ML) |

---

# 🇬🇧 English

## 📌 Project Overview

**SIDIK-ML** (`v0.5.4`) is an independent Python package extending the deterministic [`SIDIK`](https://github.com/radityobw/SIDIK) (`>=v1.3.4`) security analyzer with **explainable classical Non-Artificial Neural Network (Non-ANN) machine learning** for false-positive reduction in hardcoded secret detection.

Instead of forking SIDIK's UI or introducing heavy deep learning weights, `SIDIK-ML` integrates through the leak-safe `SecretCandidate` (`schema_version = "1.0.0"`) contract and the standard `sidik.plugins` entry-point group. When launched via `sidik-ml gui`, `sidik-ml interactive`, or `--enable-ml`, the shared UI switches cleanly into an **Amber Orange (`#f59e0b` / `#ea580c`)** identity with `NON-ANN ML` badges, amber-yellow `sidik-ml>` console prompts, and warm orange-themed HTML/Markdown triage dashboards, clearly distinguishing hybrid ML analysis from Core SIDIK's **Ocean Blue (`#3b82f6`)** deterministic edition.

## ✨ Key Capabilities

* **Strict Non-ANN Model Suite:** Evaluates *Logistic Regression*, *Decision Tree*, *Random Forest*, *Extra Trees*, and *Gradient Boosting*, backed by `validate_non_ann_model_name()` to reject neural network architectures.
* **30 Leakage-Free Static Features (`FEATURE_SCHEMA_VERSION = "1.0.0"`):** Extracts string statistics, Shannon entropy metrics, rule indicators, lexical cues, code structure, and file context without persisting raw credentials.
* **Zero Data Leakage Splitting & Verification:** Enforces cross-repo `value_sha256` collision resolution, `StratifiedGroupKFold` grouped by `repo_id`, and explicit `assert_no_leakage()` checks.
* **Fail-Safe Hybrid `TriagePolicy`:** Runs in advisory mode by default (`suppress_false_positives=False`) and shields `CRITICAL` findings (`never_suppress_severities=("CRITICAL",)`).
* **Hardened Offline `skops` Loading:** Validates `.skops` model digests against explicit SHA-256 checksums or companion `.manifest.json` files and restricts deserialization to `SAFE_SKOPS_TYPE_PREFIXES`.

## 📊 Verified Empirical Highlights (`CredData` Held-Out `N = 64`, 25 Repos, & `CredDetX`)

| Model / Configuration | *5-Fold CV F1* | Test Precision | Test Recall | Test F1 | Test MCC | Test FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition A: `SIDIK v1.1.0`** | - | `0.8103` | `0.8868` | `0.8468` | `-0.1465` | `1.0000` |
| *Logistic Regression* | `0.8550` | **`1.0000`** | `0.8302` | `0.9072` | `0.6757` | **`0.0000`** |
| *Random Forest* | `0.8949` | `0.9298` | **`1.0000`** | **`0.9636`** | **`0.7692`** | `0.3636` |
| **`Gradient Boosting` (CV Winner)** | **`0.9028`** | `0.9038` | `0.8868` | `0.8952` | `+0.4178` | `0.4545` |

On 25 external Python repositories (`1,193` `.py` files), the CV-selected model flags **`35.0%` (`7/20`) of deterministic alerts as false positives** with only **`+3.6 ms` (`+0.15%`) runtime overhead**. Under zero-shot domain shift on `CredDetX` (`1,094` candidates), fail-safe Hybrid Condition `C1` preserves **`F1 = 0.7969` and `MCC = +0.2075`**, outperforming both standalone deterministic SIDIK and unguarded ML classifiers.

## 🚀 Quick Start

```bash
# Global install from PyPI (Core SIDIK is installed automatically)
pip install sidik-ml
pip install "sidik-ml[gui]"

# Check version and launch Amber-Orange SIDIK-ML Interactive CLI (TUI) or Desktop GUI
sidik-ml --version
sidik-ml -i --lang en          # or: sidik-ml interactive / python -m sidik_ml -i / run_sidik_ml_cli.bat
sidik-ml --gui                 # or: sidik-ml gui / sidik-ml-gui / python -m sidik_ml --gui / run_sidik_ml.bat

# Run unified Secret + SCA scan (defaults to console output and --fail-on any)
sidik-ml . --fail-on high
sidik-ml --target path/to/project --include-deps --model model.skops --format html -o ml_report.html --lang en
sidik-ml fetch-model https://example.com/gradient_boosting.skops --sha256 <hex_digest>

# Run tests
pytest tests -v

# Run via GitHub Container Registry (ghcr.io) or install .whl from GitHub Releases
docker run --rm -v "${PWD}:/workspace" ghcr.io/radityobw/sidik-ml:latest /workspace --include-deps
```

> **Windows PowerShell Note:** When running executables directly inside `SIDIK\.venv\Scripts` without activating `.venv` first, prefix commands with `.\` (e.g., `.\sidik-ml -i`, `.\sidik-ml --gui`) or invoke via `python -m sidik_ml ...` (or `run_sidik_ml.bat` / `run_sidik_ml_cli.bat`).

## 📄 License

Distributed under the **MIT License**. See [LICENSE](https://github.com/radityobw/SIDIK-ML/blob/main/LICENSE).
