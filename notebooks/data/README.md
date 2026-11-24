# Leak-Free Static Feature Corpus for `quickstart.ipynb`

This directory contains `creddata_features.jsonl` (`621` raw rows extracted from the public **Samsung/CredData** corpus, Apache-2.0 license) for running reproducible offline Non-ANN machine learning experiments in [`quickstart.ipynb`](https://github.com/radityobw/SIDIK-ML/blob/main/notebooks/quickstart.ipynb).

## Privacy & Zero-Leakage Design

* **Zero Raw Secrets Stored:** Every record contains only masked code context (`****`), a one-way `value_sha256` cryptographic digest, `repo_id`, ground-truth label (`1` = true secret, `0` = false positive), and `30` scan-time numeric features (`FEATURE_SCHEMA_VERSION = "1.0.0"`).
* **Deduplication & Unseen-Repo Splitting:** Running `deduplicate_records()` and `resolve_cross_repo_value_duplicates()` reduces the `621` raw extractions to **`322` clean unique candidates** (`264` true secrets, `58` false positives) across **`109` distinct repositories**, then `stratified_repository_split()` splits them into **`258` training samples (`86` repos)** and **`64` held-out test samples (`23` unseen repos)** with zero repository or SHA-256 overlap.
* **Lightweight Package Guarantee:** This dataset is scoped strictly to `notebooks/data/` for documentation and interactive Colab/Jupyter execution and is excluded from the PyPI distribution wheel (`sidik-ml`).
