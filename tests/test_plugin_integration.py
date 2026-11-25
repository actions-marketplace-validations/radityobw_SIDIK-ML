"""End-to-end integration tests for SIDIK-ML plugin discovery, unified SAST+SCA, and GUI hooks."""

import importlib.metadata
import json
import os
import re
import tempfile
import unittest
from unittest.mock import patch

from sidik import SecurityScanner
from sidik.plugins import PostScanPlugin, get_plugin
from sidik_ml import __version__ as ML_VERSION
from sidik_ml.cli import build_parser, main as cli_main
from sidik_ml.dataset import LabeledCandidateRecord, records_to_matrix
from sidik_ml.features import FEATURE_NAMES
from sidik_ml.models import create_non_ann_model
from sidik_ml.plugin import SidikMLPlugin, resolve_model_and_sha256
from sidik_ml.serialization import save_verified_skops_model


def _build_contract_fixture_records() -> list[LabeledCandidateRecord]:
    """Builds self-contained synthetic contract-test records for plugin integration checks."""
    records = []
    for repo_idx in range(10):
        repo = f"contract_repo_{repo_idx}"
        for is_pos in (1, 0):
            feats = {name: 0.0 for name in FEATURE_NAMES}
            feats["str_length"] = 32.0 if is_pos else 16.0
            feats["shannon_entropy"] = 4.3 + (0.05 * (repo_idx % 3)) if is_pos else 2.2 + (0.05 * (repo_idx % 3))
            feats["meets_entropy_threshold"] = 1.0 if is_pos else 0.0
            feats["has_positive_keyword"] = 1.0 if is_pos else 0.0
            feats["is_test_path"] = 0.0 if is_pos else 1.0
            feats["is_placeholder_or_mock"] = 0.0 if is_pos else 1.0
            feats["passed_deterministic_filter"] = 1.0
            records.append(LabeledCandidateRecord(
                candidate_id=f"cid_{repo_idx}_{is_pos}",
                repo_id=repo,
                file_path="src/cfg.py" if is_pos else "tests/test_cfg.py",
                value_sha256=f"sha_{repo_idx}_{is_pos}",
                masked_context=f"ctx_{repo_idx}_{is_pos}",
                label=is_pos,
                features=feats,
                deterministic_passed=True,
                severity="HIGH" if is_pos else "MEDIUM",
            ))
    return records


class TestSidikMLPluginIntegration(unittest.TestCase):
    """Validates sidik.plugins integration across CLI, scanner, SCA, reports, and GUI."""

    def test_entry_point_discovery_finds_sidik_ml(self):
        self.assertRegex(ML_VERSION, r"^\d+\.\d+\.\d+$")
        self.assertEqual(importlib.metadata.version("sidik-ml"), ML_VERSION)
        self.assertEqual(importlib.metadata.metadata("sidik-ml")["Author"], "Radityo Budi Waskito")
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo_root, "CHANGELOG.md"), "r", encoding="utf-8") as cf:
            self.assertIn(f"## [{ML_VERSION}]", cf.read())

        plug = get_plugin("sidik-ml")
        self.assertIsNotNone(plug, "sidik-ml entry point must be registered under sidik.plugins")
        self.assertIsInstance(plug, PostScanPlugin)
        self.assertEqual(plug.name, "sidik-ml")
        self.assertEqual(plug.version, ML_VERSION)

    def test_unified_secret_plus_sca_with_critical_protection_and_tamper_guard(self):
        recs = _build_contract_fixture_records()
        X, y, _ = records_to_matrix(recs)
        clf = create_non_ann_model("random_forest", random_state=42)
        clf.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp:
            m_path = os.path.join(tmp, "rf_test.skops")
            manifest = save_verified_skops_model(clf, m_path, metadata={"model": "random_forest"})
            resolved_path, resolved_sha = resolve_model_and_sha256(m_path)
            self.assertEqual(resolved_path, m_path)
            self.assertEqual(resolved_sha, manifest["sha256"])

            proj_dir = os.path.join(tmp, "proj")
            os.makedirs(proj_dir)
            with open(os.path.join(proj_dir, "tests_cfg.py"), "w", encoding="utf-8") as f:
                f.write("aws_prod_key = 'AKIAJ2M4K8P9Q3R7W5T1'\n")
                f.write("jwt_sample = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c'\n")
            with open(os.path.join(proj_dir, "requirements.txt"), "w", encoding="utf-8") as f:
                f.write("requests==2.19.0\n")

            plugin = SidikMLPlugin()
            scanner = SecurityScanner(
                enable_secret_scan=True,
                enable_dependency_scan=True,
                post_scan_plugins=[plugin],
                plugin_options={
                    "model_path": m_path,
                    "threshold": 0.99,
                    "suppress_false_positives": True,
                },
            )
            report = scanner.scan_path(proj_dir)

            dep_findings = [f for f in report.findings if f.finding_type == "dependency"]
            sec_findings = [f for f in report.findings if f.finding_type == "secret"]
            self.assertGreaterEqual(len(dep_findings), 1)
            self.assertNotIn("ml_triage", dep_findings[0].metadata)

            # CRITICAL finding must remain preserved and annotated with ml_triage
            self.assertEqual(len(sec_findings), 1)
            self.assertEqual(sec_findings[0].severity.value, "CRITICAL")
            self.assertIn("ml_triage", sec_findings[0].metadata)
            self.assertIn("top_feature_contributions", sec_findings[0].metadata["ml_triage"])

            # Check tampered digest rejection
            with self.assertRaises(ValueError):
                plugin.triage_report(
                    scanner,
                    proj_dir,
                    report,
                    options={"model_path": m_path, "sha256": "0" * 64},
                )

            # Check missing manifest without sha256 fails closed
            os.remove(f"{m_path}.manifest.json")
            with self.assertRaises(ValueError):
                resolve_model_and_sha256(m_path)

    def test_cli_scan_html_with_sca_and_launcher_subcommands(self):
        parser = build_parser()
        args_tui = parser.parse_args(["interactive", "--lang", "en"])
        self.assertEqual(args_tui.command, "interactive")
        args_gui = parser.parse_args(["gui", "--threshold", "0.7"])
        self.assertEqual(args_gui.command, "gui")

        recs = _build_contract_fixture_records()
        X, y, _ = records_to_matrix(recs)
        clf = create_non_ann_model("logistic_regression", random_state=42)
        clf.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp:
            m_path = os.path.join(tmp, "lr_test.skops")
            save_verified_skops_model(clf, m_path, metadata={"model": "logistic_regression"})

            proj_dir = os.path.join(tmp, "app")
            os.makedirs(proj_dir)
            with open(os.path.join(proj_dir, "main.py"), "w", encoding="utf-8") as f:
                f.write("aws_key = 'AKIAJ2M4K8P9Q3R7W5T1'\n")
            with open(os.path.join(proj_dir, "requirements.txt"), "w", encoding="utf-8") as f:
                f.write("requests==2.19.0\n")

            out_html = os.path.join(tmp, "out", "report.html")
            rc = cli_main([
                "scan",
                proj_dir,
                "--include-deps",
                "--model",
                m_path,
                "--format",
                "html",
                "--output",
                out_html,
            ])
            self.assertEqual(rc, 1)
            rc_allow = cli_main([
                "scan",
                proj_dir,
                "--include-deps",
                "--model",
                m_path,
                "--fail-on",
                "none",
                "--format",
                "html",
                "--output",
                out_html,
            ])
            self.assertEqual(rc_allow, 0)
            self.assertTrue(os.path.isfile(out_html))
            with open(out_html, "r", encoding="utf-8") as fh:
                html_content = fh.read()
            self.assertIn("ML Probability:", html_content)
            self.assertIn("#ea580c", html_content)
            self.assertNotIn("Research Edition", html_content)
            self.assertIn("NON-ANN ML TRIAGE", html_content)

            # Interactive launcher forwards to sidik.cli.run_interactive_cli
            with patch("sidik.cli.run_interactive_cli", return_value=0) as mock_tui:
                rc_tui = cli_main(["interactive", "--lang", "en", "--model", m_path])
                self.assertEqual(rc_tui, 0)
                mock_tui.assert_called_once()
                _, kwargs = mock_tui.call_args
                self.assertTrue(kwargs["enable_ml_plugin"])
                self.assertEqual(kwargs["banner_title"], "SIDIK-ML")

            # GUI launcher forwards to sidik.gui.main
            with patch("sidik.gui.main") as mock_gui:
                rc_gui = cli_main(["gui", "--model", m_path, "--suppress-fp"])
                self.assertEqual(rc_gui, 0)
                mock_gui.assert_called_once()
                _, kwargs = mock_gui.call_args
                self.assertTrue(kwargs["enable_ml"])
                self.assertEqual(kwargs["app_brand"], "SIDIK-ML")

    def test_headless_gui_mounts_ml_plugin_controls(self):
        try:
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                with self.assertRaises(SystemExit) as ctx:
                    cli_main(["--version"])
            self.assertEqual(ctx.exception.code, 0)
            self.assertIn(f"SIDIK-ML v{ML_VERSION}", buf.getvalue())

            import tkinter as tk
            from sidik.gui import SecurityAnalyzerGUI
            root = tk.Tk()
        except Exception as exc:
            self.skipTest(f"Tkinter display unavailable: {exc}")
        root.withdraw()
        try:
            app_plain = SecurityAnalyzerGUI(root, enable_ml=False, app_brand="SIDIK")
            self.assertFalse(app_plain.is_ml_edition)
            self.assertEqual(app_plain._accent["primary"], "#3b82f6")
            self.assertIn("Deterministic Edition", app_plain.version_badge.cget("text"))
            self.assertNotIn("Research Edition", app_plain.version_badge.cget("text"))
        finally:
            root.destroy()

        ml_root = tk.Tk()
        ml_root.withdraw()
        try:
            app_ml = SecurityAnalyzerGUI(ml_root, enable_ml=True, app_brand="SIDIK-ML")
            self.assertIsNotNone(app_ml.ml_plugin)
            self.assertTrue(app_ml.is_ml_edition)
            self.assertEqual(app_ml._accent["primary"], "#f59e0b")
            self.assertNotIn("Research Edition", app_ml.version_badge.cget("text"))
            self.assertTrue(hasattr(app_ml, "ml_edition_pill"))
            self.assertTrue(hasattr(app_ml, "chk_ml"))
            self.assertTrue(app_ml.var_ml_enabled.get())
            self.assertIn(f"SIDIK-ML v{ML_VERSION}", ml_root.title())
        finally:
            ml_root.destroy()

    def test_cli_symmetric_shorthands_and_module_entrypoint(self):
        import subprocess
        import sys

        with patch("sidik.cli.run_interactive_cli", return_value=0) as mock_tui:
            for argv in (["-i", "--lang", "en"], ["--interactive"], ["--cli"]):
                mock_tui.reset_mock()
                rc = cli_main(argv)
                self.assertEqual(rc, 0)
                mock_tui.assert_called_once()
                _, kwargs = mock_tui.call_args
                self.assertTrue(kwargs["enable_ml_plugin"])
                self.assertEqual(kwargs["banner_title"], "SIDIK-ML")

        with patch("sidik.gui.main") as mock_gui:
            rc_gui1 = cli_main(["--gui", "--suppress-fp"])
            self.assertEqual(rc_gui1, 0)
            mock_gui.assert_called_once()
            _, kwargs1 = mock_gui.call_args
            self.assertTrue(kwargs1["enable_ml"])
            self.assertTrue(kwargs1["ml_suppress_fp"])
            self.assertEqual(kwargs1["app_brand"], "SIDIK-ML")

            mock_gui.reset_mock()
            rc_gui2 = cli_main(["--gui", "--ml-threshold", "0.7"])
            self.assertEqual(rc_gui2, 0)
            mock_gui.assert_called_once()
            _, kwargs2 = mock_gui.call_args
            self.assertTrue(kwargs2["enable_ml"])
            self.assertAlmostEqual(kwargs2["ml_threshold"], 0.7)
            self.assertEqual(kwargs2["app_brand"], "SIDIK-ML")

        with tempfile.TemporaryDirectory() as tmp:
            proj_dir = os.path.join(tmp, "proj")
            os.makedirs(proj_dir)
            with open(os.path.join(proj_dir, "app.py"), "w", encoding="utf-8") as f:
                f.write("aws_key = 'AKIAJ2M4K8P9Q3R7W5T1'\n")
            with open(os.path.join(proj_dir, "requirements.txt"), "w", encoding="utf-8") as f:
                f.write("requests==2.19.0\n")

            out_json = os.path.join(tmp, "out1.json")
            rc1 = cli_main([proj_dir, "--format", "json", "-o", out_json])
            self.assertEqual(rc1, 1)
            with open(out_json, "r", encoding="utf-8") as fh:
                doc1 = json.load(fh)
            self.assertEqual(doc1["summary"]["total_secrets_found"], 1)

            out_json2 = os.path.join(tmp, "out2.json")
            rc2 = cli_main(["-t", proj_dir, "--include-deps", "--fail-on", "none", "--format", "json", "-o", out_json2])
            self.assertEqual(rc2, 0)
            with open(out_json2, "r", encoding="utf-8") as fh:
                doc2 = json.load(fh)
            self.assertGreaterEqual(doc2["summary"]["total_vulnerabilities_found"], 1)

        proc = subprocess.run(
            [sys.executable, "-m", "sidik_ml", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn(f"SIDIK-ML v{ML_VERSION}", proc.stdout)

    def test_v054_cli_parity_single_pass_skew_and_fetch_model(self):
        import hashlib
        import io
        import warnings
        from contextlib import redirect_stderr, redirect_stdout

        # 1. Missing target path returns exit code 2 with stderr error
        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            rc_missing = cli_main(["nonexistent_target_dir_v054"])
        self.assertEqual(rc_missing, 2)
        self.assertIn("does not exist", err_buf.getvalue())

        # 2. Clean repo returns 0 with non-empty output and pass-through notice on stderr
        with tempfile.TemporaryDirectory() as tmp:
            clean_dir = os.path.join(tmp, "clean_proj")
            os.makedirs(clean_dir)
            with open(os.path.join(clean_dir, "app.py"), "w", encoding="utf-8") as f:
                f.write("print('ok')\n")
            out_buf = io.StringIO()
            err_buf2 = io.StringIO()
            with patch("sidik_ml.cli.discover_default_model", return_value=None):
                with redirect_stdout(out_buf), redirect_stderr(err_buf2):
                    rc_clean = cli_main([clean_dir, "--format", "compact"])
            self.assertEqual(rc_clean, 0)
            self.assertIn("Clean!", out_buf.getvalue())
            self.assertIn("pass-through mode", err_buf2.getvalue())

            # 3. Single-pass plugin reuses _preextracted_candidates (does not re-scan files)
            dirty_dir = os.path.join(tmp, "dirty_proj")
            os.makedirs(dirty_dir)
            with open(os.path.join(dirty_dir, "app.py"), "w", encoding="utf-8") as f:
                f.write("aws_key = 'AKIAJ2M4K8P9Q3R7W5T1'\n")
            plugin = SidikMLPlugin()
            scanner = SecurityScanner(enable_dependency_scan=False, post_scan_plugins=[plugin])
            with patch.object(scanner, "extract_secret_candidates", side_effect=AssertionError("Must reuse pre-extracted candidates")):
                rep = scanner.scan_path(dirty_dir)
            self.assertEqual(len(rep.findings), 1)
            self.assertIn("ml_triage", rep.findings[0].metadata)

            # 4. sklearn_version recorded in manifest and mismatch triggers RuntimeWarning
            recs = _build_contract_fixture_records()
            X, y, _ = records_to_matrix(recs)
            clf = create_non_ann_model("logistic_regression", random_state=42)
            clf.fit(X, y)
            m_path = os.path.join(tmp, "skew_test.skops")
            manifest = save_verified_skops_model(clf, m_path)
            self.assertIn("sklearn_version", manifest["metadata"])
            manifest["metadata"]["sklearn_version"] = "0.0.1"
            with open(f"{m_path}.manifest.json", "w", encoding="utf-8") as mf:
                json.dump(manifest, mf)
            from sidik_ml.serialization import fetch_verified_model, load_verified_skops_model
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_verified_skops_model(m_path, manifest["sha256"])
            self.assertTrue(any("scikit-learn 0.0.1" in str(w.message) for w in caught))

            # 5. fetch_verified_model rejects non-HTTPS & SHA mismatch, accepts matching HTTPS payload
            with self.assertRaises(ValueError):
                fetch_verified_model("http://example.com/m.skops", "a" * 64)
            fake_bytes = open(m_path, "rb").read()
            fake_sha = hashlib.sha256(fake_bytes).hexdigest()

            class _FakeResp:
                def __init__(self, data: bytes) -> None:
                    self._data = data
                def read(self, _n: int = -1) -> bytes:
                    return self._data
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    return False

            with patch("urllib.request.urlopen", return_value=_FakeResp(fake_bytes)):
                dest, got_sha = fetch_verified_model(
                    "https://example.com/releases/model.skops",
                    fake_sha,
                    output_path=os.path.join(tmp, "fetched", "model.skops"),
                )
            self.assertEqual(got_sha, fake_sha)
            self.assertTrue(os.path.isfile(dest))
            self.assertTrue(os.path.isfile(f"{dest}.manifest.json"))

    def test_bundled_portable_gbdt_out_of_the_box_and_roundtrip_equivalence(self):
        import io
        from contextlib import redirect_stderr
        from sidik_ml.explainability import explain_single_candidate
        from sidik_ml.plugin import discover_default_model
        from sidik_ml.serialization import (
            PortableGradientBoostingClassifier,
            export_portable_gbdt_model,
            get_bundled_default_model_path,
            load_verified_model,
        )

        # 1. Bundled model exists, auto-discovered, loads safely, and rejects tamper
        bundled = get_bundled_default_model_path()
        self.assertIsNotNone(bundled)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SIDIK_ML_MODEL_PATH", None)
            discovered = discover_default_model()
        self.assertIsNotNone(discovered)
        model_path, model_sha = discovered
        self.assertEqual(os.path.basename(model_path), "default_gradient_boosting.json")
        portable = load_verified_model(model_path, model_sha)
        self.assertIsInstance(portable, PortableGradientBoostingClassifier)
        self.assertEqual(portable.n_features_in_, len(FEATURE_NAMES))
        with self.assertRaises(ValueError):
            load_verified_model(model_path, "0" * 64)

        # 2. Round-trip export matches scikit-learn GradientBoostingClassifier within 1e-9
        recs = _build_contract_fixture_records()
        X, y, _ = records_to_matrix(recs)
        clf = create_non_ann_model("gradient_boosting", random_state=42)
        clf.fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            exp_path = os.path.join(tmp, "rt_gbdt.json")
            manifest = export_portable_gbdt_model(clf, exp_path)
            loaded = load_verified_model(exp_path, manifest["sha256"])
            sk_probs = clf.predict_proba(X)
            pt_probs = loaded.predict_proba(X)
            max_diff = max(abs(float(sp[1]) - float(pp[1])) for sp, pp in zip(sk_probs, pt_probs))
            self.assertLess(max_diff, 1e-9)
            self.assertEqual(
                explain_single_candidate(clf, X[0]),
                explain_single_candidate(loaded, X[0]),
            )

            # 3. CLI & SidikMLPlugin automatically use bundled GBDT out-of-the-box without --model
            proj_dir = os.path.join(tmp, "oob_app")
            os.makedirs(proj_dir)
            with open(os.path.join(proj_dir, "app.py"), "w", encoding="utf-8") as fh:
                fh.write("aws_prod_key = 'AKIAJ2M4K8P9Q3R7W5T1'\n")
            out_json = os.path.join(tmp, "oob.json")
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                rc = cli_main([proj_dir, "--mode", "secret", "--fail-on", "none", "--format", "json", "-o", out_json])
            self.assertEqual(rc, 0)
            self.assertNotIn("pass-through mode", err_buf.getvalue())
            with open(out_json, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            self.assertEqual(len(doc["findings"]), 1)
            ml_meta = doc["findings"][0]["metadata"]["ml_triage"]
            self.assertEqual(ml_meta["model_name"], "default_gradient_boosting.json")
            self.assertEqual(ml_meta["model_version"], ML_VERSION)

    def test_all_five_bundled_models_preset_resolution_roundtrip_and_cli_models(self):
        import io
        from contextlib import redirect_stdout
        from sidik_ml.explainability import explain_single_candidate
        from sidik_ml.models import SUPPORTED_MODELS
        from sidik_ml.serialization import (
            PortableGradientBoostingClassifier,
            PortableLogisticRegressionClassifier,
            PortableTreeEnsembleClassifier,
            export_portable_model,
            list_bundled_models,
            load_bundled_model,
            load_verified_model,
        )

        # 1. All 5 Non-ANN models are bundled with valid SHA-256 manifests and reject tampered checksums
        catalog = list_bundled_models()
        self.assertEqual(set(catalog.keys()), set(SUPPORTED_MODELS))
        expected_types = {
            "gradient_boosting": PortableGradientBoostingClassifier,
            "logistic_regression": PortableLogisticRegressionClassifier,
            "decision_tree": PortableTreeEnsembleClassifier,
            "random_forest": PortableTreeEnsembleClassifier,
            "extra_trees": PortableTreeEnsembleClassifier,
        }
        for name, exp_cls in expected_types.items():
            clf_b = load_bundled_model(name)
            self.assertIsInstance(clf_b, exp_cls)
            self.assertEqual(clf_b.n_features_in_, len(FEATURE_NAMES))
            res_p, bad_sha = resolve_model_and_sha256(name, "0" * 64)
            self.assertEqual(res_p, catalog[name]["path"])
            with self.assertRaises(ValueError):
                load_verified_model(res_p, bad_sha)

        # 2. Export -> load round-trip equivalence across all 5 Non-ANN architectures
        recs = _build_contract_fixture_records()
        X, y, _ = records_to_matrix(recs)
        with tempfile.TemporaryDirectory() as tmp:
            for name in SUPPORTED_MODELS:
                sk_clf = create_non_ann_model(name, random_state=42)
                sk_clf.fit(X, y)
                out_p = os.path.join(tmp, f"rt_{name}.json")
                manifest = export_portable_model(sk_clf, out_p, metadata={"model_name": name})
                pt_clf = load_verified_model(out_p, manifest["sha256"])
                sk_probs = sk_clf.predict_proba(X)
                pt_probs = pt_clf.predict_proba(X)
                max_diff = max(abs(float(a[1]) - float(b[1])) for a, b in zip(sk_probs, pt_probs))
                self.assertLess(max_diff, 1e-9, f"{name} round-trip max_abs_diff={max_diff}")
                sk_exp = explain_single_candidate(sk_clf, X[0])
                pt_exp = explain_single_candidate(pt_clf, X[0])
                self.assertEqual(list(sk_exp.keys()), list(pt_exp.keys()))

            # 3. CLI 'sidik-ml models --json' and preset names ('random_forest', 'lr', 'preset:decision_tree')
            models_buf = io.StringIO()
            with redirect_stdout(models_buf):
                rc_models = cli_main(["models", "--json"])
            self.assertEqual(rc_models, 0)
            parsed_cat = json.loads(models_buf.getvalue())
            self.assertEqual(set(parsed_cat.keys()), set(SUPPORTED_MODELS))

            proj_dir = os.path.join(tmp, "preset_app")
            os.makedirs(proj_dir)
            with open(os.path.join(proj_dir, "app.py"), "w", encoding="utf-8") as fh:
                fh.write("aws_prod_key = 'AKIAJ2M4K8P9Q3R7W5T1'\n")

            for preset_arg, expected_fname in [
                ("random_forest", "default_random_forest.json"),
                ("lr", "default_logistic_regression.json"),
                ("preset:decision_tree", "default_decision_tree.json"),
            ]:
                out_json = os.path.join(tmp, f"out_{expected_fname}.json")
                rc = cli_main([
                    proj_dir,
                    "--mode", "secret",
                    "--model", preset_arg,
                    "--fail-on", "none",
                    "--format", "json",
                    "--quiet",
                    "-o", out_json,
                ])
                self.assertEqual(rc, 0)
                with open(out_json, "r", encoding="utf-8") as fh:
                    doc = json.load(fh)
                self.assertEqual(doc["findings"][0]["metadata"]["ml_triage"]["model_name"], expected_fname)

    def test_v059_polyglot_multi_ecosystem_scan_and_db_cli_passthrough(self):
        import argparse
        import io
        from contextlib import redirect_stderr, redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            proj_dir = os.path.join(tmp, "polyglot_app")
            os.makedirs(proj_dir)
            with open(os.path.join(proj_dir, "service.ts"), "w", encoding="utf-8") as fh:
                fh.write(
                    'const awsKey = "AKIAJ2M4K8P9Q3R7W5T1";\n'
                    '// const dummy_key = "AKIAJ2M4K8P9Q3R7W5T2";\n'
                    '// const suppressedKey = "AKIAJ2M4K8P9Q3R7W5T3"; // nosec\n'
                    "const token = process.env.API_TOKEN;\n"
                )
            with open(os.path.join(proj_dir, "main.go"), "w", encoding="utf-8") as fh:
                fh.write(
                    'package main\n'
                    'import "os"\n'
                    'const ghToken = "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"\n'
                    'func getTok() string { return os.Getenv("GITHUB_TOKEN") }\n'
                )
            with open(os.path.join(proj_dir, "package.json"), "w", encoding="utf-8") as fh:
                json.dump({"name": "poly-demo", "dependencies": {"lodash": "4.17.15"}}, fh)
            with open(os.path.join(proj_dir, "go.mod"), "w", encoding="utf-8") as fh:
                fh.write("module example.com/demo\n\ngo 1.21\n\nrequire golang.org/x/net v0.15.0\n")
            with open(os.path.join(proj_dir, "pom.xml"), "w", encoding="utf-8") as fh:
                fh.write(
                    "<project><dependencies><dependency>"
                    "<groupId>org.apache.logging.log4j</groupId>"
                    "<artifactId>log4j-core</artifactId>"
                    "<version>2.14.1</version>"
                    "</dependency></dependencies></project>\n"
                )

            out_json = os.path.join(tmp, "poly_out.json")
            rc = cli_main([
                proj_dir,
                "--model", "random_forest",
                "--fail-on", "none",
                "--format", "json",
                "--quiet",
                "-o", out_json,
            ])
            self.assertEqual(rc, 0)
            with open(out_json, "r", encoding="utf-8") as fh:
                doc = json.load(fh)

            sec_findings = [f for f in doc["findings"] if f["finding_type"] == "secret"]
            dep_findings = [f for f in doc["findings"] if f["finding_type"] == "dependency"]

            self.assertEqual(len(sec_findings), 2)
            sec_exts = {os.path.splitext(f["file_path"])[1] for f in sec_findings}
            self.assertEqual(sec_exts, {".ts", ".go"})
            for sf in sec_findings:
                self.assertEqual(sf["metadata"]["ml_triage"]["model_name"], "default_random_forest.json")

            dep_ecos = {df["metadata"].get("ecosystem") for df in dep_findings}
            self.assertTrue({"npm", "Go", "Maven"}.issubset(dep_ecos))
            dep_cves = {df["metadata"].get("cve") for df in dep_findings}
            self.assertTrue({"CVE-2020-8203", "CVE-2023-44487", "CVE-2021-44228"}.issubset(dep_cves))
            for df in dep_findings:
                self.assertNotIn("ml_triage", df["metadata"])

        # db-info subcommand and --db-info flag passthrough
        for argv in (["db-info"], ["--db-info"]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc_db = cli_main(argv)
            self.assertEqual(rc_db, 0)
            out_txt = buf.getvalue()
            self.assertIn("PyPI Advisories", out_txt)
            self.assertIn("Multi-Eco Built-in", out_txt)

        # update-db subcommand and --update-db flag passthrough
        with patch("sidik.cli.run_cli_with_args", return_value=0) as mock_run:
            rc_u1 = cli_main(["update-db", "--ecosystem", "npm,go"])
            self.assertEqual(rc_u1, 0)
            mock_run.assert_called_once_with(["--update-db", "--ecosystem", "npm,go"], exit_on_finish=False)

            mock_run.reset_mock()
            rc_u2 = cli_main(["--update-db", "--ecosystem", "all"])
            self.assertEqual(rc_u2, 0)
            mock_run.assert_called_once_with(["--update-db", "--ecosystem", "all"], exit_on_finish=False)

        # Fail closed when an older SIDIK parser lacks --ecosystem
        old_parser = argparse.ArgumentParser()
        err_buf = io.StringIO()
        with patch("sidik.cli.build_arg_parser", return_value=old_parser), redirect_stderr(err_buf):
            rc_old = cli_main(["update-db", "--ecosystem", "npm"])
        self.assertEqual(rc_old, 2)
        self.assertIn("SIDIK >=1.4.0", err_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
