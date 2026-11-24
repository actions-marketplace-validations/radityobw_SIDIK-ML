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


if __name__ == "__main__":
    unittest.main()
