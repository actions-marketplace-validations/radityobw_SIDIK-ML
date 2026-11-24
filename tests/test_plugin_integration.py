"""End-to-end integration tests for SIDIK-ML plugin discovery, unified SAST+SCA, and GUI hooks."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from sidik import SecurityScanner
from sidik.plugins import PostScanPlugin, get_plugin
from sidik_ml.cli import build_parser, main as cli_main
from sidik_ml.dataset import records_to_matrix
from sidik_ml.models import create_non_ann_model
from sidik_ml.plugin import SidikMLPlugin, resolve_model_and_sha256
from sidik_ml.serialization import save_verified_skops_model
from tests.test_models_and_training import _build_contract_fixture_records


class TestSidikMLPluginIntegration(unittest.TestCase):
    """Validates sidik.plugins integration across CLI, scanner, SCA, reports, and GUI."""

    def test_entry_point_discovery_finds_sidik_ml(self):
        plug = get_plugin("sidik-ml")
        self.assertIsNotNone(plug, "sidik-ml entry point must be registered under sidik.plugins")
        self.assertIsInstance(plug, PostScanPlugin)
        self.assertEqual(plug.name, "sidik-ml")

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
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.isfile(out_html))
            html_content = open(out_html, "r", encoding="utf-8").read()
            self.assertIn("ML Probability:", html_content)

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
            import tkinter as tk
            from sidik.gui import SecurityAnalyzerGUI
            root = tk.Tk()
        except Exception as exc:
            self.skipTest(f"Tkinter display unavailable: {exc}")
        root.withdraw()
        try:
            app = SecurityAnalyzerGUI(root, enable_ml=True, app_brand="SIDIK-ML")
            self.assertIsNotNone(app.ml_plugin)
            self.assertTrue(hasattr(app, "chk_ml"))
            self.assertTrue(app.var_ml_enabled.get())
            self.assertIn("SIDIK-ML", root.title())
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
