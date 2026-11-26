"""Command-line interface, interactive console launcher, and GUI launcher for SIDIK-ML."""

import argparse
import json
import os
import sys
from typing import List, Optional

from sidik import SecurityScanner, __version__ as SIDIK_VERSION
from sidik.cli import (
    _safe_launch_gui,
    apply_baseline_filter,
    apply_severity_filter,
    evaluate_fail_on_threshold,
    write_baseline_file,
    write_github_step_summary,
)
from sidik.core import load_pyproject_sidik_config
from sidik.reporting.formatter import ReportFormatter
from sidik_ml import __version__ as ML_VERSION
from sidik_ml.dataset import load_jsonl_dataset
from sidik_ml.features import extract_feature_dict
from sidik_ml.integration import HybridTriagePipeline
from sidik_ml.plugin import SidikMLPlugin, discover_default_model, resolve_model_and_sha256
from sidik_ml.schemas import TriagePolicy
from sidik_ml.serialization import (
    fetch_verified_model,
    list_bundled_models,
    load_verified_model,
    verify_artifact_sha256,
)
from sidik_ml.training import train_and_compare_models


KNOWN_SUBCOMMANDS = {
    "scan",
    "interactive",
    "cli",
    "tui",
    "gui",
    "models",
    "update-db",
    "db-info",
    "extract-features",
    "verify-model",
    "train",
    "fetch-model",
}


def _normalize_argv(raw_argv: List[str]) -> List[str]:
    """Normalizes symmetric SIDIK-style shorthands (`-i`, `--gui`, `.`, `-t`, `--update-db`, `--db-info`) into subcommands."""
    if not raw_argv:
        return []
    head = raw_argv[0]
    if head in KNOWN_SUBCOMMANDS or head in ("-v", "--version"):
        return list(raw_argv)

    interactive_flags = {"-i", "--interactive", "--cli"}
    if any(tok in interactive_flags for tok in raw_argv):
        rest = [tok for tok in raw_argv if tok not in interactive_flags]
        return ["interactive", *rest]

    if "--gui" in raw_argv:
        rest = [tok for tok in raw_argv if tok != "--gui"]
        return ["gui", *rest]

    if head in ("-h", "--help"):
        return list(raw_argv)

    if "--update-db" in raw_argv:
        rest = [tok for tok in raw_argv if tok != "--update-db"]
        return ["update-db", *rest]

    if "--db-info" in raw_argv:
        rest = [tok for tok in raw_argv if tok != "--db-info"]
        return ["db-info", *rest]

    if (not head.startswith("-")) or ("-t" in raw_argv or "--target" in raw_argv):
        return ["scan", *raw_argv]

    return list(raw_argv)


def _add_shared_ml_args(subparser: argparse.ArgumentParser) -> None:
    """Registers symmetric `--model/--ml-model`, `--sha256/--ml-sha256`, `--threshold`, and `--suppress-fp` flags."""
    subparser.add_argument(
        "--model", "--ml-model",
        dest="model",
        default=None,
        help="Bundled preset (gradient_boosting, random_forest, extra_trees, logistic_regression, decision_tree) or .skops/.json path",
    )
    subparser.add_argument(
        "--sha256", "--ml-sha256",
        dest="sha256",
        default=None,
        help="Optional expected SHA-256 checksum (auto-reads <model>.manifest.json if omitted)",
    )
    subparser.add_argument(
        "--threshold", "--ml-threshold",
        dest="threshold",
        type=float,
        default=0.5,
        help="Probability threshold for ML triage (default: 0.5)",
    )
    subparser.add_argument(
        "--suppress-fp", "--ml-suppress-fp",
        dest="suppress_fp",
        action="store_true",
        help="Suppress non-CRITICAL findings classified as false positives",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sidik-ml",
        description="SIDIK-ML: Explainable Non-ANN ML Extension for SIDIK Polyglot SAST & Multi-Ecosystem SCA",
        epilog=(
            "Examples (symmetric with SIDIK):\n"
            "  sidik-ml models                                    # List all 5 bundled Non-ANN models\n"
            "  sidik-ml -i                                        # Launch Amber-Orange Interactive Console (TUI)\n"
            "  sidik-ml --gui                                     # Launch SIDIK-ML Desktop GUI (alias: sidik-ml gui)\n"
            "  sidik-ml . --model random_forest                   # Polyglot scan using bundled Random Forest preset\n"
            "  sidik-ml --target . --include-deps --format html   # Full Polyglot Secret + 8-Ecosystem SCA HTML dashboard\n"
            "  sidik-ml update-db --ecosystem npm,go,crates       # Sync offline OSV advisories (8 ecosystems or 'all')\n"
            "  sidik-ml db-info                                   # Show PyPI & multi-ecosystem SQLite advisory stats\n"
            "  sidik-ml train data.jsonl -o artifacts/ --stratified"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--interactive", "-i", "--cli",
        action="store_true",
        help="Launch interactive CLI console window (TUI) with SIDIK-ML active",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch SIDIK-ML Desktop GUI",
    )
    parser.add_argument(
        "--update-db",
        action="store_true",
        help="Download offline Google OSV advisories (PyPI default, or --ecosystem npm,go,crates,all)",
    )
    parser.add_argument(
        "--db-info",
        action="store_true",
        help="Show offline OSV advisory database statistics (PyPI + multi-ecosystem SQLite cache)",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"SIDIK-ML v{ML_VERSION} (Explainable Non-ANN ML Extension | Core SIDIK v{SIDIK_VERSION})",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    p_scan = sub.add_parser("scan", help="Scan a path via SIDIK and apply verified ML triage & optional SCA")
    p_scan.add_argument("target", nargs="?", default=None, help="File or directory path to scan")
    p_scan.add_argument("--target", "-t", dest="target_opt", default=None, help="File or directory path to scan")
    _add_shared_ml_args(p_scan)
    p_scan.add_argument(
        "--mode", "-m",
        choices=["all", "secret", "secrets", "dep", "deps", "dependency", "dependencies"],
        default=None,
        help="Scan mode: all (default), secret (secret scan only), or dep/dependency (dependency scan only)",
    )
    p_scan.add_argument("--secret-only", action="store_true", help="Perform secret detection scan only")
    p_scan.add_argument("--dep-only", action="store_true", help="Perform dependency vulnerability scan only")
    p_scan.add_argument(
        "--include-deps",
        action="store_true",
        help="Include multi-ecosystem dependency manifest scanning across 8 ecosystems (enabled by default in 'all' mode)",
    )
    p_scan.add_argument(
        "--format", "-f",
        choices=["console", "compact", "json", "markdown", "csv", "html", "sarif", "github"],
        default="console",
        help="Report format (default: console Amber-Orange dashboard summary)",
    )
    p_scan.add_argument("--output", "-o", default=None, help="Optional output file path for report")
    p_scan.add_argument("--sarif-output", default=None, help="Optional secondary SARIF v2.1.0 output file path")
    p_scan.add_argument("--lang", choices=["id", "en"], default="id", help="Report language (default: id)")
    p_scan.add_argument("--json", action="store_true", help="Emit JSON output")
    p_scan.add_argument(
        "--severity", "-s",
        choices=["low", "medium", "high", "critical"],
        default=None,
        help="Filter reported findings by minimum severity level",
    )
    p_scan.add_argument(
        "--fail-on",
        choices=["none", "any", "low", "medium", "high", "critical"],
        default=None,
        help="Exit code 1 threshold: none, any, low, medium, high, or critical (default: any)",
    )
    p_scan.add_argument("--no-color", action="store_true", help="Disable ANSI color codes in terminal output")
    p_scan.add_argument("--quiet", "-q", action="store_true", help="Suppress progress and informational notices")
    p_scan.add_argument("--no-step-summary", action="store_true", help="Do not append Markdown summary to $GITHUB_STEP_SUMMARY")
    p_scan.add_argument(
        "--extensions",
        default=None,
        help="Comma-separated file extensions for polyglot secret scanning (e.g., '.py,.js,.ts,.go,.rs,.env')",
    )
    p_scan.add_argument("--exclude", action="append", default=[], help="Glob pattern to exclude (repeatable)")
    p_scan.add_argument("--baseline", default=None, help="Path to JSON baseline file of suppressed finding IDs")
    p_scan.add_argument("--write-baseline", default=None, help="Write current finding IDs to JSON baseline file")
    p_scan.add_argument("--db", default=None, help="Optional path to custom vulnerability database JSON file")

    p_tui = sub.add_parser("interactive", aliases=["cli", "tui"], help="Launch SIDIK Interactive CLI Console with SIDIK-ML plugin active")
    p_tui.add_argument("--lang", choices=["id", "en"], default="id", help="Initial console language (default: id)")
    _add_shared_ml_args(p_tui)

    p_gui = sub.add_parser("gui", help="Launch SIDIK Desktop GUI with SIDIK-ML plugin pre-enabled")
    _add_shared_ml_args(p_gui)

    p_models = sub.add_parser("models", help="List all 5 bundled SHA-256-verified pre-trained Non-ANN models")
    p_models.add_argument("--json", action="store_true", help="Emit JSON output")

    p_upd = sub.add_parser(
        "update-db",
        help="Download offline Google OSV advisories via SIDIK (PyPI or 8 ecosystems)",
    )
    p_upd.add_argument(
        "--ecosystem",
        default="PyPI",
        help="OSV ecosystem(s): PyPI (default), comma-separated (npm,go,crates,maven,packagist,rubygems,nuget), or 'all'",
    )
    p_upd.add_argument(
        "--db", "--output", "-o",
        dest="db",
        default=None,
        help="Optional custom output path for PyPI JSON advisory snapshot",
    )

    p_dbi = sub.add_parser(
        "db-info",
        help="Show offline OSV advisory database statistics (PyPI + multi-ecosystem SQLite cache)",
    )
    p_dbi.add_argument(
        "--db",
        default=None,
        help="Optional path to custom vulnerability database JSON file",
    )

    p_feat = sub.add_parser("extract-features", help="Export leakage-free feature JSONL from target candidates")
    p_feat.add_argument("target", help="File or directory path to inspect")
    p_feat.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    p_feat.add_argument("--repo-id", default="local_target", help="Repository identifier tag for exported records")

    p_ver = sub.add_parser("verify-model", help="Verify SHA-256 checksum of an offline .skops artifact")
    p_ver.add_argument("model_path", help="Path to .skops artifact")
    p_ver.add_argument("--sha256", required=True, help="Expected SHA-256 hex digest")

    p_train = sub.add_parser("train", help="Train and evaluate classical Non-ANN models on a labeled JSONL dataset")
    p_train.add_argument("dataset", help="Path to labeled JSONL dataset")
    p_train.add_argument("--output-dir", "-o", required=True, help="Directory for model artifact, manifest, and report")
    p_train.add_argument("--stratified", action="store_true", help="Use StratifiedGroupKFold split and CV selection")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_train.add_argument("--bootstrap", type=int, default=300, help="Bootstrap resamples for 95%% CI (default: 300)")

    p_fetch = sub.add_parser("fetch-model", help="Download and SHA-256 verify a remote .skops model into user cache")
    p_fetch.add_argument("url", help="HTTPS URL to the .skops model artifact")
    p_fetch.add_argument("--sha256", required=True, help="Expected SHA-256 hex digest of the artifact")
    p_fetch.add_argument("--output", "-o", default=None, help="Optional destination .skops path")

    return parser


def _render_report_text(report, fmt: str, lang: str, use_color: Optional[bool] = None) -> str:
    if fmt == "json":
        return ReportFormatter.to_json(report)
    if fmt == "markdown":
        return ReportFormatter.to_markdown(report, lang=lang)
    if fmt == "csv":
        return ReportFormatter.to_csv(report)
    if fmt == "html":
        return ReportFormatter.to_html(report, lang=lang)
    if fmt == "sarif":
        return ReportFormatter.to_sarif(report)
    if fmt == "github":
        return ReportFormatter.to_github_annotations(report, lang=lang)
    return ReportFormatter.to_console_summary(report, lang=lang, use_color=use_color)


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = list(sys.argv[1:]) if argv is None else list(argv)
    if not raw_argv and argv is None and sys.stdin.isatty() and sys.stdout.isatty():
        raw_argv = ["interactive"]

    parser = build_parser()
    args = parser.parse_args(_normalize_argv(raw_argv))

    if args.command == "models":
        catalog = list_bundled_models()
        if args.json:
            print(json.dumps(catalog, indent=2))
        else:
            print("Bundled SHA-256-Verified Non-ANN Models (Out-of-the-Box):")
            for name, info in catalog.items():
                marker = " [DEFAULT]" if info.get("is_default") else ""
                print(
                    f"  - {name:20s} | {info['size_kb']:6.2f} KB | sha256={info['sha256'][:12]}...{marker}"
                    f" | {info['profile']}"
                )
        return 0

    if args.command in ("interactive", "cli", "tui") or (args.command is None and args.interactive):
        from sidik.cli import run_interactive_cli
        m_arg = getattr(args, "model", None)
        s_arg = getattr(args, "sha256", None)
        if m_arg:
            try:
                m_arg, s_arg = resolve_model_and_sha256(m_arg, s_arg)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 2
        return run_interactive_cli(
            initial_lang=getattr(args, "lang", "id"),
            enable_ml_plugin=True,
            ml_model_path=m_arg,
            ml_sha256=s_arg,
            ml_threshold=getattr(args, "threshold", 0.5),
            ml_suppress_fp=getattr(args, "suppress_fp", False),
            banner_title="SIDIK-ML",
        )

    if args.command == "gui" or (args.command is None and args.gui):
        m_arg = getattr(args, "model", None)
        s_arg = getattr(args, "sha256", None)
        if m_arg:
            try:
                m_arg, s_arg = resolve_model_and_sha256(m_arg, s_arg)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 2
        return _safe_launch_gui(
            enable_ml=True,
            ml_model_path=m_arg,
            ml_sha256=s_arg,
            ml_threshold=getattr(args, "threshold", 0.5),
            ml_suppress_fp=getattr(args, "suppress_fp", False),
            app_brand="SIDIK-ML",
        )

    if args.command == "update-db" or (args.command is None and args.update_db):
        from sidik.cli import build_arg_parser as build_sidik_parser, run_cli_with_args

        eco = getattr(args, "ecosystem", "PyPI") or "PyPI"
        fwd = ["--update-db"]
        if "--ecosystem" in build_sidik_parser()._option_string_actions:
            fwd.extend(["--ecosystem", eco])
        elif eco.strip().lower() not in ("pypi", ""):
            print(
                f"Error: --ecosystem '{eco}' requires SIDIK >=1.4.0 (installed: v{SIDIK_VERSION}).",
                file=sys.stderr,
            )
            return 2
        db_opt = getattr(args, "db", None)
        if db_opt:
            fwd.extend(["--db", db_opt])
        return run_cli_with_args(fwd, exit_on_finish=False)

    if args.command == "db-info" or (args.command is None and args.db_info):
        from sidik.cli import run_cli_with_args

        fwd = ["--db-info"]
        db_opt = getattr(args, "db", None)
        if db_opt:
            fwd.extend(["--db", db_opt])
        return run_cli_with_args(fwd, exit_on_finish=False)

    if args.command is None:
        parser.print_help()
        return 2

    if args.command == "fetch-model":
        try:
            dest, digest = fetch_verified_model(args.url, args.sha256, output_path=args.output)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        print(f"Verified model saved to: {dest}")
        print(f"SHA-256: {digest}")
        return 0

    if args.command == "train":
        records = load_jsonl_dataset(args.dataset)
        report = train_and_compare_models(
            records,
            random_state=args.seed,
            output_dir=args.output_dir,
            dataset_path=args.dataset,
            use_stratified_split=args.stratified,
            n_bootstrap=args.bootstrap,
        )
        print(json.dumps({"best_model": report["best_model"], "saved_artifact": report["saved_artifact"]}, indent=2))
        return 0

    if args.command == "verify-model":
        ok = verify_artifact_sha256(args.model_path, args.sha256)
        print("VERIFIED" if ok else "CHECKSUM_MISMATCH")
        return 0 if ok else 1

    if args.command == "extract-features":
        scanner = SecurityScanner(enable_dependency_scan=False, measure_memory=False)
        cands = scanner.extract_secret_candidates(args.target, include_filtered=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for c in cands:
                rec = {
                    "candidate_id": c.candidate_id,
                    "repo_id": args.repo_id,
                    "file_path": c.file_path,
                    "value_sha256": c.value_sha256,
                    "masked_context": c.code_context.get("masked_line", c.masked_value),
                    "deterministic_passed": bool(c.deterministic_decision.get("passed_deterministic_filter")),
                    "severity": c.severity.value if hasattr(c.severity, "value") else str(c.severity),
                    "features": extract_feature_dict(c),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Exported {len(cands)} candidate feature rows to {args.output}")
        return 0

    scan_target = args.target_opt or args.target
    if not scan_target:
        parser.error("Target path is required (e.g., 'sidik-ml .' or 'sidik-ml scan <path>' or 'sidik-ml -i')")

    if not os.path.exists(scan_target):
        print(f"Error: Target path '{scan_target}' does not exist.", file=sys.stderr)
        return 2

    model_path_arg = args.model
    resolved_sha: Optional[str] = args.sha256
    if model_path_arg:
        try:
            model_path_arg, resolved_sha = resolve_model_and_sha256(model_path_arg, args.sha256)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    else:
        discovered = discover_default_model()
        if discovered is not None:
            model_path_arg, resolved_sha = discovered

    if not model_path_arg and not args.quiet:
        print(
            "[SIDIK-ML Notice] Running in deterministic pass-through mode (no verified ML model loaded). "
            "Supply --model <path.skops|.json> or run 'sidik-ml fetch-model' / 'sidik-ml train' to enable Non-ANN ML scoring.",
            file=sys.stderr,
        )

    cfg_base = scan_target if os.path.isdir(scan_target) else os.path.dirname(os.path.abspath(scan_target))
    proj_cfg = load_pyproject_sidik_config(cfg_base)

    ext_spec = args.extensions or (
        ",".join(proj_cfg["extensions"]) if isinstance(proj_cfg.get("extensions"), list) else proj_cfg.get("extensions")
    )
    parsed_extensions = None
    if isinstance(ext_spec, str) and ext_spec.strip():
        parsed_extensions = tuple(
            e.strip() if e.strip().startswith(".") else f".{e.strip()}"
            for e in ext_spec.split(",")
            if e.strip()
        )

    eff_severity = args.severity or (str(proj_cfg["severity"]) if proj_cfg.get("severity") else None)
    eff_baseline = args.baseline or (str(proj_cfg["baseline"]) if proj_cfg.get("baseline") else None)
    fail_on = (args.fail_on or str(proj_cfg.get("fail_on", "any"))).lower()

    if args.format == "compact" and not args.include_deps and args.mode is None and not args.output and not args.sarif_output and not args.write_baseline:
        classifier = None
        model_name = "deterministic_pass_through"
        if model_path_arg and resolved_sha:
            classifier = load_verified_model(model_path_arg, resolved_sha)
            model_name = os.path.basename(model_path_arg)

        compact_scanner = SecurityScanner(
            enable_dependency_scan=False,
            custom_vulnerability_db=args.db,
            ignore_patterns=list(args.exclude) if args.exclude else None,
            secret_extensions=parsed_extensions,
            measure_memory=False,
        )
        pipeline = HybridTriagePipeline(
            scanner=compact_scanner,
            classifier=classifier,
            model_name=model_name,
            policy=TriagePolicy(
                decision_threshold=args.threshold,
                suppress_false_positives=args.suppress_fp,
            ),
        )
        findings = pipeline.scan_path(scan_target, include_filtered=True)
        if eff_severity:
            sev_order = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
            min_r = sev_order.get(eff_severity.lower(), 1)
            findings = [f for f in findings if sev_order.get(f.severity.lower(), 0) >= min_r]

        if args.json:
            print(json.dumps([f.to_dict() for f in findings], indent=2))
        elif findings:
            for f in findings:
                print(f"[{f.severity}] {f.final_action} p={f.ml_prediction.secret_probability:.2f} {f.file_path}:{f.line_number} {f.masked_value}")
        else:
            print("[OK] SIDIK-ML: Clean! 0 findings detected.")

        active_alerts = [f for f in findings if not f.suppressed_by_ml]
        if fail_on == "none":
            return 0
        if fail_on == "any":
            return 1 if active_alerts else 0
        sev_order = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        th_r = sev_order.get(fail_on, 1)
        return 1 if any(sev_order.get(f.severity.lower(), 0) >= th_r for f in active_alerts) else 0

    mode_val = (args.mode or str(proj_cfg.get("mode", "all"))).lower()
    is_secret_only = args.secret_only or mode_val in ("secret", "secrets")
    is_dep_only = args.dep_only or mode_val in ("dep", "deps", "dependency", "dependencies")
    fmt = "json" if args.json else ("console" if args.format == "compact" else args.format)

    plugin = SidikMLPlugin()
    scanner = SecurityScanner(
        enable_secret_scan=not is_dep_only,
        enable_dependency_scan=(not is_secret_only) or bool(args.include_deps),
        custom_vulnerability_db=args.db,
        ignore_patterns=list(args.exclude) if args.exclude else None,
        secret_extensions=parsed_extensions,
        measure_memory=True,
        post_scan_plugins=[plugin],
        plugin_options={
            "model_path": model_path_arg,
            "sha256": resolved_sha,
            "threshold": args.threshold,
            "suppress_false_positives": args.suppress_fp,
        },
    )
    scan_report = scanner.scan_path(scan_target)
    apply_severity_filter(scan_report, eff_severity)
    apply_baseline_filter(scan_report, eff_baseline)

    if args.write_baseline:
        write_baseline_file(scan_report, args.write_baseline)
        if not args.quiet:
            print(f"Baseline written ({len(scan_report.findings)} findings): {args.write_baseline}", file=sys.stderr)

    if args.sarif_output:
        sarif_dir = os.path.dirname(os.path.abspath(args.sarif_output))
        if sarif_dir:
            os.makedirs(sarif_dir, exist_ok=True)
        with open(args.sarif_output, "w", encoding="utf-8") as sf:
            sf.write(ReportFormatter.to_sarif(scan_report))

    out_text = _render_report_text(
        scan_report,
        fmt=fmt,
        lang=args.lang,
        use_color=(False if args.no_color else None),
    )
    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_text)
        if not args.quiet:
            print(f"Report written to {args.output}")
    else:
        print(out_text)

    write_github_step_summary(scan_report, lang=args.lang, disable=args.no_step_summary)
    return evaluate_fail_on_threshold(scan_report.findings, fail_on)


def launch_gui(argv: Optional[List[str]] = None) -> int:
    """Dedicated entry point for ``sidik-ml-gui`` wrapping SIDIK's desktop GUI."""
    parser = argparse.ArgumentParser(prog="sidik-ml-gui", description="Launch SIDIK-ML Desktop GUI")
    _add_shared_ml_args(parser)
    args = parser.parse_args(argv)
    return main([
        "gui",
        *(["--model", args.model] if args.model else []),
        *(["--sha256", args.sha256] if args.sha256 else []),
        "--threshold", str(args.threshold),
        *(["--suppress-fp"] if args.suppress_fp else []),
    ])


if __name__ == "__main__":
    sys.exit(main())
