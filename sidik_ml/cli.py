"""Command-line interface, interactive console launcher, and GUI launcher for SIDIK-ML."""

import argparse
import json
import os
import sys
from typing import List, Optional

from sidik import SecurityScanner, __version__ as SIDIK_VERSION
from sidik.reporting.formatter import ReportFormatter
from sidik_ml import __version__ as ML_VERSION
from sidik_ml.dataset import load_jsonl_dataset
from sidik_ml.features import extract_feature_dict
from sidik_ml.integration import HybridTriagePipeline
from sidik_ml.plugin import SidikMLPlugin, resolve_model_and_sha256
from sidik_ml.schemas import TriagePolicy
from sidik_ml.serialization import load_verified_skops_model, verify_artifact_sha256
from sidik_ml.training import train_and_compare_models


KNOWN_SUBCOMMANDS = {
    "scan",
    "interactive",
    "cli",
    "tui",
    "gui",
    "extract-features",
    "verify-model",
    "train",
}


def _normalize_argv(raw_argv: List[str]) -> List[str]:
    """Normalizes symmetric SIDIK-style shorthands (`-i`, `--gui`, `.`, `-t`) into subcommands."""
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

    if (not head.startswith("-")) or ("-t" in raw_argv or "--target" in raw_argv):
        return ["scan", *raw_argv]

    return list(raw_argv)


def _add_shared_ml_args(subparser: argparse.ArgumentParser) -> None:
    """Registers symmetric `--model/--ml-model`, `--sha256/--ml-sha256`, `--threshold`, and `--suppress-fp` flags."""
    subparser.add_argument(
        "--model", "--ml-model",
        dest="model",
        default=None,
        help="Optional path to verified .skops model artifact",
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
        description="SIDIK-ML: Explainable Non-ANN ML Extension for SIDIK Static Security Analysis",
        epilog=(
            "Examples (symmetric with SIDIK):\n"
            "  sidik-ml -i                                        # Launch Electric Violet Interactive Console (TUI)\n"
            "  sidik-ml --gui                                     # Launch SIDIK-ML Desktop GUI (alias: sidik-ml gui)\n"
            "  sidik-ml .                                         # Scan current directory with ML triage\n"
            "  sidik-ml --target . --include-deps --format html   # Full Secret + SCA HTML dashboard\n"
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
        "--include-deps",
        action="store_true",
        help="Also inspect Python dependency manifests against offline OSV advisories",
    )
    p_scan.add_argument(
        "--format", "-f",
        choices=["compact", "console", "json", "markdown", "csv", "html", "sarif", "github"],
        default="compact",
        help="Report format (compact preserves single-line ML triage; others use SIDIK ReportFormatter)",
    )
    p_scan.add_argument("--output", "-o", default=None, help="Optional output file path for report")
    p_scan.add_argument("--lang", choices=["id", "en"], default="id", help="Report language (default: id)")
    p_scan.add_argument("--json", action="store_true", help="Emit JSON output (equivalent to --format json in compact mode)")

    p_tui = sub.add_parser("interactive", aliases=["cli", "tui"], help="Launch SIDIK Interactive CLI Console with SIDIK-ML plugin active")
    p_tui.add_argument("--lang", choices=["id", "en"], default="id", help="Initial console language (default: id)")
    _add_shared_ml_args(p_tui)

    p_gui = sub.add_parser("gui", help="Launch SIDIK Desktop GUI with SIDIK-ML plugin pre-enabled")
    _add_shared_ml_args(p_gui)

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

    return parser


def _render_report_text(report, fmt: str, lang: str) -> str:
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
    return ReportFormatter.to_console_summary(report, lang=lang)


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = list(sys.argv[1:]) if argv is None else list(argv)
    if not raw_argv and argv is None and sys.stdin.isatty() and sys.stdout.isatty():
        raw_argv = ["interactive"]

    parser = build_parser()
    args = parser.parse_args(_normalize_argv(raw_argv))

    if args.command in ("interactive", "cli", "tui") or (args.command is None and args.interactive):
        from sidik.cli import run_interactive_cli
        return run_interactive_cli(
            initial_lang=getattr(args, "lang", "id"),
            enable_ml_plugin=True,
            ml_model_path=getattr(args, "model", None),
            ml_sha256=getattr(args, "sha256", None),
            ml_threshold=getattr(args, "threshold", 0.5),
            ml_suppress_fp=getattr(args, "suppress_fp", False),
            banner_title="SIDIK-ML",
        )

    if args.command == "gui" or (args.command is None and args.gui):
        from sidik.gui import main as sidik_gui_main
        sidik_gui_main(
            enable_ml=True,
            ml_model_path=getattr(args, "model", None),
            ml_sha256=getattr(args, "sha256", None),
            ml_threshold=getattr(args, "threshold", 0.5),
            ml_suppress_fp=getattr(args, "suppress_fp", False),
            app_brand="SIDIK-ML",
        )
        return 0

    if args.command is None:
        parser.print_help()
        return 2

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

    resolved_sha: Optional[str] = args.sha256
    if args.model:
        try:
            _, resolved_sha = resolve_model_and_sha256(args.model, args.sha256)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    use_full_formatter = (args.format != "compact") or bool(args.include_deps) or bool(args.output)
    if use_full_formatter:
        fmt = "json" if (args.format == "compact" and args.json) else ("console" if args.format == "compact" else args.format)
        plugin = SidikMLPlugin()
        scanner = SecurityScanner(
            enable_secret_scan=True,
            enable_dependency_scan=bool(args.include_deps),
            measure_memory=True,
            post_scan_plugins=[plugin],
            plugin_options={
                "model_path": args.model,
                "sha256": resolved_sha,
                "threshold": args.threshold,
                "suppress_false_positives": args.suppress_fp,
            },
        )
        scan_report = scanner.scan_path(scan_target)
        out_text = _render_report_text(scan_report, fmt=fmt, lang=args.lang)
        if args.output:
            out_dir = os.path.dirname(os.path.abspath(args.output))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out_text)
            print(f"Report written to {args.output}")
        else:
            print(out_text)
        return 0

    classifier = None
    model_name = "deterministic_pass_through"
    if args.model and resolved_sha:
        classifier = load_verified_skops_model(args.model, resolved_sha)
        model_name = args.model

    pipeline = HybridTriagePipeline(
        classifier=classifier,
        model_name=model_name,
        policy=TriagePolicy(
            decision_threshold=args.threshold,
            suppress_false_positives=args.suppress_fp,
        ),
    )
    findings = pipeline.scan_path(scan_target, include_filtered=True)
    if args.json:
        print(json.dumps([f.to_dict() for f in findings], indent=2))
    else:
        for f in findings:
            print(f"[{f.severity}] {f.final_action} p={f.ml_prediction.secret_probability:.2f} {f.file_path}:{f.line_number} {f.masked_value}")
    return 0


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
