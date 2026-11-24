"""Command-line interface for SIDIK-ML hybrid scanning, feature export, and artifact verification."""

import argparse
import json
import sys
from typing import List, Optional

from sidik import SecurityScanner
from sidik_ml.dataset import load_jsonl_dataset
from sidik_ml.features import extract_feature_dict
from sidik_ml.integration import HybridTriagePipeline
from sidik_ml.schemas import TriagePolicy
from sidik_ml.serialization import load_verified_skops_model, verify_artifact_sha256
from sidik_ml.training import train_and_compare_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sidik-ml",
        description="SIDIK-ML: Explainable Non-ANN ML Extension for SIDIK Secret Detection",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Scan a path via SIDIK and apply optional verified ML triage")
    p_scan.add_argument("target", help="File or directory path to scan")
    p_scan.add_argument("--model", default=None, help="Path to verified .skops model artifact")
    p_scan.add_argument("--sha256", default=None, help="Expected SHA-256 checksum for the .skops model")
    p_scan.add_argument("--threshold", type=float, default=0.5, help="Probability threshold (default: 0.5)")
    p_scan.add_argument(
        "--suppress-fp",
        action="store_true",
        help="Hide non-CRITICAL findings classified as false positives (default keeps all)",
    )
    p_scan.add_argument("--json", action="store_true", help="Emit JSON output")

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


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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

    classifier = None
    model_name = "deterministic_pass_through"
    if args.model:
        if not args.sha256:
            print("Error: --sha256 is required whenever --model is specified.", file=sys.stderr)
            return 2
        classifier = load_verified_skops_model(args.model, args.sha256)
        model_name = args.model

    pipeline = HybridTriagePipeline(
        classifier=classifier,
        model_name=model_name,
        policy=TriagePolicy(
            decision_threshold=args.threshold,
            suppress_false_positives=args.suppress_fp,
        ),
    )
    findings = pipeline.scan_path(args.target, include_filtered=True)
    if args.json:
        print(json.dumps([f.to_dict() for f in findings], indent=2))
    else:
        for f in findings:
            print(f"[{f.severity}] {f.final_action} p={f.ml_prediction.secret_probability:.2f} {f.file_path}:{f.line_number} {f.masked_value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
