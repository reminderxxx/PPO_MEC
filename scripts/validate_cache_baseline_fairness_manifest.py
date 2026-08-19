"""Validate a G07 fairness manifest and emit a structured report/diff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.cache_baseline_fairness import validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate cache baseline fairness manifest")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--report_path", default="")
    parser.add_argument("--pairwise_diff_path", default="")
    parser.add_argument("--skip_file_checks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest_path).read_text(encoding="utf-8-sig"))
    report = validate_manifest(manifest, root=ROOT, check_files=not args.skip_file_checks)
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.pairwise_diff_path:
        diff_path = Path(args.pairwise_diff_path)
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(json.dumps(report["pairwise_protocol_diff"], ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
