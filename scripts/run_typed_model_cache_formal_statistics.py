"""Collect frozen formal row artifacts and run the preregistered statistics consumer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_execution import (
    FormalExecutionError,
    validate_protocol_v1_1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(Path(args.protocol_path).read_text(encoding="utf-8-sig"))
    validate_protocol_v1_1(protocol)
    input_root = Path(args.input_root)
    rows = sorted(input_root.glob("formal_controller/**/benchmark_rows.csv"))
    if not rows:
        raise FormalExecutionError("formal controller rows are missing")
    command = [
        sys.executable,
        str(ROOT / "scripts/analyze_top_journal_statistics.py"),
    ]
    for path in rows:
        command.extend(["--rows_path", str(path.resolve())])
    command.extend(
        [
            "--output_root",
            str(Path(args.output_root)),
            "--metrics",
            *protocol["endpoints"]["primary"],
            "--outer_cluster_keys",
            "source_segment_run_id",
            "window_id",
            "--inner_cluster_keys",
            "seed",
            "workflow_id",
            "--bootstrap_samples",
            "10000",
            "--random_seed",
            "1401",
        ]
    )
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    print(
        json.dumps(
            {
                "status": "pass",
                "row_artifact_count": len(rows),
                "output_root": str(Path(args.output_root).resolve()),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
