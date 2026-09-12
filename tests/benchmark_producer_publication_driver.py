"""Run the real benchmark producer and emit its transaction descriptor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import benchmark_main_results
from src.evaluators.formal_cell_transaction import write_child_output_descriptor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--setting-id", required=True)
    args, benchmark_args = parser.parse_known_args()
    staging = Path(args.staging).resolve()
    artifact = staging / "artifact"
    benchmark_parent = artifact / "benchmark"
    before = set(benchmark_parent.iterdir()) if benchmark_parent.exists() else set()
    sys.argv = [str(ROOT / "scripts/benchmark_main_results.py"), *benchmark_args]
    benchmark_main_results.main()
    created = sorted(set(benchmark_parent.iterdir()) - before)
    if len(created) != 1:
        raise RuntimeError(f"real benchmark producer created {len(created)} run directories")
    producer = created[0]
    write_child_output_descriptor(
        staging / "cell_child_output.json",
        cell_id=args.cell_id,
        phase=args.phase,
        logical_setting_id=args.setting_id,
        output_root=staging,
        artifact_root=artifact,
        producer_kind="benchmark_main_results_real_nonformal",
        required_payload=[
            f"benchmark/{producer.name}/aggregate_summary.json",
            f"benchmark/{producer.name}/benchmark_rows.csv",
            f"benchmark/{producer.name}/artifact_integrity_manifest.json",
        ],
    )


if __name__ == "__main__":
    main()
