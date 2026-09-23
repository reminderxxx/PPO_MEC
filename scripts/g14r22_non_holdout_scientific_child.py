#!/usr/bin/env python3
"""Small non-holdout scientific producer used only by G14R22 acceptance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


AGENTS = [
    "reactive_lru", "reactive_fifo", "reactive_lfu", "reactive_aging_lfu", "reactive_random",
    "sa_ghmappo", "ppo", "mappo", "dqn", "dueling_dqn", "qmix", "controller_mat",
    "dag_offload_drl", "cache_offload_drl", "dt_handoff_drl",
]
METRICS = [
    "full_service_ready_byte_hit_rate", "joint_base_adapter_hit_rate",
    "full_service_ready_request_rate", "transfer_mb_per_request",
    "workflow_continuity_rate", "end_to_end_workflow_delay",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.output_root / f"main_results_non_holdout_{args.capacity}"
    artifact.mkdir(parents=True)
    rows_path = artifact / "benchmark_rows.csv"
    fields = ["agent_name", "seed", "window_id", "source_segment_run_id", "workflow_id", "capacity_label", *METRICS]
    with rows_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for window in range(2):
            for agent_index, agent in enumerate(AGENTS):
                base = 0.50 + agent_index / 1000 + window / 10000
                writer.writerow({
                    "agent_name": agent, "seed": 7, "window_id": f"public_non_holdout_{window}",
                    "source_segment_run_id": "public_non_holdout", "workflow_id": "wf_public",
                    "capacity_label": args.capacity,
                    "full_service_ready_byte_hit_rate": base,
                    "joint_base_adapter_hit_rate": base + 0.01,
                    "full_service_ready_request_rate": base + 0.02,
                    "transfer_mb_per_request": 5.0 - base,
                    "workflow_continuity_rate": base + 0.03,
                    "end_to_end_workflow_delay": 10.0 - base,
                })
    inventory = [{"path": rows_path.name, "size_bytes": rows_path.stat().st_size, "sha256": digest(rows_path)}]
    manifest = artifact / "artifact_integrity_manifest.json"
    manifest.write_text(json.dumps({"version": "g14r22_non_holdout_v1", "files": inventory}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
