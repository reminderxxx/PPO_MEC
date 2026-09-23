#!/usr/bin/env python3
"""Run the real public G14R22 entry with a strictly non-holdout tiny fixture."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.dedicated_holdout_execution import (
    CAPACITIES, HOLDOUT_COMMAND_PACKAGE_VERSION, PRIMARY_METRICS, canonical_sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    args.work_root.mkdir(parents=True, exist_ok=False)
    output_root = args.work_root / "public_entry_output"
    python = sys.executable
    scientific = [
        [python, str(ROOT / "scripts/g14r22_non_holdout_scientific_child.py"),
         "--capacity", capacity, "--output-root", "{G14R22_CELL_OUTPUT_ROOT}"]
        for capacity in CAPACITIES
    ]
    statistics = [
        python, str(ROOT / "scripts/analyze_top_journal_statistics.py"),
        "--rows_path", "{G14R22_ROWS_0}", "--rows_path", "{G14R22_ROWS_1}",
        "--rows_path", "{G14R22_ROWS_2}", "--candidate_agent", "sa_ghmappo",
        "--baseline_agents", "reactive_lru", "reactive_fifo", "reactive_lfu", "reactive_aging_lfu",
        "reactive_random", "ppo", "mappo", "dqn", "dueling_dqn", "qmix", "controller_mat",
        "dag_offload_drl", "cache_offload_drl", "dt_handoff_drl",
        "--metrics", *PRIMARY_METRICS,
        "--pair_keys", "seed", "window_id", "workflow_id", "capacity_label",
        "--outer_cluster_keys", "source_segment_run_id", "window_id",
        "--inner_cluster_keys", "seed", "workflow_id", "capacity_label",
        "--ci_method", "bca", "--bootstrap_samples", "20", "--random_seed", "1401",
        "--formal-agent-order-contract-path",
        str(ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/formal_agent_order_contract.json"),
        "--output_root", "{G14R22_STATISTICS_OUTPUT_ROOT}",
    ]
    package = {
        "command_package_version": HOLDOUT_COMMAND_PACKAGE_VERSION,
        "executor_checkout": str(ROOT), "executor_commit": "non_holdout_acceptance",
        "output_root": str(output_root),
        "phases": ["scientific", "statistics", "publication", "integrity"],
        "scientific_matrix": {"profile": "tiny public non-holdout"},
        "commands": {"scientific": scientific, "statistics": statistics},
        "automatic_retry_count": 0, "acceptance_non_holdout": True,
        "acceptance_request_sha256": canonical_sha256({"profile": "non_holdout"}),
        "acceptance_checkpoint_audit": {
            "status": "pass", "actual_scope": {"models": 0, "total_bytes": 0},
            "models_canonical_sha256": canonical_sha256([]), "models": [],
        },
    }
    package["command_package_sha256"] = canonical_sha256(package)
    package_path = args.work_root / "acceptance_command_package.json"
    package_path.write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n")
    command = [python, str(ROOT / "scripts/run_dedicated_public_holdout.py"),
               "--command-package-path", str(package_path), "--check", "acceptance"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    receipt = {
        "acceptance_version": "g14r22_non_holdout_public_entry_v1",
        "command": command, "return_code": completed.returncode,
        "stdout": completed.stdout, "stderr": completed.stderr,
        "output_root": str(output_root),
        "scientific_children": 3, "statistics_rows": 84,
        "holdout_opened": False, "holdout_policy_runs": 0,
        "passed": completed.returncode == 0,
    }
    (args.work_root / "acceptance_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
