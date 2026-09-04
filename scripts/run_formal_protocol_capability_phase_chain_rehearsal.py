"""Run the G14R13 non-formal 13-phase capability-routing rehearsal."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_typed_model_cache_formal_protocol import resolved_expansion_context
from src.evaluators.typed_model_cache_formal_execution import (
    AppendOnlyPhaseRunner,
    CommandResult,
    PHASE_ORDER,
    validate_command_templates,
    validate_phase_ledger,
    validate_protocol_v1_1,
)
from src.runtime.active_formal_bundle import validate_active_formal_bundle
from src.runtime.formal_protocol_capabilities import (
    protocol_capability_matrix,
    require_live_execution_protocol,
)


PHASE_CONSUMERS = {
    "preflight": ["outer protocol runner", "nested restart validator"],
    "tests": ["wrapper and capability regression suite"],
    "train": ["training binding and provenance"],
    "dev_select": ["dev selection"],
    "checkpoint_freeze": ["checkpoint freeze"],
    "formal_cache_policy": ["cache-policy benchmark"],
    "formal_controller": ["controller benchmark"],
    "formal_ablation": ["ablation benchmark"],
    "formal_support": ["support benchmark"],
    "formal_scalability": ["scalability benchmark"],
    "formal_statistics": ["statistics"],
    "formal_gate": ["integrity and paper gate"],
    "complete_without_holdout": ["sealed holdout terminal"],
}


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant in {path}: {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--allow-pending-index", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    python_executable = Path(args.python_executable)
    if not python_executable.is_absolute() or not python_executable.is_file():
        raise ValueError("an existing absolute --python-executable is required")
    if Path(sys.executable).resolve() != python_executable.resolve():
        raise ValueError("rehearsal Python differs from the explicit Python executable")
    observed_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    if observed_commit != args.candidate_commit:
        raise ValueError("candidate commit mismatch")

    bundle = validate_active_formal_bundle(
        repository_root=ROOT,
        require_ready=not args.allow_pending_index,
    )
    protocol = bundle["protocol"]
    validate_protocol_v1_1(protocol)
    capabilities = require_live_execution_protocol(
        protocol["typed_model_cache_formal_protocol_version"]
    )
    expansion = resolved_expansion_context(
        protocol,
        protocol_path=bundle["protocol_path"],
        output_root=str(output_root),
        python_executable=str(python_executable.resolve()),
        active_formal_bundle_sha256=bundle["active_formal_bundle_sha256"],
        active_protocol_index_path=str(
            ROOT
            / "configs/experiment/typed_model_cache_formal_protocol_v2_4_20260905/protocol_index.json"
        ),
        active_bundle_resource_resolution_audit_sha256="0" * 64,
    )
    matrix = validate_command_templates(
        protocol["execution_contract"]["command_templates"], expansion
    )
    commands = [
        command
        for phase in matrix["expanded"].values()
        for command in phase["commands"]
    ]
    if matrix["phase_count"] != 15 or matrix["command_count"] != 186:
        raise ValueError("frozen 15-phase/186-command matrix drift")
    if len(matrix["expanded"]["train"]["commands"]) != 150:
        raise ValueError("frozen training command count drift")
    if any(command[0] != str(python_executable.resolve()) for command in commands):
        raise ValueError("command matrix Python identity drift")
    if any("{" in token or "}" in token or "/ABSOLUTE/" in token for command in commands for token in command):
        raise ValueError("command matrix contains unresolved location state")

    runner = AppendOnlyPhaseRunner(protocol=protocol, output_root=output_root)
    for phase in PHASE_ORDER:
        marker = f"nonformal_phase_markers/{phase}.json"

        def execute(
            _command: list[str], *, selected_phase: str = phase, marker_path: str = marker
        ) -> CommandResult:
            write_json(
                output_root / marker_path,
                {
                    "phase": selected_phase,
                    "status": "pass",
                    "consumer_routes": PHASE_CONSUMERS[selected_phase],
                    "formal": False,
                    "performance_evidence": False,
                    "holdout_opened": False,
                    "protocol_version": capabilities.version,
                    "active_bundle_core_sha256": bundle["active_bundle_core_sha256"],
                },
            )
            return CommandResult(returncode=0)

        runner.run_phase(
            phase,
            command=[] if phase == "complete_without_holdout" else ["g14r13", phase],
            input_hash=canonical_sha256(
                {
                    "phase": phase,
                    "protocol": protocol["hashes"]["semantic_sha256"],
                    "active_bundle": bundle["active_formal_bundle_sha256"],
                    "capabilities": protocol_capability_matrix(),
                }
            ),
            expected_outputs=[] if phase == "complete_without_holdout" else [marker],
            executor=execute,
        )

    events = runner.events()
    ledger = validate_phase_ledger(events)
    summary = {
        "status": "pass",
        "scope": "non-formal capability-routing transaction-chain rehearsal",
        "acceptance_bootstrap_pending_index": args.allow_pending_index,
        "formal": False,
        "performance_evidence": False,
        "candidate_commit": observed_commit,
        "protocol_version": capabilities.version,
        "active_bundle_core_sha256": bundle["active_bundle_core_sha256"],
        "active_formal_bundle_sha256": bundle["active_formal_bundle_sha256"],
        "phase_order": list(PHASE_ORDER),
        "phase_count": len(PHASE_ORDER),
        "completed_phase_order": [row["phase"] for row in events if row["status"] == "completed"],
        "phase_ledger_validation": ledger,
        "complete_without_holdout": events[-1]["phase"] == "complete_without_holdout",
        "frozen_internal_phase_count": matrix["phase_count"],
        "frozen_command_count": matrix["command_count"],
        "frozen_training_command_count": len(matrix["expanded"]["train"]["commands"]),
        "command_matrix_sha256": matrix["command_matrix_sha256"],
        "single_absolute_python": str(python_executable.resolve()),
        "unresolved_placeholder_count": 0,
        "absolute_sentinel_count": 0,
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "holdout_capability": False,
        "holdout_sealed_unopened_unconsumed": True,
        "consumer_routes": PHASE_CONSUMERS,
    }
    write_json(output_root / "phase_chain_rehearsal.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
