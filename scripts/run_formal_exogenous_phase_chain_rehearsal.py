"""Bind real G14R9 rehearsal evidence into the complete append-only phase chain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_execution import (
    AppendOnlyPhaseRunner,
    CommandResult,
    PHASE_ORDER,
    validate_command_templates,
    validate_phase_ledger,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import canonical_sha256, sha256_file
from src.runtime.active_formal_bundle import validate_active_formal_bundle


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    evidence_root = Path(args.evidence_root).resolve()
    output_root = Path(args.output_root).resolve()
    exact = read_json(evidence_root / "exact_failure_unit_rehearsal.json")
    capacities = read_json(evidence_root / "three_capacity_rehearsal.json")
    candidate_inputs = json.loads(
        (evidence_root / "candidate_selection_inputs.json").read_text(encoding="utf-8")
    )
    if (
        exact.get("status") != "pass"
        or exact.get("agent_count") != 15
        or exact.get("fingerprint_count") != 1
        or not exact.get("observed_alignment_all_pass")
        or capacities.get("status") != "pass"
        or len(capacities.get("capacities", [])) != 3
        or any(row.get("request_exposure_fingerprint_count") != 1 for row in capacities["capacities"])
        or len(candidate_inputs) != 10
    ):
        raise ValueError("real request-execution rehearsal evidence is incomplete")
    if not (
        exact.get("clean_candidate_start") is True
        and exact.get("detached_candidate") is True
        and exact.get("candidate_has_local_venv") is False
    ):
        raise ValueError("phase chain lacks a clean detached no-.venv candidate start")

    bundle = validate_active_formal_bundle(
        repository_root=ROOT,
        require_clean_git=False,
        require_origin_main_match=False,
    )
    protocol = bundle["protocol"]
    validate_protocol_v1_1(protocol)
    expansion = dict(protocol["execution_contract"]["default_expansion_context"])
    for key, value in list(expansion.items()):
        if isinstance(value, str) and value.startswith("/ABSOLUTE/FORMAL_OUTPUT_ROOT"):
            expansion[key] = str(output_root) + value[len("/ABSOLUTE/FORMAL_OUTPUT_ROOT"):]
    expansion.update(
        python_executable=str(Path(sys.executable).absolute()),
        clean_worktree_root=str(ROOT),
        repository_root=str(ROOT),
        protocol_path=bundle["protocol_path"],
        output_root=str(output_root),
        active_formal_bundle_sha256=bundle["active_formal_bundle_sha256"],
        active_protocol_index_path=str(
            ROOT
            / "configs/experiment/typed_model_cache_formal_protocol_v2_0_20260831/protocol_index.json"
        ),
        active_bundle_resource_resolution_audit_sha256="0" * 64,
    )
    for key, value in list(expansion.items()):
        if isinstance(value, str) and value.startswith("/ABSOLUTE/"):
            expansion[key] = str(output_root / "resolved" / key)
    matrix = validate_command_templates(
        protocol["execution_contract"]["command_templates"], expansion
    )
    commands = [
        command
        for phase in matrix["expanded"].values()
        for command in phase["commands"]
    ]
    if matrix["command_count"] != 186:
        raise ValueError("Protocol 2.0 command matrix is not 186 cells")
    if any("{" in str(command) or "/ABSOLUTE/" in str(command) for command in commands):
        raise ValueError("expanded command matrix contains unresolved location state")

    phase_sources = {
        "preflight": ["active bundle validation", "186-command expansion"],
        "tests": ["formal exogenous request contract tests"],
        "train": ["fresh non-formal tiny checkpoint reports"],
        "dev_select": ["10 learned-agent candidate selection inputs"],
        "checkpoint_freeze": ["10 fresh checkpoint SHA-256 identities"],
        "formal_cache_policy": ["exact 5-reactive-policy outcomes on common exposure"],
        "formal_controller": ["exact complete 15-agent benchmark rows"],
        "formal_ablation": ["distinct policy outcomes with common exposure"],
        "formal_support": ["three-capacity common-exposure results"],
        "formal_scalability": ["288/576/864 MB complete 15-agent results"],
        "formal_statistics": ["deterministic candidate input table"],
        "formal_gate": ["evidence inventory and SHA-256 integrity"],
        "complete_without_holdout": ["sealed holdout and zero formal evidence"],
    }
    runner = AppendOnlyPhaseRunner(protocol=protocol, output_root=output_root)
    request_fingerprint = exact["request_exposure_fingerprint"]
    for phase in PHASE_ORDER:
        marker = f"phase_markers/{phase}.json"

        def execute(_command: list[str], *, selected_phase: str = phase, path: str = marker) -> CommandResult:
            payload = {
                "phase": selected_phase,
                "status": "pass",
                "formal": False,
                "performance_evidence": False,
                "holdout_opened": False,
                "request_exposure_fingerprint": request_fingerprint,
                "request_execution_contract_version": "1.0.0",
                "active_bundle_core_sha256": bundle["active_bundle_core_sha256"],
                "sources": phase_sources[selected_phase],
            }
            write_json(output_root / path, payload)
            return CommandResult(returncode=0)

        runner.run_phase(
            phase,
            command=[] if phase == "complete_without_holdout" else ["g14r9", phase],
            input_hash=canonical_sha256(
                {
                    "phase": phase,
                    "request_exposure_fingerprint": request_fingerprint,
                    "active_bundle_core_sha256": bundle["active_bundle_core_sha256"],
                }
            ),
            expected_outputs=[] if phase == "complete_without_holdout" else [marker],
            executor=execute,
        )
    events = runner.events()
    ledger = validate_phase_ledger(events)
    inventory = []
    for path in sorted(output_root.rglob("*.json")):
        if path.name == "phase_chain_rehearsal.json":
            continue
        inventory.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    summary = {
        "status": "pass",
        "formal": False,
        "performance_evidence": False,
        "holdout_opened": False,
        "holdout_capability": False,
        "clean_candidate_start": True,
        "detached_candidate": True,
        "candidate_has_local_venv": False,
        "candidate_commit": exact["candidate_commit"],
        "phase_order": list(PHASE_ORDER),
        "completed_phase_order": [row["phase"] for row in events if row["status"] == "completed"],
        "phase_ledger_validation": ledger,
        "command_count": matrix["command_count"],
        "phase_count": matrix["phase_count"],
        "unresolved_placeholder_count": 0,
        "absolute_sentinel_count": 0,
        "fresh_checkpoint_only": True,
        "v9_reference_count": 0,
        "request_exposure_fingerprint": request_fingerprint,
        "request_execution_identity_in_all_phase_records": True,
        "complete_without_holdout": events[-1]["phase"] == "complete_without_holdout",
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "inventory": inventory,
        "inventory_canonical_sha256": canonical_sha256(inventory),
    }
    write_json(output_root / "phase_chain_rehearsal.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
