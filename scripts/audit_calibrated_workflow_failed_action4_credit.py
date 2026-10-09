"""Read-only semantic audit of the proposed failed-action-4 advantage cap.

The audit replays an existing behavior ledger and runs five predeclared local
environment witnesses.  It never constructs an agent or calls an optimizer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.analyze_calibrated_workflow_selected_checkpoint_alignment import (  # noqa: E402
    _load_protocol,
)
from src.envs.core.calibrated_continuous_workflow_env import (  # noqa: E402
    CalibratedContinuousWorkflowEnv,
)


SOURCE_COMMIT = "858bc797e23b4e56051663f28d6fd7681f5ee77d"
ENV_PATH = Path("src/envs/core/calibrated_continuous_workflow_env.py")
CAP_PREDICATE = (
    "executed_action==4 AND service_completed==false AND progressed==false "
    "AND migration_success==false"
)
DIRECT_MISSING_FIELDS = (
    "cache_before_residents",
    "cache_after_residents",
    "cache_event_reason",
    "cache_event_admitted",
    "cache_event_victims",
    "prepared_state_before",
    "prepared_state_after",
    "future_bundle_reuse",
    "future_victim_reload",
    "counterfactual_avoided_loading",
)
SYNTHETIC_CASES = (
    {
        "case_id": "committed_warm_then_reuse",
        "purpose": "current service fails, target admission commits, next actual arrival reuses the bundle",
        "kind": "reuse",
    },
    {
        "case_id": "contact_budget_rollback",
        "purpose": "temporary admission exceeds contact budget and is atomically rolled back",
        "kind": "rollback",
    },
    {
        "case_id": "unused_warm_with_victim_reload",
        "purpose": "wrong target warm is unused, evicts a bundle, and the victim is later reloaded",
        "kind": "eviction",
    },
    {
        "case_id": "committed_warm_never_reused",
        "purpose": "target admission commits but the fixed future never visits that target",
        "kind": "unused",
    },
    {
        "case_id": "target_ready_noop_lru_touch",
        "purpose": "target bundle is already resident; action 4 changes no resident set but touches LRU metadata",
        "kind": "noop",
    },
    {
        "case_id": "capacity_rejection_no_change",
        "purpose": "bundle exceeds total capacity and admission is rejected without resident change",
        "kind": "rejection",
    },
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_blob_sha(commit: str, path: Path) -> str:
    payload = subprocess.run(
        ["git", "show", f"{commit}:{path.as_posix()}"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def _integrity(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_integrity.json"
    )
    _write_json(
        root / "artifact_integrity.json",
        {
            "schema_version": "calibrated_workflow_failed_action4_credit_audit_integrity_v1",
            "files": [
                {
                    "path": str(path.relative_to(root)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            ],
        },
    )


def _cache_snapshot(env: CalibratedContinuousWorkflowEnv) -> dict[str, Any]:
    return {
        str(rsu_id): {
            "residents": list(cache.residents),
            "last_used": dict(cache.last_used),
        }
        for rsu_id, cache in env.caches.items()
    }


def _row_bool(row: dict[str, str], name: str) -> bool:
    value = str(row[name]).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"invalid boolean {name}={row[name]!r}")
    return value == "true"


def _trigger(row: dict[str, str]) -> bool:
    return bool(
        int(row["executed_action"]) == 4
        and not _row_bool(row, "service_completed")
        and not _row_bool(row, "progressed")
        and not _row_bool(row, "migration_success")
    )


def _episode_replay(
    config: dict[str, Any], instance: dict[str, Any], rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], int]:
    env = CalibratedContinuousWorkflowEnv(config, deepcopy(instance))
    env.reset()
    replay: list[dict[str, Any]] = []
    mismatch_count = 0
    for row in sorted(rows, key=lambda item: int(item["step_index"])):
        semantic_before = env._semantic_state()
        node_before = deepcopy(semantic_before.get("current_workflow_node") or {})
        cache_before = _cache_snapshot(env)
        prepared_before = deepcopy(env.prepared_state)
        completed_before = len(env.completed)
        _, reward, terminated, truncated, info = env.step(int(row["executed_action"]))
        transition = deepcopy(info["transition"])
        cache_after = _cache_snapshot(env)
        prepared_after = deepcopy(env.prepared_state)
        reconstructed = {
            "service_completed": bool(transition["service_completed"]),
            "migration_success": bool(transition["migration_success"]),
            "progressed": len(env.completed) > completed_before,
            "model_transfer_bytes": int(transition["model_transfer_bytes"]),
            "state_transfer_bytes": int(transition["state_transfer_bytes"]),
            "recompute_seconds": float(transition["recompute_seconds"]),
            "service_operation_seconds": float(transition["service_operation_seconds"]),
            "clock_seconds_after": float(transition["clock_seconds_after"]),
        }
        for name, value in reconstructed.items():
            recorded = row[name]
            if isinstance(value, bool):
                matches = _row_bool(row, name) == value
            elif isinstance(value, int):
                matches = int(recorded) == value
            else:
                matches = abs(float(recorded) - value) <= 1e-6
            mismatch_count += int(not matches)
        replay.append(
            {
                "ledger": row,
                "semantic_before": semantic_before,
                "node_before": node_before,
                "cache_before": cache_before,
                "cache_after": cache_after,
                "prepared_before": prepared_before,
                "prepared_after": prepared_after,
                "transition": transition,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
    return replay, mismatch_count


def _future_evidence(
    trigger_index: int,
    steps: list[dict[str, Any]],
    target_rsu: str,
    adapter_id: str,
    admitted: set[str],
    victims: set[str],
    required_bundle: set[str],
) -> dict[str, Any]:
    target_visit_step: int | None = None
    same_bundle_need_step: int | None = None
    reuse_step: int | None = None
    victim_reload_step: int | None = None
    victim_failure_step: int | None = None
    admitted_lineage_intact = bool(admitted)
    factual_reuse_zero_transfer = False
    future_cost_seconds = 0.0
    for step in steps[trigger_index + 1 :]:
        transition = step["transition"]
        ledger = step["ledger"]
        future_cost_seconds += float(transition["step_cost_seconds"])
        current_rsu = str(transition["current_rsu_id"])
        node = step["node_before"]
        bundle = set()
        required_adapter = node.get("required_adapter")
        if required_adapter is not None:
            bundle = set(
                step["semantic_before"]
                .get("calibrated_context", {})
                .get("adapter_to_bundle", {})
                .get(required_adapter, [])
            )
        before_target = set(
            step["cache_before"].get(target_rsu, {}).get("residents", [])
        )
        if admitted and not admitted.issubset(before_target):
            admitted_lineage_intact = False
        if current_rsu == target_rsu and target_visit_step is None:
            target_visit_step = int(ledger["step_index"])
        if (
            current_rsu == target_rsu
            and str(required_adapter) == adapter_id
            and same_bundle_need_step is None
        ):
            same_bundle_need_step = int(ledger["step_index"])
        if (
            current_rsu == target_rsu
            and str(required_adapter) == adapter_id
            and required_bundle.issubset(before_target)
            and int(ledger["executed_action"]) != 2
            and bool(transition["service_completed"])
            and reuse_step is None
        ):
            reuse_step = int(ledger["step_index"])
            factual_reuse_zero_transfer = int(transition["model_transfer_bytes"]) == 0
        for cache_event in transition.get("cache_events", []):
            if victims & set(cache_event.get("admitted", [])) and victim_reload_step is None:
                victim_reload_step = int(ledger["step_index"])
        if (
            current_rsu == target_rsu
            and victims & bundle
            and not bool(transition["service_completed"])
            and victim_failure_step is None
        ):
            victim_failure_step = int(ledger["step_index"])
    return {
        "future_target_visit_step": target_visit_step,
        "future_same_bundle_need_step": same_bundle_need_step,
        "future_reuse_step": reuse_step,
        "admitted_lineage_intact_until_episode_end": admitted_lineage_intact,
        "factual_reuse_had_zero_model_transfer": factual_reuse_zero_transfer,
        "future_victim_reload_step": victim_reload_step,
        "future_victim_service_failure_step": victim_failure_step,
        "future_recorded_cost_seconds_after_trigger": future_cost_seconds,
        "counterfactual_avoided_loading": "unknown_not_identifiable_from_factual_ledger",
    }


def _classification(
    event: dict[str, Any], future: dict[str, Any], residents_unchanged: bool
) -> str:
    committed = bool(event.get("committed", False))
    admitted = bool(event.get("admitted", []))
    victims = bool(event.get("victims", []))
    reused = future["future_reuse_step"] is not None
    victim_cost = bool(
        future["future_victim_reload_step"] is not None
        or future["future_victim_service_failure_step"] is not None
    )
    if committed and not admitted:
        return "noop_resident_set_no_new_admission"
    if not committed and residents_unchanged:
        return "rollback_or_rejection_no_resident_change"
    if committed and admitted and reused and victim_cost:
        return "committed_warm_reused_with_observed_victim_cost"
    if committed and admitted and reused:
        return "committed_warm_reused_no_observed_victim_cost"
    if committed and admitted and not reused and victim_cost:
        return "committed_warm_not_reused_with_observed_victim_cost"
    if committed and admitted and not reused and victims:
        return "committed_warm_not_reused_unresolved_victim_externality"
    if committed and admitted:
        return "committed_warm_not_reused_no_victims"
    return "unexpected_or_incomplete_reconstruction"


def _audit_existing(
    source_root: Path, config: dict[str, Any], instances: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with (source_root / "behavior_ledger.csv").open(encoding="utf-8", newline="") as handle:
        ledger_rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str, str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in ledger_rows:
        groups[
            (
                row["arm"],
                row["split"],
                row["method"],
                int(row["seed"]),
                row["design_id"],
            )
        ].append(row)
    trigger_rows: list[dict[str, Any]] = []
    mismatch_count = 0
    for identity, rows in sorted(groups.items()):
        steps, mismatches = _episode_replay(config, instances[identity[-1]], rows)
        mismatch_count += mismatches
        for index, step in enumerate(steps):
            ledger = step["ledger"]
            if not _trigger(ledger):
                continue
            transition = step["transition"]
            events = list(transition.get("cache_events", []))
            event = dict(events[0]) if events else {}
            target = str(transition.get("predicted_handoff_target_rsu_id") or "")
            adapter = str(event.get("adapter_id") or step["node_before"].get("required_adapter") or "")
            required_bundle = set(
                step["semantic_before"]
                .get("calibrated_context", {})
                .get("adapter_to_bundle", {})
                .get(adapter, [])
            )
            before_target = step["cache_before"].get(target, {})
            after_target = step["cache_after"].get(target, {})
            residents_unchanged = before_target.get("residents", []) == after_target.get(
                "residents", []
            )
            last_used_changed = before_target.get("last_used", {}) != after_target.get(
                "last_used", {}
            )
            admitted = set(map(str, event.get("admitted", [])))
            victims = set(map(str, event.get("victims", [])))
            future = _future_evidence(
                index,
                steps,
                target,
                adapter,
                admitted,
                victims,
                required_bundle,
            )
            trigger_rows.append(
                {
                    "arm": identity[0],
                    "split": identity[1],
                    "method": identity[2],
                    "seed": identity[3],
                    "design_id": identity[4],
                    "step_index": int(ledger["step_index"]),
                    "cap_predicate": CAP_PREDICATE,
                    "direct_ledger_cache_fields": "unknown_not_recorded",
                    "replay_validation_mismatch_count_for_episode": mismatches,
                    "current_rsu_id": transition["current_rsu_id"],
                    "target_rsu_id": target,
                    "adapter_id": adapter,
                    "cache_event_committed": bool(event.get("committed", False)),
                    "cache_event_reason": event.get("reason", "missing"),
                    "admitted": json.dumps(sorted(admitted), separators=(",", ":")),
                    "victims": json.dumps(sorted(victims), separators=(",", ":")),
                    "target_residents_before": json.dumps(
                        before_target.get("residents", []), separators=(",", ":")
                    ),
                    "target_residents_after": json.dumps(
                        after_target.get("residents", []), separators=(",", ":")
                    ),
                    "resident_set_changed": not residents_unchanged,
                    "target_last_used_changed": last_used_changed,
                    "prepared_state_changed": step["prepared_before"]
                    != step["prepared_after"],
                    "model_transfer_bytes": int(transition["model_transfer_bytes"]),
                    "state_transfer_bytes": int(transition["state_transfer_bytes"]),
                    **future,
                    "classification": _classification(event, future, residents_unchanged),
                }
            )
    if mismatch_count:
        raise RuntimeError(f"existing ledger replay mismatches: {mismatch_count}")
    return trigger_rows, {
        "behavior_row_count": len(ledger_rows),
        "episode_count": len(groups),
        "replay_validation_mismatch_count": mismatch_count,
        "trigger_count": len(trigger_rows),
    }


def _node(node_id: str, adapter: str, predecessors: list[str], successors: list[str]) -> dict[str, Any]:
    base = "base:family_b" if adapter == "helmet_distinct" else "base:family_a"
    return {
        "node_id": node_id,
        "node_name": node_id,
        "required_base_model": base,
        "required_adapter": adapter,
        "input_size": 192757,
        "output_size": 2195,
        "input_bytes": 192757,
        "state_bytes": 2195,
        "compute_seconds": 3.807234,
        "predecessors": predecessors,
        "successors": successors,
    }


def _synthetic_instance(config: dict[str, Any], spec: dict[str, str]) -> tuple[dict[str, Any], list[int], list[int]]:
    kind = spec["kind"]
    tight = int(config["capacity_bytes"]["shared_tight"])
    ample = int(config["capacity_bytes"]["shared_ample"])
    base_a = ["base:family_a"]
    alpr = list(config["adapter_to_bundle"]["alpr"])
    helmet = list(config["adapter_to_bundle"]["helmet_shared"])
    nodes = [_node("n0", "alpr", [], [])]
    actual = ["rsu_0", "rsu_1", "rsu_1"]
    predicted = ["rsu_0", "rsu_1", "rsu_1"]
    capacity = ample
    residents = {"rsu_0": base_a, "rsu_1": base_a, "rsu_2": base_a}
    prefix: list[int] = []
    suffix = [3]
    actual_mbps = 1000.0
    if kind == "rollback":
        residents["rsu_1"] = []
        suffix = [2]
    elif kind == "eviction":
        capacity = tight
        residents["rsu_1"] = helmet
        actual = ["rsu_0", "rsu_2", "rsu_1", "rsu_1"]
        predicted = ["rsu_0", "rsu_1", "rsu_1", "rsu_1"]
        nodes = [
            _node("n0", "alpr", [], ["n1"]),
            _node("n1", "helmet_shared", ["n0"], []),
        ]
        suffix = [2, 0]
    elif kind == "unused":
        actual = ["rsu_0", "rsu_2", "rsu_2"]
        predicted = ["rsu_0", "rsu_1", "rsu_1"]
        suffix = [2]
    elif kind == "noop":
        residents["rsu_1"] = alpr
        actual = ["rsu_0", "rsu_0", "rsu_2", "rsu_2"]
        predicted = ["rsu_0", "rsu_0", "rsu_1", "rsu_1"]
        nodes = [
            _node("n0", "alpr", [], ["n1"]),
            _node("n1", "alpr", ["n0"], []),
        ]
        prefix = [2]
        suffix = [2]
    elif kind == "rejection":
        capacity = 100_000_000
        residents = {"rsu_0": [], "rsu_1": [], "rsu_2": []}
        suffix = [2]
    return (
        {
            "design_id": f"synthetic_{spec['case_id']}",
            "split": "synthetic_semantic_witness",
            "window_id": f"synthetic_{spec['case_id']}",
            "workflow_id": f"wf_{spec['case_id']}",
            "trace_features": {"handoff_pressure": "fixed", "mean_speed_proxy": 20.0},
            "workflow_features": {"topology_class": "chain"},
            "factors": {
                "sharing": "fixed_by_case",
                "capacity": "fixed_by_case",
                "state_scale": 1,
                "initial_target": "fixed_by_case",
            },
            "rsu_ids": ["rsu_0", "rsu_1", "rsu_2"],
            "rsu_sequence": actual,
            "predicted_rsu_sequence": predicted,
            "prediction_profile": {"confidence": 0.9, "uncertainty": 0.1},
            "link_profile": {
                "actual_mbps": actual_mbps,
                "estimated_mbps": 1000.0,
                "error_class": "fixed_synthetic",
            },
            "cache_capacity_bytes": capacity,
            "initial_residents": residents,
            "nodes": nodes,
            "edges": [["n0", "n1"]] if len(nodes) == 2 else [],
            "execution_order": [node["node_id"] for node in nodes],
            "deadline_seconds": 60.0,
            "max_steps": max(len(actual), len(prefix) + 1 + len(suffix)),
            "source_classes": {
                "measured": [],
                "trace_derived": [],
                "literature": [],
                "artificial": ["all_fields_synthetic_semantic_witness"],
            },
        },
        prefix,
        suffix,
    )


def _synthetic_witnesses(config: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for spec in SYNTHETIC_CASES:
        instance, prefix, suffix = _synthetic_instance(config, spec)
        env = CalibratedContinuousWorkflowEnv(config, instance)
        env.reset()
        for action in prefix:
            if action not in env.valid_actions():
                raise RuntimeError(f"illegal prefix action in {spec['case_id']}")
            env.step(action)
        cache_before = _cache_snapshot(env)
        prepared_before = deepcopy(env.prepared_state)
        completed_before = len(env.completed)
        if 4 not in env.valid_actions():
            raise RuntimeError(f"action 4 is not legal in {spec['case_id']}")
        _, _, _, _, info = env.step(4)
        transition = deepcopy(info["transition"])
        cache_after = _cache_snapshot(env)
        prepared_after = deepcopy(env.prepared_state)
        predicate = bool(
            not transition["service_completed"]
            and len(env.completed) == completed_before
            and not transition["migration_success"]
        )
        if not predicate:
            raise RuntimeError(f"synthetic case does not trigger cap predicate: {spec['case_id']}")
        future: list[dict[str, Any]] = []
        for action in suffix:
            if env.terminated:
                break
            if action not in env.valid_actions():
                raise RuntimeError(f"illegal suffix action in {spec['case_id']}: {action}")
            before = _cache_snapshot(env)
            node = deepcopy(env._current_node())
            _, _, terminated, truncated, step_info = env.step(action)
            future.append(
                {
                    "action": action,
                    "node_id": node["node_id"],
                    "required_adapter": node["required_adapter"],
                    "cache_before": before,
                    "transition": step_info["transition"],
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
        event = dict(transition["cache_events"][0]) if transition["cache_events"] else {}
        outputs.append(
            {
                "case_id": spec["case_id"],
                "purpose": spec["purpose"],
                "predeclared_kind": spec["kind"],
                "prefix_actions": prefix,
                "probe_action": 4,
                "suffix_actions": suffix,
                "cap_predicate_triggered": predicate,
                "public_state_exposes_actual_mbps": "actual_mbps"
                in info["semantic_state"]["calibrated_context"]["link"],
                "cache_event": event,
                "target_cache_before": cache_before.get(
                    str(transition["predicted_handoff_target_rsu_id"]), {}
                ),
                "target_cache_after": cache_after.get(
                    str(transition["predicted_handoff_target_rsu_id"]), {}
                ),
                "prepared_state_before": prepared_before,
                "prepared_state_after": prepared_after,
                "transition": transition,
                "future": future,
                "final_summary": env.summary(),
            }
        )
    return outputs


def run(source_root: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    design, config, _ = _load_protocol()
    manifest = _read_json(ROOT_DIR / design["workload_manifest"])
    instances = {str(row["design_id"]): row for row in manifest["instances"]}
    source_blob_sha = _git_blob_sha(SOURCE_COMMIT, ENV_PATH)
    current_sha = _sha256(ROOT_DIR / ENV_PATH)
    if source_blob_sha != current_sha:
        raise RuntimeError("current environment differs from scientific execution source")
    trigger_rows, replay_summary = _audit_existing(
        source_root, config, instances
    )
    witnesses = _synthetic_witnesses(config)
    class_counts = Counter(row["classification"] for row in trigger_rows)
    summary_rows = [
        {"classification": name, "trigger_count": count}
        for name, count in sorted(class_counts.items())
    ]
    output_root.mkdir(parents=True)
    _write_csv(output_root / "trigger_samples.csv", trigger_rows)
    _write_csv(output_root / "trigger_classification_summary.csv", summary_rows)
    _write_json(output_root / "synthetic_witnesses.json", witnesses)
    _write_json(
        output_root / "field_coverage.json",
        {
            "direct_behavior_ledger_fields": list(
                csv.DictReader(
                    (source_root / "behavior_ledger.csv").open(
                        encoding="utf-8", newline=""
                    )
                ).fieldnames
                or []
            ),
            "direct_missing_fields": list(DIRECT_MISSING_FIELDS),
            "missing_fields_direct_status": "unknown_not_recorded",
            "reconstructed_fields_source": (
                "deterministic replay of recorded executed actions against hash-matched "
                "scientific environment and frozen instances"
            ),
            "counterfactual_avoided_loading_status": "unknown_not_identifiable_from_factual_ledger",
        },
    )
    committed_new = sum(
        bool(row["cache_event_committed"] and json.loads(row["admitted"]))
        for row in trigger_rows
    )
    reused = sum(row["future_reuse_step"] is not None for row in trigger_rows)
    committed_new_reused = sum(
        bool(
            row["cache_event_committed"]
            and json.loads(row["admitted"])
            and row["future_reuse_step"] is not None
        )
        for row in trigger_rows
    )
    victim_cost = sum(
        row["future_victim_reload_step"] is not None
        or row["future_victim_service_failure_step"] is not None
        for row in trigger_rows
    )
    _write_json(
        output_root / "audit_summary.json",
        {
            "schema_version": "calibrated_workflow_failed_action4_credit_audit_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git("rev-parse", "HEAD"),
            "source_scientific_commit": SOURCE_COMMIT,
            "source_environment_sha256": source_blob_sha,
            "current_environment_sha256": current_sha,
            "source_run": str(source_root.relative_to(ROOT_DIR)),
            "source_behavior_ledger_sha256": _sha256(
                source_root / "behavior_ledger.csv"
            ),
            "cap_predicate": CAP_PREDICATE,
            "existing_replay": replay_summary,
            "classification_counts": dict(sorted(class_counts.items())),
            "cache_event_reason_counts": dict(
                sorted(Counter(row["cache_event_reason"] for row in trigger_rows).items())
            ),
            "committed_new_admission_count": committed_new,
            "all_trigger_future_same_bundle_reuse_count": reused,
            "committed_new_admission_future_reuse_count": committed_new_reused,
            "committed_new_admission_no_observed_reuse_count": committed_new
            - committed_new_reused,
            "observed_victim_reload_or_failure_count": victim_cost,
            "synthetic_witness_count": len(witnesses),
            "training_steps": 0,
            "optimizer_updates": 0,
            "checkpoint_reads": 0,
            "model_calls": 0,
            "decision": "CAP_PREDICATE_INSUFFICIENT_NOT_READY",
            "reason": (
                "migration_success=false does not imply no durable cache side effect or "
                "no later value; committed target warming and eviction externalities both occur"
            ),
            "automatic_replacement_candidate_authorized": False,
        },
    )
    _integrity(output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default="artifacts/benchmarks/calibrated_workflow_value_normalization_ab_20261009_v2",
    )
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run(Path(args.source_root).resolve(), Path(args.output_root).resolve())


if __name__ == "__main__":
    main()
