from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.analyze_top_journal_statistics import required_identity_value
from scripts.benchmark_main_results import resolve_verified_capacity_identity


ROOT = Path(__file__).resolve().parents[1]
STATISTICS = ROOT / "scripts/analyze_top_journal_statistics.py"


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def base_row(agent: str, capacity: str, value: float) -> dict[str, str]:
    return {
        "seed": "7",
        "window_id": "w0",
        "workflow_id": "j0",
        "capacity_label": capacity,
        "source_segment_run_id": "segment0",
        "agent_name": agent,
        "metric": str(value),
    }


def run_statistics(tmp_path: Path, paths: list[Path]) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(STATISTICS)]
    for path in paths:
        command.extend(("--rows_path", str(path)))
    command.extend(
        (
            "--candidate_agent", "candidate",
            "--baseline_agents", "baseline",
            "--metrics", "metric",
            "--pair_keys", "seed", "window_id", "workflow_id", "capacity_label",
            "--outer_cluster_keys", "source_segment_run_id", "window_id",
            "--inner_cluster_keys", "seed", "workflow_id", "capacity_label",
            "--bootstrap_samples", "20",
            "--random_seed", "1401",
            "--output_root", str(tmp_path / "output"),
        )
    )
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def test_cross_capacity_pairs_are_not_merged_or_overwritten(tmp_path: Path) -> None:
    rows = [
        base_row("candidate", "constrained_288mb", 3.0),
        base_row("baseline", "constrained_288mb", 1.0),
        base_row("candidate", "medium_576mb", 11.0),
        base_row("baseline", "medium_576mb", 1.0),
    ]
    path = tmp_path / "rows.csv"
    write_rows(path, rows)
    completed = run_statistics(tmp_path, [path])
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((tmp_path / "output/paired_statistics.json").read_text())
    row = payload["rows"][0]
    assert row["paired_count"] == 2
    assert row["inner_cluster_count"] == 2
    assert row["outer_cluster_count"] == 1
    assert row["mean_delta"] == 6.0
    assert payload["identity_validation"]["validated_pair_coordinate_count"] == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.__setitem__("capacity_label", ""), "pair key field blank: capacity_label"),
        (lambda row: row.__setitem__("capacity_label", "null"), "pair key field null sentinel: capacity_label"),
    ],
)
def test_required_capacity_identity_fails_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    rows = [base_row("candidate", "medium_576mb", 2.0), base_row("baseline", "medium_576mb", 1.0)]
    mutation(rows[0])
    path = tmp_path / "rows.csv"
    write_rows(path, rows)
    completed = run_statistics(tmp_path, [path])
    assert completed.returncode != 0
    assert message in completed.stderr


def test_missing_identity_field_is_distinguished() -> None:
    with pytest.raises(ValueError, match="field missing"):
        required_identity_value({}, "capacity_label", identity_kind="pair key")


def test_wrong_identity_type_is_distinguished() -> None:
    with pytest.raises(ValueError, match="invalid type"):
        required_identity_value(
            {"capacity_label": ["medium_576mb"]},
            "capacity_label",
            identity_kind="pair key",
        )
    with pytest.raises(ValueError, match="field null"):
        required_identity_value(
            {"capacity_label": None}, "capacity_label", identity_kind="pair key"
        )


def test_conflicting_capacity_causes_incomplete_pair_failure(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    write_rows(
        path,
        [
            base_row("candidate", "constrained_288mb", 2.0),
            base_row("baseline", "medium_576mb", 1.0),
        ],
    )
    completed = run_statistics(tmp_path, [path])
    assert completed.returncode != 0
    assert "incomplete statistics pair matrix" in completed.stderr


def test_duplicate_coordinate_across_files_is_rejected(tmp_path: Path) -> None:
    rows = [base_row("candidate", "medium_576mb", 2.0), base_row("baseline", "medium_576mb", 1.0)]
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    write_rows(first, rows)
    write_rows(second, rows)
    completed = run_statistics(tmp_path, [first, second])
    assert completed.returncode != 0
    assert "duplicate statistics pair row" in completed.stderr


def test_cluster_conflict_is_rejected_even_when_metric_is_null(tmp_path: Path) -> None:
    candidate = base_row("candidate", "medium_576mb", 2.0)
    baseline = base_row("baseline", "medium_576mb", 1.0)
    candidate["metric"] = ""
    baseline["metric"] = ""
    baseline["source_segment_run_id"] = "different_segment"
    path = tmp_path / "rows.csv"
    write_rows(path, [candidate, baseline])
    completed = run_statistics(tmp_path, [path])
    assert completed.returncode != 0
    assert "conflicting cluster identity within pair" in completed.stderr


def test_runtime_capacity_identity_requires_resource_and_matching_mb(tmp_path: Path) -> None:
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text("capacity: 576\n", encoding="utf-8")
    digest = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    audit = {
        "status": "pass",
        "resource_registry_semantic_sha256": "b" * 64,
        "resolutions": [
            {
                "logical_resource_id": "runtime_config.medium_576mb",
                "resource_role": "runtime_config",
                "status": "compatible",
                "resolved_path": str(runtime_path),
                "observed_sha256": digest,
                "observed_size_bytes": runtime_path.stat().st_size,
                "semantic_identity_fingerprint": "c" * 64,
            }
        ],
    }
    runtime = {
        "model_cache_profile": "typed_base_adapter_state_v1",
        "runtime_contract_sha256": "a" * 64,
        "cache_capacity_profile": {"enabled": True, "unit": "mb", "capacity_mb": 576.0},
    }
    identity = resolve_verified_capacity_identity(
        runtime_contract=runtime,
        runtime_config_resource_id="runtime_config.medium_576mb",
        portable_resource_audit=audit,
        resolved_runtime_config_path=str(runtime_path),
    )
    assert identity and identity["capacity_label"] == "medium_576mb"
    with pytest.raises(ValueError, match="resource/runtime mismatch"):
        resolve_verified_capacity_identity(
            runtime_contract=runtime,
            runtime_config_resource_id="runtime_config.constrained_288mb",
            portable_resource_audit=audit,
            resolved_runtime_config_path=str(runtime_path),
        )
    with pytest.raises(ValueError, match="portable resource"):
        resolve_verified_capacity_identity(
            runtime_contract=runtime,
            runtime_config_resource_id="",
            portable_resource_audit=audit,
            resolved_runtime_config_path=str(runtime_path),
        )
    with pytest.raises(ValueError, match="successful portable resource resolution"):
        resolve_verified_capacity_identity(
            runtime_contract=runtime,
            runtime_config_resource_id="runtime_config.medium_576mb",
            portable_resource_audit={"status": "legacy_no_resource_registry", "resolutions": []},
            resolved_runtime_config_path=str(runtime_path),
        )
    spoofed = json.loads(json.dumps(audit))
    spoofed["resolutions"][0]["observed_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="resolution identity mismatch"):
        resolve_verified_capacity_identity(
            runtime_contract=runtime,
            runtime_config_resource_id="runtime_config.medium_576mb",
            portable_resource_audit=spoofed,
            resolved_runtime_config_path=str(runtime_path),
        )
