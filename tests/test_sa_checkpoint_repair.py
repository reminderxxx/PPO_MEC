from __future__ import annotations

from pathlib import Path

from scripts import train_sa_ghmappo_real_sample as train_script


class _Args:
    profile = "smoke"
    episodes = 2
    audit_update_checkpoints = False
    post_training_audit_mode = "full"
    post_training_audit_max_windows = None
    post_training_audit_max_workflows = None


class _WorkflowState:
    workflow_id = "wf_1"


def test_checkpoint_audit_uses_recorded_best_source_without_full_update_audit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    latest = checkpoint_root / "latest.pt"
    source_update = checkpoint_root / "update_0006.pt"
    best_by_reward = checkpoint_root / "best_by_reward.pt"
    for path in [latest, source_update, best_by_reward]:
        path.write_bytes(b"checkpoint")

    def fake_evaluate_checkpoint_protocol(**kwargs):
        checkpoint_path = Path(kwargs["checkpoint_path"])
        reward = 118.0 if checkpoint_path.name == "update_0006.pt" else 90.0
        return {
            "aggregate_by_agent": {
                "sa_ghmappo": {
                    "total_reward": reward,
                    "workflow_continuity_rate": 0.9,
                    "handoff_failure_rate": 0.0,
                    "backhaul_traffic_cost": 1.0,
                    "handoff_ready_ratio": 0.5,
                    "mechanism_realization_rate": 0.6,
                    "adapter_state_migration_overhead": 0.0,
                }
            },
            "aggregate_policy_diagnostics_by_agent": {"sa_ghmappo": {}},
            "rows": [],
            "eval_window_ids": ["w1"],
            "workflow_ids": ["wf_1"],
        }

    def fake_load_checkpoint_metadata(path: str):
        return {"update_count": 6 if Path(path).name == "update_0006.pt" else 16}

    monkeypatch.setattr(train_script, "evaluate_checkpoint_protocol", fake_evaluate_checkpoint_protocol)
    monkeypatch.setattr(train_script, "load_checkpoint_metadata", fake_load_checkpoint_metadata)
    monkeypatch.setattr(train_script, "build_checkpoint_fingerprint", lambda _path: {"mock": True})

    audit = train_script.run_checkpoint_consistency_audit(
        current_agent_name="sa_ghmappo",
        checkpoint_root=checkpoint_root,
        workflow_states=[_WorkflowState()],
        eval_windows=[{"window_id": "w1"}],
        args=_Args(),
        best_record={
            "best_by_reward": {
                "path": str(best_by_reward),
                "source_checkpoint_path": str(source_update),
                "metrics": {"total_reward": 118.0},
            }
        },
    )

    assert audit["expected_best_sources"]["best_by_reward"] == str(source_update)
    assert any(
        item["checkpoint_label"] == "source_best_by_reward"
        for item in audit["audited_checkpoints"]
        if item["exists"]
    )


def test_compact_checkpoint_audit_limits_protocol_eval_and_disables_repair(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    latest = checkpoint_root / "latest.pt"
    source_update = checkpoint_root / "update_0006.pt"
    best_by_reward = checkpoint_root / "best_by_reward.pt"
    for path in [latest, source_update, best_by_reward]:
        path.write_bytes(b"checkpoint")

    class CompactArgs(_Args):
        post_training_audit_mode = "compact"

    calls: list[dict[str, object]] = []

    def fake_evaluate_checkpoint_protocol(**kwargs):
        calls.append(
            {
                "checkpoint_path": str(kwargs["checkpoint_path"]),
                "max_windows": kwargs.get("max_windows"),
                "max_workflows": kwargs.get("max_workflows"),
                "protocol_name": kwargs.get("protocol_name"),
            }
        )
        return {
            "aggregate_by_agent": {
                "sa_ghmappo": {
                    "total_reward": 7.0,
                    "workflow_continuity_rate": 0.9,
                    "handoff_failure_rate": 0.0,
                    "backhaul_traffic_cost": 1.0,
                    "handoff_ready_ratio": 0.5,
                    "mechanism_realization_rate": 0.6,
                    "adapter_state_migration_overhead": 0.0,
                }
            },
            "aggregate_policy_diagnostics_by_agent": {"sa_ghmappo": {}},
            "rows": [],
            "eval_window_ids": ["w1"],
            "workflow_ids": ["wf_1"],
        }

    monkeypatch.setattr(train_script, "evaluate_checkpoint_protocol", fake_evaluate_checkpoint_protocol)
    monkeypatch.setattr(train_script, "load_checkpoint_metadata", lambda _path: {"update_count": 6})
    monkeypatch.setattr(train_script, "build_checkpoint_fingerprint", lambda _path: {"mock": True})

    audit = train_script.run_checkpoint_consistency_audit(
        current_agent_name="sa_ghmappo",
        checkpoint_root=checkpoint_root,
        workflow_states=[_WorkflowState()],
        eval_windows=[{"window_id": "w1"}, {"window_id": "w2"}],
        args=CompactArgs(),
        best_record={
            "best_by_reward": {
                "path": str(best_by_reward),
                "source_checkpoint_path": str(source_update),
                "metrics": {"total_reward": 7.0},
            }
        },
    )

    assert audit["audit_scope"]["mode"] == "compact"
    assert audit["audit_scope"]["full_protocol"] is False
    assert audit["audit_scope"]["repair_best_record_eligible"] is False
    assert any(call["max_windows"] == 1 for call in calls)
    assert any(call["max_workflows"] == 1 for call in calls)
