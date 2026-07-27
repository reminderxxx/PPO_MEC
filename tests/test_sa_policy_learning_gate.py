from __future__ import annotations

from scripts.train_sa_ghmappo_real_sample import (
    build_policy_learning_gate,
    maybe_update_best_checkpoint,
)


def _update_eval(*, update_index: int, digest: str, reward: float) -> dict:
    return {
        "update_index": update_index,
        "raw_policy_eval": {
            "policy_action_signatures_by_agent": {
                "sa_ghmappo": {"digest": digest},
            },
            "aggregate_by_agent": {
                "sa_ghmappo": {
                    "total_reward": reward,
                    "workflow_continuity_rate": 0.5,
                },
            },
        },
    }


def test_policy_learning_gate_waits_for_two_raw_policy_evaluations() -> None:
    gate = build_policy_learning_gate(
        [_update_eval(update_index=1, digest="first", reward=1.0)],
        "sa_ghmappo",
    )

    assert gate["status"] == "pending_insufficient_checkpoints"
    assert gate["selection_eligible"] is False


def test_policy_learning_gate_blocks_invariant_raw_policy() -> None:
    gate = build_policy_learning_gate(
        [
            _update_eval(update_index=1, digest="same", reward=1.0),
            _update_eval(update_index=2, digest="same", reward=1.0),
        ],
        "sa_ghmappo",
    )

    assert gate["status"] == "blocked_policy_invariant"
    assert gate["policy_invariant"] is True
    assert gate["selection_eligible"] is False


def test_policy_learning_gate_allows_selection_after_raw_policy_changes() -> None:
    gate = build_policy_learning_gate(
        [
            _update_eval(update_index=1, digest="first", reward=1.0),
            _update_eval(update_index=2, digest="second", reward=1.0),
        ],
        "sa_ghmappo",
    )

    assert gate["status"] == "passed_policy_changed"
    assert gate["policy_invariant"] is False
    assert gate["selection_eligible"] is True


def test_checkpoint_selector_uses_raw_policy_metrics(tmp_path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    checkpoint_path = checkpoint_root / "update_0002.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    update_eval = {
        "aggregate_by_agent": {"sa_ghmappo": {"total_reward": 99.0}},
        "raw_policy_eval": {
            "protocol_name": "update_eval_raw_policy",
            "policy_evaluation_mode": "raw_policy",
            "deterministic_eval": True,
            "aggregate_by_agent": {"sa_ghmappo": {"total_reward": 3.0}},
            "aggregate_policy_diagnostics_by_agent": {"sa_ghmappo": {}},
            "rows": [],
            "eval_window_ids": ["w1"],
            "workflow_ids": ["wf1"],
        },
        "policy_learning_gate": {
            "status": "passed_policy_changed",
            "selection_eligible": True,
        },
    }

    result = maybe_update_best_checkpoint(
        current_agent_name="sa_ghmappo",
        checkpoint_path=checkpoint_path,
        checkpoint_root=checkpoint_root,
        update_index=2,
        episode_index=4,
        update_eval=update_eval,
        best_record={},
    )

    assert result["best_by_reward"]["score"] == 3.0
    assert result["best_by_reward"]["selection_protocol"]["policy_evaluation_mode"] == "raw_policy"
