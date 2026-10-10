"""Contract checks for the optional synthetic elapsed-time sensitivity profile."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest


SOURCE = Path(os.environ.get("CSCWD_SOURCE_ROOT", ""))
pytestmark = pytest.mark.skipif(not os.environ.get("CSCWD_SOURCE_ROOT"), reason="set CSCWD_SOURCE_ROOT to frozen source worktree")


def _setup():
    audit_path = Path(__file__).resolve().parents[1] / "scripts/audit_cscwd_mobility_elapsed_sensitivity.py"
    spec = importlib.util.spec_from_file_location("cscwd_mobility_audit", audit_path)
    assert spec and spec.loader
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    sys.path.insert(0, str(SOURCE))
    from scripts.run_calibrated_workflow_event_aux_abstention_ab import _load_candidate_inputs
    from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv, TwoStepCostRule

    protocol = json.loads((SOURCE / "configs/experiment/calibrated_workflow_event_aux_abstention_ab_v1.json").read_text())
    config, splits, _, _ = _load_candidate_inputs(protocol)
    instances = {row["design_id"]: row for split in ("dev", "regression", "frozen_check") for row in splits[split]}
    return audit, CalibratedContinuousWorkflowEnv, audit.build_elapsed_env_class(CalibratedContinuousWorkflowEnv), TwoStepCostRule, config, instances


def _signature(env, observation, info):
    return {
        "observation": observation.tolist(), "mask": info["action_mask"],
        "semantic": info["semantic_state"], "transition": info.get("transition"),
        "clock": env.clock_seconds, "step_index": env.step_index,
        "metrics": env.metrics, "cache": audit_state(env),
    }


def audit_state(env):
    return {
        "cache": {key: (list(cache.residents), dict(cache.last_used)) for key, cache in env.caches.items()},
        "prepared": deepcopy(env.prepared_state), "completed": list(env.completed),
    }


def test_legacy_is_native_class_and_transition_equivalent():
    audit, Base, Elapsed, _, config, instances = _setup()
    assert Base is not Elapsed
    old_profile = Base(config, instances["regression_05"])
    reference = Base(config, instances["regression_05"])
    a_ob, a_info = old_profile.reset()
    b_ob, b_info = reference.reset()
    assert _signature(old_profile, a_ob, a_info) == _signature(reference, b_ob, b_info)
    for action in (3, 2, 3):
        assert a_info["action_mask"][action]
        a_ob, a_reward, a_done, a_trunc, a_info = old_profile.step(action)
        b_ob, b_reward, b_done, b_trunc, b_info = reference.step(action)
        assert (a_reward, a_done, a_trunc) == (b_reward, b_done, b_trunc)
        assert _signature(old_profile, a_ob, a_info) == _signature(reference, b_ob, b_info)


def test_two_failed_waits_advance_only_legacy_mobility():
    _, Base, Elapsed, _, config, instances = _setup()
    old = Base(config, instances["regression_05"])
    new = Elapsed(config, instances["regression_05"])
    old.reset(); new.reset()
    for _ in range(2):
        assert old.valid_actions()[3] == 3 and new.valid_actions()[3] == 3
        old_info = old.step(3)[4]
        new_info = new.step(3)[4]
        assert not old_info["transition"]["service_completed"]
        assert not new_info["transition"]["service_completed"]
    assert old.clock_seconds == new.clock_seconds == 4.0
    assert old._mobility_index() == 2 and new._mobility_index() == 0
    assert old._current_rsu_id() != new._current_rsu_id()
    assert old_info["transition"]["current_rsu_id"] == new_info["transition"]["current_rsu_id"] == "rsu_0"


def test_short_control_and_successful_long_service_use_same_start_rsu_cost():
    _, Base, Elapsed, _, config, instances = _setup()
    for design_id, crosses_five_seconds in (("dev_01", False), ("dev_00", True)):
        old = Base(config, instances[design_id])
        new = Elapsed(config, instances[design_id])
        old.reset(); new.reset()
        old_event = old.step(3)[4]["transition"]
        new_event = new.step(3)[4]["transition"]
        assert old_event == new_event
        assert old_event["service_completed"]
        assert (new.clock_seconds >= 5.0) is crosses_five_seconds
        assert new._current_rsu_id() == old._current_rsu_id() == "rsu_0"
        assert audit_state(old) == audit_state(new)


def test_long_atomic_action_crosses_multiple_slots_only_after_completion():
    _, Base, Elapsed, _, config, instances = _setup()
    instance = instances["frozen_check_04"]
    old = Base(config, instance)
    new = Elapsed(config, instance)
    old.reset(); new.reset()
    assert old._physical_contact_budget_seconds() == new._physical_contact_budget_seconds() == 10.0
    assert 2 in old.valid_actions() and 2 in new.valid_actions()
    old_event = old.step(2)[4]["transition"]
    new_event = new.step(2)[4]["transition"]
    assert old_event == new_event  # same native action/cost/cache/state/reward at starting RSU
    assert new_event["current_rsu_id"] == "rsu_0"
    assert new.clock_seconds > 15.0
    assert old._mobility_index() == 1 and new._mobility_index() >= 3
    assert old._current_rsu_id() != new._current_rsu_id()
    assert old_event["model_transfer_bytes"] == new_event["model_transfer_bytes"] == 0
    assert audit_state(old) == audit_state(new)


def test_public_forecast_is_prefix_only_and_rule_preview_unchanged():
    audit, Base, Elapsed, Rule, config, instances = _setup()
    original = instances["regression_05"]
    mutated = deepcopy(original)
    mutated["rsu_sequence"] = list(mutated["rsu_sequence"])
    mutated["rsu_sequence"][2:] = ["rsu_0"] * (len(mutated["rsu_sequence"]) - 2)
    a = Elapsed(config, original)
    b = Elapsed(config, mutated)
    a_obs, a_info = a.reset(); b_obs, b_info = b.reset()
    assert a_obs.tolist() == b_obs.tolist()
    assert a_info["semantic_state"] == b_info["semantic_state"]
    assert a_info["action_mask"] == b_info["action_mask"]
    assert a._physical_contact_budget_seconds() != b._physical_contact_budget_seconds()
    from scripts.run_calibrated_workflow_strong_baselines import _build_learned
    for method in ("old_sa", "event_abstention_sa", "ppo", "mappo"):
        run_id, algorithm, expected_hash = audit.CHECKPOINTS[method]
        checkpoint = SOURCE / "artifacts/experiments" / run_id / "checkpoints" / f"{algorithm}_seed7_selected.pt"
        assert audit.sha(checkpoint) == expected_hash
        arm_config = {**config, "mechanism_aux_missing_current_event_abstention_enabled": method == "event_abstention_sa"}
        agent = _build_learned(algorithm, 7, arm_config, popart_enabled=False)
        agent.load(str(checkpoint))
        agent._deterministic_action = True
        action_a, payload_a = agent.act(a_obs, a_info)
        action_b, payload_b = agent.act(b_obs, b_info)
        assert action_a == action_b
        assert payload_a["env_action_probs"] == payload_b["env_action_probs"]
        assert payload_a["raw_env_action"] == payload_b["raw_env_action"]
    counter = [0]
    rule = Rule()
    chosen = rule.select_action(a)
    counted = rule.select_action(audit.PreviewFacade(a, counter))
    assert chosen == counted
    assert counter[0] > 0
    assert a.clock_seconds == 0 and a.step_index == 0
