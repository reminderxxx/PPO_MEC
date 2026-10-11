"""Independent no-training audit of B's frozen event-supervision abstention."""

from __future__ import annotations

import csv
from copy import deepcopy
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
B_ROOT = Path("/Users/howen/.codex/worktrees/causal-budget-extension/PPO_MEC")
SOURCE = ROOT / "artifacts/analysis/cscwd_service_feasible_action_branches_20261010_v1"
OUTPUT = ROOT / "artifacts/analysis/cscwd_conditional_abstention_audit_20261010_v2"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(B_ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _logits() -> dict[str, torch.Tensor]:
    return {"slow_logits": torch.tensor([0.3, -0.2, 0.1], requires_grad=True),
            "fast_logits": torch.tensor([0.15, -0.1], requires_grad=True),
            "event_logits": torch.tensor([-0.25, 0.45], requires_grad=True)}


def _grad(agent, states: list[dict]) -> tuple[float, list[dict[str, torch.Tensor]]]:
    outputs = [_logits() for _ in states]
    loss = agent.agent._compute_auxiliary_loss(states, outputs)
    loss.backward()
    return float(loss.detach()), [{name: value.grad.detach().clone() for name, value in output.items()}
                                   for output in outputs]


def main() -> None:
    if (OUTPUT / "abstention_receipt.json").exists():
        raise RuntimeError("create-only abstention audit already exists")
    if _git("status", "--porcelain"):
        raise RuntimeError("B implementation worktree is not frozen clean")
    commit, tree = _git("rev-parse", "HEAD"), _git("rev-parse", "HEAD^{tree}")
    # The candidate modules must resolve from B, never this A checkout.
    sys.path = [entry for entry in sys.path if entry not in {str(ROOT), str(ROOT / "scripts") }]
    sys.path.insert(0, str(B_ROOT))
    from scripts.run_calibrated_workflow_strong_baselines import _build_learned
    from scripts.run_calibrated_workflow_interface_repair import _sha256, _write_json

    manifest = json.loads((SOURCE / "analysis_manifest.json").read_text(encoding="utf-8"))
    snapshot_manifest = json.loads((SOURCE / "snapshot_export_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if _sha256(SOURCE / name) != digest:
            raise RuntimeError("frozen branch artifact hash drift")
    for entry in snapshot_manifest["state_files"].values():
        if _sha256(SOURCE / entry["path"]) != entry["sha256"]:
            raise RuntimeError("frozen complete state hash drift")
    mappings = _rows(SOURCE / "source_to_state_rows.csv")
    states = []
    for row in mappings:
        snapshot = json.loads((SOURCE / "state_snapshots" / f"{row['full_state_sha256']}.json").read_text(encoding="utf-8"))
        states.append((row, snapshot["config"], snapshot["policy_input_observation"], snapshot["policy_input_info"]))
    if len(states) != 30:
        raise RuntimeError("frozen state mapping count drift")
    config = deepcopy(states[0][1])
    if any(item[1] != config for item in states):
        raise RuntimeError("branch states use different shared environment config")
    legacy_config = {**config, "mechanism_aux_current_service_feasibility_gate_enabled": False,
                     "mechanism_aux_missing_current_event_abstention_enabled": False}
    candidate_config = {**legacy_config, "mechanism_aux_missing_current_event_abstention_enabled": True}
    legacy = _build_learned("sa_ghmappo", 7, legacy_config, popart_enabled=False)
    candidate = _build_learned("sa_ghmappo", 7, candidate_config, popart_enabled=False)
    legacy._deterministic_action = True
    candidate._deterministic_action = True
    old_weights = legacy.agent._network.state_dict()
    new_weights = candidate.agent._network.state_dict()
    if set(old_weights) != set(new_weights) or any(not torch.equal(old_weights[key], new_weights[key]) for key in old_weights):
        raise RuntimeError("candidate changed initial network parameter identity")
    all_semantic = [info["semantic_state"] for _, _, _, info in states]
    stats = candidate.agent._auxiliary_event_supervision_stats(all_semantic)
    ready = []
    missing = []
    forward_equal = 0
    gradient_equal_ready = 0
    gradient_zero_missing = 0
    for row, _, observation, info in states:
        semantic = info["semantic_state"]
        is_ready = row["current_bundle_ready"] == "True"
        (ready if is_ready else missing).append(semantic)
        if legacy.agent._build_mechanism_targets(semantic) != candidate.agent._build_mechanism_targets(semantic):
            raise RuntimeError("abstention changed hard/soft or slow/fast pseudo target values")
        if candidate.agent._mechanism_aux_current_service_feasibility_gate_enabled:
            raise RuntimeError("rejected hard-zero event target gate was re-enabled")
        if info.get("run_metadata", {}).get("policy_evaluation_mode") != "raw_policy":
            raise RuntimeError("frozen evaluation mode drift")
        old_action, old_info = legacy.act(observation, deepcopy(info))
        new_action, new_info = candidate.act(observation, deepcopy(info))
        checked_fields = ("env_action_probs", "env_action_log_prob", "env_action_entropy",
                          "raw_env_action", "projected_env_action", "final_env_action", "action_mask")
        if old_action != new_action or any(old_info[field] != new_info[field] for field in checked_fields):
            raise RuntimeError(f"raw policy/action likelihood drift at {row['source_id']}")
        if legacy.evaluate_value(observation, info) != candidate.evaluate_value(observation, info):
            raise RuntimeError("value prediction drift")
        forward_equal += 1
        old_loss, old_grads = _grad(legacy, [semantic])
        new_loss, new_grads = _grad(candidate, [semantic])
        for name in ("slow_logits", "fast_logits"):
            if not torch.equal(old_grads[0][name], new_grads[0][name]):
                raise RuntimeError(f"{name} auxiliary gradient drift")
        if is_ready:
            if old_loss != new_loss or not torch.equal(old_grads[0]["event_logits"], new_grads[0]["event_logits"]):
                raise RuntimeError("current-ready event auxiliary gradient drift")
            gradient_equal_ready += 1
        else:
            if torch.count_nonzero(new_grads[0]["event_logits"]):
                raise RuntimeError("current-missing event auxiliary gradient not zero")
            gradient_zero_missing += 1
        changed = deepcopy(semantic)
        changed["predictions"] = {"unrelated_future_probe": ["not_a_policy_input"]}
        if candidate.agent._event_auxiliary_supervision_weight(changed) != float(is_ready):
            raise RuntimeError("abstention depends on prediction suffix")
    if gradient_equal_ready != 6 or gradient_zero_missing != 24:
        raise RuntimeError("readiness denominator drift")
    if stats["eligible_count"] != 30 or stats["supervised_count"] != 6 or stats["abstained_count"] != 24:
        raise RuntimeError("effective supervision receipt drift")
    if candidate.agent._mechanism_aux_coef != legacy.agent._mechanism_aux_coef:
        raise RuntimeError("other mechanism auxiliary loss coefficient changed")
    if not math.isclose(float(stats["supervised_fraction"]), 0.2, abs_tol=1e-12):
        raise RuntimeError("effective supervision fraction drift")
    old_profile = deepcopy(missing[0])
    old_profile["interface_profile"] = "calibrated_workflow_interface_v3_prefix_only"
    try:
        candidate.agent._event_auxiliary_supervision_weight(old_profile)
    except ValueError as error:
        if "requires the v4 public observation profile" not in str(error):
            raise
    else:
        raise RuntimeError("candidate accepted old public observation profile")
    with tempfile.TemporaryDirectory(prefix="cscwd_abstention_checkpoint_audit_") as directory:
        legacy_path = Path(directory) / "legacy.pt"
        candidate_path = Path(directory) / "candidate.pt"
        legacy.agent.save(str(legacy_path))
        candidate.agent.save(str(candidate_path))
        legacy.agent.load(str(legacy_path))
        candidate.agent.load(str(candidate_path))
        for receiver, foreign_path in ((legacy, candidate_path), (candidate, legacy_path)):
            try:
                receiver.agent.load(str(foreign_path))
            except ValueError as error:
                if "event-supervision abstention checkpoint mismatch" not in str(error):
                    raise
            else:
                raise RuntimeError("cross-semantics SA checkpoint unexpectedly loaded")
    # Isolate the two event auxiliary consumers. Both must have zero gradient on missing states.
    isolated = []
    for event_weight, temporal_weight in ((1.0, 0.0), (0.0, 0.7)):
        for agent in (legacy, candidate):
            agent.agent._auxiliary_event_weight = event_weight
            agent.agent._temporal_consistency_coef = temporal_weight
        _, old_grad = _grad(legacy, [missing[0]])
        _, new_grad = _grad(candidate, [missing[0]])
        if torch.count_nonzero(old_grad[0]["event_logits"]) == 0 or torch.count_nonzero(new_grad[0]["event_logits"]) != 0:
            raise RuntimeError("event CE or temporal margin abstention failed")
        isolated.append({"event_ce_weight": event_weight, "temporal_margin_weight": temporal_weight,
                         "legacy_event_grad_nonzero": True, "candidate_event_grad_zero": True})
    # The original eligible-sample denominator is retained rather than scaling by the ready count.
    legacy.agent._auxiliary_event_weight = candidate.agent._auxiliary_event_weight = 1.0
    legacy.agent._temporal_consistency_coef = candidate.agent._temporal_consistency_coef = 0.35
    _, ready_single = _grad(candidate, [ready[0]])
    _, mixed_grads = _grad(candidate, [ready[0], missing[0]])
    if not torch.allclose(mixed_grads[0]["event_logits"], ready_single[0]["event_logits"] / 2.0,
                          atol=1e-8, rtol=1e-7):
        raise RuntimeError("event auxiliary denominator renormalized by ready subset")
    if torch.count_nonzero(mixed_grads[1]["event_logits"]) != 0:
        raise RuntimeError("mixed batch missing sample receives event gradient")
    all_missing_loss, all_missing_grads = _grad(candidate, [missing[0], missing[1]])
    if not math.isfinite(all_missing_loss) or any(
        torch.count_nonzero(item["event_logits"]) != 0 for item in all_missing_grads
    ):
        raise RuntimeError("all-missing event auxiliary loss is unstable or supervised")
    event_logits = torch.tensor([-0.25, 0.45], requires_grad=True)
    selected_action = torch.tensor(1)
    old_log_prob = Categorical(logits=event_logits.detach()).log_prob(selected_action)
    ratio = (Categorical(logits=event_logits).log_prob(selected_action) - old_log_prob).exp()
    (-torch.minimum(ratio, torch.clamp(ratio, 0.8, 1.2))).backward()
    if torch.count_nonzero(event_logits.grad) == 0:
        raise RuntimeError("executed-action PPO surrogate has zero event gradient")
    baseline_receipts = {}
    for method in ("ppo", "mappo", "dt_handoff_drl"):
        old = _build_learned(method, 7, legacy_config, popart_enabled=False)
        new = _build_learned(method, 7, candidate_config, popart_enabled=False)
        if new.agent._mechanism_aux_missing_current_event_abstention_enabled:
            raise RuntimeError(f"candidate flag leaked into {method} baseline")
        old._deterministic_action = new._deterministic_action = True
        old_state = old.agent._network.state_dict()
        new_state = new.agent._network.state_dict()
        if set(old_state) != set(new_state) or any(not torch.equal(old_state[key], new_state[key]) for key in old_state):
            raise RuntimeError(f"{method} baseline initial weights changed")
        observation, info = states[0][2], states[0][3]
        old_action, old_info = old.act(observation, deepcopy(info))
        new_action, new_info = new.act(observation, deepcopy(info))
        if old_action != new_action or old_info["env_action_probs"] != new_info["env_action_probs"]:
            raise RuntimeError(f"{method} baseline raw probability drift")
        baseline_receipts[method] = {"flag_disabled": True, "network_state_keys": len(old_state),
                                     "raw_action_probability_equal": True}
    env_path = B_ROOT / "src/envs/core/calibrated_continuous_workflow_env.py"
    if _sha256(env_path) != "c6ded6ce84230519160a40f29e276b5823eb204a35e31c35f914d1c6bfee3a59":
        raise RuntimeError("shared environment implementation changed from frozen science")
    receipt = {"schema_version": "cscwd_event_aux_abstention_independent_audit_v1",
               "created_at": datetime.now(timezone.utc).isoformat(), "status": "PASS",
               "implementation_commit": commit, "implementation_tree": tree,
               "event_target_semantics": "missing_current_event_abstention_v1", "sa_only": True,
               "candidate_commit": commit, "candidate_tree": tree,
               "candidate_profile": "calibrated_workflow_interface_v4_prepared_state_prefix",
               "source_analysis_manifest_sha256": _sha256(SOURCE / "analysis_manifest.json"),
               "source_snapshot_export_manifest_sha256": _sha256(SOURCE / "snapshot_export_manifest.json"),
               "source_rows": len(states), "forward_equal_rows": forward_equal,
               "ready_gradient_identical_rows": gradient_equal_ready,
               "missing_event_gradient_zero_rows": gradient_zero_missing,
               "effective_supervision_stats": stats, "isolated_event_terms": isolated,
               "fixed_denominator_half_ready_gradient": True,
               "all_missing_auxiliary_finite_event_gradient_zero": True,
               "executed_action_ppo_surrogate_event_gradient_nonzero": True,
               "baseline_invariance": baseline_receipts,
               "old_profile_rejected": True, "cross_semantics_checkpoint_rejected": True,
               "shared_environment_sha256": _sha256(env_path),
               "new_training_steps": 0, "new_env_steps": 0, "formal_or_holdout_reads": 0}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT / "abstention_receipt.json", receipt)
    print(json.dumps({"status": "PASS", "candidate_commit": commit,
                      "source_rows": len(states), "supervised_fraction": stats["supervised_fraction"]}))


if __name__ == "__main__":
    main()
