"""Audit public causal rules against the actual raw event-time environment.

The audit is deliberately result-blind and training-free.  It constructs two
synthetic traces with the same observed prefix and different hidden suffixes,
then checks that public rules and an untrained PPO distribution are invariant.
The historical exact-clone rules are retained only as privileged references.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
from typing import Any


SCHEMA_VERSION = "cscwd_public_rule_raw_fairness_audit_v1"
DEFAULT_OUTPUT = "artifacts/analysis/cscwd_public_rule_raw_fairness_20261011_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_public_estimator(public_root: Path) -> Any:
    path = public_root / "src/agents/causal_public_action_estimator.py"
    spec = importlib.util.spec_from_file_location("_cscwd_public_action_estimator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load public estimator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_lightweight_project_namespaces(raw_root: Path) -> None:
    """Avoid unrelated eager imports while loading the isolated raw env.

    Some package ``__init__`` modules import training-only PyTorch components.
    The raw environment contract itself needs only the named environment and
    schema modules, so namespace packages keep this audit runnable in a
    data/audit-only runtime.  A full PPO numeric check remains conditional on
    the training runtime being available.
    """

    for name, path in (("src.envs.core", raw_root / "src/envs/core"),):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _source_guard(env: Any) -> dict[str, Any]:
    return {
        "caches": deepcopy(env.caches),
        "prepared_state": deepcopy(env.prepared_state),
        "clock_seconds": float(env.clock_seconds),
        "step_index": int(env.step_index),
        "metrics": deepcopy(env.metrics),
        "rng_state": deepcopy(env._rng.bit_generator.state),
    }


def _preview(env: Any, action: int) -> dict[str, Any]:
    preview = env.clone_for_decision_model()
    _, _, terminated, truncated, info = preview.step(action)
    transition = dict(info["transition"])
    return {
        "action": int(action),
        "service_completed": bool(transition.get("service_completed", False)),
        "admission_rejection_reason": transition.get("admission_rejection_reason"),
        "step_cost_seconds": float(transition["step_cost_seconds"]),
        "clock_seconds_after": float(transition["clock_seconds_after"]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "physical_contact_budget_seconds": float(env._physical_contact_budget_seconds()),
    }


def _ppo_distribution(agent: Any, info: dict[str, Any], action: int) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        policy_output = agent._forward_policy(
            info["semantic_state"],
            run_metadata=info["run_metadata"],
        )
        log_prob, entropy, probabilities = agent._env_action_distribution_statistics(
            policy_output=policy_output,
            env_action=action,
            action_mask=info["action_mask"],
        )
    return {
        "action": int(action),
        "probabilities": probabilities,
        "selected_action_log_probability": float(log_prob.item()),
        "entropy": float(entropy.item()),
    }


def _build_trace_pair(raw_trace_type: Any) -> dict[str, Any]:
    times = tuple(round(index * 0.1, 9) for index in range(101))
    shared_prefix = ((0.0, 0.0), (0.0, 0.02))
    slow_suffix = tuple((0.0, 0.02 + 0.02 * (index - 1)) for index in range(2, 101))
    fast_suffix = tuple((0.0, 100.0) for _ in range(2, 101))
    return {
        "slow_hidden_suffix": raw_trace_type(
            times,
            shared_prefix + slow_suffix,
            "same_public_prefix_vehicle",
        ),
        "fast_hidden_suffix": raw_trace_type(
            times,
            shared_prefix + fast_suffix,
            "same_public_prefix_vehicle",
        ),
    }


def _execute_selected(env_type: Any, config: dict[str, Any], row: dict[str, Any], trace: Any, action: int) -> dict[str, Any]:
    env = env_type(config, row, trace)
    mask = list(env._info()["action_mask"])
    _, _, terminated, truncated, info = env.step(action)
    transition = dict(info["transition"])
    return {
        "requested_action": int(action),
        "requested_action_legal": bool(mask[action]),
        "executed_action": int(transition["action"]),
        "action_projection_applied": int(transition["action"]) != int(action),
        "service_completed": bool(transition.get("service_completed", False)),
        "admission_rejection_reason": transition.get("admission_rejection_reason"),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def run_audit(raw_root: Path, public_root: Path) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    public_root = public_root.resolve()
    if str(raw_root) not in sys.path:
        sys.path.insert(0, str(raw_root))
    _install_lightweight_project_namespaces(raw_root)

    from scripts.freeze_calibrated_continuous_workflow_pilot import (  # noqa: E402
        _load_experiment_config,
    )
    from src.envs.core.calibrated_continuous_workflow_env import (  # noqa: E402
        ImmediateCostRule,
        TwoStepCostRule,
    )
    from src.envs.core.raw_ngsim_event_time_env import (  # noqa: E402
        RawNGSIMEventTimeEnv,
        RawVehicleTrace,
    )

    estimator = _load_public_estimator(public_root)
    config_path = raw_root / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json"
    manifest_path = raw_root / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json"
    config, _ = _load_experiment_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    row = deepcopy(next(item for item in manifest["instances"] if item["design_id"] == "dev_01"))
    traces = _build_trace_pair(RawVehicleTrace)
    envs = {
        name: RawNGSIMEventTimeEnv(config, deepcopy(row), trace)
        for name, trace in traces.items()
    }
    infos = {name: env._info() for name, env in envs.items()}
    names = list(envs)
    public_prefix_equal = infos[names[0]] == infos[names[1]]
    if not public_prefix_equal:
        raise AssertionError("hidden suffix changed RawNGSIMEventTimeEnv._info()")

    public_rule_types = (
        estimator.CausalPublicImmediateRule,
        estimator.CausalPublicTwoStepRule,
    )
    public_rules: dict[str, Any] = {}
    for rule_type in public_rule_types:
        actions: dict[str, int] = {}
        side_effect_free: dict[str, bool] = {}
        for name, env in envs.items():
            before = _source_guard(env)
            actions[name] = int(rule_type().select_action(env))
            side_effect_free[name] = _source_guard(env) == before
        invariant = len(set(actions.values())) == 1
        if not invariant or not all(side_effect_free.values()):
            raise AssertionError(f"public rule contract failed: {rule_type.method_name}")
        public_rules[rule_type.method_name] = {
            "capability_profile": rule_type.capability_profile,
            "objective_profile": rule_type.objective_profile,
            "actions": actions,
            "hidden_suffix_invariant": invariant,
            "source_state_unchanged": side_effect_free,
        }

    privileged_rules: dict[str, Any] = {}
    for rule_type in (ImmediateCostRule, TwoStepCostRule):
        actions = {name: int(rule_type().select_action(env)) for name, env in envs.items()}
        declared = estimator.PRIVILEGED_REFERENCE_PROFILES[rule_type.method_name]
        privileged_rules[rule_type.method_name] = {
            "capability_profile": declared["capability_profile"],
            "objective_profile": declared["objective_profile"],
            "public_information_matched": declared["public_information_matched"],
            "uses_environment_preview": declared["uses_environment_preview"],
            "actions": actions,
            "hidden_suffix_invariant": len(set(actions.values())) == 1,
        }

    privileged_action3_previews = {
        name: _preview(env, 3) for name, env in envs.items()
    }
    preview_changed = (
        privileged_action3_previews[names[0]]["service_completed"]
        != privileged_action3_previews[names[1]]["service_completed"]
        or privileged_action3_previews[names[0]]["admission_rejection_reason"]
        != privileged_action3_previews[names[1]]["admission_rejection_reason"]
    )
    if not preview_changed:
        raise AssertionError("privileged exact preview did not expose hidden-suffix contrast")

    selected_action = public_rules["causal_public_immediate_rule"]["actions"][names[0]]
    ppo_input_identity = all(
        infos[names[0]][field] == infos[names[1]][field]
        for field in ("semantic_state", "action_mask", "run_metadata")
    )
    ppo_numeric_status = "not_run"
    ppo_numeric_reason = None
    ppo_rows: dict[str, Any] = {}
    ppo_invariant: bool | None = None
    ppo_runtime = {
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "torch_version": None,
        "torch_module_path": None,
    }
    try:
        import torch

        ppo_runtime["torch_version"] = str(torch.__version__)
        ppo_runtime["torch_module_path"] = str(torch.__file__)

        from scripts.run_calibrated_workflow_interface_repair import (  # noqa: E402
            _build_agent,
        )

        ppo_config = deepcopy(config)
        ppo_config["interface_profile"] = infos[names[0]]["semantic_state"]["interface_profile"]
        ppo = _build_agent(
            "ppo",
            7,
            ppo_config,
            agent_overrides={"value_normalization_enabled": False},
        )
        ppo_rows = {
            name: _ppo_distribution(ppo, info, selected_action)
            for name, info in infos.items()
        }
        ppo_invariant = ppo_rows[names[0]] == ppo_rows[names[1]]
        ppo_numeric_status = "passed" if ppo_invariant else "failed"
        if not ppo_invariant:
            raise AssertionError("PPO distribution changed under a hidden-suffix perturbation")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        ppo_numeric_status = "unavailable"
        ppo_numeric_reason = "training runtime has no torch module"

    executions: dict[str, Any] = {}
    for rule_name, rule_row in public_rules.items():
        action = rule_row["actions"][names[0]]
        executions[rule_name] = {
            name: _execute_selected(
                RawNGSIMEventTimeEnv,
                config,
                deepcopy(row),
                trace,
                action,
            )
            for name, trace in traces.items()
        }
        if any(
            not item["requested_action_legal"] or item["action_projection_applied"]
            for item in executions[rule_name].values()
        ):
            raise AssertionError(f"mask/executed-action mismatch for {rule_name}")

    return {
        "schema_version": SCHEMA_VERSION,
        "audit_scope": "synthetic_hidden_suffix_actual_raw_env_no_training_no_selection",
        "raw_environment": "RawNGSIMEventTimeEnv",
        "actual_info_method_used": True,
        "design_id": "dev_01",
        "observed_prefix_samples": 2,
        "hidden_suffix_variants": names,
        "public_prefix_equal": public_prefix_equal,
        "public_action_masks_equal": infos[names[0]]["action_mask"] == infos[names[1]]["action_mask"],
        "public_rules": public_rules,
        "ppo_probability_consistency": {
            "seed": 7,
            "checkpoint_loaded": False,
            "parameter_updates": 0,
            "selected_public_action": selected_action,
            "public_policy_input_identity": ppo_input_identity,
            "rows": ppo_rows,
            "numeric_check_status": ppo_numeric_status,
            "numeric_check_reason": ppo_numeric_reason,
            "numeric_hidden_suffix_invariant": ppo_invariant,
            "runtime_identity": ppo_runtime,
            "claim_boundary": (
                "input identity is confirmed; numerical probability equality is claimed "
                "only when numeric_check_status is passed"
            ),
        },
        "executed_action_consistency": executions,
        "privileged_references": privileged_rules,
        "privileged_action3_preview_contrast": {
            "changed": preview_changed,
            "rows": privileged_action3_previews,
            "interpretation": (
                "exact transition preview reads hidden realized contact; selected privileged "
                "actions need not differ for the capability contrast to be present"
            ),
        },
        "claim_boundary": {
            "training_runs": 0,
            "evaluation_matrix_runs": 0,
            "checkpoint_selection": False,
            "raw_data_exported": False,
            "scientific_performance_claim": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    script_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--raw-source-root", type=Path, required=True)
    parser.add_argument("--public-rule-source-root", type=Path, default=script_root)
    parser.add_argument("--output-dir", type=Path, default=script_root / DEFAULT_OUTPUT)
    args = parser.parse_args()

    raw_root = args.raw_source_root.resolve()
    public_root = args.public_rule_source_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = run_audit(raw_root, public_root)
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_paths = {
        "public_estimator": public_root / "src/agents/causal_public_action_estimator.py",
        "privileged_rule_runtime": raw_root / "src/envs/core/calibrated_continuous_workflow_env.py",
        "raw_environment": raw_root / "src/envs/core/raw_ngsim_event_time_env.py",
        "audit_script": Path(__file__).resolve(),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_run_id": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "literature_cutoff": "2026-10-11",
        "target_venue": "top-journal-methodology-support",
        "policy_version": "tmc_review_policy_v3_20260621",
        "evidence_level": "L2_bounded_actual_environment_contract_audit",
        "git_commits": {
            "public_rule_source": _git_commit(public_root),
            "raw_environment_source": _git_commit(raw_root),
        },
        "inputs": {
            "raw_source_root": str(raw_root),
            "public_rule_source_root": str(public_root),
            "synthetic_trace_only": True,
            "real_ngsim_rows": 0,
        },
        "files": {
            "audit.json": _sha256(audit_path),
            **{
                str(path.relative_to(root)): _sha256(path)
                for (_, path), root in zip(
                    source_paths.items(),
                    (public_root, raw_root, raw_root, public_root),
                )
            },
        },
        "claim_boundary": audit["claim_boundary"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"audit": str(audit_path), "manifest": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
