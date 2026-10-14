"""Launch a reproducible, trace-driven eight-request joint-control experiment.

This is an internal simulation protocol. It uses original registry agents and
does not claim measured RSU resources, physical vehicles, or paper readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.registry import ALGO_REGISTRY, build_agent
from src.data.mobility.ngsim_provider import NGSIMProvider
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.envs.core.multi_request_joint_control_env import MultiRequestJointControlEnv
from src.envs.core.multi_request_replay_env import ReplayCosts, select_continuous_window
from src.evaluators.real_sample_support import build_sample_rsus, discover_ngsim_csv, summarize_mobility_frames

AGENTS = ("sa_ghmappo", "ppo", "mappo", "dqn", "dag_offload_drl", "cache_offload_drl", "dt_handoff_drl")
PROTOCOL = "eight_vehicle_joint_sim_v1_1"
PPO_AGENTS = {"sa_ghmappo", "ppo", "mappo", "dag_offload_drl", "cache_offload_drl", "dt_handoff_drl"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobility_csv_path", type=Path, default=discover_ngsim_csv(ROOT))
    parser.add_argument("--workflow_csv_path", type=Path, default=ROOT / "data/raw/workflow/alibaba2018/batch_task.csv")
    parser.add_argument("--max_mobility_rows", type=int, default=800_000)
    parser.add_argument("--window_length", type=int, default=120)
    parser.add_argument("--train_windows", type=int, default=20)
    parser.add_argument("--dev_windows", type=int, default=8)
    parser.add_argument("--formal_windows", type=int, default=12)
    parser.add_argument("--holdout_windows", type=int, default=12)
    parser.add_argument("--gap_windows", type=int, default=1)
    parser.add_argument("--holdout_gap_windows", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--update_every", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=list(AGENTS))
    parser.add_argument("--output_root", type=Path, default=ROOT / "artifacts/eight_vehicle_joint_sim")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Short plumbing check; output is not formal evidence")
    parser.add_argument("--environment_revision", choices=("v1", "v2"), default="v1")
    parser.add_argument("--development_only", action="store_true", help="Train/dev only; do not evaluate formal or holdout")
    parser.add_argument("--model_cache_config", type=Path, help="v2 controlled byte-cache JSON configuration")
    parser.add_argument("--vehicle_adapter_config", type=Path,
                        help="v2 JSON list of statically preloaded adapters for all vehicles, or per-vehicle map")
    parser.add_argument("--prediction_horizon_frames", type=int, default=3)
    parser.add_argument("--disable_prediction", action="store_true")
    parser.add_argument("--strict_cold_cache_mask", action="store_true",
                        help="v2 diagnostic: allow steady offload only after the current adapter is ready")
    parser.add_argument("--disable_shared_state", action="store_true")
    parser.add_argument("--shared_state_delay_frames", type=int, default=1)
    parser.add_argument("--shared_state_interval_frames", type=int, default=1)
    parser.add_argument("--sa_auxiliary_coef", type=float, default=0.0,
                        help="v2 explicit auxiliary ablation; default is the corrected PPO objective alone")
    parser.add_argument("--rsu_work_units_per_frame", type=float, default=1.0)
    parser.add_argument("--vehicle_work_units_per_frame", type=float, default=0.25)
    parser.add_argument("--workload_bytes_per_unit", type=float, default=1024.0)
    parser.add_argument("--workflow_profile_unit_bytes", type=int, default=1)
    parser.add_argument("--state_bytes_per_link_frame", type=int, default=1024)
    return parser.parse_args()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def _load_protocol(args: argparse.Namespace) -> tuple[dict[str, list[dict]], list[Any], list[Any]]:
    if min(args.train_windows, args.dev_windows, args.formal_windows, args.holdout_windows,
           args.episodes, args.update_every) < 1 or args.window_length < 2:
        raise ValueError("all split and training counts must be positive")
    if args.update_every > args.episodes:
        raise ValueError("update_every cannot exceed episodes")
    if args.gap_windows < 0 or args.holdout_gap_windows < 0:
        raise ValueError("split gaps must be nonnegative")
    if not args.smoke and not getattr(args, "development_only", False) and (
        args.formal_windows < 12 or args.holdout_windows < 12 or len(args.seeds) < 3
    ):
        raise ValueError("formal protocol requires >=12 formal/holdout windows and >=3 seeds")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be distinct")
    for path in (args.mobility_csv_path, args.workflow_csv_path):
        if not path.is_file() or path.stat().st_size < 1024:
            raise FileNotFoundError(f"raw data missing or an LFS pointer: {path}")
    frames = NGSIMProvider(args.mobility_csv_path, max_rows=args.max_mobility_rows).get_loaded_frames()
    workflows = WorkflowDatasetBuilder().build_selected_alibaba_workflow_states(
        csv_path=args.workflow_csv_path, max_workflows=8, workflow_selector="ordered",
        min_tasks=5, max_tasks=20, adapter_assignment_profile="semantic_ai_service",
    )
    if len(workflows) != 8 or len({w.workflow_id for w in workflows}) != 8:
        raise RuntimeError("eight distinct Alibaba DAGs are required")
    adapter_ids = {node.required_adapter for workflow in workflows for node in workflow.nodes}
    if len(adapter_ids) < 3:
        raise RuntimeError("adapter competition is unidentifiable with fewer than three labels")
    needed = (sum((args.train_windows, args.dev_windows, args.formal_windows, args.holdout_windows))
              + 2 * args.gap_windows + args.holdout_gap_windows)
    candidates = []
    for offset in range(0, len(frames) - args.window_length + 1, args.window_length):
        try:
            selected, vehicle_ids, _ = select_continuous_window(
                frames[offset:offset + args.window_length], vehicle_count=8, window_length=args.window_length,
            )
        except RuntimeError:
            continue
        candidates.append({
            "frame_offset": offset,
            "time_start": int(selected[0]["time_index"]),
            "time_end": int(selected[-1]["time_index"]),
            "source_segment_id": selected[0].get("source_segment_id"),
            "vehicle_ids": vehicle_ids,
            "frames": selected,
        })
        if len(candidates) >= needed:
            break
    if len(candidates) < needed:
        raise RuntimeError(f"only {len(candidates)} eligible windows; need {needed}")
    cuts = [args.train_windows, args.dev_windows, args.formal_windows, args.holdout_windows]
    splits: dict[str, list[dict]] = {}
    position = 0
    for index, (name, count) in enumerate(zip(("train", "dev", "formal", "holdout"), cuts)):
        splits[name] = candidates[position:position + count]
        position += count + (args.gap_windows if index < 2 else
                             args.holdout_gap_windows if index == 2 else 0)
    all_windows = [window for group in splits.values() for window in group]
    for index, left in enumerate(all_windows):
        for right in all_windows[index + 1:]:
            if left["source_segment_id"] == right["source_segment_id"] and not (
                left["time_end"] < right["time_start"] or right["time_end"] < left["time_start"]
            ):
                raise RuntimeError("split raw time intervals overlap")
    training_frames = [frame for window in splits["train"] for frame in window["frames"]]
    axis = summarize_mobility_frames(training_frames)["dominant_axis"]
    rsus, _ = build_sample_rsus(training_frames, rsu_layout=f"custom:axis={axis},count=4,coverage_factor=1.2")
    if len(rsus) != 4:
        raise RuntimeError("four simulated RSUs are required")
    return splits, workflows, rsus


def _environment(window: dict, workflows: list[Any], rsus: list[Any],
                 environment_options: dict | None = None) -> MultiRequestJointControlEnv:
    return MultiRequestJointControlEnv(
        frames=window["frames"], vehicle_workflows=dict(zip(window["vehicle_ids"], workflows)),
        rsu_states=rsus, costs=ReplayCosts(compute_slots_per_rsu=1, adapter_slots_per_rsu=2,
                                          link_slots_per_rsu=1, adapter_fetch_frames=1, handoff_state_frames=1),
        **(environment_options or {}),
    )


def _episode(agent: Any, window: dict, workflows: list[Any], rsus: list[Any],
             *, deterministic: bool, collect: bool,
             environment_options: dict | None = None) -> tuple[dict, list[dict], list[dict]]:
    env = _environment(window, workflows, rsus, environment_options)
    trajectories: dict[str, list[dict]] = defaultdict(list)
    team_rewards: list[float] = []
    total_reward = 0.0
    while True:
        states = env.decision_states()
        actions: dict[str, int] = {}
        action_info: dict[str, dict] = {}
        decision_info: dict[str, dict] = {}
        for vehicle_id, info in sorted(states.items()):
            policy_info = {**info, "deterministic_policy": deterministic,
                           "run_metadata": {"policy_evaluation_mode": "raw_policy"}}
            action, details = agent.act([], policy_info)
            actions[vehicle_id] = int(action)
            action_info[vehicle_id] = dict(details)
            decision_info[vehicle_id] = policy_info
        _, reward, done, _ = env.step_actions(actions)
        total_reward += reward
        team_rewards.append(float(reward))
        following = env.decision_states() if not done else {}
        if collect:
            for vehicle_id, info in states.items():
                details = action_info[vehicle_id]
                trajectories[vehicle_id].append({
                    "observation": [], "action": actions[vehicle_id], "reward": reward,
                    "environment_reward": reward, "team_frame_index": len(team_rewards) - 1,
                    "terminated": done or vehicle_id not in following, "truncated": False,
                    "log_prob": float(details.get("log_prob", 0.0)),
                    "value": float(details.get("value", 0.0)), "next_value": 0.0,
                    "next_observation": [], "action_info": details,
                    "decision_info": decision_info[vehicle_id],
                    "env_info": following.get(vehicle_id, {}),
                })
        if done:
            break
    rows = []
    team_returns = [0.0] * len(team_rewards)
    future = 0.0
    for index in range(len(team_rewards) - 1, -1, -1):
        future = team_rewards[index] + 0.99 * future
        team_returns[index] = future
    for vehicle_rows in trajectories.values():
        # A finished request has no more actions, but its cache/state choices
        # still affect the team's later rewards. Preserve that tail for PPO's
        # return and for the terminal DQN transition (whose next state is absent).
        last = vehicle_rows[-1]
        next_index = last["team_frame_index"] + 1
        tail = 0.99 * team_returns[next_index] if next_index < len(team_returns) else 0.0
        last["team_terminal_continuation_reward"] = tail
        last["reward"] += tail
        for row in vehicle_rows:
            row["return"] = team_returns[row["team_frame_index"]]
            row["advantage"] = row["return"] - row["value"]
        rows.extend(vehicle_rows)
    summary = env.summary()
    summary["reward"] = round(total_reward, 6)
    summary["frame_offset"] = window["frame_offset"]
    summary["time_start"] = window["time_start"]
    summary["time_end"] = window["time_end"]
    summary["policy_evaluation_mode"] = "raw_policy"
    summary["action_selection"] = "greedy" if deterministic else "sample"
    summary["collected_decisions"] = len(rows)
    return summary, rows, env.events


def _environment_options(args: argparse.Namespace) -> dict:
    if args.environment_revision == "v1":
        return {"environment_revision": "v1"}
    options = {
        "environment_revision": "v2",
        "prediction_enabled": not args.disable_prediction,
        "strict_cold_cache_mask": args.strict_cold_cache_mask,
        "prediction_horizon_frames": args.prediction_horizon_frames,
        "shared_state_enabled": not args.disable_shared_state,
        "shared_state_delay_frames": args.shared_state_delay_frames,
        "shared_state_interval_frames": args.shared_state_interval_frames,
        "rsu_work_units_per_frame": args.rsu_work_units_per_frame,
        "vehicle_work_units_per_frame": args.vehicle_work_units_per_frame,
        "workload_bytes_per_unit": args.workload_bytes_per_unit,
        "workflow_profile_unit_bytes": args.workflow_profile_unit_bytes,
        "state_bytes_per_link_frame": args.state_bytes_per_link_frame,
    }
    if args.model_cache_config is not None:
        options["model_cache_config"] = json.loads(args.model_cache_config.read_text(encoding="utf-8"))
    if args.vehicle_adapter_config is not None:
        options["vehicle_adapter_ids"] = json.loads(args.vehicle_adapter_config.read_text(encoding="utf-8"))
    return options


def _agent_options(name: str, args: argparse.Namespace) -> dict:
    if args.environment_revision != "v2":
        return {}
    options = {"joint_observation_enabled": True}
    if name in PPO_AGENTS:
        options["actor_objective"] = "masked_env_action_v1"
    if name == "sa_ghmappo":
        options.update(auxiliary_coef=args.sa_auxiliary_coef, event_logit_temperature=1.0,
                       event_logit_temperature_final=1.0, event_temperature_decay_updates=0,
                       event_logit_sharpening_final_scale=1.0)
    return options


def main() -> None:
    args = arguments()
    if args.strict_cold_cache_mask and args.environment_revision != "v2":
        raise ValueError("strict_cold_cache_mask requires environment revision v2")
    if args.environment_revision == "v2" and not (args.development_only or args.smoke or args.preflight_only):
        raise ValueError("v2 is development-only until a new independent evaluation protocol is frozen; use --development_only")
    if not np.isfinite(args.sa_auxiliary_coef) or args.sa_auxiliary_coef < 0:
        raise ValueError("sa_auxiliary_coef must be finite and nonnegative")
    if len(set(args.agents)) != len(args.agents):
        raise ValueError("agents must be distinct")
    for name in args.agents:
        if name not in ALGO_REGISTRY or ALGO_REGISTRY[name]["support_level"] != "trainable":
            raise ValueError(f"not a registered trainable algorithm: {name}")
    splits, workflows, rsus = _load_protocol(args)
    environment_options = _environment_options(args)
    # Instantiate before writing any run manifest to reject invalid configuration.
    probe_env = _environment(splits["train"][0], workflows, rsus, environment_options)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_{'smoke' if args.smoke else 'development' if args.development_only else 'formal_sim'}"
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    public_splits = {name: [{key: value for key, value in window.items() if key != "frames"}
                            for window in windows] for name, windows in splits.items()}
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "run_id": run_id, "protocol": PROTOCOL if args.environment_revision == "v1" else (
            "eight_vehicle_joint_sim_v2_cold_mask_development" if args.strict_cold_cache_mask
            else "eight_vehicle_joint_sim_v2_development"),
        "action_contract": probe_env.action_contract, "environment_revision": args.environment_revision,
        "environment_options": environment_options,
        "observation_contract": getattr(probe_env, "observation_contract", "legacy_semantic_v1"),
        "agent_options": {name: _agent_options(name, args) for name in args.agents},
        "training_contract": "raw_policy_rollout_metadata_preserved_v2",
        "return_contract": "discounted_team_return_to_episode_end_with_terminal_request_tail",
        "development_only": args.development_only or args.smoke,
        "evidence_scope": "internal_trace_driven_simulation; not physical RSUs or measured links",
        "paper_claim_ready": False, "holdout_status": "sealed_not_evaluated",
        "holdout_exclusion_note": "Earlier proposed offsets 5400..6720 were inspected for mobility handoff opportunities before freeze; this run uses later uninspected intervals.",
        "git_commit": git_commit, "command": sys.argv, "agents": args.agents, "seeds": args.seeds,
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "episodes_per_agent_seed": args.episodes, "update_every": args.update_every,
        "window_length": args.window_length, "max_mobility_rows": args.max_mobility_rows,
        "mobility_csv_path": str(args.mobility_csv_path.resolve()),
        "workflow_csv_path": str(args.workflow_csv_path.resolve()),
        "raw_data_sha256": {"ngsim": _hash_file(args.mobility_csv_path),
                            "alibaba": _hash_file(args.workflow_csv_path)},
        "adapter_assignment_profile": "semantic_ai_service_synthetic_from_dag_structure",
        "workflow_ids": [workflow.workflow_id for workflow in workflows],
        "adapter_ids": sorted({node.required_adapter for workflow in workflows for node in workflow.nodes}),
        "rsu_layout_source": "train_windows_only", "rsu_layout": [rsu.to_dict() for rsu in rsus],
        "costs": {"compute_slots_per_rsu": 1, "adapter_slots_per_rsu": 2,
                  "link_slots_per_rsu": 1, "adapter_fetch_frames": 1,
                  "handoff_state_frames": 1, "vehicle_compute_frames": 4,
                  "cost_source": "configuration_assumptions_not_measured"},
        "reward": "completed_nodes - 0.01*unfinished_requests - 0.02*vehicle_compute_frames - 0.01*adapter_fetches - 0.01*state_transfer_frames",
        "splits": public_splits,
        "limitations": ["cross-source NGSIM/Alibaba pairing is synthetic",
                        "first-party direction agents are not reproductions of outside papers",
                        "same NGSIM segment can correlate nonoverlapping windows",
                        "equal episodes do not imply equal decision counts",
                        "v2 byte sizes, work units and shared JSON wire cost are controlled proxies",
                        "shared controller actors are not full vehicle/RSU-level MARL"],
    }
    if args.environment_revision == "v2":
        manifest["costs"] = probe_env.configuration()
        manifest["reward"] = "completed_nodes minus unfinished/vehicle/model-transfer/state-transfer/shared-information costs; see environment configuration"
        work = [probe_env._work_required(node) for workflow in workflows for node in workflow.nodes]
        working_set = sum(obj.resident_size_bytes for obj in probe_env.model_cache.objects.values())
        manifest["mechanism_qualification"] = {
            "scope": "static_input_check_not_realized_performance",
            "node_work_units_min": min(work), "node_work_units_max": max(work),
            "node_work_units_unique": sorted(set(work)),
            "multi_frame_rsu_node_fraction": sum(value > probe_env.rsu_work_units_per_frame for value in work) / len(work),
            "model_working_set_bytes": working_set,
            "model_cache_capacity_by_rsu": probe_env.model_cache.capacity_bytes,
            "cache_pressure_possible_by_rsu": {rsu: working_set > capacity
                                               for rsu, capacity in probe_env.model_cache.capacity_bytes.items()},
        }
    _write_json(output / "manifest.json", manifest)
    print(f"PREFLIGHT_OK run_id={run_id} output={output} splits="
          f"{','.join(f'{key}:{len(value)}' for key, value in splits.items())}", flush=True)
    if "mechanism_qualification" in manifest:
        print("MECHANISM_INPUT_CHECK " + json.dumps(manifest["mechanism_qualification"], sort_keys=True), flush=True)
    if args.preflight_only:
        return
    for seed in args.seeds:
        for name in args.agents:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            agent = build_agent(name, random_seed=seed, train_epochs=2, batch_size=64,
                                **_agent_options(name, args))
            pending: list[dict] = []
            for episode_index in range(args.episodes):
                window = splits["train"][episode_index % len(splits["train"])]
                summary, rows, _ = _episode(agent, window, workflows, rsus,
                                            deterministic=False, collect=True, environment_options=environment_options)
                pending.extend(rows)
                _append_jsonl(output / "train.jsonl", {"agent": name, "seed": seed,
                              "episode": episode_index, **summary})
                if (episode_index + 1) % args.update_every == 0 or episode_index + 1 == args.episodes:
                    update = agent.learn(pending)
                    _append_jsonl(output / "updates.jsonl", {"agent": name, "seed": seed,
                                  "episode": episode_index, "sample_count": len(pending),
                                  "update": update})
                    pending = []
                if episode_index == 0 or (episode_index + 1) % args.update_every == 0:
                    print(f"TRAIN agent={name} seed={seed} episode={episode_index+1}/{args.episodes} "
                          f"completed={summary['completed_requests']}/8", flush=True)
            checkpoint = output / "checkpoints" / f"{name}_seed{seed}.pt"
            agent.save(str(checkpoint))
            _append_jsonl(output / "checkpoints.jsonl", {"agent": name, "seed": seed,
                          "path": str(checkpoint), "sha256": _hash_file(checkpoint),
                          "environment_revision": args.environment_revision,
                          "observation_contract": manifest["observation_contract"],
                          "agent_options": _agent_options(name, args)})
            for split_name in ("dev", "formal"):
                if (args.smoke or args.development_only) and split_name == "formal":
                    continue
                for window in splits[split_name]:
                    summary, _, events = _episode(agent, window, workflows, rsus,
                                                  deterministic=True, collect=False, environment_options=environment_options)
                    _append_jsonl(output / f"{split_name}.jsonl", {"agent": name, "seed": seed, **summary})
                    if split_name == "dev" and args.environment_revision == "v2":
                        for event in events:
                            _append_jsonl(output / "dev_events.jsonl", {"agent": name, "seed": seed,
                                          "action_selection": "greedy", "window_frame_offset": window["frame_offset"], **event})
                        sampled, _, sampled_events = _episode(agent, window, workflows, rsus,
                                                deterministic=False, collect=False,
                                                environment_options=environment_options)
                        _append_jsonl(output / "dev_sampled.jsonl", {"agent": name, "seed": seed, **sampled})
                        for event in sampled_events:
                            _append_jsonl(output / "dev_events.jsonl", {"agent": name, "seed": seed,
                                          "action_selection": "sample", "window_frame_offset": window["frame_offset"], **event})
                    if split_name == "formal":
                        for event in events:
                            _append_jsonl(output / "formal_events.jsonl",
                                          {"agent": name, "seed": seed,
                                           "window_frame_offset": window["frame_offset"], **event})
            print(f"EVALUATED agent={name} seed={seed}", flush=True)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["checkpoint_count"] = len(args.agents) * len(args.seeds)
    _write_json(output / "manifest.json", manifest)
    print(f"RUN_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
