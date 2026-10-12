"""Train one exploratory shared-actor PPO run on eight-vehicle replay windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.ppo_agent import TraceDrivenMultiRequestPPO
from src.data.mobility.ngsim_provider import NGSIMProvider
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.envs.core.multi_request_replay_env import (
    MultiRequestReplayEnv,
    ReplayCosts,
    select_continuous_window,
)
from src.evaluators.real_sample_support import (
    build_sample_rsus,
    discover_ngsim_csv,
    summarize_mobility_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobility_csv_path", type=Path, default=None)
    parser.add_argument(
        "--workflow_csv_path", type=Path,
        default=ROOT_DIR / "data/raw/workflow/alibaba2018/batch_task.csv",
    )
    parser.add_argument("--max_mobility_rows", type=int, default=250_000)
    parser.add_argument("--window_length", type=int, default=120)
    parser.add_argument("--train_windows", type=int, default=24)
    parser.add_argument("--eval_windows", type=int, default=8)
    parser.add_argument("--gap_windows", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=320)
    parser.add_argument("--update_every", type=int, default=8)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--reward_scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rsu_coverage_factor", type=float, default=0.75)
    parser.add_argument(
        "--output_root", type=Path,
        default=ROOT_DIR / "artifacts/training/eight_vehicle_ppo_diagnostic",
    )
    return parser.parse_args()


def _sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _prepare_windows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[Any]]:
    if min(args.train_windows, args.eval_windows, args.window_length, args.episodes) < 1:
        raise ValueError("window counts, window length and episodes must be positive")
    if min(args.update_every, args.ppo_epochs, args.minibatch_size) < 1:
        raise ValueError("PPO update counts and minibatch size must be positive")
    if not (0.0 < args.gamma <= 1.0 and args.reward_scale > 0.0):
        raise ValueError("gamma and reward_scale must be positive")
    if args.gap_windows < 0 or args.rsu_coverage_factor <= 0:
        raise ValueError("gap_windows must be nonnegative and coverage factor positive")
    mobility_path = args.mobility_csv_path or discover_ngsim_csv(ROOT_DIR)
    frames = NGSIMProvider(mobility_path, max_rows=args.max_mobility_rows).get_loaded_frames()
    workflows = WorkflowDatasetBuilder().build_selected_alibaba_workflow_states(
        csv_path=args.workflow_csv_path,
        max_workflows=8,
        workflow_selector="ordered",
        min_tasks=5,
        max_tasks=20,
        adapter_assignment_profile="legacy_batch_type",
    )
    if len(workflows) != 8:
        raise RuntimeError("eight distinct Alibaba DAGs are required")
    needed = args.train_windows + args.gap_windows + args.eval_windows
    windows = []
    for offset in range(0, len(frames) - args.window_length + 1, args.window_length):
        try:
            selected, vehicle_ids, _ = select_continuous_window(
                frames[offset : offset + args.window_length],
                vehicle_count=8,
                window_length=args.window_length,
            )
        except RuntimeError:
            continue
        axis = summarize_mobility_frames(selected)["dominant_axis"]
        layout = f"custom:axis={axis},count=4,coverage_factor={args.rsu_coverage_factor}"
        rsus, _ = build_sample_rsus(selected, rsu_layout=layout)
        if len(rsus) != 4:
            raise RuntimeError("expected four simulated RSUs")
        identity = [{
            "time_index": frame["time_index"],
            "vehicles": [(vehicle.vehicle_id, vehicle.position_x, vehicle.position_y) for vehicle in frame["vehicles"]],
        } for frame in selected]
        windows.append({
            "frame_offset": offset,
            "time_start": int(selected[0]["time_index"]),
            "time_end": int(selected[-1]["time_index"]),
            "source_segment_id": selected[0].get("source_segment_id"),
            "vehicle_ids": vehicle_ids,
            "trajectory_sha256": _sha256(identity),
            "rsu_layout": [rsu.to_dict() for rsu in rsus],
            "frames": selected,
            "rsus": rsus,
        })
        if len(windows) == needed:
            break
    if len(windows) < needed:
        raise RuntimeError(f"found {len(windows)} eligible nonoverlapping windows; need {needed}")
    train = windows[: args.train_windows]
    evaluation = windows[args.train_windows + args.gap_windows :]
    for left_index, left in enumerate(train + evaluation):
        for right in (train + evaluation)[left_index + 1 :]:
            if left["source_segment_id"] == right["source_segment_id"]:
                if not (left["time_end"] < right["time_start"] or right["time_end"] < left["time_start"]):
                    raise RuntimeError("train/eval raw time intervals overlap")
    return train + evaluation, workflows


def _environment(window: dict[str, Any], workflows: list[Any]) -> MultiRequestReplayEnv:
    return MultiRequestReplayEnv(
        frames=window["frames"],
        vehicle_workflows=dict(zip(window["vehicle_ids"], workflows)),
        rsu_states=window["rsus"],
        costs=ReplayCosts(),
    )


def _episode(
    env: MultiRequestReplayEnv,
    policy: TraceDrivenMultiRequestPPO | None,
    *, deterministic: bool,
    gamma: float,
    reward_scale: float,
    collect: bool,
) -> tuple[dict[str, Any], list[dict[str, torch.Tensor]]]:
    observation = env.reset()
    raw_reward = 0.0
    step_samples = []
    scaled_rewards = []
    while True:
        current_samples = []
        targets: dict[str, str] = {}
        if policy is not None:
            vehicle_ids, rsu_ids, features, mask = policy.encode(
                observation, window_length=len(env.frames)
            )
            if vehicle_ids:
                actions, old_logprobs, old_values = policy.act(
                    features, mask, deterministic=deterministic
                )
                targets = {
                    vehicle_id: rsu_ids[int(action)]
                    for vehicle_id, action in zip(vehicle_ids, actions)
                }
                if collect:
                    for index in range(len(vehicle_ids)):
                        current_samples.append({
                            "features": features[index],
                            "mask": mask[index],
                            "action": actions[index],
                            "old_logprob": old_logprobs[index],
                            "old_value": old_values[index],
                        })
        observation, reward, done, _ = env.step(targets)
        raw_reward += reward
        scaled_rewards.append(reward * reward_scale)
        step_samples.append(current_samples)
        if done:
            break
    samples = []
    if collect:
        future_return = 0.0
        for step_index in reversed(range(len(step_samples))):
            future_return = scaled_rewards[step_index] + gamma * future_return
            for sample in step_samples[step_index]:
                sample["return"] = torch.tensor(future_return, dtype=torch.float32)
                samples.append(sample)
    summary = env.summary()
    summary["reward"] = round(raw_reward, 6)
    return summary, samples


def _ppo_update(
    policy: TraceDrivenMultiRequestPPO,
    optimizer: torch.optim.Optimizer,
    samples: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> dict[str, float]:
    if not samples:
        raise RuntimeError("no policy decisions collected")
    features = torch.stack([item["features"] for item in samples])
    masks = torch.stack([item["mask"] for item in samples])
    actions = torch.stack([item["action"] for item in samples])
    old_logprobs = torch.stack([item["old_logprob"] for item in samples])
    returns = torch.stack([item["return"] for item in samples])
    old_values = torch.stack([item["old_value"] for item in samples])
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1.0e-6)
    last = {}
    for _ in range(args.ppo_epochs):
        for batch_ids in torch.randperm(len(samples)).split(args.minibatch_size):
            distribution, values = policy.distribution_and_value(features[batch_ids], masks[batch_ids])
            ratio = (distribution.log_prob(actions[batch_ids]) - old_logprobs[batch_ids]).exp()
            surrogate = ratio * advantages[batch_ids]
            clipped = ratio.clamp(1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages[batch_ids]
            actor_loss = -torch.minimum(surrogate, clipped).mean()
            critic_loss = (values - returns[batch_ids]).square().mean()
            entropy = distribution.entropy().mean()
            loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            last = {
                "actor_loss": float(actor_loss.detach()),
                "critic_loss": float(critic_loss.detach()),
                "entropy": float(entropy.detach()),
                "samples": len(samples),
            }
    return last


def _public_window(window: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in window.items() if key not in {"frames", "rsus"}}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    windows, workflows = _prepare_windows(args)
    train_windows = windows[: args.train_windows]
    eval_windows = windows[args.train_windows :]
    policy = TraceDrivenMultiRequestPPO()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True).strip()
    manifest = {
        "experiment_type": "cross_source_multi_request_ppo_training_diagnostic",
        "algorithm": "shared_actor_independent_decision_ppo",
        "existing_ppo_agent_contract_reused": False,
        "existing_sa_ghmappo_checkpoint_reused": False,
        "formal_protocol": False,
        "joint_vehicle_request_log": False,
        "physical_rsus": False,
        "model_inference_executed": False,
        "rsu_and_link_costs_measured": False,
        "simultaneous_arrival_assumed": True,
        "rsu_layout_uses_full_window_geometry": True,
        "service_frames_per_node": 1,
        "input_output_bytes_consumed": False,
        "git_commit": commit,
        "command": sys.argv,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "mobility_csv_path": str((args.mobility_csv_path or discover_ngsim_csv(ROOT_DIR)).resolve()),
        "workflow_csv_path": str(args.workflow_csv_path.resolve()),
        "max_mobility_rows": args.max_mobility_rows,
        "window_length": args.window_length,
        "scan_stride": args.window_length,
        "train_window_count": args.train_windows,
        "eval_window_count": args.eval_windows,
        "gap_eligible_windows": args.gap_windows,
        "training_episodes": args.episodes,
        "hyperparameters": {
            "update_every": args.update_every,
            "ppo_epochs": args.ppo_epochs,
            "minibatch_size": args.minibatch_size,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "clip_epsilon": args.clip_epsilon,
            "entropy_coef": args.entropy_coef,
            "value_coef": args.value_coef,
            "reward_scale_for_gradient": args.reward_scale,
        },
        "reward_contract": "completed_nodes - 0.01 * unfinished_requests per frame",
        "action_contract": "one covered RSU target per active vehicle per frame",
        "observation_features": list(policy.feature_names),
        "simulated_costs": vars(ReplayCosts()),
        "workflow_ids": [workflow.workflow_id for workflow in workflows],
        "workflow_sha256": _sha256([workflow.to_dict() for workflow in workflows]),
        "train_windows": [_public_window(window) for window in train_windows],
        "eval_windows": [_public_window(window) for window in eval_windows],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    train_rows = []
    batch = []
    for episode in range(1, args.episodes + 1):
        window_index = rng.randrange(len(train_windows))
        result, samples = _episode(
            _environment(train_windows[window_index], workflows), policy,
            deterministic=False, gamma=args.gamma, reward_scale=args.reward_scale, collect=True,
        )
        batch.extend(samples)
        row = {
            "episode": episode,
            "train_window_index": window_index,
            "frame_offset": train_windows[window_index]["frame_offset"],
            "reward": result["reward"],
            "completed_requests": result["completed_requests"],
            "completed_nodes": result["completed_nodes"],
            "steps": result["steps"],
        }
        if episode % args.update_every == 0 or episode == args.episodes:
            row["update"] = _ppo_update(policy, optimizer, batch, args)
            batch = []
        train_rows.append(row)
        if episode % max(args.update_every * 5, 1) == 0 or episode == args.episodes:
            print(f"episode={episode}/{args.episodes} reward={row['reward']:.3f} completed={row['completed_requests']}/8", flush=True)

    checkpoint_path = output / "final_checkpoint.pt"
    torch.save({
        "model_state_dict": policy.state_dict(),
        "feature_names": policy.feature_names,
        "git_commit": commit,
        "episodes": args.episodes,
    }, checkpoint_path)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    eval_rows = []
    eval_event_rows = []
    for index, window in enumerate(eval_windows):
        learned_env = _environment(window, workflows)
        learned, _ = _episode(
            learned_env, policy,
            deterministic=True, gamma=args.gamma, reward_scale=args.reward_scale, collect=False,
        )
        nearest_env = _environment(window, workflows)
        nearest, _ = _episode(
            nearest_env, None,
            deterministic=True, gamma=args.gamma, reward_scale=args.reward_scale, collect=False,
        )
        for label, evaluated_env in (("ppo_final", learned_env), ("nearest", nearest_env)):
            eval_event_rows.extend({
                "eval_window_index": index,
                "policy": label,
                **event,
            } for event in evaluated_env.events)
        eval_rows.append({
            "eval_window_index": index,
            "frame_offset": window["frame_offset"],
            "time_start": window["time_start"],
            "time_end": window["time_end"],
            "ppo_reward": learned["reward"],
            "nearest_reward": nearest["reward"],
            "ppo_completed_requests": learned["completed_requests"],
            "nearest_completed_requests": nearest["completed_requests"],
            "ppo_completed_nodes": learned["completed_nodes"],
            "nearest_completed_nodes": nearest["completed_nodes"],
            "ppo_steps": learned["steps"],
            "nearest_steps": nearest["steps"],
            "ppo_per_request": learned["per_request"],
            "nearest_per_request": nearest["per_request"],
        })
    report = {
        "run_id": run_id,
        "checkpoint_sha256": checkpoint_hash,
        "train_first_32_mean_reward": mean(row["reward"] for row in train_rows[:min(32, len(train_rows))]),
        "train_last_32_mean_reward": mean(row["reward"] for row in train_rows[-min(32, len(train_rows)):]),
        "eval_ppo_mean_reward": mean(row["ppo_reward"] for row in eval_rows),
        "eval_nearest_mean_reward": mean(row["nearest_reward"] for row in eval_rows),
        "eval_paired_mean_reward_delta": mean(row["ppo_reward"] - row["nearest_reward"] for row in eval_rows),
        "eval_ppo_completed_requests": sum(row["ppo_completed_requests"] for row in eval_rows),
        "eval_nearest_completed_requests": sum(row["nearest_completed_requests"] for row in eval_rows),
        "train_rows": train_rows,
        "eval_rows": eval_rows,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "eval_request_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in eval_event_rows:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"train_rows", "eval_rows"}}, indent=2))
    print(f"output_dir={output}")


if __name__ == "__main__":
    main()
