"""Train and compare a fixed diagnostic policy pool on one eight-vehicle replay split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import torch
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.dqn_agent import TraceDrivenMultiRequestQNetwork
from src.agents.mappo_agent import TraceDrivenMultiRequestMAPPO
from src.agents.ppo_agent import TraceDrivenMultiRequestPPO
from scripts.train_eight_vehicle_ppo import (
    _environment, _episode, _ppo_update, _prepare_windows, _public_window, _sha256,
)
from src.envs.core.multi_request_replay_env import ReplayCosts
from src.evaluators.real_sample_support import discover_ngsim_csv


LEARNED = ("trace_ppo", "trace_mappo", "trace_dqn", "trace_ddqn", "trace_dueling_dqn", "trace_dueling_ddqn")
HEURISTICS = ("nearest", "uniform_random", "least_associated", "cache_affinity")
EXCLUDED = {
    "sa_ghmappo": "semantic_discrete_5/controller action, not per-vehicle RSU selection",
    "ppo": "registered single-controller five-action PPO; trace_ppo is a separate policy",
    "dqn/ddqn/dueling_dqn/dueling_ddqn": "registered five-action Q agents; trace_* use a separate four-RSU network",
    "mappo": "controller-level CTDE action, not vehicle-level multi-agent action",
    "qmix": "controller-level value-decomposition action, not vehicle-level action",
    "controller_mat": "controller-level transformer action, not vehicle-level action",
    "dag_offload_drl": "semantic_discrete_5 action, not per-vehicle RSU selection",
    "cache_offload_drl": "semantic_discrete_5 action, not per-vehicle RSU selection",
    "dt_handoff_drl": "semantic_discrete_5 action, not per-vehicle RSU selection",
    "reactive_greedy": "original semantic action contract differs; nearest is a new replay heuristic",
    "popularity_cache_heuristic": "original semantic action contract differs; cache_affinity is a new replay heuristic",
    "reactive_lru/reactive_fifo/reactive_lfu/reactive_aging_lfu/reactive_random": "cache-control action contract differs from per-vehicle RSU selection",
    "td3/sac/maddpg/ippo": "not live trainable agents on the replay action contract",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobility_csv_path", type=Path)
    parser.add_argument("--workflow_csv_path", type=Path,
                        default=ROOT_DIR / "data/raw/workflow/alibaba2018/batch_task.csv")
    parser.add_argument("--max_mobility_rows", type=int, default=250_000)
    parser.add_argument("--window_length", type=int, default=120)
    parser.add_argument("--train_windows", type=int, default=24)
    parser.add_argument("--gap_windows", type=int, default=1)
    parser.add_argument("--eval_windows", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=320)
    parser.add_argument("--update_every", type=int, default=8)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--reward_scale", type=float, default=0.1)
    parser.add_argument("--rsu_coverage_factor", type=float, default=0.75)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 29])
    parser.add_argument("--output_root", type=Path,
                        default=ROOT_DIR / "artifacts/training/eight_vehicle_algorithm_diagnostic")
    return parser.parse_args()


def _heuristic_targets(name: str, observation: dict, window_length: int,
                       rng: random.Random) -> dict[str, str]:
    vehicle_ids, rsu_ids, features, mask = TraceDrivenMultiRequestPPO.encode(
        observation, window_length=window_length)
    targets = {}
    for index, vehicle_id in enumerate(vehicle_ids):
        legal = [j for j, allowed in enumerate(mask[index]) if allowed]
        if name == "uniform_random":
            choice = rng.choice(legal)
        elif name == "nearest":
            choice = min(legal, key=lambda j: (float(features[index, j, 0]), j))
        elif name == "least_associated":
            choice = min(legal, key=lambda j: (float(features[index, j, 4]),
                                                float(features[index, j, 0]), j))
        elif name == "cache_affinity":
            choice = max(legal, key=lambda j: (
                3 * float(features[index, j, 2]) + float(features[index, j, 3])
                + float(features[index, j, 7]) - float(features[index, j, 0])
                - 0.1 * float(features[index, j, 4]), -j))
        else:
            raise ValueError(name)
        targets[vehicle_id] = rsu_ids[choice]
    return targets


def _heuristic_episode(env, name: str, rng: random.Random) -> dict:
    observation = env.reset()
    total = 0.0
    while True:
        observation, reward, done, _ = env.step(
            _heuristic_targets(name, observation, len(env.frames), rng))
        total += reward
        if done:
            break
    return {**env.summary(), "reward": round(total, 6)}


def _mappo_episode(env, model: TraceDrivenMultiRequestMAPPO, *, deterministic: bool,
                   gamma: float, reward_scale: float, collect: bool) -> tuple[dict, list[dict]]:
    observation = env.reset()
    total = 0.0
    steps = []
    rewards = []
    while True:
        vehicle_ids, rsu_ids, features, mask = model.encode(observation, window_length=len(env.frames))
        targets = {}
        samples = []
        if vehicle_ids:
            context = features.mean(dim=0).flatten().expand(len(vehicle_ids), -1)
            actions, logprobs, values = model.act(features, mask, context,
                                                  deterministic=deterministic)
            targets = {vehicle_id: rsu_ids[int(action)]
                       for vehicle_id, action in zip(vehicle_ids, actions)}
            if collect:
                samples = [{"features": features[i], "mask": mask[i],
                            "context": context[i], "action": actions[i],
                            "old_logprob": logprobs[i], "old_value": values[i]}
                           for i in range(len(vehicle_ids))]
        observation, reward, done, _ = env.step(targets)
        total += reward
        rewards.append(reward * reward_scale)
        steps.append(samples)
        if done:
            break
    collected = []
    if collect:
        discounted = 0.0
        for i in reversed(range(len(steps))):
            discounted = rewards[i] + gamma * discounted
            for sample in steps[i]:
                sample["return"] = torch.tensor(discounted, dtype=torch.float32)
                collected.append(sample)
    return {**env.summary(), "reward": round(total, 6)}, collected


def _mappo_update(model, optimizer, samples, args) -> dict:
    if not samples:
        raise RuntimeError("no MAPPO decisions collected")
    features = torch.stack([item["features"] for item in samples])
    masks = torch.stack([item["mask"] for item in samples])
    contexts = torch.stack([item["context"] for item in samples])
    actions = torch.stack([item["action"] for item in samples])
    old_logprobs = torch.stack([item["old_logprob"] for item in samples])
    returns = torch.stack([item["return"] for item in samples])
    old_values = torch.stack([item["old_value"] for item in samples])
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    last = {}
    for _ in range(args.ppo_epochs):
        for ids in torch.randperm(len(samples)).split(args.minibatch_size):
            distribution, values = model.distribution_and_value(features[ids], masks[ids], contexts[ids])
            ratio = (distribution.log_prob(actions[ids]) - old_logprobs[ids]).exp()
            surrogate = ratio * advantages[ids]
            clipped = ratio.clamp(1 - args.clip_epsilon, 1 + args.clip_epsilon) * advantages[ids]
            actor_loss = -torch.minimum(surrogate, clipped).mean()
            critic_loss = (values - returns[ids]).square().mean()
            entropy = distribution.entropy().mean()
            loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last = {"actor_loss": float(actor_loss.detach()),
                    "critic_loss": float(critic_loss.detach()),
                    "entropy": float(entropy.detach()), "samples": len(samples)}
    return last


def _q_episode(env, model: TraceDrivenMultiRequestQNetwork, *, epsilon: float,
               rng: random.Random, collect: bool) -> tuple[dict, list[tuple]]:
    observation = env.reset()
    transitions = []
    total = 0.0
    while True:
        vehicle_ids, rsu_ids, features, mask = TraceDrivenMultiRequestPPO.encode(
            observation, window_length=len(env.frames))
        targets = {}
        decisions = []
        if vehicle_ids:
            with torch.no_grad():
                q_values = model(features, mask)
            for i, vehicle_id in enumerate(vehicle_ids):
                legal = torch.where(mask[i])[0].tolist()
                action = rng.choice(legal) if rng.random() < epsilon else int(q_values[i].argmax())
                targets[vehicle_id] = rsu_ids[action]
                if collect:
                    decisions.append((vehicle_id, features[i], mask[i], action))
        next_observation, reward, done, _ = env.step(targets)
        total += reward
        if collect and decisions:
            next_ids, _, next_features, next_mask = TraceDrivenMultiRequestPPO.encode(
                next_observation, window_length=len(env.frames)) if not done else ([], [], None, None)
            next_index = {vehicle_id: i for i, vehicle_id in enumerate(next_ids)}
            for vehicle_id, state, legal_mask, action in decisions:
                if vehicle_id in next_index:
                    j = next_index[vehicle_id]
                    successor = next_features[j]
                    successor_mask = next_mask[j]
                    terminal = False
                else:
                    successor = torch.zeros_like(state)
                    successor_mask = torch.tensor([True, False, False, False])
                    terminal = True
                transitions.append((state, legal_mask, action, reward * 0.1,
                                    successor, successor_mask, terminal))
        if done:
            break
        observation = next_observation
    return {**env.summary(), "reward": round(total, 6)}, transitions


def _q_update(online, target, optimizer, replay, *, double: bool,
              gamma: float, rng: random.Random) -> float | None:
    if len(replay) < 256:
        return None
    losses = []
    for _ in range(4):
        batch = rng.sample(list(replay), min(256, len(replay)))
        states = torch.stack([item[0] for item in batch])
        masks = torch.stack([item[1] for item in batch])
        actions = torch.tensor([item[2] for item in batch], dtype=torch.long)
        rewards = torch.tensor([item[3] for item in batch], dtype=torch.float32)
        next_states = torch.stack([item[4] for item in batch])
        next_masks = torch.stack([item[5] for item in batch])
        terminals = torch.tensor([item[6] for item in batch], dtype=torch.float32)
        selected = online(states, masks).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_target = target(next_states, next_masks)
            if double:
                next_actions = online(next_states, next_masks).argmax(1)
                next_value = next_target.gather(1, next_actions[:, None]).squeeze(1)
            else:
                next_value = next_target.max(1).values
            td_target = rewards + gamma * (1 - terminals) * next_value
        loss = nn.functional.smooth_l1_loss(selected, td_target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return mean(losses)


def _train(name: str, seed: int, args, windows, workflows, output: Path) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_windows = windows[:args.train_windows]
    episode_windows = [rng.randrange(len(train_windows)) for _ in range(args.episodes)]
    rows = []
    if name in ("trace_ppo", "trace_mappo"):
        model = TraceDrivenMultiRequestPPO() if name == "trace_ppo" else TraceDrivenMultiRequestMAPPO()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        batch = []
        for episode, window_index in enumerate(episode_windows, 1):
            if name == "trace_ppo":
                result, samples = _episode(_environment(train_windows[window_index], workflows), model,
                                           deterministic=False, gamma=args.gamma,
                                           reward_scale=args.reward_scale, collect=True)
            else:
                result, samples = _mappo_episode(_environment(train_windows[window_index], workflows), model,
                                                  deterministic=False, gamma=args.gamma,
                                                  reward_scale=args.reward_scale, collect=True)
            batch.extend(samples)
            row = {"episode": episode, "window_index": window_index, "reward": result["reward"],
                   "completed_requests": result["completed_requests"]}
            if episode % args.update_every == 0 or episode == args.episodes:
                row["update"] = (_ppo_update(model, optimizer, batch, args) if name == "trace_ppo"
                                 else _mappo_update(model, optimizer, batch, args))
                batch = []
            rows.append(row)
    else:
        double = name in ("trace_ddqn", "trace_dueling_ddqn")
        dueling = name in ("trace_dueling_dqn", "trace_dueling_ddqn")
        model = TraceDrivenMultiRequestQNetwork(dueling=dueling)
        target = TraceDrivenMultiRequestQNetwork(dueling=dueling)
        target.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        replay = deque(maxlen=4096)
        for episode, window_index in enumerate(episode_windows, 1):
            epsilon = 0.25 + (0.02 - 0.25) * min((episode - 1) / max(args.episodes - 1, 1), 1)
            result, transitions = _q_episode(_environment(train_windows[window_index], workflows),
                                             model, epsilon=epsilon, rng=rng, collect=True)
            replay.extend(transitions)
            row = {"episode": episode, "window_index": window_index, "reward": result["reward"],
                   "completed_requests": result["completed_requests"], "epsilon": epsilon,
                   "replay_size": len(replay)}
            if episode % args.update_every == 0 or episode == args.episodes:
                row["update_loss"] = _q_update(model, target, optimizer, replay,
                                               double=double, gamma=args.gamma, rng=rng)
                target.load_state_dict(model.state_dict())
            rows.append(row)
    checkpoint = output / f"{name}_seed{seed}.pt"
    torch.save({"model_state_dict": model.state_dict(), "algorithm": name,
                "seed": seed, "episodes": args.episodes}, checkpoint)
    return {"model": model, "train_rows": rows, "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}


def main() -> None:
    args = parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be nonempty and distinct")
    torch.set_num_threads(1)
    windows, workflows = _prepare_windows(args)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True).strip()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id, "experiment_type": "eight_vehicle_algorithm_pool_diagnostic",
        "formal_protocol": False, "git_commit": commit, "command": sys.argv,
        "algorithm_ids": list(LEARNED + HEURISTICS), "excluded_original_agents": EXCLUDED,
        "original_registry_agents_reused": False,
        "action_contract": "one covered RSU target per active vehicle per frame",
        "reward_contract": "completed_nodes - 0.01 * unfinished_requests per frame",
        "shared_observation_features": list(TraceDrivenMultiRequestPPO.feature_names),
        "training_episodes_per_learned_algorithm_per_seed": args.episodes,
        "seeds": args.seeds, "train_window_count": args.train_windows,
        "eval_window_count": args.eval_windows, "gap_eligible_windows": args.gap_windows,
        "window_length": args.window_length, "max_mobility_rows": args.max_mobility_rows,
        "rsu_coverage_factor": args.rsu_coverage_factor,
        "simulated_costs": vars(ReplayCosts()), "simultaneous_arrival_assumed": True,
        "physical_rsus": False, "model_inference_executed": False,
        "rsu_and_link_costs_measured": False, "rsu_layout_uses_full_window_geometry": True,
        "mobility_csv_path": str((args.mobility_csv_path or discover_ngsim_csv(ROOT_DIR)).resolve()),
        "workflow_csv_path": str(args.workflow_csv_path.resolve()),
        "workflow_ids": [workflow.workflow_id for workflow in workflows],
        "workflow_sha256": _sha256([workflow.to_dict() for workflow in workflows]),
        "train_windows": [_public_window(window) for window in windows[:args.train_windows]],
        "eval_windows": [_public_window(window) for window in windows[args.train_windows:]],
        "hyperparameters": {"learning_rate": args.learning_rate, "gamma": args.gamma,
                            "update_every": args.update_every, "ppo_epochs": args.ppo_epochs,
                            "ppo_minibatch_size": args.minibatch_size, "q_replay_capacity": 4096,
                            "q_batch_size": 256, "q_gradient_steps_per_update": 4,
                            "q_target_sync_every_episodes": args.update_every,
                            "q_epsilon_start": 0.25, "q_epsilon_end": 0.02,
                            "reward_scale_for_gradient": args.reward_scale},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    eval_rows = []
    train_rows = []
    checkpoints = []
    with (output / "eval_request_events.jsonl").open("w", encoding="utf-8") as event_file:
        for seed in args.seeds:
            for name in LEARNED + HEURISTICS:
                trained = None
                if name in LEARNED:
                    trained = _train(name, seed, args, windows, workflows, output)
                    train_rows.extend({"algorithm": name, "seed": seed, **row}
                                      for row in trained["train_rows"])
                    checkpoints.append({"algorithm": name, "seed": seed,
                                        "path": trained["checkpoint"],
                                        "sha256": trained["checkpoint_sha256"]})
                for index, window in enumerate(windows[args.train_windows:]):
                    env = _environment(window, workflows)
                    if name == "trace_ppo":
                        result, _ = _episode(env, trained["model"], deterministic=True,
                                             gamma=args.gamma, reward_scale=args.reward_scale,
                                             collect=False)
                    elif name == "trace_mappo":
                        result, _ = _mappo_episode(env, trained["model"], deterministic=True,
                                                    gamma=args.gamma, reward_scale=args.reward_scale,
                                                    collect=False)
                    elif name in LEARNED:
                        result, _ = _q_episode(env, trained["model"], epsilon=0.0,
                                               rng=random.Random(seed), collect=False)
                    else:
                        result = _heuristic_episode(env, name, random.Random(seed + index * 1009))
                    eval_rows.append({"algorithm": name, "seed": seed, "eval_window_index": index,
                                      "frame_offset": window["frame_offset"],
                                      "time_start": window["time_start"],
                                      "time_end": window["time_end"],
                                      "reward": result["reward"],
                                      "completed_requests": result["completed_requests"],
                                      "completed_nodes": result["completed_nodes"],
                                      "steps": result["steps"], "per_request": result["per_request"],
                                      "status_counts": result["status_counts"]})
                    for event in env.events:
                        event_file.write(json.dumps({"algorithm": name, "seed": seed,
                                                     "eval_window_index": index, **event},
                                                    ensure_ascii=False) + "\n")
                print(f"seed={seed} algorithm={name} evaluated={args.eval_windows}", flush=True)
    summary = []
    for name in LEARNED + HEURISTICS:
        rows = [row for row in eval_rows if row["algorithm"] == name]
        summary.append({"algorithm": name, "mean_reward": mean(row["reward"] for row in rows),
                        "completed_requests": sum(row["completed_requests"] for row in rows),
                        "request_count": len(rows) * 8,
                        "mean_completed_nodes": mean(row["completed_nodes"] for row in rows),
                        "evaluations": len(rows)})
    report = {"run_id": run_id, "summary": summary, "checkpoints": checkpoints,
              "train_rows": train_rows, "eval_rows": eval_rows}
    (output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "summary": summary}, ensure_ascii=False, indent=2))
    print(f"output_dir={output}")


if __name__ == "__main__":
    main()
