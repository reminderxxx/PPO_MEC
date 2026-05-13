"""toy benchmark 批量运行器。"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from src.agents.registry import build_agent
from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.workflow.toy_workflow_generator import ToyWorkflowGenerator
from src.envs.core.predictor_manager import PredictorManager
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import RSUState
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.metrics.recorder import EpisodeRecorder
from src.trainers.on_policy_trainer import OnPolicyTrainer


@dataclass(frozen=True)
class ToyScenarioSpec:
    """toy benchmark 场景配置。"""

    scenario_name: str
    handoff_profile: str
    trajectory_mode: str
    primary_speed: float
    secondary_speed: float
    workflow_node_count: int
    rsu_positions_x: list[float]
    rsu_coverages: list[float]
    cache_plan: dict[str, list[str]]
    primary_start_x: float = 5.0
    secondary_start_x: float = 48.0
    frame_count: int = 7
    oscillation_centers_x: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """运行多 seed、多场景的 toy benchmark。"""

    def __init__(
        self,
        seeds: list[int],
        agent_names: list[str],
        output_root: str | Path,
        max_steps: int = 24,
    ) -> None:
        self._seeds = list(seeds)
        self._agent_names = list(agent_names)
        self._output_root = Path(output_root)
        self._max_steps = max_steps
        self._catalog_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "data"
            / "model_catalog"
            / "sample_model_catalog.json"
        )

    def run(self, scenarios: list[ToyScenarioSpec]) -> dict[str, Any]:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._output_root / run_id
        episode_dir = run_dir / "episodes"
        episode_rows: list[dict[str, Any]] = []

        for scenario in scenarios:
            for seed in self._seeds:
                for agent_name in self._agent_names:
                    summary = self._run_single_episode(
                        scenario=scenario,
                        seed=seed,
                        agent_name=agent_name,
                        episode_dir=episode_dir,
                        run_id=run_id,
                    )
                    episode_rows.append(self._summary_to_row(summary, scenario_name=scenario.scenario_name, seed=seed))

        aggregate_summary = {
            "benchmark_run_id": run_id,
            "seeds": self._seeds,
            "agents": self._agent_names,
            "scenarios": [scenario.to_dict() for scenario in scenarios],
            "episode_count": len(episode_rows),
            "aggregate_by_agent": self._aggregate_rows(episode_rows, group_keys=["agent_name"]),
            "aggregate_by_scenario_and_agent": self._aggregate_rows(
                episode_rows,
                group_keys=["scenario_name", "agent_name"],
            ),
            "episode_rows": episode_rows,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "aggregate_summary.json").write_text(
            json.dumps(aggregate_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_csv(run_dir / "benchmark_rows.csv", episode_rows)
        return aggregate_summary

    def _run_single_episode(
        self,
        scenario: ToyScenarioSpec,
        seed: int,
        agent_name: str,
        episode_dir: Path,
        run_id: str,
    ) -> dict[str, Any]:
        trainer = self._build_trainer(scenario=scenario, seed=seed, agent_name=agent_name)
        summary = trainer.run_episode(
            run_metadata={
                "script": "scripts/benchmark_toy_runs.py",
                "benchmark_run_id": run_id,
                "scenario_name": scenario.scenario_name,
                "agent_name": agent_name,
                "seed": seed,
                "env_name": "toy_vec_env",
                "wrapper_name": "gym_vec_env",
                "predictor_name": "baseline_predictor_v1",
            }
        )
        summary["episode_success"] = bool(summary.get("episode_status", {}).get("completed", False))

        summary_path = (
            episode_dir
            / scenario.scenario_name
            / agent_name
            / f"seed_{seed}.summary.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary["run_info"]["summary_path"] = str(summary_path)
        summary["run_info"]["scenario_name"] = scenario.scenario_name
        summary["run_info"]["seed"] = seed
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    def _build_trainer(
        self,
        scenario: ToyScenarioSpec,
        seed: int,
        agent_name: str,
    ) -> OnPolicyTrainer:
        trajectory_frames = self._build_trajectory_frames(scenario=scenario, seed=seed)
        mobility_provider = ReplayProvider(trajectory_frames=trajectory_frames)
        workflow_state = ToyWorkflowGenerator().generate(
            workflow_id=f"wf_{scenario.scenario_name}_{seed}",
            node_count=scenario.workflow_node_count,
        )
        adapter_catalog = self._build_catalog(scenario.cache_plan)
        predictor_manager = PredictorManager()
        recorder = EpisodeRecorder(prefetch_validation_window=2)
        rsu_states = self._build_rsu_states(scenario)
        core_env = VecWorkflowCoreEnv(
            mobility_provider=mobility_provider,
            workflow_state=workflow_state,
            adapter_catalog=adapter_catalog,
            rsu_states=rsu_states,
            predictor_manager=predictor_manager,
            max_steps=self._max_steps,
        )
        env = GymVecEnv(core_env=core_env, recorder=recorder)
        agent = build_agent(agent_name, random_seed=seed, deterministic_action=True)
        return OnPolicyTrainer(
            env=env,
            agent=agent,
            recorder=recorder,
            max_steps=self._max_steps,
        )

    def _build_catalog(self, cache_plan: dict[str, list[str]]) -> AdapterCatalog:
        base_catalog = AdapterCatalog.from_json(self._catalog_path)
        payload = base_catalog.to_dict()
        existing_profiles = {item["rsu_id"]: item for item in payload["rsu_adapter_caches"]}
        for rsu_id, adapters in cache_plan.items():
            if rsu_id in existing_profiles:
                existing_profiles[rsu_id]["cached_adapter_ids"] = list(adapters)
            else:
                payload["rsu_adapter_caches"].append(
                    {"rsu_id": rsu_id, "cached_adapter_ids": list(adapters)}
                )
        return AdapterCatalog.from_dict(payload)

    def _build_rsu_states(self, scenario: ToyScenarioSpec) -> list[RSUState]:
        rsu_states: list[RSUState] = []
        for index, position_x in enumerate(scenario.rsu_positions_x):
            rsu_states.append(
                RSUState(
                    rsu_id=f"rsu_{chr(ord('a') + index)}",
                    position_x=position_x,
                    position_y=0.0,
                    coverage_radius=scenario.rsu_coverages[index],
                )
            )
        return rsu_states

    def _build_trajectory_frames(
        self,
        scenario: ToyScenarioSpec,
        seed: int,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        primary_positions = self._build_primary_positions(scenario=scenario, rng=rng)
        secondary_positions = self._build_secondary_positions(scenario=scenario, rng=rng)
        frame_count = min(len(primary_positions), len(secondary_positions))
        frames: list[dict[str, Any]] = []
        for frame_index in range(frame_count):
            frames.append(
                {
                    "time_index": frame_index,
                    "vehicles": [
                        {
                            "vehicle_id": "veh_1",
                            "position_x": round(primary_positions[frame_index], 3),
                            "position_y": 0.0,
                            "speed": float(scenario.primary_speed),
                            "base_model_id": "veh_base_v1",
                            "active_workflow_id": "wf_primary",
                        },
                        {
                            "vehicle_id": "veh_2",
                            "position_x": round(secondary_positions[frame_index], 3),
                            "position_y": 4.0,
                            "speed": float(scenario.secondary_speed),
                            "base_model_id": "veh_base_v1",
                            "active_workflow_id": "wf_auxiliary",
                        },
                    ],
                }
            )
        return frames

    def _build_primary_positions(
        self,
        scenario: ToyScenarioSpec,
        rng: random.Random,
    ) -> list[float]:
        if scenario.trajectory_mode == "linear":
            start_x = scenario.primary_start_x + rng.uniform(-1.0, 1.0)
            return [start_x + scenario.primary_speed * index for index in range(scenario.frame_count)]

        if scenario.trajectory_mode == "linear_dense":
            start_x = scenario.primary_start_x + rng.uniform(-0.5, 0.5)
            dense_speed = scenario.primary_speed + rng.uniform(-1.0, 1.0)
            return [start_x + dense_speed * index for index in range(scenario.frame_count)]

        raise ValueError(f"未知 toy trajectory_mode: {scenario.trajectory_mode}")

    def _build_secondary_positions(
        self,
        scenario: ToyScenarioSpec,
        rng: random.Random,
    ) -> list[float]:
        start_x = scenario.secondary_start_x + rng.uniform(-2.0, 2.0)
        return [start_x + scenario.secondary_speed * index for index in range(scenario.frame_count)]

    def _summary_to_row(
        self,
        summary: dict[str, Any],
        scenario_name: str,
        seed: int,
    ) -> dict[str, Any]:
        return {
            "scenario_name": scenario_name,
            "agent_name": summary["run_info"]["agent_name"],
            "seed": seed,
            "episode_success": summary["episode_success"],
            "total_reward": summary["reward_breakdown"]["total"]["sum"],
            "handoff_ready_ratio": summary["system_metrics"]["handoff_ready_ratio"],
            "predictive_prefetch_precision": summary["system_metrics"]["predictive_prefetch_precision"],
            "backhaul_traffic_cost": summary["system_metrics"]["backhaul_traffic_cost"],
            "handoff_total_count": summary["handoff_summary"]["handoff_total_count"],
            "validated_predictive_prefetch_count": summary["validated_predictive_prefetch_count"],
        }

    def _aggregate_rows(
        self,
        rows: list[dict[str, Any]],
        group_keys: list[str],
    ) -> dict[str, Any]:
        metrics = [
            "total_reward",
            "handoff_ready_ratio",
            "predictive_prefetch_precision",
            "backhaul_traffic_cost",
        ]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            group_key = "|".join(str(row[key]) for key in group_keys)
            grouped.setdefault(group_key, []).append(row)

        aggregate: dict[str, Any] = {}
        for group_key, group_rows in grouped.items():
            group_descriptor = {
                key: group_rows[0][key]
                for key in group_keys
            }
            metric_summary = {
                metric: self._metric_stats([float(item[metric]) for item in group_rows])
                for metric in metrics
            }
            aggregate[group_key] = {
                "group": group_descriptor,
                "episode_count": len(group_rows),
                "metrics": metric_summary,
            }
        return aggregate

    def _metric_stats(self, values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(fmean(values), 6),
            "std": round(pstdev(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }

    def _write_csv(self, output_path: Path, rows: list[dict[str, Any]]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "scenario_name",
            "agent_name",
            "seed",
            "episode_success",
            "total_reward",
            "handoff_ready_ratio",
            "predictive_prefetch_precision",
            "backhaul_traffic_cost",
            "handoff_total_count",
            "validated_predictive_prefetch_count",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def build_default_toy_scenarios() -> list[ToyScenarioSpec]:
    """返回 low / medium / high handoff 三类 toy 场景。"""
    return [
        ToyScenarioSpec(
            scenario_name="low_handoff",
            handoff_profile="low",
            trajectory_mode="linear",
            primary_speed=8.0,
            secondary_speed=6.0,
            workflow_node_count=5,
            rsu_positions_x=[0.0, 80.0, 160.0],
            rsu_coverages=[65.0, 65.0, 65.0],
            cache_plan={
                "rsu_a": ["adapter_perception", "adapter_lane"],
                "rsu_b": ["adapter_tracking", "adapter_fusion", "adapter_intent"],
                "rsu_c": ["adapter_control", "adapter_risk", "adapter_coop"],
            },
            primary_start_x=30.0,
            secondary_start_x=92.0,
            frame_count=6,
        ),
        ToyScenarioSpec(
            scenario_name="medium_handoff",
            handoff_profile="medium",
            trajectory_mode="linear_dense",
            primary_speed=18.0,
            secondary_speed=9.0,
            workflow_node_count=6,
            rsu_positions_x=[0.0, 60.0, 110.0],
            rsu_coverages=[40.0, 40.0, 35.0],
            cache_plan={
                "rsu_a": ["adapter_perception"],
                "rsu_b": ["adapter_tracking", "adapter_fusion"],
                "rsu_c": ["adapter_intent", "adapter_control"],
            },
            primary_start_x=5.0,
            secondary_start_x=58.0,
            frame_count=7,
        ),
        ToyScenarioSpec(
            scenario_name="high_handoff",
            handoff_profile="high",
            trajectory_mode="linear_dense",
            primary_speed=20.0,
            secondary_speed=12.0,
            workflow_node_count=8,
            rsu_positions_x=[0.0, 45.0, 90.0, 135.0],
            rsu_coverages=[24.0, 24.0, 24.0, 24.0],
            cache_plan={
                "rsu_a": ["adapter_perception"],
                "rsu_b": ["adapter_lane", "adapter_tracking"],
                "rsu_c": ["adapter_fusion", "adapter_intent"],
                "rsu_d": ["adapter_control", "adapter_risk", "adapter_coop"],
            },
            primary_start_x=4.0,
            secondary_start_x=30.0,
            frame_count=9,
        ),
    ]

