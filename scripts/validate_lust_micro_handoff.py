"""LuST micro-BS 高频切换极速验证。"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.envs.core.predictor_manager import PredictorManager
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.evaluators.main_results_support import build_selected_workflow_states
from src.evaluators.real_sample_support import load_real_mobility_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LuST micro-BS handoff pressure")
    parser.add_argument("--mobility_csv_path", type=str, default="")
    parser.add_argument(
        "--lust_scenario_root",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "mobility" / "LuSTScenario" / "LuSTScenario-master" / "scenario"),
    )
    parser.add_argument(
        "--workflow_csv_path",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"),
    )
    parser.add_argument("--max_mobility_rows", type=int, default=80000)
    parser.add_argument("--window_length", type=int, default=120)
    parser.add_argument("--frame_offset", type=int, default=0)
    parser.add_argument("--rsu_layout", type=str, default="lust_micro")
    parser.add_argument("--window_selector", type=str, default="max_handoff_candidate")
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--total_steps", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mobility_bundle = load_real_mobility_bundle(
        root_dir=ROOT_DIR,
        mobility_source="lust",
        mobility_csv_path=args.mobility_csv_path,
        lust_scenario_root=args.lust_scenario_root,
        max_mobility_rows=args.max_mobility_rows,
        rsu_layout=args.rsu_layout,
        frame_offset=args.frame_offset,
        window_length=args.window_length,
        window_selector=args.window_selector,
        random_seed=args.random_seed,
    )
    workflow_state = build_selected_workflow_states(
        workflow_csv_path=args.workflow_csv_path,
        max_workflows=1,
        workflow_selector=args.workflow_selector,
        min_tasks=args.min_tasks,
        max_tasks=args.max_tasks,
        random_seed=args.random_seed,
    )[0]
    core_env = VecWorkflowCoreEnv(
        mobility_provider=mobility_bundle.provider,
        workflow_state=workflow_state,
        adapter_catalog=AdapterCatalog.from_json(ROOT_DIR / "src" / "data" / "model_catalog" / "sample_model_catalog.json"),
        rsu_states=mobility_bundle.rsu_states,
        predictor_manager=PredictorManager(),
        max_steps=max(args.total_steps + 32, 128),
        mobility_source="lust",
    )
    env = GymVecEnv(core_env=core_env)
    rng = random.Random(args.random_seed)

    env.reset()
    reset_count = 1
    cumulative_handoff_count = 0
    cumulative_handoff_ready_count = 0

    for _ in range(args.total_steps):
        action = rng.randrange(env.action_space.n)
        _, _, terminated, truncated, info = env.step(action)
        metrics = info.get("metrics_protocol", {})
        cumulative_handoff_count += int(metrics.get("handoff_event_count", 0))
        cumulative_handoff_ready_count += int(bool(metrics.get("handoff_ready", False)))
        reached_end = getattr(env.core_env._mobility_provider, "reached_end", lambda: False)()
        if terminated or truncated or reached_end:
            env.reset()
            reset_count += 1

    handoff_ready_ratio = (
        float(cumulative_handoff_ready_count) / float(cumulative_handoff_count)
        if cumulative_handoff_count > 0
        else 0.0
    )
    print("LuST micro validation")
    print(f"window_id: {mobility_bundle.rsu_metadata.get('window_id')}")
    print(f"workflow_id: {workflow_state.workflow_id}")
    print(f"effective_rsu_layout: {mobility_bundle.rsu_metadata.get('effective_rsu_layout')}")
    print(f"rsu_count: {mobility_bundle.rsu_metadata.get('rsu_count')}")
    print(f"coverage_radius: {mobility_bundle.rsu_metadata.get('coverage_radius')}")
    print(f"spacing: {mobility_bundle.rsu_metadata.get('spacing')}")
    print(f"reset_count: {reset_count}")
    print(f"cumulative_handoff_count: {cumulative_handoff_count}")
    print(f"cumulative_handoff_ready_count: {cumulative_handoff_ready_count}")
    print(f"handoff_ready_ratio: {handoff_ready_ratio:.6f}")


if __name__ == "__main__":
    main()
