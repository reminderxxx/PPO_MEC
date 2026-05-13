"""运行真实 sample 的最小 dry-run。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.registry import build_agent
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.envs.core.predictor_manager import PredictorManager
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.evaluators.real_sample_support import load_real_mobility_bundle
from src.metrics.recorder import EpisodeRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行真实 mobility + Alibaba workflow 的 sample dry-run")
    parser.add_argument("--mobility_source", choices=["ngsim", "lust"], default="ngsim")
    parser.add_argument("--primary_vehicle_selection", choices=["stable_first", "handoff_pressure"], default="stable_first")
    parser.add_argument("--workflow_source", choices=["alibaba"], default="alibaba")
    parser.add_argument("--mobility_csv_path", type=str, default="", help="真实 mobility CSV 路径")
    parser.add_argument(
        "--lust_scenario_root",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "mobility" / "LuSTScenario" / "LuSTScenario-master" / "scenario"),
        help="当 mobility_source=lust 时使用的 scenario 目录",
    )
    parser.add_argument(
        "--workflow_csv_path",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"),
        help="Alibaba batch_task.csv 路径",
    )
    parser.add_argument("--max_mobility_rows", type=int, default=2000, help="真实 mobility 采样读取上限")
    parser.add_argument("--max_workflows", type=int, default=3, help="用于 workflow 选择的候选 sample 数上限")
    parser.add_argument("--max_steps", type=int, default=12, help="dry-run 的最大交互步数")
    parser.add_argument("--prefetch_validation_window", type=int, default=6, help="predictive prefetch 的验证窗口步数")
    parser.add_argument(
        "--workflow_selector",
        type=str,
        default="ordered",
        help="workflow 选择器，支持 ordered / random / job_name:<job> / index:<zero_based_index>",
    )
    parser.add_argument(
        "--rsu_layout",
        type=str,
        default="auto_dominant_tight",
        help=(
            "RSU 布局，支持 auto / auto_dominant_tight / auto_dominant_wide / "
            "tight_x / tight_y / wide_x / wide_y / custom:axis=x,count=4,coverage=10,spacing=8"
        ),
    )
    parser.add_argument("--frame_offset", type=int, default=0, help="从已加载真实 mobility frames 中选择窗口的起始偏移")
    parser.add_argument("--window_length", type=int, default=24, help="真实 mobility 窗口长度；<=0 表示使用剩余所有帧")
    parser.add_argument(
        "--window_selector",
        type=str,
        default="max_handoff_candidate",
        choices=["ordered", "random", "max_handoff_candidate", "max_axis_crossing"],
        help="真实 mobility 窗口选择策略",
    )
    parser.add_argument("--min_tasks", type=int, default=5, help="Alibaba workflow 最小任务数")
    parser.add_argument("--max_tasks", type=int, default=20, help="Alibaba workflow 最大任务数")
    parser.add_argument("--random_seed", type=int, default=7, help="用于随机 workflow/window 选择")
    parser.add_argument("--agent_name", choices=["sa_ghmappo"], default="sa_ghmappo")
    return parser.parse_args()


def load_selected_workflow_state(args: argparse.Namespace) -> tuple[object, list[str], str]:
    if args.workflow_source != "alibaba":
        raise ValueError(f"不支持的 workflow_source: {args.workflow_source}")
    workflow_csv_path = Path(args.workflow_csv_path)
    if not workflow_csv_path.exists():
        raise FileNotFoundError(
            f"Alibaba batch_task.csv 不存在: {workflow_csv_path}。请检查原始数据路径。"
        )
    builder = WorkflowDatasetBuilder()
    selected_workflow_states = builder.build_selected_alibaba_workflow_states(
        csv_path=workflow_csv_path,
        max_workflows=args.max_workflows,
        workflow_selector=args.workflow_selector,
        min_tasks=args.min_tasks,
        max_tasks=args.max_tasks,
        random_seed=args.random_seed,
    )
    if not selected_workflow_states:
        raise RuntimeError(
            f"Alibaba workflow builder 未返回可用 WorkflowGraphState。输入文件: {workflow_csv_path}"
        )
    workflow_ids = [workflow_state.workflow_id for workflow_state in selected_workflow_states]
    return selected_workflow_states[0], workflow_ids, str(workflow_csv_path)


def main() -> None:
    args = parse_args()
    mainline_label = "LuST(SUMO) + Alibaba" if args.mobility_source == "lust" else "NGSIM + Alibaba"
    mobility_bundle = load_real_mobility_bundle(
        root_dir=ROOT_DIR,
        mobility_source=args.mobility_source,
        mobility_csv_path=args.mobility_csv_path,
        lust_scenario_root=args.lust_scenario_root,
        max_mobility_rows=args.max_mobility_rows,
        rsu_layout=args.rsu_layout,
        frame_offset=args.frame_offset,
        window_length=args.window_length,
        window_selector=args.window_selector,
        random_seed=args.random_seed,
    )
    workflow_state, workflow_candidate_ids, workflow_source_path = load_selected_workflow_state(args)

    catalog_path = ROOT_DIR / "src" / "data" / "model_catalog" / "sample_model_catalog.json"
    adapter_catalog = AdapterCatalog.from_json(catalog_path)
    predictor_manager = PredictorManager()
    recorder = EpisodeRecorder(prefetch_validation_window=args.prefetch_validation_window)
    core_env = VecWorkflowCoreEnv(
        mobility_provider=mobility_bundle.provider,
        workflow_state=workflow_state,
        adapter_catalog=adapter_catalog,
        rsu_states=mobility_bundle.rsu_states,
        predictor_manager=predictor_manager,
        max_steps=max(args.max_steps + 2, 8),
        mobility_source=args.mobility_source,
        primary_vehicle_selection=args.primary_vehicle_selection,
    )
    env = GymVecEnv(core_env=core_env, recorder=recorder)
    agent = build_agent(args.agent_name, random_seed=args.random_seed, deterministic_action=True)

    recorder.start_episode(
        run_metadata={
            "script": "scripts/run_real_sample_dryrun.py",
            "mainline": mainline_label,
            "mobility_source": args.mobility_source,
            "primary_vehicle_selection": args.primary_vehicle_selection,
            "workflow_source": args.workflow_source,
            "agent_name": args.agent_name,
            "workflow_selector": args.workflow_selector,
            "rsu_layout": args.rsu_layout,
            "frame_offset": args.frame_offset,
            "window_length": args.window_length,
            "window_selector": args.window_selector,
            "random_seed": args.random_seed,
            "window_id": mobility_bundle.rsu_metadata.get("window_id"),
            "prefetch_validation_window": args.prefetch_validation_window,
        }
    )
    observation, info = env.reset()

    print("real sample dry-run 初始化完成")
    print(f"mobility_source: {args.mobility_source}")
    print(f"primary_vehicle_selection: {args.primary_vehicle_selection}")
    print(f"mobility_source_path: {mobility_bundle.source_path}")
    print(f"workflow_source: {args.workflow_source}")
    print(f"workflow_source_path: {workflow_source_path}")
    print(f"workflow_selector: {args.workflow_selector}")
    print(f"workflow_candidate_ids: {workflow_candidate_ids}")
    print(f"selected_workflow_id: {workflow_state.workflow_id}")
    print(f"workflow_node_count: {len(workflow_state.nodes)}")
    print(f"loaded_frame_count: {len(mobility_bundle.frames)}")
    print(f"window_id: {mobility_bundle.rsu_metadata.get('window_id')}")
    print(f"window_time_range: {mobility_bundle.rsu_metadata.get('time_index_start')} -> {mobility_bundle.rsu_metadata.get('time_index_end')}")
    print(f"dominant_axis: {mobility_bundle.rsu_metadata.get('dominant_axis')}")
    print(f"chosen_rsu_axis: {mobility_bundle.rsu_metadata.get('chosen_rsu_axis')}")
    print(f"coverage: {mobility_bundle.rsu_metadata.get('coverage_radius')}")
    print(f"spacing: {mobility_bundle.rsu_metadata.get('spacing')}")
    print(f"estimated_association_change_count: {mobility_bundle.rsu_metadata.get('estimated_association_change_count')}")
    print(f"prefetch_validation_window: {args.prefetch_validation_window}")

    terminated = False
    truncated = False
    step_index = 0
    while not terminated and not truncated and step_index < args.max_steps:
        action, action_info = agent.act(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        step_index += 1
        metrics = info.get("metrics_protocol", {})
        print(
            f"step={step_index} action={info.get('action_name')} policy_mode={action_info.get('policy_mode')} "
            f"time_index={metrics.get('time_index')} node={metrics.get('current_node_id')} "
            f"assoc={metrics.get('post_action_associated_rsu_id')} handoff={metrics.get('handoff_event_count')} "
            f"prefetch={metrics.get('predictive_prefetch_requested')} migration={metrics.get('migration_during_handoff')} "
            f"cache_hit={metrics.get('cache_hit')} reward={reward:.3f}"
        )

    summary = recorder.build_summary()
    summary["episode_success"] = bool(summary.get("episode_status", {}).get("completed", False))
    summary["run_info"]["mobility_source_path"] = mobility_bundle.source_path
    summary["run_info"]["workflow_source_path"] = workflow_source_path
    summary["run_info"]["workflow_candidate_ids"] = workflow_candidate_ids
    summary["run_info"]["selected_workflow_id"] = workflow_state.workflow_id
    summary["run_info"]["rsu_metadata"] = mobility_bundle.rsu_metadata
    summary["run_info"].update(
        {
            "window_id": mobility_bundle.rsu_metadata.get("window_id"),
            "prefetch_validation_window": args.prefetch_validation_window,
            "primary_vehicle_selection": args.primary_vehicle_selection,
            "frame_offset": mobility_bundle.rsu_metadata.get("frame_offset"),
            "window_length": mobility_bundle.rsu_metadata.get("window_length"),
            "window_selector": mobility_bundle.rsu_metadata.get("window_selector"),
            "time_index_start": mobility_bundle.rsu_metadata.get("time_index_start"),
            "time_index_end": mobility_bundle.rsu_metadata.get("time_index_end"),
            "dominant_axis": mobility_bundle.rsu_metadata.get("dominant_axis"),
            "chosen_rsu_axis": mobility_bundle.rsu_metadata.get("chosen_rsu_axis"),
            "coverage_radius": mobility_bundle.rsu_metadata.get("coverage_radius"),
            "spacing": mobility_bundle.rsu_metadata.get("spacing"),
        }
    )

    output_dir = ROOT_DIR / "artifacts" / "runs" / datetime.now().strftime("real_sample_dryrun_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("final_summary:")
    print(f"summary_path: {summary_path}")
    print(f"episode_success: {summary['episode_success']}")
    print(f"total_steps: {summary['episode_status']['total_steps']}")
    print(f"total_reward: {summary['reward_breakdown']['total']['sum']:.3f}")
    print(f"handoff_total_count: {summary['handoff_summary']['handoff_total_count']}")
    print(f"predictive_prefetch_request_count: {summary['prefetch_summary']['true_predictive_prefetch_count']}")
    print(f"prefetch_validated_hit_count: {summary['prefetch_summary']['prefetch_validated_hit_count']}")
    print(f"prefetch_expired_miss_count: {summary['prefetch_summary']['prefetch_expired_miss_count']}")
    print(f"migration_during_handoff_count: {summary['handoff_summary']['migration_during_handoff_count']}")
    print(f"handoff_ready_count: {summary['handoff_summary']['handoff_ready_count']}")
    print(f"migration_prepare_count: {summary['handoff_summary']['migration_prepare_count']}")
    print(f"workflow_continuity_rate: {summary['system_metrics']['workflow_continuity_rate']:.3f}")
    print(f"handoff_failure_rate: {summary['system_metrics']['handoff_failure_rate']:.3f}")
    print(f"predictive_prefetch_precision: {summary['system_metrics']['predictive_prefetch_precision']:.3f}")
    print(f"handoff_ready_ratio: {summary['system_metrics']['handoff_ready_ratio']:.3f}")


if __name__ == "__main__":
    main()


