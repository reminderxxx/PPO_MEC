"""Run a config-driven baseline train/eval/benchmark loop."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only on incomplete envs.
    raise SystemExit("PyYAML is required to read configs/experiment/baseline/*.yaml") from exc

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.registry import checkpoint_required_agents, get_algo_spec, list_evaluable_agents


CORE_METRICS = [
    "total_reward",
    "end_to_end_workflow_delay",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "handoff_ready_ratio",
    "adapter_warm_hit_ratio",
    "cross_rsu_cold_start_frequency",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
    "predictive_prefetch_precision",
]
MECHANISM_DIAGNOSTIC_METRICS = [
    "predictive_prefetch_request_count",
    "validated_predictive_prefetch_count",
    "migration_prepare_count",
    "handoff_ready_count",
    "handoff_total_count",
    "mechanism_realization_rate",
]
DETAILED_METRICS = list(dict.fromkeys(CORE_METRICS + MECHANISM_DIAGNOSTIC_METRICS))
WINDOW_CLASSES = ["mechanism_activating", "active_non_mechanism", "idle_or_sparse"]
METRIC_IMPLEMENTATION_STATUS = {
    **{metric_name: "recorder_system_metric" for metric_name in CORE_METRICS},
    "predictive_prefetch_request_count": "recorder_prefetch_count",
    "validated_predictive_prefetch_count": "recorder_prefetch_validation_count",
    "migration_prepare_count": "recorder_handoff_count",
    "handoff_ready_count": "recorder_handoff_count",
    "handoff_total_count": "recorder_handoff_count",
    "mechanism_realization_rate": "derived_from_validated_prefetch_or_handoff_ready_or_migration_during_handoff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiment from configs/experiment/baseline/*.yaml")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "experiment" / "baseline" / "smoke.yaml"))
    parser.add_argument("--agents", nargs="*", default=None, help="Optional subset of agents from the config.")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_benchmark", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"experiment config must be a mapping: {path}")
    return payload


def string_arg(name: str, value: Any) -> list[str]:
    if value is None:
        return []
    return [name, str(value)]


def data_args(config: dict[str, Any], *, include_window_selector: bool = False) -> list[str]:
    data = dict(config.get("data", {}))
    scenario = dict(config.get("scenario", {}))
    args: list[str] = []
    args += string_arg("--mobility_source", data.get("mobility_source", "ngsim"))
    args += string_arg("--mobility_csv_path", data.get("mobility_csv_path", ""))
    args += string_arg(
        "--workflow_csv_path",
        data.get(
            "workflow_csv_path",
            str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"),
        ),
    )
    args += string_arg("--max_mobility_rows", data.get("max_mobility_rows", 500))
    args += string_arg("--max_workflows", data.get("max_workflows", 1))
    args += string_arg("--workflow_selector", data.get("workflow_selector", "ordered"))
    args += string_arg("--rsu_layout", scenario.get("rsu_layout", "auto_dominant_tight"))
    args += string_arg("--frame_offset", scenario.get("frame_offset", 0))
    args += string_arg("--window_length", scenario.get("window_length", 8))
    args += string_arg("--max_steps", scenario.get("max_steps", 3))
    args += string_arg("--min_tasks", data.get("min_tasks", 5))
    args += string_arg("--max_tasks", data.get("max_tasks", 20))
    if include_window_selector:
        args += string_arg("--window_selector", scenario.get("window_selector", "max_handoff_candidate"))
        args += string_arg("--window_count", scenario.get("window_count", 1))
        args += string_arg("--window_scan_stride", scenario.get("window_scan_stride", 2))
        args += string_arg("--window_mode", scenario.get("window_mode", "mixed_informative"))
    return args


def run_command(
    command: list[str],
    *,
    dry_run: bool,
    command_log: list[dict[str, Any]],
    stage: str,
    agent_name: str = "",
    seed: int | None = None,
) -> dict[str, Any]:
    command_text = " ".join(command)
    record = {
        "stage": stage,
        "agent_name": agent_name,
        "seed": seed,
        "command": command_text,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "dry_run": bool(dry_run),
    }
    if dry_run:
        command_log.append(record)
        print(f"dry-run: {command_text}")
        return record

    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    record.update(
        {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )
    command_log.append(record)
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with returncode={completed.returncode}: {command_text}")
    return record


def parse_labeled_path(stdout: str, label: str) -> str:
    prefix = f"{label}:"
    for line in stdout.splitlines():
        if line.strip().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def first_labeled_path(stdout: str, labels: list[str]) -> str:
    for label in labels:
        path = parse_labeled_path(stdout, label)
        if path:
            return path
    return ""


def selected_agents(config: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    agents = list(config.get("agents", []))
    if requested:
        requested_set = set(requested)
        agents = [agent for agent in agents if str(agent.get("name")) in requested_set]
    return [dict(agent) for agent in agents]


def build_checkpoint_args(checkpoints: dict[str, str]) -> list[str]:
    args: list[str] = []
    if checkpoints.get("sa_ghmappo"):
        args += ["--sa_ghmappo_checkpoint_path", checkpoints["sa_ghmappo"]]
    ppo_checkpoint = checkpoints.get("ppo") or checkpoints.get("flat_ppo")
    mappo_checkpoint = checkpoints.get("mappo") or checkpoints.get("flat_mappo")
    if ppo_checkpoint:
        args += ["--flat_ppo_checkpoint_path", ppo_checkpoint]
    if mappo_checkpoint:
        args += ["--flat_mappo_checkpoint_path", mappo_checkpoint]
    return args


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def representative_checkpoint(checkpoints_by_seed: dict[str, str], seeds: list[int]) -> str:
    for seed in seeds:
        checkpoint_path = checkpoints_by_seed.get(str(seed), "")
        if checkpoint_path:
            return checkpoint_path
    return next((path for path in checkpoints_by_seed.values() if path), "")


def init_agent_records(agents: list[dict[str, Any]], seeds: list[int]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    records: dict[str, dict[str, Any]] = {}
    seed_checkpoint_manifest: dict[str, dict[str, str]] = {}
    evaluable_agents = set(list_evaluable_agents())
    for agent_config in agents:
        agent_name = str(agent_config.get("name"))
        spec = get_algo_spec(agent_name)
        checkpoint_path = str(agent_config.get("checkpoint_path", "") or "")
        records[agent_name] = {
            "agent_name": agent_name,
            "role": str(agent_config.get("role", "")),
            "support_level": spec.get("support_level", ""),
            "status": "pending",
            "checkpoint_path": checkpoint_path,
            "checkpoint_paths_by_seed": {},
            "train_artifacts_by_seed": {},
            "eval_artifacts_by_seed": {},
            "notes": spec.get("notes", ""),
        }
        if checkpoint_path:
            seed_checkpoint_manifest[agent_name] = {
                str(seed): checkpoint_path
                for seed in seeds
            }
            records[agent_name]["checkpoint_paths_by_seed"] = dict(seed_checkpoint_manifest[agent_name])
        if agent_name not in evaluable_agents:
            records[agent_name]["status"] = "skipped_skeleton"
    return records, seed_checkpoint_manifest


def train_agent_for_seed(
    *,
    agent_name: str,
    seed: int,
    config: dict[str, Any],
    output_root: Path,
    command_log: list[dict[str, Any]],
    dry_run: bool,
) -> dict[str, str]:
    training = dict(config.get("training", {}))
    if agent_name == "sa_ghmappo":
        command = [
            sys.executable,
            "-B",
            "scripts/train_sa_ghmappo_real_sample.py",
            "--agent_name",
            agent_name,
            "--profile",
            str(training.get("profile", "smoke")),
            "--episodes",
            str(training.get("episodes", 1)),
            "--update_every",
            str(training.get("update_every", 1)),
            "--batch_size",
            str(training.get("batch_size", 8)),
            "--random_seed",
            str(seed),
            "--output_root",
            str(output_root / "training" / "main_agents" / f"seed_{seed}"),
        ]
    else:
        command = [
            sys.executable,
            "-B",
            "scripts/train_algo_pool_real_sample.py",
            "--agent_name",
            agent_name,
            "--profile",
            str(training.get("profile", "smoke")),
            "--episodes",
            str(training.get("episodes", 1)),
            "--update_every",
            str(training.get("update_every", 1)),
            "--batch_size",
            str(training.get("batch_size", 8)),
            "--random_seed",
            str(seed),
            "--output_root",
            str(output_root / "training" / "algo_pool" / f"seed_{seed}"),
        ]
    command += data_args(config)
    result = run_command(
        command,
        dry_run=dry_run,
        command_log=command_log,
        stage="train",
        agent_name=agent_name,
        seed=seed,
    )
    output_dir = parse_labeled_path(result["stdout"], "output_dir")
    summary_path = parse_labeled_path(result["stdout"], "summary_path")
    train_summary_path = str(Path(output_dir) / "train_summary.json") if output_dir else summary_path
    checkpoint_path = first_labeled_path(
        result["stdout"],
        [
            "latest_checkpoint_path",
            "best_by_reward_path",
            "best_by_continuity_path",
            "best_by_mechanism_balanced_path",
        ],
    )
    if agent_name == "sa_ghmappo" and output_dir and not summary_path:
        summary_path = train_summary_path
    return {
        "seed": str(seed),
        "checkpoint_path": checkpoint_path,
        "train_csv_path": parse_labeled_path(result["stdout"], "train_csv_path"),
        "summary_path": summary_path,
        "train_summary_path": train_summary_path,
        "output_dir": output_dir,
    }


def eval_agent_for_seed(
    *,
    agent_name: str,
    seed: int,
    checkpoint_path: str,
    config: dict[str, Any],
    output_root: Path,
    command_log: list[dict[str, Any]],
    dry_run: bool,
) -> dict[str, str]:
    if agent_name == "sa_ghmappo":
        command = [
            sys.executable,
            "-B",
            "scripts/eval_sa_ghmappo_real_sample.py",
            "--agent_name",
            agent_name,
            "--checkpoint_path",
            checkpoint_path,
            "--random_seed",
            str(seed),
            "--output_root",
            str(output_root / "eval" / "main_agents" / f"seed_{seed}"),
        ]
    else:
        command = [
            sys.executable,
            "-B",
            "scripts/eval_algo_pool_real_sample.py",
            "--agent_name",
            agent_name,
            "--random_seed",
            str(seed),
            "--output_root",
            str(output_root / "eval" / "algo_pool" / f"seed_{seed}"),
        ]
        if checkpoint_path:
            command += ["--checkpoint_path", checkpoint_path]
    command += data_args(config)
    result = run_command(
        command,
        dry_run=dry_run,
        command_log=command_log,
        stage="eval",
        agent_name=agent_name,
        seed=seed,
    )
    summary_path = parse_labeled_path(result["stdout"], "summary_path")
    return {
        "seed": str(seed),
        "eval_csv_path": parse_labeled_path(result["stdout"], "eval_csv_path"),
        "summary_path": summary_path,
        "eval_summary_path": summary_path,
        "agent_comparison_path": parse_labeled_path(result["stdout"], "agent_comparison_path"),
    }


def metric_mean_from_aggregate(payload: dict[str, Any], metric_name: str) -> Any:
    return dict(payload.get("metrics", {})).get(metric_name, {}).get("mean", "")


def agent_metric_fallback_means(aggregate: dict[str, Any]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in list(aggregate.get("rows", [])):
        grouped[str(row.get("agent_name", ""))].append(dict(row))
    means_by_agent: dict[str, dict[str, float]] = {}
    for agent_name, rows in grouped.items():
        means_by_agent[agent_name] = {}
        for metric_name in DETAILED_METRICS:
            values = [
                float(row.get(metric_name, 0.0))
                for row in rows
                if row.get(metric_name, "") != ""
            ]
            if values:
                means_by_agent[agent_name][metric_name] = mean(values)
    return means_by_agent


def load_aggregate(aggregate_path: str) -> dict[str, Any]:
    if not aggregate_path:
        return {}
    return json.loads(Path(aggregate_path).read_text(encoding="utf-8"))


def trace_fields_for_seed(
    *,
    agent_name: str,
    seed: int,
    record: dict[str, Any],
    aggregate_path: str,
    benchmark_rows_path: str,
) -> dict[str, Any]:
    seed_key = str(seed)
    checkpoint_paths_by_seed = dict(record.get("checkpoint_paths_by_seed", {}))
    train_artifacts = dict(dict(record.get("train_artifacts_by_seed", {})).get(seed_key, {}))
    eval_artifacts = dict(dict(record.get("eval_artifacts_by_seed", {})).get(seed_key, {}))
    checkpoint_path = checkpoint_paths_by_seed.get(seed_key, record.get("checkpoint_path", ""))
    train_summary_path = train_artifacts.get("train_summary_path", train_artifacts.get("summary_path", ""))
    eval_summary_path = eval_artifacts.get("eval_summary_path", eval_artifacts.get("summary_path", ""))
    return {
        "agent": agent_name,
        "agent_name": agent_name,
        "seed": seed,
        "status": record.get("status", "not_run"),
        "support_level": record.get("support_level", ""),
        "role": record.get("role", ""),
        "checkpoint_path": checkpoint_path,
        "train_summary_path": train_summary_path,
        "train_csv_path": train_artifacts.get("train_csv_path", ""),
        "train_output_dir": train_artifacts.get("output_dir", ""),
        "eval_summary_path": eval_summary_path,
        "eval_csv_path": eval_artifacts.get("eval_csv_path", ""),
        "agent_comparison_path": eval_artifacts.get("agent_comparison_path", ""),
        "benchmark_aggregate_path": aggregate_path,
        "benchmark_rows_path": benchmark_rows_path,
        "notes": record.get("notes", ""),
    }


def metric_mean_from_group(payload: dict[str, Any], metric_name: str) -> Any:
    return dict(payload.get("metrics", {})).get(metric_name, {}).get("mean", "")


def build_comparison_rows(
    aggregate: dict[str, Any],
    agent_records: dict[str, dict[str, Any]],
    aggregate_path: str,
    benchmark_rows_path: str,
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate_by_agent = dict(aggregate.get("aggregate_by_agent", {}))
    aggregate_by_seed_and_agent = dict(aggregate.get("aggregate_by_seed_and_agent", {}))
    fallback_means = agent_metric_fallback_means(aggregate)
    for agent_name, record in sorted(agent_records.items()):
        for seed in seeds:
            row = trace_fields_for_seed(
                agent_name=agent_name,
                seed=seed,
                record=record,
                aggregate_path=aggregate_path,
                benchmark_rows_path=benchmark_rows_path,
            )
            aggregate_payload = aggregate_by_seed_and_agent.get(f"{seed}|{agent_name}", {})
            if not aggregate_payload:
                aggregate_payload = aggregate_by_agent.get(agent_name, {})
            row["benchmark_episode_count"] = aggregate_payload.get("episode_count", "")
            row["train_seed_count"] = len(dict(record.get("train_artifacts_by_seed", {})))
            row["eval_seed_count"] = len(dict(record.get("eval_artifacts_by_seed", {})))
            row["seed_count"] = len(seeds)
            for metric_name in DETAILED_METRICS:
                value = metric_mean_from_group(aggregate_payload, metric_name)
                if value == "":
                    value = fallback_means.get(agent_name, {}).get(metric_name, "")
                row[f"{metric_name}_mean"] = value
            rows.append(row)
    return rows


def build_agent_summary_rows(
    comparison_rows: list[dict[str, Any]],
    agent_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        rows_by_agent[str(row.get("agent_name", ""))].append(row)

    output_rows: list[dict[str, Any]] = []
    for agent_name, seed_rows in sorted(rows_by_agent.items()):
        record = agent_records.get(agent_name, {})
        output_row = {
            "agent": agent_name,
            "agent_name": agent_name,
            "status": record.get("status", "not_run"),
            "support_level": record.get("support_level", ""),
            "role": record.get("role", ""),
            "seed_count": len(seed_rows),
            "checkpoint_paths_by_seed": json.dumps(record.get("checkpoint_paths_by_seed", {}), ensure_ascii=False, sort_keys=True),
            "train_artifacts_by_seed": json.dumps(record.get("train_artifacts_by_seed", {}), ensure_ascii=False, sort_keys=True),
            "eval_artifacts_by_seed": json.dumps(record.get("eval_artifacts_by_seed", {}), ensure_ascii=False, sort_keys=True),
            "notes": record.get("notes", ""),
        }
        for metric_name in DETAILED_METRICS:
            values = [
                float(row[f"{metric_name}_mean"])
                for row in seed_rows
                if row.get(f"{metric_name}_mean", "") != ""
            ]
            output_row[f"{metric_name}_mean"] = mean(values) if values else ""
        output_rows.append(output_row)
    return output_rows


def mean(values: list[float]) -> float:
    return round(fmean(values), 6) if values else 0.0


def build_window_class_rows(
    aggregate: dict[str, Any],
    agents: list[str],
    agent_records: dict[str, dict[str, Any]],
    aggregate_path: str,
    benchmark_rows_path: str,
    seeds: list[int],
) -> list[dict[str, Any]]:
    benchmark_rows = list(aggregate.get("rows", []))
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in benchmark_rows:
        grouped[
            (
                str(row.get("window_class", "unknown")),
                str(row.get("agent_name", "")),
                str(row.get("seed", "")),
            )
        ].append(dict(row))

    output_rows: list[dict[str, Any]] = []
    for window_class in WINDOW_CLASSES:
        for agent_name in sorted(agents):
            record = agent_records.get(agent_name, {})
            for seed in seeds:
                group_rows = grouped.get((window_class, agent_name, str(seed)), [])
                output_row: dict[str, Any] = {
                    "window_class": window_class,
                    **trace_fields_for_seed(
                        agent_name=agent_name,
                        seed=seed,
                        record=record,
                        aggregate_path=aggregate_path,
                        benchmark_rows_path=benchmark_rows_path,
                    ),
                    "episode_count": len(group_rows),
                }
                for metric_name in DETAILED_METRICS:
                    if group_rows:
                        output_row[f"{metric_name}_mean"] = mean(
                            [float(row.get(metric_name, 0.0)) for row in group_rows]
                        )
                    else:
                        output_row[f"{metric_name}_mean"] = ""
                output_rows.append(output_row)
    return output_rows


def window_metric(
    by_window_class: list[dict[str, Any]],
    *,
    window_class: str,
    agent_name: str,
    metric_name: str,
) -> float | None:
    target_key = f"{metric_name}_mean"
    values: list[float] = []
    for row in by_window_class:
        if row.get("window_class") == window_class and row.get("agent_name") == agent_name:
            value = row.get(target_key, "")
            if value != "":
                values.append(float(value))
    if not values:
        return None
    return mean(values)


def delta(
    by_window_class: list[dict[str, Any]],
    *,
    window_class: str,
    agent_name: str,
    reference_agent: str,
    metric_name: str,
) -> float | None:
    value = window_metric(
        by_window_class,
        window_class=window_class,
        agent_name=agent_name,
        metric_name=metric_name,
    )
    reference = window_metric(
        by_window_class,
        window_class=window_class,
        agent_name=reference_agent,
        metric_name=metric_name,
    )
    if value is None or reference is None:
        return None
    return round(value - reference, 6)


def build_mechanism_separability_diagnosis(
    aggregate: dict[str, Any],
    by_window_class: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_by_strata = dict(aggregate.get("selected_window_plan_by_strata", {}))
    diagnosis: dict[str, Any] = {
        "window_class_availability": {
            window_class: len(list(selected_by_strata.get(window_class, [])))
            for window_class in WINDOW_CLASSES
        },
        "popularity_vs_reactive_on_mechanism_activating": {},
        "sa_ghmappo_vs_baselines_on_mechanism_activating": {},
        "active_non_mechanism_extra_cost": {},
        "unseparated_reasons": [],
    }

    for metric_name in [
        "predictive_prefetch_request_count",
        "validated_predictive_prefetch_count",
        "workflow_continuity_rate",
    ]:
        diagnosis["popularity_vs_reactive_on_mechanism_activating"][metric_name] = {
            "popularity_minus_reactive": delta(
                by_window_class,
                window_class="mechanism_activating",
                agent_name="popularity_cache_heuristic",
                reference_agent="reactive_greedy",
                metric_name=metric_name,
            )
        }

    for baseline in ["reactive_greedy", "popularity_cache_heuristic", "ppo", "mappo", "ippo"]:
        diagnosis["sa_ghmappo_vs_baselines_on_mechanism_activating"][baseline] = {
            metric_name: delta(
                by_window_class,
                window_class="mechanism_activating",
                agent_name="sa_ghmappo",
                reference_agent=baseline,
                metric_name=metric_name,
            )
            for metric_name in [
                "total_reward",
                "workflow_continuity_rate",
                "predictive_prefetch_request_count",
                "validated_predictive_prefetch_count",
                "handoff_ready_count",
                "mechanism_realization_rate",
            ]
        }

    for metric_name in [
        "predictive_prefetch_request_count",
        "migration_prepare_count",
        "backhaul_traffic_cost",
        "workflow_continuity_rate",
    ]:
        diagnosis["active_non_mechanism_extra_cost"][metric_name] = {
            "popularity_minus_reactive": delta(
                by_window_class,
                window_class="active_non_mechanism",
                agent_name="popularity_cache_heuristic",
                reference_agent="reactive_greedy",
                metric_name=metric_name,
            ),
            "sa_minus_reactive": delta(
                by_window_class,
                window_class="active_non_mechanism",
                agent_name="sa_ghmappo",
                reference_agent="reactive_greedy",
                metric_name=metric_name,
            ),
        }

    if diagnosis["window_class_availability"].get("mechanism_activating", 0) <= 0:
        diagnosis["unseparated_reasons"].append("mechanism_activating_window_missing")
    if diagnosis["window_class_availability"].get("active_non_mechanism", 0) <= 0:
        diagnosis["unseparated_reasons"].append("active_non_mechanism_window_missing")

    mechanism_prefetch = [
        window_metric(
            by_window_class,
            window_class="mechanism_activating",
            agent_name=agent_name,
            metric_name="predictive_prefetch_request_count",
        )
        for agent_name in ["reactive_greedy", "popularity_cache_heuristic", "sa_ghmappo"]
    ]
    if all(value in {None, 0.0} for value in mechanism_prefetch):
        diagnosis["unseparated_reasons"].append("predictive_prefetch_not_triggered_on_mechanism_windows")

    mechanism_realization = [
        window_metric(
            by_window_class,
            window_class="mechanism_activating",
            agent_name=agent_name,
            metric_name="mechanism_realization_rate",
        )
        for agent_name in [
            "reactive_greedy",
            "popularity_cache_heuristic",
            "sa_ghmappo",
            "ppo",
            "mappo",
            "ippo",
        ]
    ]
    if all(value in {None, 0.0} for value in mechanism_realization):
        diagnosis["unseparated_reasons"].append("mechanism_realization_not_observed")
    return diagnosis


def build_manifest(
    *,
    run_id: str,
    experiment_name: str,
    config_path: Path,
    output_root: Path,
    seeds: list[int],
    agents: dict[str, dict[str, Any]],
    benchmark_aggregate_path: str,
    benchmark_rows_path: str,
    seed_checkpoint_manifest_path: Path,
    command_log_path: Path,
    comparison_summary_path: Path,
    comparison_summary_detailed_path: Path,
    comparison_summary_by_window_class_path: Path,
    agent_seed_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "config_path": str(config_path),
        "output_root": str(output_root),
        "seeds": seeds,
        "agents": agents,
        "benchmark": {
            "aggregate_summary_path": benchmark_aggregate_path,
            "benchmark_rows_path": benchmark_rows_path,
            "seed_checkpoint_manifest_path": str(seed_checkpoint_manifest_path),
        },
        "summaries": {
            "comparison_summary_path": str(comparison_summary_path),
            "comparison_summary_detailed_path": str(comparison_summary_detailed_path),
            "comparison_summary_by_window_class_path": str(comparison_summary_by_window_class_path),
        },
        "agent_seed_runs": agent_seed_runs,
        "command_log_path": str(command_log_path),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    experiment_name = str(config.get("experiment_name", "baseline_experiment"))
    run_id = datetime.now().strftime(f"{experiment_name}_%Y%m%d_%H%M%S")
    output_root = ROOT_DIR / str(config.get("output_root", "artifacts/experiments/baseline")) / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / "experiment_config.yaml")

    command_log: list[dict[str, Any]] = []
    required_checkpoints = checkpoint_required_agents()
    evaluable_agents = set(list_evaluable_agents())
    agents = selected_agents(config, args.agents)
    seeds = [int(seed) for seed in list(config.get("seeds", [7]))]
    agent_records, seed_checkpoint_manifest = init_agent_records(agents, seeds)

    for agent_config in agents:
        agent_name = str(agent_config.get("name"))
        spec = get_algo_spec(agent_name)
        if agent_name not in evaluable_agents:
            continue
        if args.skip_train or not bool(agent_config.get("train", False)):
            if agent_records[agent_name].get("status") == "pending":
                agent_records[agent_name]["status"] = "not_trained"
            continue
        if spec.get("support_level") != "trainable":
            agent_records[agent_name]["status"] = "skipped_non_learning"
            continue

        for seed in seeds:
            train_artifacts = train_agent_for_seed(
                agent_name=agent_name,
                seed=seed,
                config=config,
                output_root=output_root,
                command_log=command_log,
                dry_run=args.dry_run,
            )
            seed_key = str(seed)
            agent_records[agent_name]["train_artifacts_by_seed"][seed_key] = train_artifacts
            checkpoint_path = train_artifacts.get("checkpoint_path", "")
            if checkpoint_path:
                agent_records[agent_name]["checkpoint_paths_by_seed"][seed_key] = checkpoint_path
                seed_checkpoint_manifest.setdefault(agent_name, {})[seed_key] = checkpoint_path
        representative = representative_checkpoint(agent_records[agent_name]["checkpoint_paths_by_seed"], seeds)
        if representative:
            agent_records[agent_name]["checkpoint_path"] = representative
        agent_records[agent_name]["status"] = "trained"

    if not args.skip_eval:
        for agent_config in agents:
            agent_name = str(agent_config.get("name"))
            if agent_name not in evaluable_agents:
                continue
            for seed in seeds:
                seed_key = str(seed)
                checkpoint_path = seed_checkpoint_manifest.get(agent_name, {}).get(seed_key, "")
                if agent_name in required_checkpoints and not checkpoint_path:
                    agent_records[agent_name]["status"] = "missing_checkpoint"
                    continue
                eval_artifacts = eval_agent_for_seed(
                    agent_name=agent_name,
                    seed=seed,
                    checkpoint_path=checkpoint_path,
                    config=config,
                    output_root=output_root,
                    command_log=command_log,
                    dry_run=args.dry_run,
                )
                agent_records[agent_name]["eval_artifacts_by_seed"][seed_key] = eval_artifacts
            if agent_records[agent_name].get("status") in {"pending", "not_trained", "trained"}:
                agent_records[agent_name]["status"] = "evaluated"

    seed_checkpoint_manifest_path = output_root / "seed_checkpoint_manifest.json"
    seed_checkpoint_manifest_path.write_text(
        json.dumps(seed_checkpoint_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    representative_checkpoints = {
        agent_name: representative_checkpoint(dict(record.get("checkpoint_paths_by_seed", {})), seeds)
        for agent_name, record in agent_records.items()
    }

    benchmark_aggregate_path = ""
    benchmark_rows_path = ""
    if not args.skip_benchmark:
        benchmark_agents = [
            str(agent.get("name"))
            for agent in agents
            if str(agent.get("name")) in evaluable_agents
            and agent_records.get(str(agent.get("name")), {}).get("status") != "missing_checkpoint"
        ]
        if benchmark_agents:
            benchmark_command = [
                sys.executable,
                "-B",
                "scripts/benchmark_main_results.py",
                "--agents",
                *benchmark_agents,
                "--seeds",
                *[str(seed) for seed in seeds],
                "--output_root",
                str(output_root / "benchmark"),
                "--seed_checkpoint_manifest_path",
                str(seed_checkpoint_manifest_path),
            ]
            benchmark_command += build_checkpoint_args(representative_checkpoints)
            benchmark_command += data_args(config, include_window_selector=True)
            benchmark_result = run_command(
                benchmark_command,
                dry_run=args.dry_run,
                command_log=command_log,
                stage="benchmark",
            )
            benchmark_aggregate_path = parse_labeled_path(benchmark_result["stdout"], "aggregate_summary_path")
            benchmark_rows_path = parse_labeled_path(benchmark_result["stdout"], "benchmark_rows_path")
            for agent_name in benchmark_agents:
                if agent_records[agent_name].get("status") != "missing_checkpoint":
                    agent_records[agent_name]["status"] = "benchmark_complete"

    aggregate = load_aggregate(benchmark_aggregate_path)
    comparison_rows = build_comparison_rows(
        aggregate,
        agent_records,
        benchmark_aggregate_path,
        benchmark_rows_path,
        seeds,
    )
    agent_summary_rows = build_agent_summary_rows(comparison_rows, agent_records)
    benchmark_agents_for_rows = [
        str(agent.get("name"))
        for agent in agents
        if str(agent.get("name")) in evaluable_agents
    ]
    by_window_class_rows = build_window_class_rows(
        aggregate,
        benchmark_agents_for_rows,
        agent_records,
        benchmark_aggregate_path,
        benchmark_rows_path,
        seeds,
    )
    separability_diagnosis = build_mechanism_separability_diagnosis(aggregate, by_window_class_rows)

    comparison_csv_path = output_root / "comparison_summary.csv"
    comparison_json_path = output_root / "comparison_summary.json"
    detailed_json_path = output_root / "comparison_summary_detailed.json"
    window_class_csv_path = output_root / "comparison_summary_by_window_class.csv"
    command_log_path = output_root / "command_log.json"
    run_manifest_path = output_root / "run_manifest.json"

    write_rows_csv(comparison_csv_path, comparison_rows)
    write_rows_csv(window_class_csv_path, by_window_class_rows)

    summary_payload = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "config_path": str(config_path),
        "output_root": str(output_root),
        "seeds": seeds,
        "comparison_summary_csv": str(comparison_csv_path),
        "comparison_summary_json": str(comparison_json_path),
        "comparison_summary_detailed_json": str(detailed_json_path),
        "comparison_summary_by_window_class_csv": str(window_class_csv_path),
        "run_manifest_path": str(run_manifest_path),
        "benchmark_aggregate_path": benchmark_aggregate_path,
        "benchmark_rows_path": benchmark_rows_path,
        "core_metrics": CORE_METRICS,
        "mechanism_diagnostic_metrics": MECHANISM_DIAGNOSTIC_METRICS,
        "metric_implementation_status": {
            metric_name: METRIC_IMPLEMENTATION_STATUS.get(metric_name, "unknown")
            for metric_name in DETAILED_METRICS
        },
        "agents": agent_records,
        "row_granularity": "agent_seed",
        "rows": comparison_rows,
    }
    detailed_payload = {
        **summary_payload,
        "agent_summary_rows": agent_summary_rows,
        "aggregate_by_agent": aggregate.get("aggregate_by_agent", {}),
        "aggregate_by_seed_and_agent": aggregate.get("aggregate_by_seed_and_agent", {}),
        "selected_window_plan_by_strata": aggregate.get("selected_window_plan_by_strata", {}),
        "window_strata_summary": aggregate.get("window_strata_summary", {}),
        "by_window_class_rows": by_window_class_rows,
        "mechanism_separability_diagnosis": separability_diagnosis,
        "seed_checkpoint_manifest": seed_checkpoint_manifest,
        "command_log": command_log,
    }
    manifest_payload = build_manifest(
        run_id=run_id,
        experiment_name=experiment_name,
        config_path=config_path,
        output_root=output_root,
        seeds=seeds,
        agents=agent_records,
        benchmark_aggregate_path=benchmark_aggregate_path,
        benchmark_rows_path=benchmark_rows_path,
        seed_checkpoint_manifest_path=seed_checkpoint_manifest_path,
        command_log_path=command_log_path,
        comparison_summary_path=comparison_json_path,
        comparison_summary_detailed_path=detailed_json_path,
        comparison_summary_by_window_class_path=window_class_csv_path,
        agent_seed_runs=[
            {
                key: row.get(key, "")
                for key in [
                    "agent",
                    "agent_name",
                    "seed",
                    "status",
                    "support_level",
                    "role",
                    "checkpoint_path",
                    "train_summary_path",
                    "eval_summary_path",
                    "benchmark_aggregate_path",
                    "benchmark_rows_path",
                ]
            }
            for row in comparison_rows
        ],
    )

    comparison_json_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    detailed_json_path.write_text(json.dumps(detailed_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    command_log_path.write_text(json.dumps(command_log, ensure_ascii=False, indent=2), encoding="utf-8")
    run_manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("baseline experiment complete")
    print(f"run_id: {run_id}")
    print(f"output_root: {output_root}")
    print(f"comparison_summary_csv: {comparison_csv_path}")
    print(f"comparison_summary_json: {comparison_json_path}")
    print(f"comparison_summary_detailed_json: {detailed_json_path}")
    print(f"comparison_summary_by_window_class_csv: {window_class_csv_path}")
    print(f"run_manifest_path: {run_manifest_path}")


if __name__ == "__main__":
    main()
