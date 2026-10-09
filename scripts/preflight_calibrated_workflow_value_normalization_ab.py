"""Create-only acceptance gate for the authorized PopArt development A/B."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.freeze_calibrated_continuous_workflow_pilot import (  # noqa: E402
    _canonical_sha256,
    _load_experiment_config,
)
from scripts.run_calibrated_workflow_interface_repair import _build_agent  # noqa: E402
from scripts.run_calibrated_workflow_service_reward_alignment import (  # noqa: E402
    _resolved_arm_config,
)
from scripts.run_calibrated_workflow_value_normalization_ab import (  # noqa: E402
    DEFAULT_AUTHORIZATION,
    _collect_exact_update_batch,
    _integrity,
    _load_json,
    _sha256,
    _validate_intervals,
    _write_json,
)


def _git_status() -> list[str]:
    output = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if line.strip()]


def _run_smoke(
    *,
    method: str,
    seed: int,
    config: dict[str, Any],
    train_instances: list[dict[str, Any]],
    popart_enabled: bool,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = _build_agent(
        method,
        seed,
        config,
        agent_overrides={
            "value_normalization_enabled": popart_enabled,
            "popart_min_std": 1.0,
            "popart_max_abs_target": 1.0e20,
        },
    )
    batch, state, episodes_started, episode_rows = _collect_exact_update_batch(
        agent=agent,
        config=config,
        train_instances=train_instances,
        order=list(range(len(train_instances))),
        rng=random.Random(seed),
        state=None,
        episodes_started=0,
        transition_count=60,
        step_cap=24,
        gamma=float(config["training"]["gamma"]),
        gae_lambda=float(config["training"]["gae_lambda"]),
    )
    update = agent.learn(batch)
    return {
        "method": method,
        "popart_enabled": popart_enabled,
        "transitions": len(batch),
        "episodes_started": episodes_started,
        "episodes_completed_or_truncated": len(episode_rows),
        "active_partial_episode": state is not None,
        "optimizer_steps": int(update["optimizer_step_count"]),
        "policy_update_skipped": bool(update["policy_update_skipped"]),
        "value_loss": float(update["value_loss"]),
        "critic_rmse_denormalized_before_update": float(
            update["critic_rmse_denormalized_before_update"]
        ),
        "popart_state": update["popart_state"],
        "all_optimizer_metrics_finite": all(
            all(
                np.isfinite(float(row[key]))
                for key in (
                    "actor_loss",
                    "env_action_ppo_loss",
                    "value_loss",
                    "auxiliary_loss",
                    "pre_clip_total_grad_norm",
                    "global_clip_scale",
                )
            )
            for row in update["optimizer_step_records"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-config", default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    output_root.mkdir(parents=True)
    authorization_path = (ROOT_DIR / args.authorization_config).resolve()
    authorization = _load_json(authorization_path)
    design_path = ROOT_DIR / authorization["design_config"]
    design = _load_json(design_path)
    base_path = ROOT_DIR / design["base_config"]
    manifest_path = ROOT_DIR / design["workload_manifest"]
    base, _ = _load_experiment_config(base_path)
    manifest = _load_json(manifest_path)
    service_config = _resolved_arm_config(
        base,
        {
            "service_aligned_reward": _load_json(
                ROOT_DIR
                / "configs/experiment/calibrated_workflow_service_reward_alignment_v1.json"
            )["service_aligned_reward"]
        },
        design["reward_profile"],
    )
    train_instances = [
        row for row in manifest["instances"] if row["split"] == "train"
    ]
    interval_identity = _validate_intervals(list(manifest["instances"]))
    smoke = [
        _run_smoke(
            method=method,
            seed=7,
            config=service_config,
            train_instances=train_instances,
            popart_enabled=enabled,
        )
        for method in design["development_ab"]["learned_methods"]
        for enabled in (False, True)
    ]
    dirty = _git_status()
    checks = {
        "authorization_enabled": bool(authorization["execution_authorized"]),
        "frozen_design_still_not_authorized": not bool(design["execution_authorized"]),
        "design_hash_match": _sha256(design_path)
        == authorization["design_config_sha256"],
        "base_hash_match": manifest["config_sha256"] == _sha256(base_path),
        "resolved_base_hash_match": manifest["resolved_config_sha256"]
        == _canonical_sha256(base),
        "interval_identity": interval_identity,
        "git_status": dirty,
        "git_clean": not dirty,
        "single_launch": authorization["launch_count"] == 1
        and not authorization["automatic_retry"],
        "aggregate_budget": {
            "cells": authorization["training_cell_count"],
            "environment_steps": authorization["total_environment_steps"],
            "updates": authorization["total_update_opportunities"],
            "optimizer_steps": authorization["total_optimizer_steps"],
        },
        "formal_or_holdout_reads": authorization["formal_or_holdout_reads"],
        "downloads": authorization["downloads"],
        "real_model_generate_calls": authorization["real_model_generate_calls"],
        "smoke": smoke,
    }
    passed = bool(
        checks["authorization_enabled"]
        and checks["frozen_design_still_not_authorized"]
        and checks["design_hash_match"]
        and checks["base_hash_match"]
        and checks["resolved_base_hash_match"]
        and interval_identity["all_identity_fields_present"]
        and interval_identity["all_intervals_pairwise_disjoint"]
        and checks["git_clean"]
        and checks["single_launch"]
        and checks["formal_or_holdout_reads"] == 0
        and checks["downloads"] == 0
        and checks["real_model_generate_calls"] == 0
        and all(
            row["transitions"] == 60
            and row["optimizer_steps"] == 8
            and not row["policy_update_skipped"]
            and row["all_optimizer_metrics_finite"]
            and (row["popart_state"] is not None) == row["popart_enabled"]
            for row in smoke
        )
    )
    _write_json(
        output_root / "preflight.json",
        {
            "schema_version": "calibrated_workflow_value_normalization_ab_preflight_v1",
            "status": "pass" if passed else "fail",
            **checks,
        },
    )
    _write_json(
        output_root / "completion_receipt.json",
        {
            "status": "complete",
            "preflight_status": "pass" if passed else "fail",
            "scientific_training_started": False,
        },
    )
    _integrity(output_root)
    if not passed:
        raise RuntimeError("PopArt A/B preflight failed")
    print(json.dumps({"status": "pass", "output_root": str(output_root)}))


if __name__ == "__main__":
    main()
