"""Validate fixed-state action-2 fast-logit partial derivative without env steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def params_sha(network: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))
    from scripts.run_calibrated_workflow_event_aux_abstention_ab import _load_candidate_inputs
    from scripts.run_calibrated_workflow_strong_baselines import _build_learned
    from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv

    probe_root = args.probe_root.resolve()
    manifest = json.loads((probe_root / "probe_manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["total_environment_steps"] != 1200:
        raise RuntimeError("probe manifest incomplete")
    protocol = json.loads((root / "configs/experiment/calibrated_workflow_event_aux_abstention_ab_v1.json").read_text())
    config, splits, _, _ = _load_candidate_inputs(protocol)
    records = []
    for slot in manifest["slots"]:
        row = json.loads((probe_root / slot["name"]).read_text())
        arm = row["arm"]
        seed = int(row["seed"])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        arm_config = {**config, "mechanism_aux_missing_current_event_abstention_enabled": arm == "candidate"}
        agent = _build_learned("sa_ghmappo", seed, arm_config, popart_enabled=False)
        agent.load(row["checkpoint"])
        before = params_sha(agent._network)
        if before != row["parameter_sha256_before"]:
            raise RuntimeError("checkpoint parameter identity drift")
        env = CalibratedContinuousWorkflowEnv(arm_config, splits["train"][0])
        _, info = env.reset()  # no env.step; one fixed public train state per checkpoint
        core = agent.agent
        output = core._forward_policy(info["semantic_state"], run_metadata=info["run_metadata"])
        mask = info["action_mask"]
        fast = output["fast_logits"].detach().clone().requires_grad_(True)

        def p2(fast_logits: torch.Tensor) -> torch.Tensor:
            partial = dict(output)
            partial["fast_logits"] = fast_logits
            scores = core._masked_flat_logits(core._hierarchical_env_action_scores(partial), mask)
            return torch.softmax(scores, dim=-1)[2]

        analytic = float(torch.autograd.grad(p2(fast), fast)[0][1].item())
        with torch.no_grad():
            step = 1e-3
            plus = fast.detach().clone(); plus[1] += step
            minus = fast.detach().clone(); minus[1] -= step
            finite = float(((p2(plus) - p2(minus)) / (2 * step)).item())
        if analytic <= 0 or finite <= 0 or abs(analytic - finite) > 1e-4:
            raise RuntimeError(f"action-2 sensitivity mismatch: {slot['name']} {analytic} {finite}")
        after = params_sha(agent._network)
        if before != after:
            raise RuntimeError("network parameter mutation")
        records.append({"slot": slot["name"], "checkpoint_sha256": row["checkpoint_sha256"],
                        "parameter_sha256_before": before, "parameter_sha256_after": after,
                        "fixed_public_state": str(splits["train"][0]["design_id"]),
                        "action2_legal": bool(mask[2]), "fast_logit1_partial_autograd": analytic,
                        "fast_logit1_partial_central_difference": finite,
                        "absolute_error": abs(analytic - finite)})
    sidecar = {"schema_version": "cscwd_fast_action2_sensitivity_validation_v1",
               "meaning": "post-forward partial derivative holding other output logits fixed; not a full network parameter perturbation",
               "environment_steps": 0, "checkpoint_slots": len(records),
               "maximum_absolute_error": max(item["absolute_error"] for item in records),
               "records": records}
    path = probe_root / "action2_sensitivity_validation.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite existing sensitivity validation")
    path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(path, sha(path))


if __name__ == "__main__":
    main()
