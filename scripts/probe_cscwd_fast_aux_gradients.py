"""Read-only fixed-policy gradient probe for the frozen CSCWD auxiliary plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

SEEDS = (7, 17, 29, 43, 61)
ENDPOINTS = ("selected", "update96")
ARMS = {
    "control": "cscwd_causal_prepared_state_visibility_matched_20261010_v1",
    "candidate": "cscwd_event_aux_abstention_ab_20261010_v1",
}
PLAN_COMMIT = "900c79e"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def params_sha(network: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def vector(grads: tuple[torch.Tensor | None, ...], names: list[str], group: str) -> torch.Tensor:
    prefixes = {
        "encoder": ("encoder.",),
        "slow_actor": ("slow_actor.",),
        "fast_actor": ("fast_actor.",),
        "event_actor": ("event_actor.",),
        "critic": ("central_critic.", "slow_critic.", "fast_critic.", "event_critic."),
    }[group]
    selected = [
        torch.zeros_like(parameter).flatten() if grad is None else grad.flatten()
        for name, parameter, grad in zip(names, _PARAMS, grads)
        if name.startswith(prefixes)
    ]
    return torch.cat(selected) if selected else torch.zeros(0)


_PARAMS: list[torch.nn.Parameter] = []


def grad_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    na = float(torch.linalg.vector_norm(a).item())
    nb = float(torch.linalg.vector_norm(b).item())
    dot = float(torch.dot(a, b).item()) if a.numel() else 0.0
    return {
        "actor_norm": na,
        "other_norm": nb,
        "dot": dot,
        "cosine": dot / (na * nb) if na > 0 and nb > 0 else "UNDEFINED_ZERO_NORM",
    }


def public_bins(semantic: dict, info: dict, bundle_ready, rsu_by_id) -> dict:
    vehicles = list(semantic.get("vehicles", []) or [])
    ident = semantic.get("primary_vehicle_id")
    vehicle = next((v for v in vehicles if str(v.get("vehicle_id")) == str(ident)), vehicles[0] if vehicles else {})
    rsu = rsu_by_id(semantic, vehicle.get("associated_rsu_id"))
    node = semantic.get("current_workflow_node") or {}
    if not rsu or not node:
        readiness = "unknown"
    else:
        readiness = "ready" if bundle_ready(semantic, rsu, node) else "missing"
    context = semantic.get("calibrated_context") or {}
    ids = context.get("required_bundle_ids")
    catalog = context.get("object_catalog")
    residents = rsu.get("typed_resident_object_ids") if rsu else None
    if not isinstance(ids, list) or not isinstance(catalog, dict) or not isinstance(residents, list):
        byte_bin = "unknown"
        missing_bytes = None
    else:
        missing = [ident for ident in ids if ident not in residents]
        try:
            missing_bytes = sum(int(catalog[ident]["resident_bytes"]) for ident in missing)
        except (KeyError, TypeError, ValueError):
            missing_bytes = None
        byte_bin = (
            "unknown" if missing_bytes is None else "zero" if missing_bytes == 0
            else "(0,100MB]" if missing_bytes <= 100_000_000 else ">100MB"
        )
    transition = info.get("transition") or {}
    try:
        remaining = float(transition["deadline_seconds"]) - float(transition["clock_seconds_after"])
        deadline_bin = "<=10s" if remaining <= 10 else "(10,40]s" if remaining <= 40 else ">40s"
    except (KeyError, TypeError, ValueError):
        deadline_bin = "unknown"
    try:
        contact = float(context["contact_budget_seconds"])
        contact_bin = "<=5s" if contact <= 5 else "(5,15]s" if contact <= 15 else ">15s"
    except (KeyError, TypeError, ValueError):
        contact_bin = "unknown"
    return {"readiness": readiness, "missing_bytes": missing_bytes, "missing_bytes_bin": byte_bin,
            "deadline_bin": deadline_bin, "predicted_contact_bin": contact_bin}


def analyze_batch(agent, rows: list[dict], bundle_ready, rsu_by_id) -> dict:
    global _PARAMS
    core = agent.agent
    _PARAMS = list(core._network.parameters())
    names = [name for name, _ in core._network.named_parameters()]
    states = [core._extract_semantic_state(row["decision_info"]) for row in rows]
    metadata = [dict(row["decision_info"].get("run_metadata") or {}) for row in rows]
    masks = [core._extract_action_mask(row["decision_info"]) for row in rows]
    outputs = [core._forward_policy(state, run_metadata=meta) for state, meta in zip(states, metadata)]
    actions = torch.tensor([int(row["action"]) for row in rows], device=core._device)
    old = torch.tensor([float(row["action_info"]["env_action_log_prob"]) for row in rows], device=core._device)
    logits = torch.stack([core._masked_flat_logits(core._hierarchical_env_action_scores(out), mask)
                          for out, mask in zip(outputs, masks)])
    new = torch.distributions.Categorical(logits=logits).log_prob(actions)
    likelihood_error = float(torch.max(torch.abs(new.detach() - old)).item())
    if likelihood_error > 2e-5:
        raise RuntimeError(f"executed likelihood mismatch: {likelihood_error}")
    raw_adv = np.asarray([float(row["advantage"]) for row in rows], dtype=np.float32)
    norm_adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)
    annotations = [core._build_mechanism_guidance_annotation(state, row) for state, row in zip(states, rows)]
    norm_adv *= np.asarray([float(item.get("transition_weight", 1.0)) for item in annotations], dtype=np.float32)
    if any((core._event_prd_advantage_enabled, core._mechanism_credit_prd_enabled, core._net_utility_prd_enabled,
            core._handoff_risk_prd_enabled, core._tail_risk_prd_enabled, core._opportunity_prd_enabled,
            core._idle_execution_prd_enabled, core._env_action_model_critic_enabled,
            core._counterfactual_teacher_prd_enabled)):
        raise RuntimeError("unexpected active advantage extension; stop before approximating")
    base = torch.tensor(norm_adv, device=core._device)
    event = base.clone()  # original default event_advantage=advantage, same normalization/weights
    actor = core._env_action_ppo_coef * core._compute_env_action_ppo_loss(
        batch_outputs=outputs, batch_action_masks=masks, batch_actions=actions,
        old_env_action_log_probs=old, base_advantage=base, event_advantage=event,
        batch_rows=rows)
    returns = torch.tensor([float(row["return"]) for row in rows], device=core._device)
    value = core._value_coef * torch.mean((returns - torch.stack([out["value"] for out in outputs])) ** 2)
    targets = [core._build_mechanism_targets(state) for state in states]
    eligible = [i for i, target in enumerate(targets) if float(target["confidence_weight"]) > 1e-6]
    event_weights = [core._event_auxiliary_supervision_weight(state) for state in states]
    pieces: dict[str, torch.Tensor] = {}
    raw_pieces: dict[str, torch.Tensor] = {}
    for head in ("slow", "fast", "event"):
        terms = [
            nn.functional.cross_entropy(outputs[i][f"{head}_logits"].unsqueeze(0),
                                        torch.tensor([int(targets[i][f"{head}_target"])], device=core._device))
            * float(targets[i]["confidence_weight"]) * (event_weights[i] if head == "event" else 1.0)
            for i in eligible
        ]
        raw = torch.stack(terms).mean() if terms else actor * 0.0
        raw_pieces[head] = raw
        weight = getattr(core, f"_auxiliary_{head}_weight")
        pieces[head] = core._auxiliary_coef * weight * raw
    # Original auxiliary loss also includes temporal consistency; isolate CE only here.
    losses = {"actor": actor, "value": value, **{f"{h}_ce": p for h, p in pieces.items()}}
    grads = {}
    for key, loss in losses.items():
        if loss.requires_grad:
            grads[key] = torch.autograd.grad(loss, _PARAMS, retain_graph=True, allow_unused=True)
        else:
            grads[key] = tuple(None for _ in _PARAMS)
    groups = ("encoder", "slow_actor", "fast_actor", "event_actor", "critic")
    comparisons = {
        other: {group: grad_metrics(vector(grads["actor"], names, group), vector(grads[other], names, group))
                for group in groups}
        for other in ("slow_ce", "fast_ce", "event_ce", "value")
    }
    raw_norms = {}
    for head, raw in raw_pieces.items():
        raw_grads = torch.autograd.grad(raw, _PARAMS, retain_graph=True, allow_unused=True)
        raw_norms[head] = {group: float(torch.linalg.vector_norm(vector(raw_grads, names, group)).item())
                           for group in groups}
    # A pure fast-logit local sensitivity checks whether the fast head can affect executable fallback.
    action2_sensitivity = []
    for out, mask in zip(outputs, masks):
        detached = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value for key, value in out.items()}
        base_prob = torch.softmax(core._masked_flat_logits(core._hierarchical_env_action_scores(detached), mask), -1)[2]
        shifted = dict(detached)
        shifted["fast_logits"] = detached["fast_logits"].clone()
        shifted["fast_logits"][1] += 1e-3
        shifted_prob = torch.softmax(core._masked_flat_logits(core._hierarchical_env_action_scores(shifted), mask), -1)[2]
        action2_sensitivity.append(float(((shifted_prob - base_prob) / 1e-3).item()))
    bins = [public_bins(state, row["decision_info"], bundle_ready, rsu_by_id) for state, row in zip(states, rows)]
    strata = {field: dict(Counter(str(b[field]) for b in bins)) for field in
              ("readiness", "missing_bytes_bin", "deadline_bin", "predicted_contact_bin")}
    # Readiness-local comparison uses the same frozen batch normalization and original full-batch eligible denominator.
    readiness_gradient = {}
    for readiness in ("ready", "missing", "unknown"):
        indices = [i for i, b in enumerate(bins) if b["readiness"] == readiness]
        if not indices:
            continue
        local_actor = core._env_action_ppo_coef * core._compute_env_action_ppo_loss(
            batch_outputs=[outputs[i] for i in indices], batch_action_masks=[masks[i] for i in indices],
            batch_actions=actions[indices], old_env_action_log_probs=old[indices],
            base_advantage=base[indices], event_advantage=event[indices], batch_rows=[rows[i] for i in indices])
        local_fast_terms = [
            nn.functional.cross_entropy(outputs[i]["fast_logits"].unsqueeze(0),
                                        torch.tensor([int(targets[i]["fast_target"])], device=core._device))
            * float(targets[i]["confidence_weight"])
            for i in indices if i in eligible
        ]
        local_fast = (core._auxiliary_coef * core._auxiliary_fast_weight *
                      torch.stack(local_fast_terms).sum() / max(len(eligible), 1)) if local_fast_terms else actor * 0.0
        ga = torch.autograd.grad(local_actor, _PARAMS, retain_graph=True, allow_unused=True)
        gf = torch.autograd.grad(local_fast, _PARAMS, retain_graph=True, allow_unused=True)
        readiness_gradient[readiness] = {"count": len(indices), "eligible": len(local_fast_terms),
                                        "shared_encoder": grad_metrics(vector(ga, names, "encoder"), vector(gf, names, "encoder")),
                                        "fast_actor": grad_metrics(vector(ga, names, "fast_actor"), vector(gf, names, "fast_actor"))}
    return {
        "steps": len(rows), "old_new_logprob_max_abs": likelihood_error,
        "action_counts": dict(Counter(str(int(row["action"])) for row in rows)),
        "target_counts": {head: dict(Counter(str(target[f"{head}_target"]) for target in targets)) for head in ("slow", "fast", "event")},
        "aux_eligible": len(eligible), "event_supervised": sum(int(event_weights[i] > 0) for i in eligible),
        "losses_weighted": {key: float(loss.detach().item()) for key, loss in losses.items()},
        "losses_raw_ce": {key: float(loss.detach().item()) for key, loss in raw_pieces.items()},
        "raw_ce_grad_norms": raw_norms, "comparisons": comparisons,
        "strata_counts": strata, "readiness_gradient": readiness_gradient,
        "action2_logit_sensitivity": {"mean": float(np.mean(action2_sensitivity)),
                                      "positive_count": sum(value > 1e-7 for value in action2_sensitivity),
                                      "zero_count": sum(abs(value) <= 1e-7 for value in action2_sensitivity),
                                      "negative_count": sum(value < -1e-7 for value in action2_sensitivity)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    from scripts.run_calibrated_workflow_event_aux_abstention_ab import _load_candidate_inputs
    from scripts.run_calibrated_workflow_strong_baselines import _build_learned
    from scripts.run_calibrated_workflow_value_normalization_ab import _collect_exact_update_batch
    from src.encoders.calibrated_workflow_features import bundle_ready, rsu_by_id

    protocol_path = source / "configs/experiment/calibrated_workflow_event_aux_abstention_ab_v1.json"
    protocol = json.loads(protocol_path.read_text())
    config, splits, _, _ = _load_candidate_inputs(protocol)
    assert len(splits["train"]) == 12
    assert protocol["training"]["transitions_per_update"] == 60
    assert protocol["unchanged"]["gamma"] == 0.99 and protocol["unchanged"]["gae_lambda"] == 0.95
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite existing probe output")
    output.mkdir(parents=True)
    manifest = {"schema_version": "cscwd_fast_aux_gradient_probe_v1", "plan_commit": PLAN_COMMIT,
                "source_commit": git_commit, "source_root": str(source),
                "protocol_sha256": sha(protocol_path), "status": "running", "slots": []}
    def save() -> None:
        (output / "probe_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    save()
    total_steps = 0
    for arm, run_id in ARMS.items():
        arm_config = {**config, "mechanism_aux_missing_current_event_abstention_enabled": arm == "candidate"}
        for seed in SEEDS:
            for endpoint in ENDPOINTS:
                checkpoint = source / "artifacts/experiments" / run_id / "checkpoints" / f"sa_ghmappo_seed{seed}_{endpoint}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                agent = _build_learned("sa_ghmappo", seed, arm_config, popart_enabled=False)
                agent.load(str(checkpoint))
                before = params_sha(agent._network)
                rows, _, episodes_started, _ = _collect_exact_update_batch(
                    agent=agent, config=arm_config, train_instances=splits["train"],
                    order=list(range(len(splits["train"]))), rng=random.Random(seed),
                    state=None, episodes_started=0, transition_count=60, step_cap=24,
                    gamma=0.99, gae_lambda=0.95)
                total_steps += len(rows)
                if total_steps > 1200:
                    raise RuntimeError("environment-step cap exceeded")
                result = analyze_batch(agent, rows, bundle_ready, rsu_by_id)
                after = params_sha(agent._network)
                if before != after:
                    raise RuntimeError("parameter mutation detected")
                result.update({"arm": arm, "seed": seed, "endpoint": endpoint,
                               "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                               "parameter_sha256_before": before, "parameter_sha256_after": after,
                               "episodes_started": episodes_started})
                name = f"{arm}_seed{seed}_{endpoint}.json"
                path = output / name
                path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
                manifest["slots"].append({"name": name, "sha256": sha(path), "steps": len(rows),
                                          "checkpoint_sha256": sha(checkpoint)})
                manifest["total_environment_steps"] = total_steps
                save()
                print(f"{arm} seed={seed} {endpoint}: {len(rows)} steps, eligible={result['aux_eligible']}", flush=True)
    manifest["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
