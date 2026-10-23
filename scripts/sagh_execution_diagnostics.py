"""Local preflight diagnostics; no reward shaping, optimizer steps or env stepping."""
from __future__ import annotations

import copy
import math
import random
from contextlib import contextmanager

import numpy as np
import torch

from diagnose_sagh_flat_reward import STAGES


@contextmanager
def isolated_rng():
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    try:
        yield
    finally:
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.random.set_rng_state(before[2])


def install_gate_observer(agent):
    if not getattr(agent, "_execution_observer_installed", False):
        active = {"enabled": False, "stages": [], "mask": None}
        def snapshot(name, output):
            scores = agent._masked_flat_logits(agent._hierarchical_env_action_scores(output), active["mask"])
            active["stages"].append({"stage": name,
                "logits": {key: output[key+"_logits"].detach().cpu().tolist() for key in ("slow", "fast", "event")},
                "env_probs": scores.softmax(-1).detach().cpu().tolist(), "env_argmax": int(scores.argmax())})
        forward = agent._network.forward_single
        def observed_forward(*args, **kwargs):
            result = forward(*args, **kwargs)
            if active["enabled"]: snapshot("network", result)
            return result
        agent._network.forward_single = observed_forward
        for name in STAGES:
            original = getattr(agent, name)
            def observed(*args, _original=original, _name=name, **kwargs):
                result = _original(*args, **kwargs)
                if active["enabled"]: snapshot(_name, result)
                return result
            setattr(agent, name, observed)
        original_act = agent.act
        def observed_act(observation, info=None):
            active.update(enabled=True, stages=[], mask=agent._extract_action_mask(info))
            try:
                action, payload = original_act(observation, info)
                return action, dict(payload, checkpoint_probe=copy.deepcopy(active["stages"]))
            finally:
                active["enabled"] = False
        agent.act = observed_act
        agent._execution_observer_installed = True
    return agent


def step_record(row, metadata):
    reward = dict(row["env_info"]["reward_dict"])
    positive = sum(float(reward[k]) for k in ("positive_offset", "service_reward", "continuity_bonus", "mechanism_exploration_bonus"))
    negative = sum(float(reward[k]) for k in ("delay_penalty", "cache_miss_penalty", "migration_cost", "constraint_penalty"))
    if not math.isclose(positive-negative, row["reward"], abs_tol=1e-6) or not math.isclose(reward["total"], row["reward"], abs_tol=1e-6):
        raise ValueError("Native step reward ledger mismatch")
    info = row["action_info"]
    return {"schema": "sagh_execution_step_v1", "metadata": metadata,
            "semantic_state": copy.deepcopy(row["decision_info"]["semantic_state"]),
            "action_mask": row["decision_info"]["action_mask"],
            "native_reward": float(row["reward"]), "reward_breakdown": reward,
            "metrics_protocol": copy.deepcopy(row["env_info"].get("metrics_protocol", {})),
            "actual_action": int(row["action"]), "action_info": copy.deepcopy(info),
            "stages": copy.deepcopy(info.get("checkpoint_probe", [])),
            "terminated": bool(row.get("terminated", False)), "truncated": bool(row.get("truncated", False))}


def enumerate_executed_distribution(agent, info):
    """Enumerate <=5x4 latent paths at a fixed state using the actual act mapper.

    Called only on predeclared first-state probes, not every training step.
    The base/option selection methods are forced; every production guard and
    final legality check still executes. No environment or optimizer call occurs.
    """
    if agent._execution_mapping_contract != "shared_train_eval_v1":
        raise ValueError("Enumeration requires the shared mapping contract")
    mask = agent._extract_action_mask(info)
    if not mask or not any(mask):
        raise ValueError("Cannot enumerate an empty legal action set")
    names = ("_sample_actions", "_sample_shared_option")
    originals = {name: (name in agent.__dict__, getattr(agent, name)) for name in names}
    forced = {"base": None, "option": None}
    def base_selection(output, deterministic, action_mask):
        base = forced["base"]
        selected = agent._head_targets_for_env_action(base)
        logp, entropy, payload = agent._selected_action_statistics(
            policy_output=output, selected_actions=selected, action_mask=action_mask)
        return selected, logp, entropy, payload, {
            "raw_env_action": base, "raw_head_actions": selected,
            "projection_applied": False, "invalid_attempt_count": 0}
    def option_selection(distribution, deterministic):
        index = forced["option"]
        return distribution.logits.argmax() if index is None else torch.tensor(index, device=distribution.logits.device)
    paths, marginal = [], [0.0]*5
    try:
        agent._sample_actions = base_selection
        agent._sample_shared_option = option_selection
        with isolated_rng():
            for base, legal in enumerate(mask):
                if not legal:
                    continue
                forced.update(base=base, option=None)
                _, probe = agent.act(None, copy.deepcopy(info))
                base_mass = float(probe["sampled_base_probs"][base])
                option = probe["option_gate"]
                valid_options = [i for i, legal in enumerate(option["option_mask"]) if legal] if option.get("enabled") else [None]
                for index in valid_options:
                    forced["option"] = index
                    action, diagnostic = agent.act(None, copy.deepcopy(info))
                    option = diagnostic["option_gate"]
                    conditional_mass = float(option["option_probs"][index]) if index is not None else 1.0
                    mass = base_mass*conditional_mass
                    if mass > 1e-30 and not math.isclose(math.log(mass), diagnostic["log_prob"], abs_tol=2e-5):
                        raise ValueError("Enumerated sampled-path log probability mismatch")
                    if not mask[action]:
                        raise ValueError("Masked final action in enumerated mapping")
                    marginal[action] += mass
                    paths.append({"sampled_base": base, "sampled_option": index,
                                  "executed_option": option.get("executed_option_action"),
                                  "executed_action": action, "mass": mass,
                                  "sampled_path_log_prob": diagnostic["log_prob"],
                                  "fallback": diagnostic["idle_popularity_fallback"],
                                  "selection_reason": option.get("selection_reason")})
    finally:
        for name, (owned, original) in originals.items():
            if owned:
                setattr(agent, name, original)
            else:
                delattr(agent, name)
    if not math.isclose(sum(marginal), 1.0, abs_tol=2e-5):
        raise ValueError("Executed marginal probability mass does not sum to one")
    entropy = -sum(p*math.log(p) for p in marginal if p > 0)
    return {"schema": "executed_action_marginal_probe_v1", "scope": "exact_enumerated_fixed_state_probe",
            "execution_mapping_contract": agent._execution_mapping_contract,
            "env_action_probs_scope": "final_executed_action_marginal_at_probe_state",
            "executed_action_probs": marginal, "executed_action_entropy": entropy,
            "path_count": len(paths), "paths": paths,
            "uncontrollable": max(marginal) >= 1.0-2e-5}
